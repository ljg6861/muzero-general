"""Hybrid Compressed Memory: Entity-Relation Graph + Passage Store

Design Goals:
- Separate crisp factual structure (entities, relations, typed edges) from verbose language evidence.
- Provide sub-GB footprint via compact integer structures + optional vector index compression.
- Offer fast two-stage retrieval:
  1. Entity linking / hop over ER graph (entity -> relation -> object, or inverse).
  2. Evidence passage retrieval + rerank for provenance (and potential citation).

Key Components:
- EntityTable: maps canonical entity string -> integer id (uint32) and stores quantized embedding.
- RelationTable: maps relation name -> relation id + small embedding.
- ER Adjacency: for each (head entity id) store contiguous slices of (relation_id, tail_entity_id).
  Stored as uint32 packed arrays to minimize memory; offsets array indexes into a single flat edge array.
- PassageIndex: optional FAISS (IVF-PQ/HNSW-PQ) or pure PyTorch approximate search fallback.

This module avoids external dependencies at import time: FAISS use is lazy and optional.
Includes versioned snapshots and verifier guard capabilities.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable, Sequence
from collections import defaultdict, Counter
import random
import threading
import math
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    _FAISS_AVAILABLE = False


@dataclass
class MemorySnapshotMeta:
    """Metadata for memory snapshots."""
    version: int
    created_ts: float
    path: str



# ----------------------------- Entity / Relation Tables -----------------------------

class QuantizedEmbedding:
    """Simple product quantization wrapper for embeddings.
    If FAISS not available, falls back to storing fp16 vectors directly.
    """
    def __init__(self, dim: int, nsubq: int = 4, codebook_bits: int = 8):
        self.dim = dim
        self.nsubq = nsubq
        self.codebook_bits = codebook_bits
        self.subdim = dim // nsubq
        self.trained = False
        self.codebooks: Optional[torch.Tensor] = None  # (nsubq, 2**codebook_bits, subdim)
        self.codes: List[torch.Tensor] = []  # list of uint8 code arrays per added vector
        self.fp_store: List[torch.Tensor] = []  # fallback store

    def _train_codebooks(self, data: torch.Tensor):
        # K-means per subspace
        n = data.size(0)
        assert n > 0
        codebook_size = 2 ** self.codebook_bits
        codebooks = []
        for si in range(self.nsubq):
            sub = data[:, si*self.subdim:(si+1)*self.subdim]
            # simple k-means++ init
            idx = torch.randint(0, sub.size(0), (codebook_size,))
            centers = sub[idx].clone()
            for _ in range(8):
                dists = (sub.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1)
                assign = dists.argmin(-1)
                for k in range(codebook_size):
                    mask = assign == k
                    if mask.any():
                        centers[k] = sub[mask].mean(0)
            codebooks.append(centers)
        self.codebooks = torch.stack(codebooks)  # (nsubq, K, subdim)
        self.trained = True

    def add(self, vectors: torch.Tensor):
        if vectors.dtype != torch.float32:
            vectors = vectors.float()
        if not self.trained and vectors.size(0) >= 32:
            self._train_codebooks(vectors)
        if self.trained:
            # encode
            codes_sub = []
            for si in range(self.nsubq):
                sub = vectors[:, si*self.subdim:(si+1)*self.subdim]
                centers = self.codebooks[si]
                dists = (sub.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1)
                assign = dists.argmin(-1)
                codes_sub.append(assign.to(torch.uint8))
            codes = torch.stack(codes_sub, dim=1)  # (nvec, nsubq)
            self.codes.append(codes.cpu())
        else:
            self.fp_store.append(vectors.half().cpu())

    def reconstruct(self) -> torch.Tensor:
        if self.trained:
            all_codes = torch.cat(self.codes, dim=0) if self.codes else torch.empty(0, self.nsubq, dtype=torch.uint8)
            if all_codes.numel() == 0:
                return torch.empty(0, self.dim)
            vecs = []
            for c in all_codes:  # c: (nsubq,)
                subs = []
                for si in range(self.nsubq):
                    subs.append(self.codebooks[si, c[si]].unsqueeze(0))
                vecs.append(torch.cat(subs, dim=-1))
            return torch.cat(vecs, dim=0)
        else:
            return torch.cat(self.fp_store, dim=0) if self.fp_store else torch.empty(0, self.dim)

@dataclass
class Entity:
    name: str
    eid: int

@dataclass
class Relation:
    name: str
    rid: int


@dataclass
class RelationSchema:
    canonical_name: str
    functional: bool = False
    symmetric: bool = False
    inverse: Optional[str] = None
    description: Optional[str] = None


DEFAULT_RELATION_SCHEMA: Dict[str, RelationSchema] = {
    "capital_of": RelationSchema("capital_of", functional=True, inverse="has_capital"),
    "has_capital": RelationSchema("has_capital", inverse="capital_of"),
    "born_in": RelationSchema("born_in", functional=True, description="Person birth place"),
    "located_in": RelationSchema("located_in", inverse="contains"),
    "contains": RelationSchema("contains", inverse="located_in"),
    "parent_of": RelationSchema("parent_of", inverse="child_of"),
    "child_of": RelationSchema("child_of", functional=True, inverse="parent_of"),
    "spouse": RelationSchema("spouse", symmetric=True),
}

RELATION_ALIASES: Dict[str, str] = {
    "capital": "capital_of",
    "capital city": "capital_of",
    "is_capital_of": "capital_of",
    "birthplace": "born_in",
    "born": "born_in",
    "located": "located_in",
    "located_at": "located_in",
    "parent": "parent_of",
    "child": "child_of",
}

class EntityRelationGraph:
    def __init__(self, emb_dim: int = 128):
        self.emb_dim = emb_dim
        self.entity2id: Dict[str, int] = {}
        self.relation2id: Dict[str, int] = {}
        self.entities: List[Entity] = []
        self.relations: List[Relation] = []
        # adjacency storage: flat arrays
        self.edge_heads: List[int] = []  # head entity id (parallel to edge_rel, edge_tail)
        self.edge_rel: List[int] = []
        self.edge_tail: List[int] = []
        self.edge_conf: List[float] = []
        self.entity_offsets: List[int] = [0]  # prefix sums for quick slice
        # Embeddings (quantized)
        self.entity_emb_q = QuantizedEmbedding(emb_dim)
        self.relation_emb_q = QuantizedEmbedding(emb_dim)
        self._lock = threading.Lock()

    def add_entity(self, name: str) -> int:
        with self._lock:
            if name in self.entity2id:
                return self.entity2id[name]
            eid = len(self.entities)
            self.entity2id[name] = eid
            self.entities.append(Entity(name, eid))
            # random init embedding
            self.entity_emb_q.add(torch.randn(1, self.emb_dim))
            self.entity_offsets.append(self.entity_offsets[-1])  # no edges yet
            return eid

    def add_relation(self, name: str) -> int:
        with self._lock:
            if name in self.relation2id:
                return self.relation2id[name]
            rid = len(self.relations)
            self.relation2id[name] = rid
            self.relations.append(Relation(name, rid))
            self.relation_emb_q.add(torch.randn(1, self.emb_dim))
            return rid

    def add_edge(self, head: str, relation: str, tail: str, confidence: float = 1.0):
        h = self.add_entity(head)
        r = self.add_relation(relation)
        t = self.add_entity(tail)
        with self._lock:
            self.edge_heads.append(h)
            self.edge_rel.append(r)
            self.edge_tail.append(t)
            self.edge_conf.append(float(confidence))
        # update offsets lazily: rebuild every few insertions
        if len(self.edge_heads) % 32 == 0:
            self._rebuild_offsets()

    def _rebuild_offsets(self):
        # Build prefix sums for edges grouped by head
        edge_by_head: Dict[int, List[int]] = {}
        for idx, h in enumerate(self.edge_heads):
            edge_by_head.setdefault(h, []).append(idx)
        flat_heads = []
        flat_rel = []
        flat_tail = []
        flat_conf = []
        offsets = [0]
        for h in range(len(self.entities)):
            indices = edge_by_head.get(h, [])
            for i in indices:
                flat_heads.append(self.edge_heads[i])
                flat_rel.append(self.edge_rel[i])
                flat_tail.append(self.edge_tail[i])
                flat_conf.append(self.edge_conf[i])
            offsets.append(len(flat_heads))
        self.edge_heads = flat_heads
        self.edge_rel = flat_rel
        self.edge_tail = flat_tail
        self.edge_conf = flat_conf
        self.entity_offsets = offsets

    def neighbors(self, entity_name: str) -> List[Tuple[str, str]]:
        if entity_name not in self.entity2id:
            return []
        eid = self.entity2id[entity_name]
        start = self.entity_offsets[eid]
        end = self.entity_offsets[eid+1]
        out = []
        for i in range(start, end):
            rid = self.edge_rel[i]
            tid = self.edge_tail[i]
            conf = self.edge_conf[i] if i < len(self.edge_conf) else 1.0
            out.append((self.relations[rid].name, self.entities[tid].name, conf))
        return out

# ----------------------------- Passage Index -----------------------------

class PassageIndex:
    def __init__(self, dim: int = 384, use_faiss: bool = True):
        self.dim = dim
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        self.passages: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None  # (N, dim) fp16
        self._index = None

    def _ensure_index(self):
        if not self.use_faiss:
            return
        if self._index is None:
            # Use flat index for small datasets to avoid clustering warnings
            # FAISS IVF needs ~39 training points per centroid
            num_passages = len(self.passages)
            if num_passages < 1000:  # Use flat index for small datasets
                self._index = faiss.IndexFlatL2(self.dim)
            else:
                # Only use IVF with proper centroid sizing
                quantizer = faiss.IndexFlatL2(self.dim)
                nlist = max(1, min(128, num_passages // 50))  # At least 50 points per centroid
                self._index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
                
            if self.embeddings is not None and self.embeddings.size(0) > 0:
                if hasattr(self._index, 'train'):  # IVF indices need training
                    self._index.train(self.embeddings.numpy())
                self._index.add(self.embeddings.numpy())

    def add_passages(self, passages: Sequence[str], encoder: "SentenceEncoder") -> None:
        if not passages:
            return
        with torch.no_grad():
            encoded = encoder.encode(passages)
        encoded = encoded.to(torch.float16).cpu()
        
        # Validate dimension compatibility
        if encoded.size(1) != self.dim:
            raise ValueError(f"Encoder produces {encoded.size(1)}D embeddings but PassageIndex expects {self.dim}D")
        
        if self.embeddings is None or self.embeddings.numel() == 0:
            self.embeddings = encoded
            self.passages = list(passages)
            if self.use_faiss:
                self._index = None
                self._ensure_index()
        else:
            self.passages.extend(passages)
            self.embeddings = torch.cat([self.embeddings, encoded], dim=0)
            if self.use_faiss:
                if self._index is None:
                    self._ensure_index()
                if self._index is not None:
                    self._index.add(encoded.numpy())

    def rebuild(self) -> None:
        if not self.use_faiss:
            return
        if self.embeddings is None or self.embeddings.numel() == 0:
            return
        self._index = None
        self._ensure_index()

    def search(self, query_vec: torch.Tensor, topk: int = 5) -> List[Tuple[str, float]]:
        if self.embeddings is None or self.embeddings.numel() == 0:
            return []
        q = query_vec.float().cpu()
        if self.use_faiss and self._index is not None:
            D, I = self._index.search(q.numpy(), topk)
            return [(self.passages[i], float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        # brute force
        sims = F.cosine_similarity(q, self.embeddings.float())
        vals, idx = sims.topk(min(topk, sims.numel()))
        return [(self.passages[i], float(vals[j])) for j, i in enumerate(idx.tolist())]

# ----------------------------- Hybrid Engine -----------------------------

class SentenceEncoder:
    """Wrapper around sentence-transformer models for embedding evidence."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        self.model_name = model_name
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model = SentenceTransformer(model_name, device=str(self.device))

    @torch.inference_mode()
    def encode(self, sentences: Sequence[str]) -> torch.Tensor:
        if not sentences:
            return torch.empty(0, self.model.get_sentence_embedding_dimension())
        embeddings = self.model.encode(
            sentences,
            batch_size=8,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return embeddings


class HybridMemoryEngine:
    def __init__(
        self,
        entity_dim: int = 128,
        passage_dim: Optional[int] = None,  # Auto-detect from encoder if not provided
        use_faiss: bool = True,
        passage_encoder: Optional[SentenceEncoder] = None,
    ):
        self.graph = EntityRelationGraph(emb_dim=entity_dim)
        self.passage_encoder = passage_encoder or SentenceEncoder()
        
        # Auto-detect passage dimension from encoder if not provided
        if passage_dim is None:
            if _SENTENCE_TRANSFORMERS_AVAILABLE:
                passage_dim = self.passage_encoder.model.get_sentence_embedding_dimension()
            else:
                passage_dim = 384  # Default fallback
        
        self.passage_index = PassageIndex(dim=passage_dim, use_faiss=use_faiss)
        self.entity_aliases: Dict[str, str] = {}
        self.relation_schema = DEFAULT_RELATION_SCHEMA.copy()
        self.relation_aliases = RELATION_ALIASES.copy()
        self.facts_by_relation: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.conflict_log: List[Dict[str, str]] = []
        self.entity_degree: Counter[str] = Counter()
        self.relation_counts: Counter[str] = Counter()
        self.ingestion_history: List[Dict[str, str]] = []
        
        # Versioning and snapshots
        self._version = 0
        self._snapshot_dir = os.environ.get("MEM_SNAPSHOT_DIR", "./mem_snapshots")
        os.makedirs(self._snapshot_dir, exist_ok=True)
        self._lock = threading.RLock()

    def ingest_fact(self, head: str, relation: str, tail: str, evidence_sentences: Optional[Iterable[str]] = None):
        with self._lock:
            info = self._prepare_fact(head, relation, tail)
            if not info["allowed"]:
                return info
            self.graph.add_edge(info["head"], info["relation"], info["tail"], confidence=info["confidence"])
            rel_heads = self.facts_by_relation[info["relation"]]
            rel_heads[info["head"]].add(info["tail"])
            self.entity_degree[info["head"]] += 1
            self.entity_degree[info["tail"]] += 1
            self.relation_counts[info["relation"]] += 1
            if evidence_sentences:
                self.passage_index.add_passages(list(evidence_sentences), self.passage_encoder)
                self.passage_index.rebuild()
            self.ingestion_history.append(info)
            
            # Increment version after successful ingestion
            self._version += 1
            
            return info

    def insert_with_verifier_guard(
        self,
        triple: Tuple[str, str, str],
        evidence_text: str,
        verifier,  # FactVerifier instance
    ) -> bool:
        """Insert fact with verifier guard for schema-aware validation."""
        head, relation, tail = triple
        canonical_head = self._canonicalize_entity(head)
        canonical_tail = self._canonicalize_entity(tail)
        canonical_relation, schema = self._canonicalize_relation(relation)
        
        hypothesis = f"{canonical_head} {canonical_relation} {canonical_tail}"
        
        # Use verifier to score the fact
        if hasattr(verifier, 'entailment_score'):
            # Root version interface
            score = verifier.entailment_score(evidence_text, hypothesis)
            threshold_loose = getattr(verifier, 'threshold_loose', 0.6)
            threshold_strict = getattr(verifier, 'threshold_strict', 0.8)
        else:
            # Core version interface - create a mock candidate for verification
            from ..data.fact_extraction import FactCandidate
            candidate = FactCandidate(
                head=canonical_head,
                relation=canonical_relation,
                tail=canonical_tail,
                sentence=evidence_text
            )
            results = verifier.verify([candidate])
            score = 1.0 if results else 0.0
            threshold_loose = verifier.threshold
            threshold_strict = verifier.threshold

        with self._lock:
            if schema and schema.functional:
                # Check for existing functional relationship
                rel_heads = self.facts_by_relation[canonical_relation]
                existing_tails = rel_heads.get(canonical_head, set())
                
                if not existing_tails:
                    # No existing relationship, use loose threshold
                    if score >= threshold_loose:
                        self.ingest_fact(canonical_head, canonical_relation, canonical_tail, [evidence_text])
                        return True
                    return False
                
                if canonical_tail in existing_tails:
                    # Same fact already exists
                    return True
                    
                # Conflicting functional relationship, use strict threshold
                if score >= threshold_strict:
                    # Replace existing relationship
                    rel_heads[canonical_head] = {canonical_tail}
                    self.conflict_log.append({
                        "type": "functional_replace",
                        "head": canonical_head,
                        "relation": canonical_relation,
                        "old_tail": str(existing_tails),
                        "new_tail": canonical_tail,
                        "evidence": evidence_text,
                        "score": score
                    })
                    self.ingest_fact(canonical_head, canonical_relation, canonical_tail, [evidence_text])
                    return True
                return False
            
            # Non-functional relationship, use loose threshold
            if score >= threshold_loose:
                self.ingest_fact(canonical_head, canonical_relation, canonical_tail, [evidence_text])
                return True
            return False

    def get_version(self) -> int:
        """Get current version number."""
        with self._lock:
            return self._version

    def save_snapshot(self) -> MemorySnapshotMeta:
        """Save current memory state to disk."""
        with self._lock:
            meta = MemorySnapshotMeta(
                version=self._version,
                created_ts=time.time(),
                path=os.path.join(self._snapshot_dir, f"mem_v{self._version}.pkl"),
            )
            
            # Serialize the state
            state = {
                "version": self._version,
                "entities": {e.name: e.eid for e in self.graph.entities},
                "relations": {r.name: r.rid for r in self.graph.relations},
                "facts_by_relation": {
                    rel: {head: list(tails) for head, tails in heads.items()}
                    for rel, heads in self.facts_by_relation.items()
                },
                "entity_aliases": dict(self.entity_aliases),
                "relation_counts": dict(self.relation_counts),
                "entity_degree": dict(self.entity_degree),
                "conflict_log": self.conflict_log.copy(),
                "passages": self.passage_index.passages.copy() if self.passage_index.passages else []
            }
            
            with open(meta.path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            return meta

    def load_snapshot(self, path: str) -> None:
        """Load memory state from disk snapshot."""
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        with self._lock:
            # Clear current state
            self.graph = EntityRelationGraph(emb_dim=self.graph.emb_dim)
            self.facts_by_relation = defaultdict(lambda: defaultdict(set))
            self.entity_aliases = {}
            self.relation_counts = Counter()
            self.entity_degree = Counter()
            self.conflict_log = []
            self.passage_index = PassageIndex(dim=self.passage_index.dim, use_faiss=self.passage_index.use_faiss)
            
            # Restore state
            self._version = state.get("version", 0)
            self.entity_aliases = state.get("entity_aliases", {})
            self.relation_counts = Counter(state.get("relation_counts", {}))
            self.entity_degree = Counter(state.get("entity_degree", {}))
            self.conflict_log = state.get("conflict_log", [])
            
            # Rebuild facts
            facts_by_relation = state.get("facts_by_relation", {})
            for relation, heads in facts_by_relation.items():
                for head, tails in heads.items():
                    for tail in tails:
                        self.graph.add_edge(head, relation, tail)
                        self.facts_by_relation[relation][head].add(tail)
            
            # Restore passages
            passages = state.get("passages", [])
            if passages:
                self.passage_index.add_passages(passages, self.passage_encoder)

    def latest_snapshot_path(self) -> Optional[str]:
        """Get path to the latest snapshot file."""
        if not os.path.exists(self._snapshot_dir):
            return None
            
        files = [
            f for f in os.listdir(self._snapshot_dir) 
            if f.startswith("mem_v") and f.endswith(".pkl")
        ]
        if not files:
            return None
            
        # Sort by version number
        def extract_version(filename):
            try:
                return int(filename.split("mem_v")[1].split(".pkl")[0])
            except (ValueError, IndexError):
                return -1
                
        files.sort(key=extract_version)
        return os.path.join(self._snapshot_dir, files[-1])

    def query(self, entity: str, topk_neighbors: int = 5, topk_passages: int = 5):
        neighbors_raw = self.graph.neighbors(self._canonicalize_entity(entity))
        neighbors = []
        for rel, tail, conf in neighbors_raw[:topk_neighbors]:
            neighbors.append((rel, tail, conf))
        # Build synthetic query embedding (average entity + relation embeddings if available)
        ent_embs = self.graph.entity_emb_q.reconstruct()
        canonical_entity = self._canonicalize_entity(entity)
        if canonical_entity in self.graph.entity2id and ent_embs.size(0) > 0:
            eid = self.graph.entity2id[canonical_entity]
            ent_vec = ent_embs[eid:eid+1]
        else:
            ent_vec = torch.randn(1, self.graph.emb_dim)
        if ent_vec.size(1) != self.passage_index.dim:
            if ent_vec.size(1) > self.passage_index.dim:
                ent_vec = ent_vec[:, : self.passage_index.dim]
            else:
                pad = torch.zeros(ent_vec.size(0), self.passage_index.dim - ent_vec.size(1))
                ent_vec = torch.cat([ent_vec, pad], dim=1)
        passages = []
        if self.passage_index.embeddings is not None:
            passages = self.passage_index.search(ent_vec, topk=topk_passages)
        return {"entity": canonical_entity, "neighbors": neighbors, "passages": passages}

    def sample_triples(self, k: int = 1) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        for relation, heads in self.facts_by_relation.items():
            for head, tails in heads.items():
                for tail in tails:
                    triples.append((head, relation, tail))
        if not triples:
            return []
        return random.sample(triples, min(k, len(triples)))

    def get_summary(self) -> Dict[str, object]:
        return {
            "entities": len(self.graph.entities),
            "relations": len(self.graph.relations),
            "facts": len(self.graph.edge_heads),
            "conflicts": len(self.conflict_log),
            "top_relations": self.relation_counts.most_common(5),
        }

    # ----------------------------- Internal helpers -----------------------------

    def _canonicalize_entity(self, name: str) -> str:
        normalized = " ".join(name.strip().split())
        key = normalized.lower()
        existing = self.entity_aliases.get(key)
        if existing:
            return existing
        self.entity_aliases[key] = normalized
        return normalized

    def _canonicalize_relation(self, relation: str) -> Tuple[str, RelationSchema | None]:
        rel_key = relation.strip().lower().replace(" ", "_")
        if rel_key in self.relation_aliases:
            rel_key = self.relation_aliases[rel_key]
        schema = self.relation_schema.get(rel_key)
        canonical = schema.canonical_name if schema else rel_key
        return canonical, schema

    def _prepare_fact(self, head: str, relation: str, tail: str) -> Dict[str, object]:
        head_c = self._canonicalize_entity(head)
        tail_c = self._canonicalize_entity(tail)
        relation_c, schema = self._canonicalize_relation(relation)
        info: Dict[str, object] = {
            "head": head_c,
            "tail": tail_c,
            "relation": relation_c,
            "confidence": 1.0,
            "schema": schema.canonical_name if schema else None,
            "flags": [],
            "allowed": True,
        }

        rel_heads = self.facts_by_relation[relation_c]
        tails = rel_heads[head_c]
        if tail_c in tails:
            info["allowed"] = False
            info["flags"].append("duplicate")
            return info

        if schema and schema.functional and tails:
            # Conflict with existing assertion
            info["flags"].append("functional_conflict")
            info["confidence"] = 0.3
            for existing_tail in tails:
                self.conflict_log.append(
                    {
                        "type": "functional_conflict",
                        "head": head_c,
                        "relation": relation_c,
                        "existing_tail": existing_tail,
                        "new_tail": tail_c,
                    }
                )

        if schema and schema.inverse:
            inverse_heads = self.facts_by_relation[schema.inverse]
            inverse_tails = inverse_heads[tail_c]
            if head_c not in inverse_tails:
                inverse_tails.add(head_c)

        if self._detect_degree_spike(head_c):
            info["flags"].append("degree_spike")

        if self._detect_small_cycle(head_c, tail_c):
            info["flags"].append("cycle")
            self.conflict_log.append(
                {
                    "type": "cycle",
                    "head": head_c,
                    "relation": relation_c,
                    "tail": tail_c,
                }
            )

        return info

    def _detect_degree_spike(self, entity: str, threshold: int = 128) -> bool:
        return self.entity_degree[entity] >= threshold

    def _detect_small_cycle(self, head: str, tail: str, max_depth: int = 3) -> bool:
        if head == tail:
            return True
        visited = set()
        queue = [(tail, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            if node == head:
                return True
            neighbors = self.graph.neighbors(node)
            for _, next_tail, _ in neighbors:
                if next_tail not in visited:
                    visited.add(next_tail)
                    queue.append((next_tail, depth + 1))
        return False

__all__ = [
    "EntityRelationGraph",
    "HybridMemoryEngine", 
    "MemorySnapshotMeta",
    "RelationSchema",
    "SentenceEncoder",
    "PassageIndex",
]
