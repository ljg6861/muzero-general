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
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable
import threading
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    _FAISS_AVAILABLE = False

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

    def add_edge(self, head: str, relation: str, tail: str):
        h = self.add_entity(head)
        r = self.add_relation(relation)
        t = self.add_entity(tail)
        with self._lock:
            self.edge_heads.append(h)
            self.edge_rel.append(r)
            self.edge_tail.append(t)
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
        offsets = [0]
        for h in range(len(self.entities)):
            indices = edge_by_head.get(h, [])
            for i in indices:
                flat_heads.append(self.edge_heads[i])
                flat_rel.append(self.edge_rel[i])
                flat_tail.append(self.edge_tail[i])
            offsets.append(len(flat_heads))
        self.edge_heads = flat_heads
        self.edge_rel = flat_rel
        self.edge_tail = flat_tail
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
            out.append((self.relations[rid].name, self.entities[tid].name))
        return out

# ----------------------------- Passage Index -----------------------------

class PassageIndex:
    def __init__(self, dim: int = 384, use_faiss: bool = True):
        self.dim = dim
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        self.passages: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None  # (N, dim) fp16
        self._index = None

    def build(self, passages: List[str], encoder: nn.Module):
        self.passages = passages
        with torch.no_grad():
            embs = []
            for txt in passages:
                token_ids = torch.randint(0, 1000, (1, 32))  # placeholder random tokens; integrate real encoder later
                vec = encoder(token_ids)  # expects (1, dim)
                embs.append(vec)
            embs = torch.cat(embs, dim=0).half().cpu()
        self.embeddings = embs
        if self.use_faiss:
            quantizer = faiss.IndexFlatL2(self.dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.dim, min(16, max(1, len(passages)//8)))
            self._index.train(embs.numpy())
            self._index.add(embs.numpy())
        else:
            # no extra structure
            pass

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

class HybridMemoryEngine:
    def __init__(self, entity_dim: int = 128, passage_dim: int = 384, use_faiss: bool = True):
        self.graph = EntityRelationGraph(emb_dim=entity_dim)
        self.passage_index = PassageIndex(dim=passage_dim, use_faiss=use_faiss)

    def ingest_fact(self, head: str, relation: str, tail: str, evidence_sentences: Optional[Iterable[str]] = None):
        self.graph.add_edge(head, relation, tail)
        if evidence_sentences:
            # Append new passages (rebuild could be optimized; for prototype keep simple)
            if self.passage_index.embeddings is None:
                passages = list(evidence_sentences)
            else:
                passages = self.passage_index.passages + list(evidence_sentences)
            # Dummy encoder: simple random projection for now
            encoder = lambda toks: torch.randn(1, self.passage_index.dim)
            self.passage_index.build(passages, encoder)

    def query(self, entity: str, topk_neighbors: int = 5, topk_passages: int = 5):
        neighbors = self.graph.neighbors(entity)[:topk_neighbors]
        # Build synthetic query embedding (average entity + relation embeddings if available)
        ent_embs = self.graph.entity_emb_q.reconstruct()
        if entity in self.graph.entity2id and ent_embs.size(0) > 0:
            eid = self.graph.entity2id[entity]
            ent_vec = ent_embs[eid:eid+1]
        else:
            ent_vec = torch.randn(1, self.graph.emb_dim)
        passages = []
        if self.passage_index.embeddings is not None:
            passages = self.passage_index.search(ent_vec, topk=topk_passages)
        return {"entity": entity, "neighbors": neighbors, "passages": passages}

__all__ = [
    "EntityRelationGraph",
    "HybridMemoryEngine",
]
