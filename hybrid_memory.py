"""Versioned hybrid memory engine with snapshotting and verifier guards."""

from __future__ import annotations

import os
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, DefaultDict, Set
from collections import defaultdict

try:  # Optional dependency
    import torch
except ImportError:  # pragma: no cover - make module importable without torch
    torch = None  # type: ignore[assignment]


@dataclass
class MemorySnapshotMeta:
    version: int
    created_ts: float
    path: str


class RelationSchema:
    def __init__(
        self,
        name: str,
        is_functional: bool = False,
        inverse_of: Optional[str] = None,
        time_scoped: bool = False,
    ) -> None:
        self.name = name
        self.is_functional = is_functional
        self.inverse_of = inverse_of
        self.time_scoped = time_scoped


SCHEMA: Dict[str, RelationSchema] = {
    "capital_of": RelationSchema("capital_of", is_functional=True, inverse_of="has_capital"),
    "has_capital": RelationSchema("has_capital", inverse_of="capital_of"),
}


class HybridMemoryEngine:
    """Thread-safe hybrid memory with single-writer snapshotting."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._version = 0
        self._graph: DefaultDict[str, DefaultDict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._evidence: DefaultDict[Tuple[str, str, str], List[str]] = defaultdict(list)
        self._history: List[Tuple[Tuple[str, str, str], float]] = []
        self._retriever = None
        self._snapshot_dir = os.environ.get("MEM_SNAPSHOT_DIR", "./mem_snapshots")
        os.makedirs(self._snapshot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Mutation helpers (single writer)
    # ------------------------------------------------------------------
    def ingest_triples(self, triples: Iterable[Tuple[str, str, str]]) -> None:
        """Ingest a batch of triples and bump the version."""

        triples = list(triples)
        if not triples:
            return
        with self._lock:
            for triple in triples:
                self._graph_add(triple)
            self._version += 1

    def save_snapshot(self) -> MemorySnapshotMeta:
        """Persist the current memory state to disk."""

        with self._lock:
            meta = MemorySnapshotMeta(
                version=self._version,
                created_ts=time.time(),
                path=os.path.join(self._snapshot_dir, f"mem_v{self._version}.pkl"),
            )
            payload = {
                "version": self._version,
                "graph": {h: {r: sorted(tails) for r, tails in rels.items()} for h, rels in self._graph.items()},
                "evidence": dict(self._evidence),
            }
            with open(meta.path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return meta

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------
    def get_version(self) -> int:
        with self._lock:
            return self._version

    def load_snapshot(self, path: str) -> None:
        """Load a snapshot from disk and replace the in-memory state."""

        with open(path, "rb") as handle:
            state = pickle.load(handle)
        graph_state = defaultdict(lambda: defaultdict(set))
        for head, rels in state["graph"].items():
            for rel, tails in rels.items():
                graph_state[head][rel].update(tails)
        with self._lock:
            self._graph = graph_state
            self._evidence = defaultdict(list, state.get("evidence", {}))
            self._version = int(state.get("version", 0))

    def latest_snapshot_path(self) -> Optional[str]:
        files = [
            f for f in os.listdir(self._snapshot_dir) if f.startswith("mem_v") and f.endswith(".pkl")
        ]
        if not files:
            return None
        files.sort(key=lambda name: int(name.split("mem_v")[1].split(".pkl")[0]))
        return os.path.join(self._snapshot_dir, files[-1])

    # ------------------------------------------------------------------
    # Graph primitives
    # ------------------------------------------------------------------
    def _graph_add(self, triple: Tuple[str, str, str], evidence: Optional[str] = None) -> None:
        head, relation, tail = triple
        rel_bucket = self._graph[head]
        rel_bucket[relation].add(tail)
        if evidence:
            key = (head, relation, tail)
            self._evidence[key].append(evidence)

    def _graph_get_functional_tail(self, head: str, relation: str) -> Optional[str]:
        tails = self._graph.get(head, {}).get(relation)
        if not tails:
            return None
        # functional relation should have at most one tail
        return next(iter(tails))

    def _graph_replace_functional_tail(self, head: str, relation: str, new_tail: str) -> None:
        bucket = self._graph[head]
        bucket[relation] = {new_tail}

    def _graph_time_scope_replace(
        self,
        head: str,
        relation: str,
        old_tail: str,
        new_tail: str,
    ) -> None:
        bucket = self._graph[head]
        bucket[relation] = {new_tail}
        timestamp = time.time()
        self._history.append(((head, relation, old_tail), timestamp))

    # ------------------------------------------------------------------
    # Verifier guarded insert
    # ------------------------------------------------------------------
    def insert_with_verifier_guard(
        self,
        triple: Tuple[str, str, str],
        evidence_text: str,
        verifier: "FactVerifier",
    ) -> bool:
        head, relation, tail = triple
        schema = SCHEMA.get(relation, RelationSchema(relation))
        hypothesis = f"{head} {relation} {tail}"
        score = verifier.entailment_score(evidence_text, hypothesis)

        with self._lock:
            if schema.is_functional:
                existing_tail = self._graph_get_functional_tail(head, relation)
                if existing_tail is None:
                    if score >= verifier.threshold_loose:
                        self._graph_add(triple, evidence_text)
                        self._version += 1
                        return True
                    return False
                if existing_tail == tail:
                    return True
                if score >= verifier.threshold_strict:
                    if schema.time_scoped:
                        self._graph_time_scope_replace(head, relation, existing_tail, tail)
                    else:
                        self._graph_replace_functional_tail(head, relation, tail)
                    self._graph_add((head, relation, tail), evidence_text)
                    self._version += 1
                    return True
                return False

            if score >= verifier.threshold_loose:
                self._graph_add(triple, evidence_text)
                self._version += 1
                return True
            return False

    # ------------------------------------------------------------------
    # Retriever helpers
    # ------------------------------------------------------------------
    def set_retriever(self, retriever) -> None:
        self._retriever = retriever

    def encode_passages_async(self, passages: List[str]):
        if self._retriever is None:
            return None
        return self._retriever.encode(passages)


__all__ = [
    "HybridMemoryEngine",
    "MemorySnapshotMeta",
    "RelationSchema",
    "SCHEMA",
]
