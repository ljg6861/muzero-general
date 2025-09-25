from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class SimpleTFIDFIndex:
    """Simple TF-IDF based sparse retrieval index"""
    def __init__(self):
        self.docs: List[str] = []
        self.term_freqs: List[Counter] = []
        self.df: Counter = Counter()
        self.N = 0
        self.token_re = re.compile(r"[A-Za-z][A-Za-z0-9_]+")

    def add(self, text: str):
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        self.docs.append(text)
        self.term_freqs.append(tf)
        for t in tf.keys():
            self.df[t] += 1
        self.N += 1

    def _tokenize(self, s: str) -> List[str]:
        return [t.lower() for t in self.token_re.findall(s)]

    def build_from_iter(self, it, max_docs: int = 5000):
        for i, t in enumerate(it):
            if i >= max_docs:
                break
            self.add(t)

    def _tfidf(self, tf: Counter) -> dict:
        vec = {}
        for term, f in tf.items():
            idf = math.log((1 + self.N) / (1 + self.df[term])) + 1.0
            vec[term] = (1 + math.log(1 + f)) * idf
        return vec

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
        qtf = Counter(self._tokenize(query))
        qvec = self._tfidf(qtf)
        scores = []
        for tf, doc in zip(self.term_freqs, self.docs):
            dvec = self._tfidf(tf)
            score = 0.0
            for term, w in qvec.items():
                score += w * dvec.get(term, 0.0)
            if score > 0:
                scores.append((score, doc))
        scores.sort(key=lambda x: -x[0])
        return scores[:k]


@dataclass
class RetrievalResult:
    text: str
    score: float


class HybridRetriever:
    """Combines sparse BM25-style retrieval with dense embeddings."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        self.tfidf = SimpleTFIDFIndex()
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.device = device or os.getenv("RETRIEVER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        if device is not None:
            os.environ["RETRIEVER_DEVICE"] = device
        os.environ.setdefault("RETRIEVER_MODEL", embedding_model)
        self.encoder = build_dense_encoder()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def add_document(self, doc_id: str, text: str) -> None:
        if not text:
            return
        self.tfidf.add(text)
        self.documents.append(text)
        embedding = self.encoder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings.append(embedding)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.documents:
            return []
        sparse_results = self.tfidf.retrieve(query, k=top_k * 5)
        sparse_scores = {doc: score for score, doc in sparse_results}
        query_embedding = self.encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        dense_scores = []
        for idx, embedding in enumerate(self.embeddings):
            score = float(np.dot(query_embedding, embedding))
            dense_scores.append((self.documents[idx], score))
        dense_scores.sort(key=lambda x: x[1], reverse=True)
        combined = {}
        for doc, score in dense_scores[: top_k * 5]:
            combined[doc] = combined.get(doc, 0.0) + self.dense_weight * score
        for doc, score in sparse_scores.items():
            combined[doc] = combined.get(doc, 0.0) + self.sparse_weight * score
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def build_dense_encoder():
    """Construct a sentence-transformer encoder on the requested device."""

    device = os.getenv("RETRIEVER_DEVICE", "cpu")
    model_name = os.getenv("RETRIEVER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    encoder = SentenceTransformer(model_name, device=device)
    encoder.max_seq_length = 256
    return encoder


__all__ = ["HybridRetriever", "RetrievalResult", "build_dense_encoder"]
