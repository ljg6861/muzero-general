import os, sys, torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'core'))

from core.models.hybrid_memory import (
    HybridMemoryEngine,
    EntityRelationGraph, 
    PassageIndex,
    QuantizedEmbedding,
    SentenceEncoder
)
import torch


class _DummyEncoder(SentenceEncoder):
    def __init__(self, dim: int = 384):
        self.dim = dim

    @torch.inference_mode()
    def encode(self, sentences):
        if not sentences:
            return torch.empty(0, self.dim)
        eye = torch.eye(self.dim)
        reps = []
        for idx, _ in enumerate(sentences):
            reps.append(eye[idx % self.dim].unsqueeze(0))
        return torch.cat(reps, dim=0)


def test_basic_ingest_and_query():
    mem = HybridMemoryEngine(entity_dim=64, passage_dim=384, use_faiss=False, passage_encoder=_DummyEncoder())
    mem.ingest_fact("Earth", "orbited_by", "Moon", ["The Moon orbits the Earth."])
    mem.ingest_fact("Earth", "category", "Planet")
    # Force rebuild offsets since we only have 2 edges (less than 32 threshold)
    mem.graph._rebuild_offsets()
    out = mem.query("Earth")
    print(f"DEBUG: Query result: {out}")
    assert out["entity"] == "Earth"
    # Debug the neighbors structure
    print(f"DEBUG: Neighbors: {out['neighbors']}")
    if out["neighbors"]:
        print(f"DEBUG: First neighbor type: {type(out['neighbors'][0])}")
    # Check that we have expected relations
    assert isinstance(out["neighbors"], list)
    assert len(out["neighbors"]) >= 1, "Should have at least one neighbor after rebuild"
    neighbor_relations = [rel for rel, *_ in out["neighbors"]]
    assert "orbited_by" in neighbor_relations, f"Expected 'orbited_by' in relations: {neighbor_relations}"
    assert "category" in neighbor_relations, f"Expected 'category' in relations: {neighbor_relations}"

if __name__ == "__main__":
    test_basic_ingest_and_query()
    print("hybrid_memory test passed")
