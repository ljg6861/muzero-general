import os, sys, torch
# Allow running this file directly: add project root so 'model' package is discoverable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.hybrid_memory import HybridMemoryEngine


def test_basic_ingest_and_query():
    mem = HybridMemoryEngine(entity_dim=64, passage_dim=64, use_faiss=False)
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
    neighbor_relations = [rel for rel, tail in out["neighbors"]]
    assert "orbited_by" in neighbor_relations, f"Expected 'orbited_by' in relations: {neighbor_relations}"
    assert "category" in neighbor_relations, f"Expected 'category' in relations: {neighbor_relations}"

if __name__ == "__main__":
    test_basic_ingest_and_query()
    print("hybrid_memory test passed")
