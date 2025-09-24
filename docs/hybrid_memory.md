# Hybrid Compressed Memory (Facts + Language)

Goal: Separate durable factual structure from verbose linguistic evidence to keep the model lean while enabling precise, schema-aware retrieval.

## Two Tiers
1. **Entity–Relation Graph (ER Graph)**
   - Canonical entities (string -> uint32 id) and relations (string -> uint32 id).
   - Edges stored as tightly packed parallel arrays `(head_ids, rel_ids, tail_ids)` with prefix offsets per head for O(1) slice.
   - Quantized embeddings for entities & relations via lightweight product quantization (PQ) when enough samples observed.
2. **Passage / Evidence Store**
   - Collection of provenance sentences or shards.
   - Optional FAISS IVF-PQ or flat brute force fallback for similarity search.
   - Embeddings can be re-encoded asynchronously / incrementally.

## Retrieval Pipeline
1. **Entity Linking (External / Placeholder)**: Upstream text identifies entity surface form → canonical id (here we approximate by direct string key).
2. **Graph Hop**: From entity id, enumerate outgoing edges, optionally filter by relation type / scoring.
3. **Evidence Retrieval**: Use entity (and optional relation) embedding to query passage index (FAISS or fallback) for top-k supporting sentences.
4. **Rerank (Future)**: Cross-encoder or lightweight attention between query and candidate passages; optional relation constraint scoring.

## Memory Footprint
- Entities: N_e * (entity_id(4B) + PQ codes). Codes: nsubq bytes per entity (e.g. 4–8). Example: 1M entities @8 bytes ≈ 8 MB + codebooks.
- Relations: N_r typically << N_e.
- Edges: 3 * 4 bytes per edge + ~0 overhead for offsets. 50M edges ≈ 600 MB raw; can delta-compress or 16-bit if <=65k entities.
- Passages: Embeddings quantized (IVF-PQ). Example: 5M passages at 96B compressed each ≈ 480 MB.
- Target: sub-GB for mid-scale factual memory.

## Quantization Strategy
- Lightweight in-module PQ (k-means per subspace) for bootstrap.
- Replace with FAISS `IndexIVFPQ` / `IndexHNSWPQ` when available for faster large-scale retrieval.

## Consistency & Updates
- Append-only ingestion; periodic rebuild of offsets & FAISS index.
- Deferred: versioning for temporal facts; soft-deletes via tombstone bitset.

## Integration Points (Planned)
- Training-time augmentation: inject retrieved fact triples and top passages into model input context (controlled token budget).
- Loss shaping: auxiliary objective aligning entity embedding predictions with retrieved neighbors (graph contrastive).
- Dynamic curriculum: prioritize batches whose text produces low graph coverage.

## Current Prototype Limitations
- Entity linking stub: assumes exact surface form.
- Passage encoding placeholder: random vectors until integrated with LM encoder or a sentence transformer.
- No persistence layer yet; purely in-memory.
- No concurrency safe incremental FAISS add (full rebuild on ingestion for simplicity).

## Extending
- Swap `encoder` in `PassageIndex.build` with a real embedding model (e.g., MiniLM or pooled LM hidden states).
- Add caching layer for frequent queries.
- Implement relation-specific scoring heuristics (e.g., inverse frequency weighting).

## Example Usage
```python
from model.hybrid_memory import HybridMemoryEngine
mem = HybridMemoryEngine()
mem.ingest_fact("Ada Lovelace", "field", "Mathematics", ["Ada Lovelace was a pioneer in computing."])
mem.ingest_fact("Ada Lovelace", "collaborated_with", "Charles Babbage")
print(mem.query("Ada Lovelace"))
```

Output keys:
- `entity`: original query string
- `neighbors`: list of (relation, tail_entity) tuples
- `passages`: list of (text, score/ distance)

## Next Steps
- Integrate factual augmentation into `main.py` training loop (optional config block `"hybrid_memory": { "enable": true }`).
- Add entity-centric auxiliary loss.
- Persist memory snapshots alongside model checkpoints.
