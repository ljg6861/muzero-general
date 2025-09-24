# Project Model Instructions

## Mission
Build a single self-contained, continuously improvable language model training loop that: 
- Uses ONLY one interactive entrypoint (`main.py`) with zero runtime flags.
- Always resumes from the most recent checkpoint and maintains a rolling window of three snapshots: `lm_current.pth`, `lm_prev.pth`, `lm_prev2.pth`.
- Learns general world + task knowledge from streaming real data (HF + PDFs) without hardcoded templates or task-specific hacks.
- Incorporates internal reasoning/planning mechanisms (MuZero-inspired adapters and future cognitive modules) so the MODEL itself gets smarter, not external tool scaffolding.

## Core Principles
1. No runtime parameters. All behavior is defined by editable JSON presets in `configs/`.
2. Fixed global training token target: **100,000,000 tokens per run** (overrides preset train_tokens field).
3. Real data only. No synthetic fallback path. If all sources fail, training halts.
4. Rolling checkpoint policy ensures safe recovery and regression analysis.
5. Extensible data ingestion with plugin-style sources: Hugging Face streaming + local PDFs + formatted multi-field Q/A.
6. Efficiency-first: streamlined model, bf16 autocast, TF32 enabled, DDP-only multi-GPU, minimal overhead.
7. Architecture evolution focuses on internal representation quality, planning, and structured credit assignment—not prompt engineering.

## Current Architecture (Baseline LM)
- Token + positional embeddings (extendable via `extend_max_seq_len`).
- Stack of lightweight transformer blocks using standard `nn.MultiheadAttention` (Flash/SDPA backend leveraged automatically by PyTorch where available).
- Causal mask cached once; position ids cached.
- Standard next-token LM objective with proper pad-masking.
- Generation path: top-p nucleus sampling + n‑gram repetition blocking.

### Tier 1 Auxiliary Modules (Implemented)
Enabled via `auxiliary` section in preset JSON (all default off):

```
"auxiliary": {
  "enable_policy": true,
  "enable_value": true,
  "enable_dynamics": true,
  "enable_span_contrastive": true,
  "policy_kl_weight": 0.05,
  "value_weight": 0.05,
  "dynamics_weight": 0.05,
  "span_contrastive_weight": 0.05,
  "span_mask_fraction": 0.15,
  "span_contrastive_temperature": 0.07,
  "span_l2_normalize": true,
  "enable_adaptive_sources": true,
  "adaptive_interval_tokens": 5000000,
  "adaptive_min_weight": 0.5,
  "adaptive_max_weight": 3.0,
  "adaptive_smoothing": 0.9
}
```

Components:
- Policy Head: next-token distribution prediction at chosen position; KL to LM teacher (stabilizes planning signal).
- Value Head: predicts future average NLL over forthcoming span (proxy for information gain / difficulty).
- Dynamics Module: latent transition using final hidden + token context → predicts future latent (short rollout consistency).
- Span Contrastive: randomly masks a span; encourage representation predictive consistency via InfoNCE.
- Adaptive Source Reweighting: dynamically rescales dataset mixing weights based on relative per-source loss (higher loss → higher weight) with smoothing & clamped range.

Losses blended into total: `CE + Σ(weight_i * aux_i)`.

### MuZero-Inspired Adapter Layer (Planned Reintegration)
(Adapters exist separately for experimentation; main loop currently focuses on baseline LM training.) Planned fusion points:
- Policy head guiding sampling distribution adjustments.
- Value head estimating long-horizon utility of partial generations.
- Dynamics / latent transition module enabling limited internal rollouts for lookahead.
- Router / planner that re-scores candidate continuations before committing tokens.

These will be reintegrated after solidifying baseline data quality + efficiency.

## Data Pipeline
Implemented via `DataRegistry` + `DataConfig`:
- Built-in streaming HF datasets (script-free): `wikipedia`, `openwebtext`, `c4`, `bookcorpus`, `pile_subset`, `common_crawl`.
- Custom plugin sources (inline in preset JSON):
  - Hugging Face: `{ "type": "hf", "dataset": "owner/name", "split": "train", "text_field": "text" }`
  - Multi-field formatting (Q/A): `{ "type": "hf", "dataset": "sentence-transformers/natural-questions", "fields": ["question", "answer"], "format": "Q: {question}\nA: {answer}" }`
  - PDFs: `{ "type": "pdf", "path": "./my_pdfs" }` (recurses; yields ~3–5k char chunks)
- Mixing strategies:
  - `round_robin` (default) – interleave sources uniformly.
  - `sequential` – exhaust sources in order.
  - `weighted` – proportionally allocate turns via `source_weights`.
- Sharded sampling across DDP ranks + DataLoader workers (global composite index) to avoid duplicate tokenization.
- Each text is tokenized, chunked to `seq_length`, padded (pad labels = -100), and streamed until global token budget met.

## Checkpointing
- Rolling rotation:
  - Save new model → write temp → rotate: `lm_prev` → `lm_prev2`, `lm_current` → `lm_prev`, write new `lm_current`.
  - Mirror legacy name `simple_lm_trained.pth` for compatibility.
- Resume always prioritizes `checkpoints/lm_current.pth`.

## Presets (`configs/*.json`)
Each file defines: `model`, `train`, and `data` sections. Example additions:
```json
{
  "name": "qa_plus_wiki",
  "model": {"hidden_size":256, "num_layers":3, "num_heads":4, "seq_length":256},
  "train": {"per_gpu_batch_size":128, "accumulation_steps":4, "eval_tokens":2000000},
  "data": {
    "sources": [
      {"type":"hf","dataset":"sentence-transformers/natural-questions","split":"train","fields":["question","answer"],"format":"Q: {question}\nA: {answer}"},
      "wikipedia"
    ],
    "mix_strategy": "round_robin",
    "min_text_length": 50
  }
}
```

### Curated Defaults
`00_best_general.json` (default on Enter) and `01_best_qa.json` are maintained as opinionated starting points; users can still add new presets without touching code.

## Training Loop Flow (main.py)
1. Rank 0 interactive preset selection; broadcast JSON string.
2. Build tokenizer (adds pad if missing) and construct `DataConfig` (forced train_tokens=100M).
3. Instantiate DataLoader (streaming) with mixing strategy.
4. Build model; optionally extend positional embeddings if resuming larger seq length.
5. Load `lm_current.pth` if present (strict=False tolerant load).
6. DDP wrap (if multi-GPU); configure optimizer (AdamW, optional fused variant).
7. Iterate streaming batches accumulating gradients.
8. Log step metrics (tokens, tokens/sec, loss) and periodic evaluation (CE + PPL parity check).
9. On completion: final eval + rolling checkpoint rotate.

## Observed Performance (Recent 100M Token Run)
- Peak throughput ~76k tokens/sec on 2×3090 (bf16/TF32 mix).
- CE improved from ~5.10 at 1.3M tokens to ~4.42 mid-run; final eval CE ~4.66.
- Loss trend remained smooth; no instability spikes.
- Q/A + Wikipedia mix produced coherent question formats and geographic/place semantic structure (even when incorrect factual answer).

## Design Intent Going Forward
Focus: Improve internal efficiency and reasoning depth without hardcoding answers or adding external retrieval scaffolding.

### Near-Term Enhancements
1. Adapter Reintegration (full planner): Use existing auxiliary heads plus router for lookahead-based token reweighting (inference path gated, training regularized).
2. Multi-step Latent Rollouts: Extend dynamics to chained 2–3 steps with value aggregation.
3. Memory Compression: Learned rolling summary token & reconstruction loss.
4. Speculative Draft Decoding: Lightweight draft module with verification to accelerate generation & internal rollouts.
5. Expanded Curriculum: Source weighting already present—extend to difficulty buckets (loss percentile based).

### Mid-Term Enhancements
- Dynamic sequence extension with position interpolation plus selective layer re-scaling.
- In-graph speculative decoding using cached adapter-guided drafts (fallback to exact corrections when mismatch beyond entropy threshold).
- Latent Consistency Distillation: Teacher = short multi-rollout ensemble; Student = single forward pass.

### Long-Term Research Directions
- Unified Planning/Language Latent: Joint objective combining next-token CE, value prediction of future information gain, and dynamics consistency.
- Sparse Mixture-of-Experts gating for targeted expansion without bloating baseline parameter count.
- On-the-fly difficulty estimation: adjust sampling temperature and per-source weighting adaptively.

## Non-Goals
- No prompt-engineered task-specific hacks.
- No external retrieval or tool APIs in core training loop.
- No synthetic data fallback or label injection.

## Contribution Guidelines
- Add/modify behavior ONLY via preset JSONs or internal modular components (model/data registry/adapters).
- Preserve zero-argument `main.py` UX.
- Any new module must degrade gracefully if omitted (no hard dependency unless core).

## Quick Start
```
# Multi-GPU
torchrun --nproc_per_node=2 main.py
# Single GPU
python main.py
```
Select a preset, let it run to 100M tokens, and the rolling checkpoints appear under `checkpoints/`.

## Troubleshooting
- All sources failed → verify network / dataset names; no synthetic fallback.
- Slow throughput → reduce `per_gpu_batch_size` or workers; ensure TF32 is enabled (Ampere+ GPUs).
- Missing pad token → tokenizer adds pad mapped to eos automatically.

---
Last updated: 2025-09-24
