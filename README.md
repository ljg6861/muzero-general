# MuZero-General (Focused LM Evolution)

Single interactive training entrypoint (`main.py`) with zero runtime flags. Edit JSON presets under `configs/` to change behavior.

## Highlights
- 1-command training (interactive preset selection)
- Fixed 100M token target per run
- Rolling checkpoints: `lm_current.pth`, `lm_prev.pth`, `lm_prev2.pth`
- Streaming real data (HF + PDF) with round-robin / sequential / weighted mixing
- Q/A formatting via multi-field Hugging Face dataset sources
- bf16 / TF32 acceleration, DDP multi-GPU, gradient accumulation
- No hardcoded tasks, no prompt hacks, no synthetic fallback

## Quick Start
```bash
# Multi-GPU
torchrun --nproc_per_node=2 main.py
# Single GPU
python main.py
```
Press Enter to accept the default top curated preset (`00_best_general.json`) or choose another.

## Presets
Curated presets (auto-discovered):
- `00_best_general.json` – balanced multi-domain + Q/A with auxiliaries + adaptive curriculum.
- `01_best_qa.json` – heavier Q/A weighting.

A preset JSON defines `model`, `train`, `data`, and optional `auxiliary` sections. Example snippet:
```json
{
  "name": "qa_plus_wiki",
  "model": {"hidden_size":256,"num_layers":3,"num_heads":4,"seq_length":256},
  "train": {"per_gpu_batch_size":128,"accumulation_steps":4,"eval_tokens":2000000},
  "data": {
    "sources": [
      {"type":"hf","dataset":"sentence-transformers/natural-questions","split":"train","fields":["question","answer"],"format":"Q: {question}\nA: {answer}"},
      "wikipedia"
    ],
    "mix_strategy": "round_robin",
    "min_text_length": 50
  },
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
    "adaptive_interval_tokens": 5000000
  }
}
```

## Data Mixing
- `round_robin` (default) – uniform interleave
- `sequential` – consume in order
- `weighted` + `source_weights` – e.g. `[3,1]` to favor the first source

## Adding Custom Sources
- Hugging Face streaming:
```json
{"type":"hf","dataset":"wikimedia/wikipedia","split":"train","text_field":"text","config":"20231101.en"}
```
- Multi-field Q/A:
```json
{"type":"hf","dataset":"sentence-transformers/natural-questions","split":"train","fields":["question","answer"],"format":"Q: {question}\nA: {answer}"}
```
- PDFs:
```json
{"type":"pdf","path":"./my_pdfs"}
```
Add these inside `data.sources` (inline dict) or `data.custom_sources`.

## Checkpoint Policy
Automatically rotates:
```
lm_prev2.pth <- lm_prev.pth <- lm_current.pth <- new save
```
Also writes `simple_lm_trained.pth` as a legacy alias. Resume always loads `lm_current.pth`.

## Generation
Periodic sample outputs with top-p nucleus sampling + n-gram (length 3) repeat blocking.

## Roadmap (High-Level)
See `.github/MODEL_INSTRUCTIONS.md` for full architecture: Tier 1 auxiliaries (policy, value, dynamics, span contrastive, adaptive source weighting) plus upcoming planning & memory modules.

## License
See `LICENSE`.

---
For full internal design & future direction: `.github/MODEL_INSTRUCTIONS.md`.
