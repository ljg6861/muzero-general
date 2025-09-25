# Meta-Cognitive Language Model

A breakthrough architecture that separates **schema knowledge** from **factual recall** to enable small models to punch above their parameter class.

## 🧠 Architecture Overview

```
Question → Schema Inference → Memory Retrieval → Constrained Generation → Self-Validation → Answer
```

### Core Components

1. **Meta-Reasoning Engine** (`core/models/`)
   - **Schema Inference**: Predicts answer types from questions
   - **Type-Constrained Generation**: Filters outputs by semantic category  
   - **Self-Correction**: Validates answers against schemas
   - **Enhanced LM**: Complete meta-cognitive language model

2. **Hybrid Memory System** (`core/models/`)
   - **Entity-Relation Graph**: Structured fact storage with type info
   - **Passage Index**: Dense/sparse retrieval with FAISS
   - **Quantized Embeddings**: Memory-efficient representation

3. **Data Processing** (`core/data/`)
   - **Multi-source loading**: Natural Questions, MS MARCO, Wikipedia
   - **Fact extraction**: Structured fact mining from text
   - **Fact verification**: Consistency and accuracy validation

## 🚀 Quick Start

**Training is now simplified! Use the dedicated training script:**

```bash
# Single GPU training
python train.py configs/simple_nq.json

# Multi-GPU training  
torchrun --nproc_per_node=2 train.py configs/simple_wikitext.json
```

**Simple config format - just specify dataset and tokens:**
```json
{
    "description": "Natural Questions training",
    "dataset": "sentence-transformers/natural-questions", 
    "text_field": "query",
    "tokens": 10000000
}
```

**For benchmarks and evaluation, use main.py:**
```bash
python main.py  # Interactive config selection
```

## 📁 Project Structure

```
├── train.py                   # Simple training script (NEW!)
├── main.py                    # Benchmarks & evaluation  
├── configs/                   
│   ├── simple_*.json          # Simple training configs
│   └── router_quiz.json       # Complex benchmark configs
├── core/
│   ├── models/               # Neural architectures
│   │   ├── enhanced_lm.py    # Meta-cognitive language model
│   │   ├── schema_reasoning.py # Schema inference system
│   │   ├── hybrid_memory.py  # Entity-relation + passage memory
│   │   └── router_retriever.py # Query routing
│   └── data/                 # Data processing
│       ├── data_registry.py  # Multi-source data loading
│       ├── fact_extraction.py # Fact mining
│       └── fact_verifier.py  # Fact validation
├── tests/                    # Test suite
├── checkpoints/              # Model checkpoints
├── logs/                     # Training logs
└── archive/                  # Archived/unused files
```

## 🎯 Key Innovation

**Traditional LM**: Scale = Intelligence (bigger models → better performance)  
**Meta-Cognitive LM**: Structure = Intelligence (smarter reasoning → better performance)

- **40M parameters** with **100M+ model capabilities**
- **Explicit reasoning** about what TYPE of answer to give
- **External fact storage** separate from model parameters  
- **Self-aware validation** of generated outputs

## 📊 Training Configuration

The `router_quiz.json` config is optimized for fact learning:
- **50M tokens** for optimal parameter coverage
- **Enhanced data mix**: Natural Questions (4) + MS MARCO (2) + Wikipedia (1)  
- **Meta-reasoning**: Schema loss weight 0.2, constraint strength 0.3
- **Self-correction**: Validation threshold 0.7, refinement enabled

## 💡 Meta-Reasoning Example

**Question**: "What is the capital of France?"

1. **Schema Inference**: Recognizes this expects a `CAPITAL_CITY` type
2. **Memory Retrieval**: Queries entity-relation graph for France→capital facts
3. **Constrained Generation**: Filters output to city names only
4. **Self-Validation**: Confirms "Paris" matches expected city schema
5. **Result**: "Paris" (precise, type-appropriate, validated)

This is the future of AI: **models that think about their own thinking**! 🧠✨
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
