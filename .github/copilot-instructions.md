# Copilot Instructions for Meta-Cognitive Language Model

## Project Overview
This is a meta-cognitive language model that separates **schema knowledge** from **factual recall** to enable small models (40M parameters) to achieve capabilities typically requiring 100M+ parameter models. The core innovation is explicit reasoning about what TYPE of answer to give, even when the model doesn't know specific facts.

## Architecture & Key Components

### Unified Model Architecture (`unified_model.py`)
- **Single unified model** containing ALL features - no interfaces or multiple models
- Contains transformer core, hybrid memory, meta-reasoning, schema inference, self-correction, router, and RL components
- Uses Flash/SDPA attention with gradient checkpointing
- Critical: Always import from `unified_model.py`, not deprecated models in `archive/old_models/`

### Core Data Pipeline (`core/data/`)
- **DataRegistry** (`data_registry.py`): Multi-source streaming datasets with token accounting
- **FactExtractor** (`fact_extraction.py`): Structured fact mining from text
- **FactVerifier** (`fact_verifier.py`): Consistency validation
- Supports HuggingFace streaming datasets with robust fallback mechanisms

### Hybrid Memory System (`core/models/hybrid_memory.py`)
- **Entity-Relation Graph**: Compressed factual storage with quantized embeddings
- **Passage Index**: FAISS-based dense/sparse retrieval (with PyTorch fallback)
- Memory footprint <1GB via uint32 packed structures
- Two-stage retrieval: entity linking → evidence passage retrieval

## Training & Configuration Patterns

### Configuration System
- **No runtime parameters** - all configs in `configs/*.json`
- Interactive preset selection at startup (rank 0 broadcasts to all ranks)
- Key configs: `router_quiz.json` (QA-heavy), `test_10m.json` (small scale)
- Config structure: `model` (architecture), `train` (hyperparams), `data` (sources + mixing)

### Training Commands
```bash
# Multi-GPU (recommended)
torchrun --nproc_per_node=2 main.py

# Single GPU
python main.py
```

### Data Mixing Strategy
- Weighted mixing: Natural Questions (4) + MS MARCO (2) + Wikipedia (1)
- Token budgets: 50M training tokens for optimal parameter coverage
- Schema loss weight: 0.2, constraint strength: 0.3, validation threshold: 0.7

## Testing & Evaluation

### Test Structure
- `tests/test_meta_reasoning.py`: Schema inference and type-constrained generation
- `tests/test_hybrid_memory.py`: Memory system functionality
- `simple_benchmark.py`: Unified model benchmarking

### Benchmark Script
```bash
# Fair evaluation (no external knowledge)
./run_fair_benchmark.sh --quick
```

### Key Metrics
- **Exact Match (EM)**: Primary QA metric
- **F1 Score**: Token-level overlap
- **Evidence Score**: Retrieval quality
- Baselines: 1B models (~26.5% EM), 3B models (~32.8% EM on NQ-Open)

## Development Workflows

### Model Checkpoints
- `checkpoints/lm_current.pth`: Latest trained model
- `checkpoints/lm_prev.pth`, `lm_prev2.pth`: Backup checkpoints
- Archive checkpoints: `archive/*_checkpoints/` with summary JSON files

### Environment Setup
- Python virtual environment with PyTorch 2.2+, transformers, sentence-transformers
- Optional: FAISS for optimized retrieval (graceful fallback to PyTorch)
- CUDA environment variables set for memory efficiency

### Key File Patterns
- **Single entry point**: `main.py` (no CLI args)
- **Unified architecture**: `unified_model.py` (replaces all old model files)
- **Config-driven**: JSON configs in `configs/` directory
- **Streaming data**: HuggingFace datasets with token accounting
- **Archive pattern**: Old implementations in `archive/` for reference

## Project-Specific Conventions

### Meta-Reasoning Pipeline
```
Question → Schema Inference → Memory Retrieval → Constrained Generation → Self-Validation → Answer
```

### Answer Type System
- Explicit schema inference: `PERSON`, `CAPITAL_CITY`, `CHEMICAL_FORMULA`, etc.
- Type-constrained generation filters outputs by semantic category
- Self-correction validates answers against inferred schemas

### Memory Integration
- Entity-relation facts stored separately from model parameters
- Quantized embeddings for memory efficiency
- Fast graph traversal: entity → relation → object lookup

### Error Handling
- Graceful degradation when FAISS unavailable
- Robust streaming with timeout handling
- Checkpoint recovery with architecture adjustment

## Critical Implementation Notes

- **Never add runtime parameters** - use config files only
- **Always use `unified_model.py`** - other model files are deprecated
- **DDP training** requires picklable entry points (`_auto_ddp_entry`)
- **Token accounting** is critical for proper data mixing
- **Schema inference** is the core innovation - not just another LM component