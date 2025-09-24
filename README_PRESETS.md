# One-Command Training Presets (No Runtime Params)

DO NOT ADD RUNTIME PARAMETERS!!

- Choose a preset interactively at startup
- All defaults are printed and used as-is
- To change behavior, edit or add a JSON under `configs/`

## Run

Multi-GPU (recommended):

```
torchrun --nproc_per_node=2 main.py
```

Single GPU:

```
python main.py
```

Rank 0 presents a menu of presets from `configs/*.json`, broadcasts your choice to all ranks, and training starts. No CLI flags needed.

## Add or modify presets

Drop a new JSON into `configs/` with this shape:

```
{
  "name": "my_mix",
  "description": "Short description",
  "model": {
    "hidden_size": 256,
    "num_layers": 3,
    "num_heads": 4,
    "seq_length": 256
  },
  "train": {
    "train_tokens": 40000000,
    "eval_tokens": 2000000,
    "per_gpu_batch_size": 96,
    "accumulation_steps": 6,
    "lr": 0.0005,
    "weight_decay": 0.01,
    "fused_adamw": false,
    "grad_checkpointing": false,
    "num_workers": 2,
    "prefetch_factor": 2
  },
  "data": {
    "sources": ["wikipedia", "openwebtext"],
    "min_text_length": 50
  }
}
```

Supported `data.sources` values (from the DataRegistry): `wikipedia`, `openwebtext`, `c4`, `bookcorpus`, `pile_subset`, `common_crawl`.

### Add your own datasets (plugin-style)

Add custom sources directly in your preset via `data.custom_sources`.

- Hugging Face streaming dataset (no scripts):

```
"data": {
  "sources": [],
  "custom_sources": [
    {"type": "hf", "dataset": "wikimedia/wikipedia", "split": "train", "text_field": "text", "config": "20231101.en"}
  ]
}
```

- Local PDF directory (recursively loads all PDFs with pypdf):

```
"data": {
  "sources": [],
  "custom_sources": [
    {"type": "pdf", "path": "./my_pdfs"}
  ]
}
```

Notes:
- `sources` and `custom_sources` are combined; you can use either or both.
- For `hf` sources, set `dataset`, optional `config`, and `text_field` if different.
- PDF extraction uses pypdf’s `extract_text()` and yields ~3–5k char chunks.

### Mixing across multiple sources

The loader supports three mix strategies across sources:

- `sequential` (default previously): consume each source to budget in order
- `round_robin` (default now): alternate across sources evenly
- `weighted`: alternate according to `data.source_weights` (same length as sources)

Example (favor Q&A 3x over Wikipedia):

```
"data": {
  "sources": [
    {"type": "hf", "dataset": "sentence-transformers/natural-questions", "split": "train",
     "fields": ["question", "answer"], "format": "Q: {question}\nA: {answer}"},
    "wikipedia"
  ],
  "mix_strategy": "weighted",
  "source_weights": [3, 1]
}
```

Q&A formatting using multiple fields:

```
{"type": "hf", "dataset": "sentence-transformers/natural-questions", "split": "train",
 "fields": ["question", "answer"], "format": "Q: {question}\nA: {answer}"}
```
