UI for inspecting and testing the Transformer models

Features:
- Visualize learned token embeddings (PCA/TSNE)
- Explore a similarity graph of tokens (pyvis interactive graph)
- Logit Lens preview per layer (top tokens for a selected position)
- Text completion playground (top-k / top-p sampling)
- Launch training: baseline-continue or muzero-adapters

Quick start:
1) Install extras (in your venv):
   pip install gradio scikit-learn matplotlib plotly pyvis
2) Run the UI:
   python -m ui.app --checkpoint baseline_continue_checkpoints/<your>.pt

Notes:
- Checkpoint loader resizes positional embeddings automatically.
- For large vocabs, limit the number of tokens to visualize with --viz-vocab N.
