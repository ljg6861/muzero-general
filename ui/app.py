import os
import json
import argparse
import torch
import gradio as gr
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
from pyvis.network import Network
from io import StringIO

from model.simple_lm_infer import SimpleLM, load_checkpoint_into_simple_lm


def load_model(checkpoint_path: str, seq_length: int = 256, hidden_size: int = 256, num_layers: int = 3, num_heads: int = 4):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SimpleLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_length
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        load_checkpoint_into_simple_lm(state_dict, model)

    return tokenizer, model


def visualize_embeddings(
    model: SimpleLM,
    tokenizer,
    top_n_tokens: int = 500,
    word_start_only: bool = True,
    alpha_only: bool = True,
    focus_token: str | None = None,
    knn: int = 5,
    layout_mode: str = 'pca',  # 'pca' (static) or 'force'
    physics: bool = False,
    random_seed: int = 42,
    stabilization_iters: int = 200,
):
    with torch.no_grad():
        emb = model.tok_embedding.weight.detach().cpu().numpy()
    vocab = tokenizer.get_vocab()
    # Build id->token list
    id_to_token = [None] * len(vocab)
    for tok, idx in vocab.items():
        if 0 <= idx < len(id_to_token):
            id_to_token[idx] = tok

    # Helper: user-facing label without BPE marker; keep raw for tooltips
    def display_token(tok: str) -> str:
        if tok is None:
            return ""
        if tok.startswith(('Ġ','▁')):
            return tok[1:]
        return tok

    # Build candidate token ids with filters
    def is_word_start(tok: str) -> bool:
        return tok.startswith('Ġ') or tok.startswith('▁')

    def is_alpha(tok: str) -> bool:
        # strip leading word-start marker and check alnum with at least one letter
        core = tok[1:] if tok.startswith(('Ġ','▁')) else tok
        return any(c.isalpha() for c in core)

    candidate_ids = []
    for idx, tok in enumerate(id_to_token):
        if tok is None:
            continue
        if word_start_only and not is_word_start(tok):
            continue
        if alpha_only and not is_alpha(tok):
            continue
        candidate_ids.append(idx)

    # Cap to top_n_tokens
    candidate_ids = candidate_ids[: max(1, min(top_n_tokens, len(candidate_ids)))]
    tokens = [id_to_token[i] for i in candidate_ids]  # raw tokens
    disp_tokens = [display_token(t) for t in tokens]
    E = emb[candidate_ids]

    # Optional focus: restrict to KNN of a chosen token
    if focus_token:
        try:
            # Accept plain words: try raw, then word-start variants, then encoding with leading space
            focus_id = tokenizer.convert_tokens_to_ids(focus_token)
            if focus_id == tokenizer.unk_token_id or focus_id < 0:
                focus_id = tokenizer.convert_tokens_to_ids('Ġ' + focus_token)
            if (focus_id == tokenizer.unk_token_id or focus_id < 0) and hasattr(tokenizer, 'convert_tokens_to_ids'):
                focus_id = tokenizer.convert_tokens_to_ids('▁' + focus_token)
            if focus_id == tokenizer.unk_token_id or focus_id < 0:
                enc = tokenizer(" " + focus_token, return_tensors='pt').input_ids[0].tolist()
                focus_id = enc[-1] if enc else -1
            if focus_id in candidate_ids:
                f_idx = candidate_ids.index(focus_id)
                Vn_all = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
                sims_focus = Vn_all @ Vn_all[f_idx]
                nn = np.argpartition(-sims_focus, min(knn, len(sims_focus)-1))[: min(knn+1, len(sims_focus))]
                # include focus + neighbors
                sub_idx = sorted(set(int(i) for i in nn))
                E = E[sub_idx]
                tokens = [tokens[i] for i in sub_idx]
                disp_tokens = [disp_tokens[i] for i in sub_idx]
                candidate_ids = [candidate_ids[i] for i in sub_idx]
        except Exception:
            pass

    # 2D projection (PCA)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(E)
    data = [{"x": float(coords[i,0]), "y": float(coords[i,1]), "token": disp_tokens[i] or str(candidate_ids[i]), "raw": tokens[i] or ""} for i in range(len(tokens))]

    # Similarity graph (k-NN by cosine) on a capped subset
    M = min(200, len(tokens))
    V = E[:M]
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    sims = Vn @ Vn.T
    np.fill_diagonal(sims, -np.inf)
    G = Network(height='500px', width='100%', directed=False, notebook=False)

    # Configure layout/physics options for vis.js via set_options
    # Use a deterministic seed to reduce layout jitter when using force layout
    layout_options = {
        "layout": {"improvedLayout": True, "randomSeed": random_seed},
        "physics": {
            "enabled": bool(physics),
            "stabilization": {"enabled": bool(physics), "iterations": stabilization_iters},
            "solver": "barnesHut",
            "barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.3, "springLength": 95},
        },
        "interaction": {"hover": True}
    }
    try:
        import json as _json
        G.set_options(_json.dumps(layout_options))
    except Exception:
        pass

    # Node placement: PCA static positions or force-directed
    if layout_mode == 'pca':
        # Scale PCA coords for better spacing in vis.js
        xs = coords[:M, 0]
        ys = coords[:M, 1]
        # Normalize to ~[-250, 250]
        def _scale(a):
            a = (a - a.mean()) / (a.std() + 1e-6)
            return a * 120.0
        xs = _scale(xs)
        ys = _scale(ys)
        for i in range(M):
            G.add_node(
                int(i),
                label=str(disp_tokens[i] or str(candidate_ids[i])),
                title=str(tokens[i] or str(candidate_ids[i])),
                x=float(xs[i]),
                y=float(ys[i]),
                fixed=True,
                physics=False,
            )
    else:
        for i in range(M):
            G.add_node(int(i), label=str(disp_tokens[i] or str(candidate_ids[i])), title=str(tokens[i] or str(candidate_ids[i])))
    k = max(1, int(knn))
    for i in range(M):
        nn_idx = np.argpartition(-sims[i], min(k, M-1))[: min(k, M-1)]
        for j in nn_idx:
            j_int = int(j)
            i_int = int(i)
            if i_int < j_int:
                G.add_edge(i_int, j_int, value=float(sims[i, j]))
    html_file = os.path.join('ui', 'graph.html')
    G.write_html(html_file)
    return data, html_file


def logit_lens(model: SimpleLM, tokenizer, text: str, position: int):
    device = next(model.parameters()).device
    ids = tokenizer(text, return_tensors='pt').input_ids[:, :model.max_seq_len].to(device)
    with torch.no_grad():
        # Collect layer-wise activations by manually stepping
        bsz, seqlen = ids.shape
        pos_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        x = model.tok_embedding(ids) + model.pos_embedding(pos_ids)
        attn_mask = model.causal_mask_full[:seqlen, :seqlen]
        hidden_states = [x.detach()]  # keep on device
        for layer in model.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=None)
            hidden_states.append(x.detach())
        # Project each layer hidden at position through lm_head
        pos = min(max(0, position), seqlen - 1)
        layer_top = []
        for h in hidden_states:
            logits = model.lm_head(h[:, pos, :])  # same device as model
            probs = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, k=5, dim=-1)
            idx_list = topi[0].detach().cpu().tolist()
            toks = [tokenizer.convert_ids_to_tokens([i])[0] for i in idx_list]
            vals = [float(v) for v in topv[0].detach().cpu().tolist()]
            layer_top.append({"tokens": toks, "probs": vals})
    return layer_top


def complete_text(model: SimpleLM, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float):
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    out_ids = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)


def launch_training(script: str, args_json: str):
    """Kick off a training run in the background. Returns command string."""
    try:
        args = json.loads(args_json) if args_json.strip() else {}
    except Exception as e:
        return f"Invalid args JSON: {e}"
    base = f".venv/bin/torchrun --nproc_per_node=2 --master_port=29550 {script}"
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                base += f" --{k}"
        else:
            base += f" --{k} {v}"
    # Run detached
    cmd = f"cd {os.getcwd()} && nohup {base} > ui/train_{os.path.basename(script)}.log 2>&1 &"
    os.system(cmd)
    return f"Launched: {base}\nLog: ui/train_{os.path.basename(script)}.log"


def build_ui(default_checkpoint: str):
    tokenizer, model = load_model(default_checkpoint)

    with gr.Blocks(title="Transformer Inspector") as demo:
        gr.Markdown("## Transformer Inspector\nExplore what the model learned, complete text, and launch training.")

        with gr.Tab("Embeddings & Graph"):
            with gr.Row():
                ckpt = gr.Textbox(value=default_checkpoint, label="Checkpoint path")
                viz_vocab = gr.Slider(100, 5000, value=800, step=50, label="#Tokens to visualize")
            with gr.Row():
                word_start = gr.Checkbox(value=True, label="Word-start tokens only (Ġ/▁)")
                alpha_only = gr.Checkbox(value=True, label="Alphabetic only")
                focus_tok = gr.Textbox(value="", label="Focus token (optional, plain ok)")
                knn = gr.Slider(2, 20, value=6, step=1, label="k-NN edges / focus size")
            with gr.Row():
                layout_choice = gr.Dropdown(choices=["Static (PCA)", "Force-Directed"], value="Static (PCA)", label="Graph layout")
                seed = gr.Number(value=42, precision=0, label="Random seed")
                stab_iters = gr.Slider(0, 2000, value=200, step=50, label="Stabilization iterations (force)")
                refresh = gr.Button("Load & Visualize")
            scatter_html = gr.HTML(label="Token Embeddings (PCA)")
            graph_html = gr.HTML(label="Similarity Graph (pyvis)")

            def do_viz(ckpt_path, topn, ws_only, a_only, focus, k, layout, rnd_seed, stab):
                nonlocal tokenizer, model
                tokenizer, model = load_model(ckpt_path)
                layout_mode = 'pca' if layout == 'Static (PCA)' else 'force'
                physics = True if layout_mode == 'force' else False
                pts, html_path = visualize_embeddings(
                    model, tokenizer,
                    int(topn), bool(ws_only), bool(a_only), focus if focus else None, int(k),
                    layout_mode=layout_mode, physics=physics, random_seed=int(rnd_seed), stabilization_iters=int(stab)
                )
                # Build Plotly HTML scatter
                try:
                    import plotly.graph_objects as go
                    xs = [p["x"] for p in pts]
                    ys = [p["y"] for p in pts]
                    labels = [p["token"] for p in pts]
                    fig = go.Figure(data=go.Scatter(x=xs, y=ys, mode='markers', text=labels, hovertext=labels, marker=dict(size=6)))
                    fig.update_layout(title='Token Embeddings (PCA)', xaxis=dict(visible=False), yaxis=dict(visible=False), height=500)
                    from plotly.io import to_html
                    scatter_html_str = to_html(fig, include_plotlyjs='cdn', full_html=False)
                except Exception as e:
                    scatter_html_str = f"<pre>Plotly error: {e}</pre>"
                # Read pyvis HTML and embed via base64 data URL in an iframe to avoid sanitization issues
                try:
                    import base64
                    with open(html_path, 'r', encoding='utf-8') as f:
                        pyvis_html = f.read()
                    b64 = base64.b64encode(pyvis_html.encode('utf-8')).decode('ascii')
                    iframe = f"<iframe src=\"data:text/html;base64,{b64}\" width=\"100%\" height=\"520\" style=\"border:0;\"></iframe>"
                except Exception as e:
                    iframe = f"<pre>Graph render error: {e}</pre>"
                return scatter_html_str, iframe

            refresh.click(do_viz, inputs=[ckpt, viz_vocab, word_start, alpha_only, focus_tok, knn, layout_choice, seed, stab_iters], outputs=[scatter_html, graph_html])

        with gr.Tab("Logit Lens"):
            text = gr.Textbox(value="The future of AI is", label="Input text")
            pos = gr.Slider(0, 128, value=5, step=1, label="Position")
            run_lens = gr.Button("Run Logit Lens")
            lens_out = gr.JSON(label="Top-5 predictions per layer at position")
            run_lens.click(lambda t, p: logit_lens(model, tokenizer, t, int(p)), inputs=[text, pos], outputs=lens_out)

        with gr.Tab("Completion"):
            prompt = gr.Textbox(value="The future of AI is", label="Prompt")
            max_tokens = gr.Slider(1, 256, value=64, step=1, label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
            top_k = gr.Slider(0, 100, value=50, step=1, label="Top-K")
            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-P")
            go = gr.Button("Generate")
            completion = gr.Textbox(label="Completion")
            go.click(lambda *a: complete_text(model, tokenizer, *a), inputs=[prompt, max_tokens, temperature, top_k, top_p], outputs=completion)

        with gr.Tab("Training"):
            gr.Markdown("Launch training jobs in background. Edit args as JSON.")
            with gr.Row():
                baseline_args = gr.Textbox(value=json.dumps({
                    "load_checkpoint": "", "train_tokens": 50_000_000, "seq_length": 128, "per_gpu_batch_size": 128, "accumulation_steps": 4
                }, indent=2), label="Baseline-Continue args (JSON)")
                adapters_args = gr.Textbox(value=json.dumps({
                    "load_lm_checkpoint": "baseline_checkpoints/baseline_50M_tokens_20250923_151005.pt",
                    "train_tokens": 20_000_000, "seq_length": 256, "per_gpu_batch_size": 64, "accumulation_steps": 8
                }, indent=2), label="MuZero-Adapters args (JSON)")
            with gr.Row():
                launch_baseline = gr.Button("Launch Baseline-Continue")
                launch_adapters = gr.Button("Launch MuZero-Adapters")
            out_baseline = gr.Textbox(label="Baseline launcher output")
            out_adapters = gr.Textbox(label="Adapters launcher output")
            launch_baseline.click(lambda a: launch_training('baseline_continue.py', a), inputs=baseline_args, outputs=out_baseline)
            launch_adapters.click(lambda a: launch_training('muzero_adapters.py', a), inputs=adapters_args, outputs=out_adapters)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui(args.checkpoint)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()
