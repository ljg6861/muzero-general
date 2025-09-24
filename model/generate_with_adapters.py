#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoTokenizer

from muzero_adapters import SimpleLM, MuZeroLM
from model.data_registry import DataRegistry, DataConfig
from model.simple_rag import SimpleTFIDFIndex


def load_muzero_adapters(ckpt_path: str, seq_length: int, hidden_size: int, num_layers: int, num_heads: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_lm = SimpleLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_length,
    )

    # Build wrapper and then load state dict
    model = MuZeroLM(base_lm=base_lm)
    # Handle PyTorch 2.6+ weights_only default with safe globals
    try:
        state = torch.load(ckpt_path, map_location='cpu')
    except Exception:
        from torch.serialization import safe_globals
        import numpy
        with safe_globals([numpy._core.multiarray._reconstruct]):
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = state['model_state_dict'] if 'model_state_dict' in state else state
    # Handle possible DataParallel prefixes
    model_sd = model.state_dict()
    needs_module = any(k.startswith('module.') for k in model_sd.keys())
    has_module = any(k.startswith('module.') for k in sd.keys())
    if needs_module and not has_module:
        sd = {f'module.{k}': v for k, v in sd.items()}
    elif has_module and not needs_module:
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)} (showing up to 8): {missing[:8]}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)} (showing up to 8): {unexpected[:8]}")

    model.eval().to(device)
    return model, tokenizer, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Path to muzero adapters checkpoint (*.pt)')
    ap.add_argument('--seq_length', type=int, default=128)
    ap.add_argument('--hidden_size', type=int, default=256)
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--prompt', type=str, default='Explain how photosynthesis works in simple terms.')
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--plan_top_k', type=int, default=5)
    ap.add_argument('--plan_depth', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--repetition_penalty', type=float, default=1.0, help='Penalty >1.0 discourages repeats (e.g., 1.1)')
    ap.add_argument('--top_p', type=float, default=None, help='Nucleus sampling probability mass cutoff (e.g., 0.9)')
    ap.add_argument('--no_repeat_ngram_size', type=int, default=0, help='Avoid repeating n-grams of this size (e.g., 3)')
    ap.add_argument('--disable_planning', action='store_true', help='Disable planning and use LM sampling only')
    ap.add_argument('--sample_top_k', type=int, default=0, help='Optional top-k cap before sampling (0=off)')
    # RAG options
    ap.add_argument('--use_rag', action='store_true')
    ap.add_argument('--rag_docs', type=int, default=2000)
    ap.add_argument('--rag_k', type=int, default=3)
    args = ap.parse_args()

    model, tokenizer, device = load_muzero_adapters(
        args.checkpoint, args.seq_length, args.hidden_size, args.num_layers, args.num_heads
    )
    prompt = args.prompt
    if args.use_rag:
        # Build a tiny TF-IDF index from a small stream of text
        registry = DataRegistry()
        cfg = DataConfig(train_tokens=1_000_000, seq_length=args.seq_length, data_sources=['wikipedia'])
        sources = registry.get_working_sources(cfg.data_sources)
        def text_iter():
            for src in sources:
                for t in src.get_stream():
                    yield t
        index = SimpleTFIDFIndex()
        index.build_from_iter(text_iter(), max_docs=args.rag_docs)
        retrieved = index.retrieve(args.prompt, k=args.rag_k)
        contexts = "\n\n".join([d for _, d in retrieved])
        prompt = f"[Context]\n{contexts}\n\n[Question]\n{args.prompt}\n\n[Answer]"

    with torch.no_grad():
        out = model.generate_with_planning(
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            plan_top_k=args.plan_top_k,
            plan_depth=args.plan_depth,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            disable_planning=args.disable_planning,
            sample_top_k=args.sample_top_k,
        )
    print("\n=== OUTPUT ===\n")
    print(out)


if __name__ == '__main__':
    main()
