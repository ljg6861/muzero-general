#!/usr/bin/env python3
"""
Math Tasks Dataset for Adapter Training
======================================
Generates simple math QA prompts (arithmetic, linear equations, sequences)
with answers. Provides:
 - input_ids: tokenized prompt + answer (for LM CE and policy)
 - attention_mask: mask over tokens
 - correctness: 1 if answer is correct, 0 if intentionally corrupted
 - raw_text: original text prompt (for router bootstrapping)
 - answer_start: index where the answer tokens begin (optional use)

Designed as an IterableDataset for infinite stream; stop conditions are
controlled by the outer training loop's token budget.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizer


@dataclass
class MathTaskConfig:
    seq_length: int = 256
    mix: Tuple[float, float, float] = (0.6, 0.25, 0.15)  # arithmetic, linear, sequence
    corrupt_prob: float = 0.15  # probability to corrupt the answer to create negatives
    max_arith: int = 999
    min_arith: int = 0
    # Linear: ax + b = c with integer solution if possible
    a_range: Tuple[int, int] = (1, 9)
    b_range: Tuple[int, int] = (-20, 20)
    c_range: Tuple[int, int] = (-50, 50)


class MathTasksIterable(IterableDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: MathTaskConfig):
        super().__init__()
        self.tok = tokenizer
        self.cfg = config
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def _fmt(self, question: str, answer: str) -> Tuple[str, str]:
        # Consistent prompt style
        # Keep explicit A: delimiter so we can locate answer start
        q = f"Q: {question}\nA: "
        return q, answer

    def _gen_arithmetic(self) -> Tuple[str, str, bool]:
        a = random.randint(self.cfg.min_arith, self.cfg.max_arith)
        b = random.randint(self.cfg.min_arith, self.cfg.max_arith)
        op = random.choice(['+', '-', '*'])
        if op == '+':
            val = a + b
        elif op == '-':
            val = a - b
        else:
            val = a * b
        q = f"{a} {op} {b} = ?"
        true_ans = str(val)
        is_correct = True
        # Corrupt with some probability
        if random.random() < self.cfg.corrupt_prob:
            delta = random.choice([1, 2, 3])
            wrong = val + random.choice([-delta, delta])
            # Avoid accidental equality
            if wrong == val:
                wrong += 1
            true_ans, is_correct = str(wrong), False
        return q, true_ans, is_correct

    def _gen_linear(self) -> Tuple[str, str, bool]:
        a = random.randint(*self.cfg.a_range)
        b = random.randint(*self.cfg.b_range)
        c = random.randint(*self.cfg.c_range)
        # Solve ax + b = c => x = (c-b)/a; prefer integer solutions sometimes
        x_val = (c - b) / a
        # Round to int if close
        if abs(x_val - round(x_val)) < 1e-6:
            x_str = str(int(round(x_val)))
        else:
            # keep one decimal
            x_str = f"{x_val:.3f}".rstrip('0').rstrip('.')
        q = f"Solve for x: {a}x + {b} = {c}"
        ans = x_str
        is_correct = True
        if random.random() < self.cfg.corrupt_prob:
            # small perturbation
            try:
                if '.' in ans:
                    val = float(ans)
                    val += random.choice([-0.5, 0.5, 1.0])
                    ans = f"{val:.3f}".rstrip('0').rstrip('.')
                else:
                    val = int(ans)
                    val += random.choice([-1, 1, 2, -2])
                    ans = str(val)
            except Exception:
                pass
            is_correct = False
        return q, ans, is_correct

    def _gen_sequence(self) -> Tuple[str, str, bool]:
        # Arithmetic sequence next term
        a1 = random.randint(1, 20)
        d = random.randint(1, 10)
        n = random.randint(3, 8)
        seq = [a1 + i * d for i in range(n)]
        q = f"Next term in sequence: {', '.join(map(str, seq))}, ?"
        ans_val = a1 + n * d
        ans = str(ans_val)
        is_correct = True
        if random.random() < self.cfg.corrupt_prob:
            ans = str(ans_val + random.choice([-2, -1, 1, 2]))
            is_correct = False
        return q, ans, is_correct

    def _sample_task(self) -> Tuple[str, str, bool]:
        r = random.random()
        a_p, l_p, s_p = self.cfg.mix
        if r < a_p:
            return self._gen_arithmetic()
        elif r < a_p + l_p:
            return self._gen_linear()
        else:
            return self._gen_sequence()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            q, ans, is_correct = self._sample_task()
            prompt, answer = self._fmt(q, ans)
            raw_text = prompt + answer
            # Tokenize
            # We also need the position where answer starts
            prompt_ids = self.tok.encode(prompt, add_special_tokens=True)
            answer_ids = self.tok.encode(answer, add_special_tokens=False)

            # Build full sequence (truncate/pad to seq_length)
            full = prompt_ids + answer_ids
            if len(full) > self.cfg.seq_length:
                full = full[: self.cfg.seq_length]
                # If we truncated answer fully, mark correctness as 0 to avoid misleading signal
                if len(prompt_ids) >= self.cfg.seq_length:
                    is_correct = 0
            attn = [1] * len(full)
            if len(full) < self.cfg.seq_length:
                pad = self.tok.pad_token_id or self.tok.eos_token_id
                full = full + [pad] * (self.cfg.seq_length - len(full))
                attn = attn + [0] * (self.cfg.seq_length - len(attn))

            input_ids = torch.tensor(full, dtype=torch.long)
            attention_mask = torch.tensor(attn, dtype=torch.long)
            answer_start = min(len(prompt_ids), self.cfg.seq_length - 1)
            correctness = torch.tensor(1 if is_correct else 0, dtype=torch.float32)

            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'answer_start': torch.tensor(answer_start, dtype=torch.long),
                'correctness': correctness,
                'raw_text': raw_text,
            }


def create_math_dataloader(tokenizer: PreTrainedTokenizer,
                           seq_length: int = 256,
                           batch_size: int = 64,
                           num_workers: int = 0,
                           corrupt_prob: float = 0.15) -> DataLoader:
    cfg = MathTaskConfig(seq_length=seq_length, corrupt_prob=corrupt_prob)
    ds = MathTasksIterable(tokenizer, cfg)
    # Note: persistent_workers only supported when num_workers > 0; leave False by default
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True)
