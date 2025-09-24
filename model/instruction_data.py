import random
from typing import Iterator, Dict, List
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizer

class InstructionTuningDataset(IterableDataset):
    """
    Create simple instruction/QA pairs from raw sentences by turning facts into Q/A.
    Pattern: "Instruction: <question>\nAnswer: <answer>" with supervised tokens limited to the answer span.
    This uses a provided text iterator (e.g., from DataRegistry) to generate prompts.
    """
    def __init__(self, text_iter: Iterator[str], tokenizer: PreTrainedTokenizer, seq_length: int = 128, max_samples: int | None = None):
        self.text_iter = text_iter
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        for text in self.text_iter:
            if self.max_samples is not None and count >= self.max_samples:
                break
            sents = split_sentences(text)
            for sent in sents:
                qa = sentence_to_qa(sent)
                if qa is None:
                    continue
                q, a = qa
                prompt = f"Instruction: {q}\nAnswer: "
                full = prompt + a
                enc = self.tokenizer(full, add_special_tokens=True, truncation=True, max_length=self.seq_length, return_tensors='pt')
                input_ids = enc['input_ids'][0]
                attn = enc['attention_mask'][0]
                # Mask labels except for answer tokens
                prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
                ans_len = len(self.tokenizer(a, add_special_tokens=False)['input_ids'])
                labels = input_ids.clone()
                # Find prompt length in tokenized full (best-effort by length)
                p = min(len(prompt_ids), len(input_ids))
                # Set -100 for tokens before the last ans_len tokens
                labels[:] = -100
                # Supervise last ans_len non-pad tokens
                # Find last non-pad position
                last = (attn == 1).nonzero(as_tuple=True)[0].max().item()
                start = max(0, last - ans_len + 1)
                labels[start:last+1] = input_ids[start:last+1]
                yield {
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attn,
                    'raw_question': q,
                    'raw_answer': a,
                }
                count += 1


def split_sentences(text: str) -> List[str]:
    # Very simple splitter; avoid heavy deps
    out = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in '.!?':
            s = ''.join(cur).strip()
            if len(s) > 10:
                out.append(s)
            cur = []
    if cur:
        s = ''.join(cur).strip()
        if len(s) > 10:
            out.append(s)
    return out[:10]


def sentence_to_qa(sent: str) -> tuple[str, str] | None:
    # Heuristic: turn named entities or numbers into questions
    # Simple patterns: dates, years, proper nouns (titlecase), counts
    words = sent.split()
    if len(words) < 6:
        return None
    # prefer a year
    for i, w in enumerate(words):
        if w.isdigit() and len(w) == 4 and 1500 <= int(w) <= 2100:
            q = sent.replace(w, 'What year').strip()
            if not q.endswith('?'):
                q += '?'
            return (q, w)
    # capitalize word as entity
    for i, w in enumerate(words):
        if w[0].isupper() and w[1:].islower() and w.isalpha():
            q = f"Who is {w}?"
            return (q, w)
    # fallback: ask about a noun-ish token (last content word)
    for w in reversed(words):
        if any(ch.isalpha() for ch in w):
            q = f"What is {w}?"
            return (q, w.strip('.,!?;:'))
    return None


def create_instruction_dataloader(tokenizer: PreTrainedTokenizer, registry, config, batch_size: int, num_workers: int = 0):
    # Build a text iterator from registry (streaming)
    sources = registry.get_working_sources(config.data_sources)
    def text_gen():
        for src in sources:
            for t in src.get_stream():
                yield t
    ds = InstructionTuningDataset(text_gen(), tokenizer, seq_length=config.seq_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
