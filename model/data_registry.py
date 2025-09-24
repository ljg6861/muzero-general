"""
Robust Data Registry for Phase A Training
=========================================
Single registry supporting HF streaming datasets with no local scripts.
Implements SizedDataset and StreamDataset patterns for proper token accounting.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import PreTrainedTokenizer
from typing import Iterator, Optional, Dict, Any, List, Union
import logging
import time
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from pathlib import Path
import os


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    # Token budgets
    train_tokens: int = 2_000_000_000  # 2B tokens for training
    eval_tokens: int = 10_000_000      # 10M tokens for evaluation
    warmup_tokens: int = 100_000_000   # 100M tokens for warmup
    
    # Sequence settings
    seq_length: int = 1024
    min_text_length: int = 50
    
    # Sources (in priority order)
    data_sources: List[str] = None
    # Custom sources (plugin style). Each item is a dict with keys like:
    # {"type": "hf", "dataset": "owner/name", "split": "train", "text_field": "text", "config": "en"}
    # {"type": "pdf", "path": "./my_pdfs"}
    custom_sources: List[Dict[str, Any]] = None
    # Mixing strategy across sources: 'round_robin', 'sequential', or 'weighted'
    mix_strategy: str = 'round_robin'
    # Optional weights aligned with working sources (only used for 'weighted')
    source_weights: Optional[List[float]] = None
    
    # Fallback settings
    allow_fallback_synthetic: bool = False
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = [
                'wikipedia',
                'openwebtext', 
                'c4',
                'bookcorpus'
            ]
        if self.custom_sources is None:
            self.custom_sources = []
        if self.mix_strategy not in ('round_robin', 'sequential', 'weighted'):
            self.mix_strategy = 'round_robin'


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def get_stream(self) -> Iterator[str]:
        """Get streaming iterator of text strings."""
        pass
    
    @abstractmethod
    def estimate_tokens(self) -> int:
        """Estimate total available tokens."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Source name for logging."""
        pass


class HuggingFaceDataSource(DataSource):
    """Streaming data source from Hugging Face datasets."""
    
    def __init__(self, dataset_name: str, split: str = 'train', 
                 text_field: Optional[str] = 'text', config_name: Optional[str] = None,
                 fields: Optional[List[str]] = None, fmt: Optional[str] = None):
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.config_name = config_name
        self._dataset = None
        self.fields = fields or None
        self.fmt = fmt or None
        
    @property
    def name(self) -> str:
        return f"{self.dataset_name}({self.split})"
    
    def get_stream(self) -> Iterator[str]:
        """Get streaming iterator from HF dataset."""
        try:
            from datasets import load_dataset
            
            # Retry on transient network failures
            max_attempts = 6
            backoff = 2.0
            last_err = None
            if self._dataset is None:
                for attempt in range(1, max_attempts + 1):
                    try:
                        # Load dataset in streaming mode (no scripts)
                        self._dataset = load_dataset(
                            self.dataset_name,
                            name=self.config_name,
                            split=self.split,
                            streaming=True,
                            trust_remote_code=False  # Explicitly disable scripts
                        )
                        break
                    except Exception as e:
                        last_err = e
                        # If it’s a dataset scripts error, do not retry
                        msg = str(e)
                        non_retryable_markers = [
                            'Dataset scripts are no longer supported',
                            'found wikipedia.py',
                            "BuilderConfig '"  # missing config
                        ]
                        if any(marker in msg for marker in non_retryable_markers):
                            logging.error(f"[{self.name}] non-retryable error: {e}")
                            break
                        # Treat 429 Too Many Requests as retryable
                        if '429' in msg or 'Too Many Requests' in msg:
                            logging.warning(f"[{self.name}] rate limited (attempt {attempt}/{max_attempts}): {e}")
                        else:
                            logging.error(f"[{self.name}] streaming load attempt {attempt}/{max_attempts} failed: {e}")
                        if attempt < max_attempts:
                            # Exponential backoff with jitter
                            sleep_s = backoff * (2 ** (attempt - 1))
                            sleep_s = sleep_s * (0.5 + random.random())
                            time.sleep(sleep_s)
            
            if self._dataset is None:
                raise RuntimeError(last_err or f"Unknown error loading {self.name}")
            
            for item in self._dataset:
                # Build text either from text_field or (fields + fmt)
                text: Optional[str] = None
                if self.text_field and self.text_field in item:
                    text = item[self.text_field]
                elif self.fields and self.fmt:
                    try:
                        values = {}
                        for k in self.fields:
                            v = item.get(k, "")
                            # If v is a list/tuple, take first element
                            if isinstance(v, (list, tuple)):
                                v = v[0] if len(v) > 0 else ""
                            # If v is dict, stringify
                            if isinstance(v, dict):
                                v = str(v)
                            values[k] = v
                        text = self.fmt.format(**values)
                    except Exception:
                        text = None
                if isinstance(text, str) and len(text.strip()) > 50:
                        # Truncate very long texts to prevent memory issues
                        # Rough estimate: 4 chars per token, so limit to ~4K chars for 1K tokens
                        max_chars = 4000
                        if len(text) > max_chars:
                            # Find a good breaking point (sentence end)
                            truncated = text[:max_chars]
                            last_period = truncated.rfind('.')
                            last_exclamation = truncated.rfind('!')
                            last_question = truncated.rfind('?')
                            
                            # Use the latest sentence ending, or just truncate
                            break_point = max(last_period, last_exclamation, last_question)
                            if break_point > max_chars // 2:  # If we found a reasonable break point
                                text = text[:break_point + 1]
                            else:
                                text = text[:max_chars]
                        
                        yield text.strip()
                        
        except Exception as e:
            logging.error(f"Failed to load {self.name}: {e}")
            return
    
    def estimate_tokens(self) -> int:
        """Estimate tokens based on dataset."""
        # Conservative estimates based on known datasets
        estimates = {
            'wikipedia': 3_000_000_000,    # ~3B tokens
            'openwebtext': 8_000_000_000,  # ~8B tokens  
            'c4': 100_000_000_000,         # ~100B tokens
            'bookcorpus': 1_000_000_000,   # ~1B tokens
        }
        
        base_name = self.dataset_name.split('/')[-1].lower()
        for known_name, estimate in estimates.items():
            if known_name in base_name:
                return estimate
        
        return 100_000_000  # Default 100M


class PDFDirectoryDataSource(DataSource):
    """Stream text from all PDFs in a directory tree using pypdf."""
    def __init__(self, directory: str):
        self.directory = directory
        self._pdf_paths: List[Path] = []
        d = Path(directory)
        if d.exists():
            self._pdf_paths = [p for p in d.rglob('*.pdf') if p.is_file()]
    
    @property
    def name(self) -> str:
        return f"pdf:{self.directory}"
    
    def get_stream(self) -> Iterator[str]:
        try:
            import importlib
            pypdf = importlib.import_module('pypdf')
            PdfReader = getattr(pypdf, 'PdfReader', None)
            if PdfReader is None:
                raise ImportError("PdfReader not found in pypdf")
        except Exception as e:
            logging.error(f"pypdf not available: {e}")
            return
        for pdf_path in self._pdf_paths:
            try:
                reader = PdfReader(str(pdf_path))
                # Accumulate per page; yield chunks of reasonable size
                buf = []
                for page in reader.pages:
                    text = page.extract_text() or ""
                    text = text.strip()
                    if text:
                        buf.append(text)
                        # Yield in ~3-5k char chunks
                        if sum(len(x) for x in buf) > 4000:
                            yield "\n\n".join(buf)
                            buf = []
                if buf:
                    yield "\n\n".join(buf)
            except Exception as e:
                logging.warning(f"Failed to parse PDF {pdf_path}: {e}")
                continue
    
    def estimate_tokens(self) -> int:
        # Very rough estimate: 1 page ~ 300 tokens, assume 10 pages per PDF
        return max(100_000, 3_000 * max(1, len(self._pdf_paths)))


# SYNTHETIC DATA CLASS REMOVED - REAL DATA ONLY

class StreamDataset(IterableDataset):
    """Streaming dataset without __len__ for token-budget training."""
    
    def __init__(self, data_sources: List[DataSource], tokenizer: PreTrainedTokenizer, 
                 config: DataConfig, is_eval: bool = False):
        self.data_sources = data_sources
        self.tokenizer = tokenizer
        self.config = config
        self.is_eval = is_eval
        # Dynamic (adaptive) weights can be set externally; default None
        self.dynamic_weights: Optional[List[float]] = None
        
        # Set token budget
        self.token_budget = config.eval_tokens if is_eval else config.train_tokens
        
        logging.info(f"Created StreamDataset:")
        logging.info(f"  Sources: {[ds.name for ds in data_sources]}")
        logging.info(f"  Token budget: {self.token_budget:,}")
        logging.info(f"  Sequence length: {config.seq_length}")
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through tokenized sequences until token budget exhausted."""
        
        tokens_processed = 0
        
        # Determine distributed rank/world_size and worker shard
        try:
            import torch.distributed as dist
            ddp_initialized = dist.is_available() and dist.is_initialized()
        except Exception:
            ddp_initialized = False
        rank = dist.get_rank() if ddp_initialized else 0
        world_size = dist.get_world_size() if ddp_initialized else 1
        
        worker_info = torch.utils.data.get_worker_info()
        # Compute global sharding parameters across DDP ranks and dataloader workers
        if worker_info is not None:
            num_workers = max(1, int(worker_info.num_workers))
            local_worker_id = int(worker_info.id)
        else:
            num_workers = 1
            local_worker_id = 0
        global_world_size = max(1, world_size * num_workers)
        global_worker_id = rank * num_workers + local_worker_id
        def _yield_from_text(text: str) -> Iterator[Dict[str, torch.Tensor]]:
            nonlocal tokens_processed
            # Skip short texts
            if len(text) < self.config.min_text_length:
                return
            # Tokenize with proper truncation to prevent oversized sequences
            try:
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=self.config.seq_length * 8,
                    truncation=True
                )
            except Exception as e:
                logging.warning(f"Tokenization failed for text: {e}")
                return
            if len(tokens) < self.config.min_text_length:
                return
            for i in range(0, len(tokens), self.config.seq_length):
                if tokens_processed >= self.token_budget:
                    break
                chunk = tokens[i:i + self.config.seq_length]
                if len(chunk) < self.config.seq_length // 4:
                    continue
                if len(chunk) > self.config.seq_length:
                    chunk = chunk[:self.config.seq_length]
                elif len(chunk) < self.config.seq_length:
                    pad_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    chunk.extend([pad_token] * (self.config.seq_length - len(chunk)))
                assert len(chunk) == self.config.seq_length
                input_ids = torch.tensor(chunk, dtype=torch.long)
                labels = input_ids.clone()
                attention_mask = torch.ones(len(chunk), dtype=torch.long)
                if self.tokenizer.pad_token_id is not None:
                    labels[input_ids == self.tokenizer.pad_token_id] = -100
                tokens_processed += len(chunk)
                yield {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}

        # Helper to get next sharded text from a source
        def _next_text_sharded(state: Dict[str, Any]) -> Optional[str]:
            it = state['it']
            idx = state['idx']
            while True:
                try:
                    text = next(it)
                except StopIteration:
                    state['exhausted'] = True
                    return None
                idx += 1
                state['idx'] = idx
                if (idx % global_world_size) != global_worker_id:
                    continue
                return text

        # Build per-source iterators and indices
        states = []
        for src in self.data_sources:
            states.append({'name': src.name, 'it': iter(src.get_stream()), 'idx': 0, 'exhausted': False})

        if self.config.mix_strategy == 'sequential':
            for s in states:
                logging.info(f"Processing source (sequential): {s['name']}")
                while tokens_processed < self.token_budget and not s['exhausted']:
                    text = _next_text_sharded(s)
                    if text is None:
                        break
                    for sample in _yield_from_text(text):
                        if tokens_processed >= self.token_budget:
                            break
                        # Attach source index metadata for adaptive weighting or analysis
                        sample['source_index'] = states.index(s)
                        yield sample
        else:
            # round_robin or weighted
            def build_order():
                if self.config.mix_strategy == 'weighted':
                    weights_src = self.dynamic_weights or self.config.source_weights
                    if weights_src:
                        weights = [max(0.0, float(w)) for w in weights_src]
                        if len(weights) != len(states) or sum(weights) <= 0:
                            weights = [1.0] * len(states)
                    else:
                        weights = [1.0] * len(states)
                    total = sum(weights)
                    slots = max(len(states), 100)
                    counts = [max(1, int(slots * (w / total))) for w in weights]
                    order_local: List[int] = []
                    for i, c in enumerate(counts):
                        order_local.extend([i] * c)
                    return order_local
                # round_robin fallback
                return list(range(len(states)))
            order: List[int] = build_order()

            active = [True] * len(states)
            pos = 0
            steps_without_progress = 0
            while tokens_processed < self.token_budget and any(active):
                # Rebuild order periodically if adaptive weights change
                if pos % 500 == 0 and self.config.mix_strategy == 'weighted':
                    order = build_order()
                si = order[pos % len(order)]
                pos += 1
                if not active[si]:
                    continue
                s = states[si]
                text = _next_text_sharded(s)
                if text is None:
                    active[si] = False
                    continue
                before = tokens_processed
                for sample in _yield_from_text(text):
                    if tokens_processed >= self.token_budget:
                        break
                    sample['source_index'] = si
                    yield sample
                if tokens_processed == before:
                    steps_without_progress += 1
                    if steps_without_progress > 10 * len(states):
                        # Avoid infinite loops on pathological data
                        break
                else:
                    steps_without_progress = 0
                    # progress made (samples already yielded by _yield_from_text)
        
        logging.info(f"StreamDataset completed: {tokens_processed:,} tokens processed")

    # External API to update adaptive mixing weights in-place
    def update_dynamic_weights(self, new_weights: List[float]):
        if not isinstance(new_weights, list):
            return
        if len(new_weights) != len(self.data_sources):
            return
        self.dynamic_weights = new_weights


class SizedDataset(Dataset):
    """Sized dataset with __len__ for epoch-based training."""
    
    def __init__(self, data_sources: List[DataSource], tokenizer: PreTrainedTokenizer,
                 config: DataConfig, is_eval: bool = False):
        self.data_sources = data_sources
        self.tokenizer = tokenizer
        self.config = config
        self.is_eval = is_eval
        
        # Pre-collect samples for sizing
        self.samples = []
        self._collect_samples()
        
    def _collect_samples(self):
        """Pre-collect samples to enable __len__."""
        
        token_budget = self.config.eval_tokens if self.is_eval else self.config.train_tokens
        tokens_collected = 0
        
        logging.info(f"Pre-collecting samples for SizedDataset...")
        
        for source in self.data_sources:
            if tokens_collected >= token_budget:
                break
                
            for text in source.get_stream():
                if tokens_collected >= token_budget:
                    break
                
                if len(text) < self.config.min_text_length:
                    continue
                
                # Tokenize with proper truncation
                try:
                    tokens = self.tokenizer.encode(
                        text,
                        add_special_tokens=True,
                        max_length=self.config.seq_length * 8,  # Allow up to 8 chunks per text
                        truncation=True
                    )
                except Exception as e:
                    logging.warning(f"Tokenization failed for text: {e}")
                    continue
                
                # Skip if too short
                if len(tokens) < self.config.min_text_length:
                    continue
                
                # Create chunks
                for i in range(0, len(tokens), self.config.seq_length):
                    if tokens_collected >= token_budget:
                        break
                    
                    chunk = tokens[i:i + self.config.seq_length]
                    
                    # Skip chunks that are too short
                    if len(chunk) < self.config.seq_length // 4:  # At least 25% of seq_length
                        continue
                    
                    # Truncate or pad to exact seq_length
                    if len(chunk) > self.config.seq_length:
                        chunk = chunk[:self.config.seq_length]
                    elif len(chunk) < self.config.seq_length:
                        pad_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                        chunk.extend([pad_token] * (self.config.seq_length - len(chunk)))
                    
                    self.samples.append(chunk)
                    tokens_collected += len(chunk)
        
        logging.info(f"SizedDataset collected: {len(self.samples)} samples, {tokens_collected:,} tokens")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.samples[idx]
        
        input_ids = torch.tensor(chunk, dtype=torch.long)
        labels = input_ids.clone()
        attention_mask = torch.ones(len(chunk), dtype=torch.long)
        
        # Don't compute loss on padding
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


class DataRegistry:
    """Central registry for managing data sources."""
    
    def __init__(self):
        # Updated dataset configurations for new HF dataset formats
        self.available_sources = {
            # Wikipedia: use official Wikimedia dataset (script-free)
            'wikipedia': lambda: HuggingFaceDataSource('wikimedia/wikipedia', split='train', text_field='text', config_name='20231101.en'),
            'wikipedia_alt': lambda: HuggingFaceDataSource('wikimedia/wikipedia', split='train', text_field='text', config_name='20231101.en'),
            
            # OpenWebText alternatives
            'openwebtext': lambda: HuggingFaceDataSource('Skylion007/openwebtext', split='train', text_field='text'),
            'openwebtext_alt': lambda: HuggingFaceDataSource('openwebtext', split='train', text_field='text'),
            
            # C4 should work as it's not script-based
            'c4': lambda: HuggingFaceDataSource('c4', split='train', text_field='text', config_name='en'),
            'c4_realnews': lambda: HuggingFaceDataSource('c4', split='train', text_field='text', config_name='realnewslike'),
            
            # Alternative text sources
            'bookcorpus': lambda: HuggingFaceDataSource('bookcorpus', split='train', text_field='text'),
            'pile_subset': lambda: HuggingFaceDataSource('monology/pile-uncopyrighted', split='train', text_field='text'),
            'common_crawl': lambda: HuggingFaceDataSource('oscar-corpus/OSCAR-2301', split='train', text_field='text', config_name='en'),
        }
    
    def _build_custom_source(self, spec: Dict[str, Any]) -> Optional[DataSource]:
        t = (spec or {}).get('type')
        if t == 'hf':
            dataset = spec.get('dataset')
            if not dataset:
                return None
            split = spec.get('split', 'train')
            text_field = spec.get('text_field')
            config_name = spec.get('config')
            fields = spec.get('fields')
            fmt = spec.get('format') or spec.get('fmt')
            return HuggingFaceDataSource(dataset_name=dataset, split=split, text_field=text_field, config_name=config_name, fields=fields, fmt=fmt)
        if t == 'pdf':
            path = spec.get('path')
            if not path:
                return None
            return PDFDirectoryDataSource(path)
        return None

    def get_working_sources(self, requested_sources: List[Union[str, Dict[str, Any]]], custom_sources: List[Dict[str, Any]] | None = None) -> List[DataSource]:
        """Get list of working data sources."""
        
        working_sources = []
        failed_sources = []
        
        for source_name in requested_sources or []:
            # Allow dict specs inline in sources
            if isinstance(source_name, dict):
                try:
                    s = self._build_custom_source(source_name)
                except Exception as e:
                    s = None
                if s is None:
                    failed_sources.append(str(source_name))
                    continue
                # Probe
                try:
                    it = s.get_stream()
                    c = 0
                    for _ in it:
                        c += 1
                        if c >= 1:
                            break
                    if c > 0:
                        working_sources.append(s)
                        logging.info(f"✓ inline custom source working: {s.name}")
                    else:
                        failed_sources.append(str(source_name))
                except Exception as e:
                    failed_sources.append(str(source_name))
                continue

            if source_name not in self.available_sources:
                failed_sources.append(source_name)
                continue
            
            try:
                source = self.available_sources[source_name]()
                
                # Test the source by getting one sample with timeout
                test_iter = source.get_stream()
                try:
                    # Try to get first item with a reasonable timeout approach
                    sample_count = 0
                    for item in test_iter:
                        sample_count += 1
                        if sample_count >= 1:  # Just test that we can get one item
                            break
                    
                    if sample_count > 0:
                        working_sources.append(source)
                        logging.info(f"✓ {source_name} source working")
                    else:
                        logging.error(f"✗ {source_name} source returned no data")
                        failed_sources.append(source_name)
                        
                except Exception as stream_error:
                    logging.error(f"✗ {source_name} streaming failed: {stream_error}")
                    failed_sources.append(source_name)
                
            except Exception as e:
                logging.error(f"✗ {source_name} source failed: {e}")
                failed_sources.append(source_name)
        
        # Try alternative sources if primary ones fail
        if not working_sources and failed_sources:
            logging.info("Trying alternative data sources...")
            
            alt_mapping = {
                'wikipedia': 'wikipedia_alt',
                'openwebtext': 'openwebtext_alt', 
                'c4': 'c4_realnews'
            }
            
            for failed_source in failed_sources:
                if failed_source in alt_mapping:
                    alt_name = alt_mapping[failed_source]
                    if alt_name in self.available_sources:
                        try:
                            alt_source = self.available_sources[alt_name]()
                            test_iter = alt_source.get_stream()
                            
                            sample_count = 0
                            for item in test_iter:
                                sample_count += 1
                                if sample_count >= 1:
                                    break
                            
                            if sample_count > 0:
                                working_sources.append(alt_source)
                                logging.info(f"✓ {alt_name} alternative source working")
                                if failed_source in failed_sources:
                                    failed_sources.remove(failed_source)
                                    
                        except Exception as e:
                            logging.error(f"✗ {alt_name} alternative failed: {e}")
        
        # Add custom plugin sources (HF/PDF)
        custom_sources = custom_sources or []
        for spec in custom_sources:
            try:
                src = self._build_custom_source(spec)
                if not src:
                    continue
                # quick probe
                test_iter = src.get_stream()
                count = 0
                for _ in test_iter:
                    count += 1
                    if count >= 1:
                        break
                if count > 0:
                    working_sources.append(src)
                    logging.info(f"✓ custom source working: {src.name}")
                else:
                    logging.error(f"✗ custom source returned no data: {spec}")
            except Exception as e:
                logging.error(f"✗ custom source failed {spec}: {e}")

        if failed_sources and not working_sources:
            logging.warning(f"Failed sources: {failed_sources}")
        
        return working_sources
    
    def create_dataloader(self, tokenizer: PreTrainedTokenizer, config: DataConfig,
                         batch_size: int = 1, is_eval: bool = False,
                         use_sized: bool = False,
                         num_workers: int = 0,
                         prefetch_factor: int | None = None,
                         persistent_workers: bool | None = None) -> DataLoader:
        """Create data loader with robust source handling."""
        
        # Get working sources
        working_sources = self.get_working_sources(config.data_sources, custom_sources=config.custom_sources)
        
        # Handle fallback - NO SYNTHETIC DATA ALLOWED
        if not working_sources:
            raise RuntimeError(
                f"❌ ALL DATA SOURCES FAILED: {config.data_sources}\n"
                f"❌ REAL DATA ONLY - No synthetic fallback allowed.\n"
                f"❌ Fix your data sources to continue training."
            )
        
        # Create dataset
        if use_sized:
            dataset = SizedDataset(working_sources, tokenizer, config, is_eval)
        else:
            dataset = StreamDataset(working_sources, tokenizer, config, is_eval)
        
        # Sensible defaults
        if num_workers is None:
            num_workers = 0
        if persistent_workers is None:
            persistent_workers = num_workers > 0
        # If num_workers == 0, PyTorch ignores prefetch_factor
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,   # Streaming data is already shuffled
            num_workers=num_workers,
            pin_memory=True, # Fast GPU transfer
            persistent_workers=persistent_workers,
        )
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
        
        return DataLoader(dataset, **loader_kwargs)


# Example usage and testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test data registry
    registry = DataRegistry()
    config = DataConfig(
        train_tokens=1_000_000,  # Small for testing
        eval_tokens=100_000,
        seq_length=512,
        data_sources=['wikipedia', 'openwebtext'],
        allow_fallback_synthetic=False  # REAL DATA ONLY
    )
    
    try:
        # Test streaming dataset
        print("\nTesting StreamDataset...")
        train_loader = registry.create_dataloader(
            tokenizer=tokenizer,
            config=config,
            batch_size=2,
            is_eval=False,
            use_sized=False
        )
        
        # Test a few batches
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            print(f"Batch {i+1}: {batch['input_ids'].shape}")
            print(f"Sample text: {tokenizer.decode(batch['input_ids'][0][:50], skip_special_tokens=True)}")
        
        print("\n✓ DataRegistry test completed successfully!")
        
    except Exception as e:
        print(f"✗ DataRegistry test failed: {e}")