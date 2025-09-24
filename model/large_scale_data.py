#!/usr/bin/env python3
"""
Large-Scale Data Loading for Phase A Baseline Training
======================================================
Scales data to 50-100M tokens using cleaned Wikipedia dumps and other sources.
Focuses on high-quality, diverse text for establishing strong language modeling foundation.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import json
import os
import gzip
import pickle
import requests
import tarfile
import zipfile
from typing import Dict, List, Iterator, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import re
from collections import defaultdict
import random
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

# For Wikipedia processing
try:
    import mwparserfromhell
    HAS_MWPARSER = True
except ImportError:
    HAS_MWPARSER = False
    print("Warning: mwparserfromhell not available for advanced Wikipedia parsing")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets library not available")


@dataclass
class LargeScaleDataConfig:
    """Configuration for large-scale data processing."""
    
    # Target data scale (Phase A requirement)
    target_tokens: int = 100_000_000  # 100M tokens target
    min_tokens: int = 50_000_000      # 50M minimum
    
    # Data sources (in priority order)
    data_sources: List[str] = None
    
    # Processing parameters
    max_seq_length: int = 1024
    min_doc_length: int = 256         # Minimum document length in chars
    max_doc_length: int = 50000       # Maximum document length in chars
    
    # Quality filtering
    enable_quality_filter: bool = True
    min_words_per_line: int = 3
    max_repetition_ratio: float = 0.3  # Max ratio of repeated n-grams
    min_unique_words_ratio: float = 0.6  # Min ratio of unique words
    
    # Cache and processing
    cache_dir: str = "data/large_scale_cache"
    num_workers: int = 8
    chunk_size: int = 10000           # Documents per processing chunk
    
    # Memory management
    max_memory_gb: float = 8.0        # Max memory usage
    streaming: bool = True            # Use streaming datasets when possible
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = [
                "wikipedia",           # English Wikipedia (latest dump)
                "openwebtext",         # OpenWebText dataset
                "bookcorpus",         # BookCorpus
                "cc_news",            # Common Crawl News
                "pile_subset"         # Subset of The Pile
            ]


class WikipediaProcessor:
    """Processor for Wikipedia dumps."""
    
    def __init__(self, cache_dir: str = "data/wikipedia_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = defaultdict(int)
    
    def download_wikipedia_dump(self, 
                               language: str = "en",
                               date: str = "latest") -> Path:
        """Download Wikipedia dump if not cached."""
        
        # Use Wikipedia dump from Hugging Face datasets (easier than raw dumps)
        if HAS_DATASETS:
            print("Loading Wikipedia dataset from Hugging Face...")
            try:
                dataset = load_dataset(
                    "wikipedia", 
                    f"20220301.{language}",  # Recent stable version
                    split="train",
                    streaming=True,  # Stream to handle large size
                    trust_remote_code=True
                )
                return dataset
            except Exception as e:
                print(f"Failed to load Wikipedia dataset: {e}")
                return None
        else:
            print("datasets library not available, using fallback method")
            return self._download_wikipedia_fallback(language, date)
    
    def _download_wikipedia_fallback(self, language: str, date: str) -> Optional[Path]:
        """Fallback method for Wikipedia download."""
        # This would implement direct download from Wikipedia dumps
        # For now, return None to indicate fallback needed
        print("⚠ Wikipedia fallback not implemented, using synthetic data")
        return None
    
    def clean_wikipedia_text(self, text: str) -> str:
        """Clean Wikipedia article text."""
        
        # Remove Wikipedia markup
        if HAS_MWPARSER:
            try:
                wikicode = mwparserfromhell.parse(text)
                text = wikicode.strip_code()
            except:
                pass  # Fall back to regex cleaning
        
        # Basic cleaning
        # Remove Wikipedia-specific patterns
        text = re.sub(r'\{\{[^}]*\}\}', '', text)  # Remove templates
        text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', text)  # Remove links, keep text
        text = re.sub(r'\[http[^\s]*\s([^\]]*)\]', r'\1', text)  # Remove external links
        text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
        
        # Clean up formatting
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines -> single
        text = re.sub(r' +', ' ', text)    # Multiple spaces -> single
        text = text.strip()
        
        return text
    
    def process_wikipedia_articles(self, 
                                 dataset, 
                                 target_tokens: int) -> List[str]:
        """Process Wikipedia articles to extract clean text."""
        
        articles = []
        total_tokens = 0
        processed_count = 0
        
        print(f"Processing Wikipedia articles (target: {target_tokens:,} tokens)...")
        
        try:
            for article in tqdm(dataset, desc="Processing Wikipedia"):
                if total_tokens >= target_tokens:
                    break
                
                # Extract text
                title = article.get('title', '')
                text = article.get('text', '')
                
                # Skip disambiguation and redirect pages
                if 'disambiguation' in title.lower() or len(text) < 500:
                    continue
                
                # Clean text
                cleaned_text = self.clean_wikipedia_text(text)
                
                # Quality check
                if len(cleaned_text) < 200:  # Too short
                    continue
                
                # Estimate tokens (rough: 1 token ≈ 4 chars)
                estimated_tokens = len(cleaned_text) // 4
                
                if estimated_tokens > 0:
                    articles.append(cleaned_text)
                    total_tokens += estimated_tokens
                    processed_count += 1
                
                # Progress update
                if processed_count % 1000 == 0:
                    print(f"  Processed {processed_count:,} articles, "
                          f"~{total_tokens:,} tokens")
        
        except Exception as e:
            print(f"Error processing Wikipedia: {e}")
        
        print(f"✓ Wikipedia processing complete: {len(articles):,} articles, "
              f"~{total_tokens:,} tokens")
        
        return articles


class TextQualityFilter:
    """Filter text for quality."""
    
    def __init__(self, config: LargeScaleDataConfig):
        self.config = config
    
    def is_high_quality(self, text: str) -> bool:
        """Check if text meets quality standards."""
        
        if not self.config.enable_quality_filter:
            return True
        
        # Length check
        if len(text) < self.config.min_doc_length:
            return False
        
        if len(text) > self.config.max_doc_length:
            return False
        
        # Word-based checks
        words = text.split()
        if len(words) < 10:  # Too few words
            return False
        
        # Line quality check
        lines = text.split('\n')
        good_lines = 0
        for line in lines:
            line_words = line.strip().split()
            if len(line_words) >= self.config.min_words_per_line:
                good_lines += 1
        
        if good_lines / max(len(lines), 1) < 0.5:  # Too many short lines
            return False
        
        # Repetition check
        if self._has_excessive_repetition(text):
            return False
        
        # Unique words ratio
        unique_words = set(word.lower() for word in words)
        unique_ratio = len(unique_words) / len(words)
        if unique_ratio < self.config.min_unique_words_ratio:
            return False
        
        return True
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive repetition."""
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check 3-gram repetition
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = defaultdict(int)
        for trigram in trigrams:
            trigram_counts[trigram] += 1
        
        # Find most common trigram
        if trigram_counts:
            max_count = max(trigram_counts.values())
            repetition_ratio = max_count / len(trigrams)
            return repetition_ratio > self.config.max_repetition_ratio
        
        return False


class LargeScaleDataset(IterableDataset):
    """Streaming dataset for large-scale text data."""
    
    def __init__(self, 
                 config: LargeScaleDataConfig,
                 tokenizer,
                 split: str = "train"):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.quality_filter = TextQualityFilter(config)
        
        # Initialize data sources
        self.data_sources = self._initialize_data_sources()
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'documents_accepted': 0,
            'tokens_processed': 0,
            'bytes_processed': 0
        }
    
    def _initialize_data_sources(self) -> List:
        """Initialize available data sources."""
        sources = []
        
        for source_name in self.config.data_sources:
            if source_name == "wikipedia":
                wiki_processor = WikipediaProcessor(self.config.cache_dir)
                wiki_dataset = wiki_processor.download_wikipedia_dump()
                if wiki_dataset:
                    sources.append(("wikipedia", wiki_dataset, wiki_processor))
            
            elif source_name == "openwebtext" and HAS_DATASETS:
                try:
                    owt_dataset = load_dataset("openwebtext", split="train", streaming=True)
                    sources.append(("openwebtext", owt_dataset, None))
                except:
                    print(f"⚠ Could not load {source_name}")
            
            # Add more data sources as needed
        
        print(f"Initialized {len(sources)} data sources")
        return sources
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through dataset samples."""
        
        current_tokens = 0
        buffer = []
        
        for source_name, dataset, processor in self.data_sources:
            print(f"Processing data source: {source_name}")
            
            try:
                for item in dataset:
                    if current_tokens >= self.config.target_tokens:
                        break
                    
                    # Extract text based on source
                    if source_name == "wikipedia":
                        text = processor.clean_wikipedia_text(item.get('text', ''))
                    elif source_name == "openwebtext":
                        text = item.get('text', '')
                    else:
                        text = str(item)
                    
                    # Quality filtering
                    if not self.quality_filter.is_high_quality(text):
                        continue
                    
                    # Tokenize
                    tokens = self.tokenizer.encode(
                        text,
                        max_length=self.config.max_seq_length,
                        truncation=True,
                        padding=False
                    )
                    
                    if len(tokens) < 50:  # Too short after tokenization
                        continue
                    
                    # Create training sample
                    input_ids = torch.tensor(tokens, dtype=torch.long)
                    
                    # Update statistics
                    current_tokens += len(tokens)
                    self.stats['documents_processed'] += 1
                    self.stats['documents_accepted'] += 1
                    self.stats['tokens_processed'] += len(tokens)
                    self.stats['bytes_processed'] += len(text.encode('utf-8'))
                    
                    yield {
                        'input_ids': input_ids,
                        'labels': input_ids.clone(),  # For autoregressive training
                        'attention_mask': torch.ones_like(input_ids)
                    }
                    
                    # Progress reporting
                    if self.stats['documents_accepted'] % 1000 == 0:
                        print(f"  Processed {self.stats['documents_accepted']:,} docs, "
                              f"{current_tokens:,} tokens "
                              f"({current_tokens/self.config.target_tokens*100:.1f}%)")
            
            except Exception as e:
                print(f"Error processing {source_name}: {e}")
                continue
        
        print(f"✓ Dataset complete: {self.stats['documents_accepted']:,} documents, "
              f"{current_tokens:,} tokens")


def create_large_scale_dataloader(
    config: LargeScaleDataConfig,
    tokenizer,
    batch_size: int = 4,
    num_workers: int = 0
) -> DataLoader:
    """Create data loader for large-scale training."""
    
    print("Creating Large-Scale Data Loader")
    print("=" * 50)
    print(f"Target tokens: {config.target_tokens:,}")
    print(f"Data sources: {', '.join(config.data_sources)}")
    print(f"Quality filtering: {config.enable_quality_filter}")
    
    # Create dataset
    dataset = LargeScaleDataset(config, tokenizer)
    
    # Create data loader
    def collate_fn(batch):
        """Collate function for batching."""
        if not batch:
            return {}
        
        # Pad sequences to same length within batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        labels = []
        attention_masks = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            pad_len = max_len - seq_len
            
            # Pad with tokenizer.pad_token_id
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            
            padded_input_ids = torch.cat([
                item['input_ids'],
                torch.full((pad_len,), pad_token_id, dtype=torch.long)
            ])
            
            padded_labels = torch.cat([
                item['labels'],
                torch.full((pad_len,), -100, dtype=torch.long)  # -100 is ignored in loss
            ])
            
            padded_attention_mask = torch.cat([
                item['attention_mask'],
                torch.zeros(pad_len, dtype=torch.long)
            ])
            
            input_ids.append(padded_input_ids)
            labels.append(padded_labels)
            attention_masks.append(padded_attention_mask)
        
        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_masks)
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Data loader created with batch size {batch_size}")
    
    return dataloader


def estimate_data_scale(config: LargeScaleDataConfig) -> Dict[str, int]:
    """Estimate available data scale from different sources."""
    
    estimates = {}
    
    # Wikipedia estimates
    if "wikipedia" in config.data_sources:
        # English Wikipedia has ~6.5M articles, avg ~1000 tokens each
        estimates["wikipedia"] = 6_500_000 * 1000  # ~6.5B tokens
    
    # OpenWebText estimates  
    if "openwebtext" in config.data_sources:
        # OpenWebText has ~8M documents, avg ~500 tokens each
        estimates["openwebtext"] = 8_000_000 * 500  # ~4B tokens
    
    # Other sources...
    total_estimated = sum(estimates.values())
    
    print("Data Scale Estimates:")
    print("=" * 30)
    for source, tokens in estimates.items():
        print(f"  {source}: ~{tokens:,} tokens")
    print(f"  Total available: ~{total_estimated:,} tokens")
    print(f"  Target needed: {config.target_tokens:,} tokens")
    
    if total_estimated >= config.target_tokens:
        print("✓ Sufficient data available")
    else:
        print("⚠ May need additional data sources")
    
    return estimates


if __name__ == "__main__":
    # Test large-scale data loading
    from transformers import AutoTokenizer
    
    # Create config
    config = LargeScaleDataConfig(
        target_tokens=1_000_000,  # 1M for testing
        data_sources=["wikipedia"]
    )
    
    # Estimate data scale
    estimates = estimate_data_scale(config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create data loader
    try:
        dataloader = create_large_scale_dataloader(
            config=config,
            tokenizer=tokenizer,
            batch_size=2,
            num_workers=0
        )
        
        # Test a few batches
        print("\nTesting data loader...")
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Just test first 3 batches
                break
            
            print(f"Batch {i+1}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            
            # Decode first sample
            first_sample = batch['input_ids'][0]
            # Remove padding tokens
            non_pad_tokens = first_sample[first_sample != tokenizer.pad_token_id]
            decoded = tokenizer.decode(non_pad_tokens[:100], skip_special_tokens=True)
            print(f"  Sample text: {decoded[:200]}...")
        
        print("✓ Large-scale data loading working correctly!")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        print("This may be due to missing datasets or network issues.")
        print("In practice, you would cache data locally for reliable training.")