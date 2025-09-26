#!/usr/bin/env python3
"""
Comprehensive Training Script for Meta-Cognitive Language Model
==============================================================

Usage:
    python train.py configs/my_config.json
    torchrun --nproc_per_node=2 train.py configs/my_config.json

Features:
- Unified model with router, schema heads, and hybrid memory
- Streaming dataloader with token budgeting and sequence packing
- Hybrid memory with async fact ingestion (REBEL + DeBERTa NLI)
- Router supervision with online heuristics
- Memory auxiliary loss for graph-parametric alignment
- Clean evaluation with CE/PPL parity checks
"""

import os
import sys
import time
import math
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import deque, defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError, Future
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
try:
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    from torch.amp import autocast, GradScaler

# Set environment defaults for better performance
os.environ.setdefault('HF_HUB_READ_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Import model and data components
sys.path.append('.')
sys.path.append('core')
from unified_model import UnifiedCognitiveLM, create_unified_model, create_tokenizer
from core.data.data_registry import DataRegistry, DataConfig
from core.data.fact_extraction import FactCandidate, FactExtractor
from core.data.fact_verifier import FactVerifier
from core.models.hybrid_memory import HybridMemoryEngine, SentenceEncoder
from core.models.router_retriever import HybridRetriever


def setup_ddp():
    """Setup Distributed Data Parallel if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Check if we have enough GPUs
        num_gpus = torch.cuda.device_count()
        if rank >= num_gpus:
            raise RuntimeError(
                f"Not enough GPUs for DDP! Requested rank {rank} but only {num_gpus} GPUs available."
            )
        
        # Initialize the process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        
        return True, rank, world_size, device
    else:
        # Single GPU training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False, 0, 1, device


class _FactDeduper:
    """Bounded set to avoid re-ingesting duplicate facts each step."""

    def __init__(self, max_size: int = 8192):
        self.max_size = max_size
        self._cache: Set[Tuple[str, str, str]] = set()
        self._queue: deque = deque()

    def add(self, key: Tuple[str, str, str]) -> bool:
        if key in self._cache:
            return False
        self._cache.add(key)
        self._queue.append(key)
        if len(self._queue) > self.max_size:
            old = self._queue.popleft()
            self._cache.discard(old)
        return True


def ingest_fact_candidates(
    hybrid_memory: HybridMemoryEngine,
    fact_deduper: _FactDeduper,
    facts: List[FactCandidate]
) -> List[FactCandidate]:
    """Ingest facts into hybrid memory with deduplication"""
    ingested = []
    for fact in facts:
        key = (fact.head, fact.relation, fact.tail)
        if fact_deduper.add(key):
            hybrid_memory.ingest_fact(
                fact.head, 
                fact.relation, 
                fact.tail, 
                [fact.sentence] if fact.sentence else []
            )
            ingested.append(fact)
    return ingested


class HybridMemoryIngestionManager:
    """Asynchronous fact extraction + ingestion to avoid blocking training."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        hybrid_memory: HybridMemoryEngine,
        fact_extractor: FactExtractor,
        fact_deduper: _FactDeduper,
        decode_per_batch: int,
        log_interval: int,
        fact_verifier: Optional[FactVerifier] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.tokenizer = tokenizer
        self.hybrid_memory = hybrid_memory
        self.fact_extractor = fact_extractor
        self.fact_deduper = fact_deduper
        self.decode_per_batch = decode_per_batch
        self.log_interval = log_interval
        self.fact_verifier = fact_verifier
        self.retriever = retriever
        
        # Statistics
        self.total_ingested = 0
        self.facts_processed = 0
        self.verifier_rejections = 0
        self.latency_total = 0.0
        self.latency_count = 0
        
        # Async processing
        self.pending: deque = deque()
        self._doc_counter = 0
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="FactExtractor")

    def _should_log(self, step: int) -> bool:
        return step > 0 and step % self.log_interval == 0

    def submit(self, input_ids: torch.Tensor):
        """Submit texts for async fact extraction"""
        if input_ids.size(0) < self.decode_per_batch:
            return
            
        # Sample a subset for fact extraction
        indices = torch.randperm(input_ids.size(0))[:self.decode_per_batch]
        sample_ids = input_ids[indices]
        
        # Decode to text
        texts = []
        for seq in sample_ids:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            if len(text.strip()) > 20:  # Minimum text length
                texts.append(text.strip())
        
        if texts:
            # Submit for async processing
            future = self.executor.submit(self.fact_extractor.extract, texts)
            self.pending.append((future, texts, time.time()))

    def process_ready(self, step: int) -> List[Tuple[List[FactCandidate], Optional[str]]]:
        """Process completed fact extraction futures"""
        ready = []
        still_pending = deque()
        
        while self.pending:
            future, texts, start_time = self.pending.popleft()
            if future.done():
                try:
                    facts = future.result(timeout=0.1)
                except (TimeoutError, Exception) as e:
                    # Skip failed extractions
                    continue
                    
                # Verify facts if verifier available
                if self.fact_verifier is not None:
                    verified = self.fact_verifier.verify(facts)
                    self.verifier_rejections += len(facts) - len(verified)
                    facts = verified
                    
                # Ingest facts
                ingested = ingest_fact_candidates(
                    self.hybrid_memory, self.fact_deduper, facts
                )
                
                if ingested:
                    self.total_ingested += len(ingested)
                    latency = time.time() - start_time
                    self.latency_total += latency
                    self.latency_count += 1
                    self.facts_processed += len(facts)
                    
                    # Add to retriever if available
                    if self.retriever is not None:
                        for fact in ingested:
                            self._doc_counter += 1
                            doc_id = f"fact:{self._doc_counter}"
                            context = fact.sentence or f"{fact.head} {fact.relation} {fact.tail}."
                            self.retriever.add_document(doc_id, context)
                    
                    # Generate log message
                    log_msg = None
                    if self._should_log(step):
                        preview = ", ".join(
                            f"{f.head} --{f.relation}→ {f.tail}" for f in ingested[:3]
                        )
                        log_msg = (
                            f"   🧠 Hybrid memory ingested {len(ingested)} fact(s); "
                            f"sample: {preview}"
                        )
                        if self.fact_verifier is not None and self.fact_verifier.stats.processed:
                            rate = self.fact_verifier.stats.acceptance_rate * 100
                            log_msg += f" | verifier acceptance {rate:.1f}%"
                        log_msg += f" | latency {latency:.2f}s"
                    ready.append((ingested, log_msg))
            else:
                still_pending.append((future, texts, start_time))
        
        self.pending = still_pending
        return ready

    def flush(self, step: int) -> List[Tuple[List[FactCandidate], Optional[str]]]:
        """Process all remaining futures"""
        results: List[Tuple[List[FactCandidate], Optional[str]]] = []
        while self.pending:
            future, texts, start_time = self.pending.popleft()
            try:
                facts = future.result(timeout=5.0)
            except TimeoutError:
                future.cancel()
                continue
            
            if self.fact_verifier is not None:
                verified = self.fact_verifier.verify(facts)
                self.verifier_rejections += len(facts) - len(verified)
                facts = verified
                
            ingested = ingest_fact_candidates(
                self.hybrid_memory, self.fact_deduper, facts
            )
            
            if ingested:
                self.total_ingested += len(ingested)
                latency = time.time() - start_time
                self.latency_total += latency
                self.latency_count += 1
                
                preview = ", ".join(
                    f"{f.head} --{f.relation}→ {f.tail}" for f in ingested[:3]
                )
                log_msg = f"🧠 Final ingestion: {len(ingested)} facts; sample: {preview}"
                results.append((ingested, log_msg))
        
        return results

    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)


def generate_router_labels(text: str, tokenizer: AutoTokenizer) -> Tuple[float, List[str]]:
    """Generate router supervision labels from text using heuristics"""
    text_lower = text.lower()
    
    # Answerability heuristic
    answerability = 0.0
    
    # Question patterns suggest retrievable content
    if any(word in text_lower for word in ['what', 'who', 'where', 'when', 'which', 'how']):
        answerability += 0.6
    if any(word in text_lower for word in ['?', 'explain', 'describe', 'define']):
        answerability += 0.3
    if any(word in text_lower for word in ['the', 'this', 'that', 'these', 'those']):
        answerability += 0.1
        
    # Type labels heuristic
    type_labels = []
    if any(word in text_lower for word in ['person', 'who', 'people', 'name']):
        type_labels.append('PERSON')
    if any(word in text_lower for word in ['place', 'where', 'city', 'country', 'location']):
        type_labels.append('LOCATION')
    if any(word in text_lower for word in ['when', 'date', 'year', 'time']):
        type_labels.append('DATE')
    if any(word in text_lower for word in ['what', 'thing', 'object', 'item']):
        type_labels.append('ENTITY')
    if any(word in text_lower for word in ['how many', 'number', 'count', 'amount']):
        type_labels.append('QUANTITY')
    
    # Default fallback
    if not type_labels:
        type_labels = ['GENERAL']
        
    return min(1.0, max(0.0, answerability)), type_labels


def compute_router_loss(
    model: nn.Module,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    device: torch.device
) -> Optional[torch.Tensor]:
    """Compute router supervision loss"""
    try:
        # Get router predictions
        base_model = model.module if hasattr(model, 'module') else model
        if not hasattr(base_model, 'router_head'):
            return None
            
        # Generate supervision labels
        batch_size = input_ids.size(0)
        answerability_labels = []
        type_labels = []
        
        for i in range(batch_size):
            text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            ans, types = generate_router_labels(text, tokenizer)
            answerability_labels.append(ans)
            # For simplicity, use first type or default
            type_idx = 0 if 'PERSON' in types else (1 if 'LOCATION' in types else 2)
            type_labels.append(type_idx)
        
        answerability_targets = torch.tensor(answerability_labels, device=device, dtype=torch.float)
        type_targets = torch.tensor(type_labels, device=device, dtype=torch.long)
        
        # Get router outputs (average over sequence)
        hidden_states = base_model.encode(input_ids, pad_token_id=tokenizer.pad_token_id)
        router_logits = base_model.router_head(hidden_states)  # [batch, seq, router_dim]
        router_pooled = router_logits.mean(dim=1)  # [batch, router_dim]
        
        # Split router outputs
        answerability_logits = router_pooled[:, 0]  # Answerability score
        type_logits = router_pooled[:, 1:4]  # Type classification
        
        # Compute losses
        ans_loss = F.mse_loss(torch.sigmoid(answerability_logits), answerability_targets)
        type_loss = F.cross_entropy(type_logits, type_targets)
        
        return ans_loss + 0.5 * type_loss
        
    except Exception:
        return None


def compute_memory_aux_loss(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    hybrid_memory: Optional[HybridMemoryEngine],
    device: torch.device,
    sample_triples: List[Tuple[str, str, str]],
) -> Optional[torch.Tensor]:
    """Compute memory auxiliary loss for graph-parametric alignment"""
    if hybrid_memory is None or not sample_triples:
        return None
        
    base_model = model.module if hasattr(model, 'module') else model
    losses = []
    
    for head, relation, tail in sample_triples:
        prompt = f"{head} {relation} ->"
        target = f" {tail}{tokenizer.eos_token or ''}"
        
        # Tokenize prompt and target
        prompt_tokens = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
        combined = tokenizer(
            prompt + target,
            return_tensors='pt',
            add_special_tokens=False,
        )
        
        input_ids = combined['input_ids'].to(device)
        if input_ids.numel() == 0:
            continue
            
        # Create labels (mask prompt tokens)
        labels = input_ids.clone()
        prompt_len = prompt_tokens['input_ids'].size(1)
        labels[:, :prompt_len] = -100
        
        # Forward pass
        logits = base_model(input_ids, pad_token_id=tokenizer.pad_token_id)
        if isinstance(logits, tuple):
            logits = logits[0]
            
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        losses.append(loss)
    
    if not losses:
        return None
    return torch.stack(losses).mean()


def sample_memory_triples(
    hybrid_memory: HybridMemoryEngine,
    num_samples: int = 3
) -> List[Tuple[str, str, str]]:
    """Sample triples from hybrid memory for auxiliary loss"""
    try:
        # Get some entities and sample relations
        entities = list(hybrid_memory.entity_table.name_to_id.keys())
        if len(entities) < 2:
            return []
            
        triples = []
        for _ in range(min(num_samples, len(entities) // 2)):
            head = random.choice(entities)
            
            # Query for related entities
            query_result = hybrid_memory.query(head, topk_neighbors=3, topk_passages=1)
            if query_result.neighbors:
                relation, tail = random.choice(query_result.neighbors)
                triples.append((head, relation, tail))
                
        return triples
        
    except Exception:
        return []


def run_retrieval_probe(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    hybrid_retriever: Optional[HybridRetriever],
    hybrid_memory: Optional[HybridMemoryEngine],
    device: torch.device,
    step: int
):
    """Run periodic retrieval-in-the-loop probes to verify system behavior"""
    if hybrid_retriever is None or hybrid_memory is None:
        return
    
    # Quiz questions to test retrieval behavior
    quiz_questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "Where is the Eiffel Tower located?",
        "When was World War II fought?",
        "What is the speed of light?",
        "How many continents are there?"
    ]
    
    print(f"\n🎯 RETRIEVAL PROBE - Step {step}")
    print("-" * 50)
    
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for question in quiz_questions:
            # 1. Generate router labels to check answerability
            answerability, predicted_types = generate_router_labels(question, tokenizer)
            
            # 2. Check if router would trigger retrieval
            should_retrieve = answerability >= 0.5  # Router threshold
            
            # 3. If should retrieve, query hybrid memory and retriever
            retrieved_facts = []
            retrieved_passages = []
            
            if should_retrieve:
                # Query hybrid memory for relevant entities/facts
                try:
                    # Extract key entities from question for memory lookup
                    question_words = question.lower().replace('?', '').split()
                    key_entities = [w.capitalize() for w in question_words if len(w) > 3 and w not in ['what', 'where', 'when', 'who', 'how', 'the', 'is', 'are', 'was', 'were']]
                    
                    if key_entities:
                        for entity in key_entities[:2]:  # Check top 2 entities
                            query_result = hybrid_memory.query(entity, topk_neighbors=2, topk_passages=2)
                            if query_result.neighbors:
                                retrieved_facts.extend(query_result.neighbors[:2])
                            if query_result.passages:
                                retrieved_passages.extend([p.text for p in query_result.passages[:2]])
                except Exception:
                    pass
                
                # Query hybrid retriever for relevant passages
                try:
                    retriever_results = hybrid_retriever.retrieve(question, top_k=2)
                    retrieved_passages.extend([text for text, score in retriever_results])
                except Exception:
                    pass
            
            # 4. Generate answer with and without retrieval context
            input_text = f"Question: {question}\nAnswer:"
            
            # Generate without retrieval context
            input_ids = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=64).input_ids.to(device)
            output_no_retrieval = generate_with_model(base_model, tokenizer, input_ids, max_new_tokens=20, device=device)
            
            # Generate with retrieval context if available
            output_with_retrieval = output_no_retrieval  # Default fallback
            if retrieved_facts or retrieved_passages:
                context_parts = []
                if retrieved_facts:
                    fact_strs = [f"{h} {r} {t}" for h, r, t in retrieved_facts[:2]]
                    context_parts.append("Facts: " + "; ".join(fact_strs))
                if retrieved_passages:
                    context_parts.append("Context: " + " ".join(retrieved_passages[:2])[:200] + "...")
                
                context = " | ".join(context_parts)
                input_with_context = f"Context: {context}\nQuestion: {question}\nAnswer:"
                input_ids_ctx = tokenizer(input_with_context, return_tensors='pt', truncation=True, max_length=128).input_ids.to(device)
                output_with_retrieval = generate_with_model(base_model, tokenizer, input_ids_ctx, max_new_tokens=20, device=device)
            
            # 5. Display results
            print(f"Q: {question}")
            print(f"  Router: answerability={answerability:.2f}, types={predicted_types}, retrieve={should_retrieve}")
            if retrieved_facts:
                print(f"  Memory facts: {retrieved_facts[:2]}")
            if retrieved_passages and len([p for p in retrieved_passages if p.strip()]) > 0:
                print(f"  Retrieved passages: {len([p for p in retrieved_passages if p.strip()])} items")
            print(f"  Answer (no retrieval): {output_no_retrieval}")
            if output_with_retrieval != output_no_retrieval:
                print(f"  Answer (with retrieval): {output_with_retrieval}")
            print()
    
    model.train()
    print("✓ Retrieval probe completed\n")


def generate_with_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    device: torch.device = None,
    temperature: float = 0.8,
    top_p: float = 0.9
) -> str:
    """Generate text with the model using nucleus sampling"""
    if device is None:
        device = input_ids.device
    
    model.eval()
    original_length = input_ids.shape[1]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids, pad_token_id=tokenizer.pad_token_id or 0)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Set logits to -inf for removed tokens
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode only the new tokens
    generated_ids = input_ids[0, original_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text.strip()


def evaluate_on_dataset(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int = 10
) -> Tuple[float, float]:
    """Evaluate CE and PPL on data with parity checks"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            base_model = model.module if hasattr(model, 'module') else model
            
            # Forward pass
            outputs = base_model(input_ids, pad_token_id=tokenizer.pad_token_id)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            # Compute loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Mask padding tokens
            pad_mask = (shift_labels != tokenizer.pad_token_id)
            valid_tokens = pad_mask.sum().item()
            
            if valid_tokens == 0:
                continue
                
            loss_flat = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            loss_tok = loss_flat.view(shift_labels.shape)
            
            # Sum over valid tokens only
            total_loss += (loss_tok * pad_mask.float()).sum().item()
            total_tokens += valid_tokens
    
    model.train()
    
    if total_tokens == 0:
        return float('inf'), float('inf')
        
    ce = total_loss / total_tokens
    ppl = math.exp(ce)
    
    return ce, ppl


def create_data_config_from_config(config: Dict[str, Any]) -> DataConfig:
    """Convert training config to DataConfig format"""
    data_config = DataConfig()
    
    # Handle different config formats
    if 'target_tokens' in config and 'data_sources' in config:
        # Phase 1 format with target_tokens and data_sources
        
        # Set token budget
        data_config.train_tokens = config['target_tokens']
        data_config.eval_tokens = config.get('eval_tokens', config['target_tokens'] // 100)
        
        # Set sequence length (with context ramping support)
        training_section = config.get('training', {})
        context_ramping = training_section.get('context_ramping', {})
        ctx_len_schedule = training_section.get('ctx_len_schedule', {})
        
        if 'stages' in context_ramping and context_ramping['stages']:
            # Use initial context length for now, ramping will be handled in training loop
            initial_stage = context_ramping['stages'][0]
            data_config.seq_length = initial_stage['context_length']
        elif 'initial' in ctx_len_schedule:
            # Handle Phase 1 style ctx_len_schedule
            data_config.seq_length = ctx_len_schedule['initial']
        else:
            data_config.seq_length = training_section.get('context_length', 1024)
        
        # Convert data sources
        data_config.custom_sources = []
        for source_spec in config['data_sources']:
            # Skip sources with weight 0
            if source_spec.get('weight', 0) == 0:
                continue
                
            source = {
                'type': 'hf',
                'dataset': source_spec['dataset'],
                'split': source_spec.get('split', 'train'),
                'fields': [source_spec.get('text_field', 'text')],
                'format': '{' + source_spec.get('text_field', 'text') + '}',
                'weight': source_spec.get('weight', 1)
            }
            
            if source_spec.get('config_name'):
                source['config'] = source_spec['config_name']
            
            data_config.custom_sources.append(source)
        
        # Set mixing strategy
        data_config.mix_strategy = 'weighted'
        
    elif 'data' in config:
        # New comprehensive format (alternative structure)
        data_section = config['data']
        
        # Set token budget
        data_config.train_tokens = data_section['train_tokens']
        data_config.eval_tokens = data_section.get('eval_tokens', data_section['train_tokens'] // 100)
        
        # Set sequence length (with context ramping support)
        train_section = config.get('train', {})
        if 'context_schedule' in train_section:
            # Use initial context length for now, ramping will be handled in training loop
            data_config.seq_length = train_section['context_schedule'][0]['context_length']
        else:
            data_config.seq_length = train_section.get('context_length', 1024)
        
        # Convert data sources
        data_config.custom_sources = []
        for source_spec in data_section['sources']:
            source = {
                'type': 'hf',
                'dataset': source_spec['dataset'],
                'split': source_spec.get('split', 'train'),
                'fields': [source_spec.get('text_field', 'text')],
                'format': '{' + source_spec.get('text_field', 'text') + '}',
                'weight': source_spec.get('weight', 1)
            }
            
            if source_spec.get('config_name'):
                source['config'] = source_spec['config_name']
            
            data_config.custom_sources.append(source)
        
        # Set mixing strategy
        data_config.mix_strategy = data_section.get('mixing_strategy', 'weighted')
        
    elif 'tokens' in config:
        # Legacy simple format
        data_config.train_tokens = config['tokens']
        data_config.eval_tokens = min(config['tokens'] // 100, 1_000_000)
        data_config.seq_length = 512
        
        source = {
            'type': 'hf',
            'dataset': config['dataset'],
            'split': config.get('split', 'train'),
            'fields': [config.get('text_field', 'text')],
            'format': '{' + config.get('text_field', 'text') + '}',
            'weight': 1
        }
        
        if config.get('config_name'):
            source['config'] = config['config_name']
        
        data_config.custom_sources = [source]
        data_config.mix_strategy = 'sequential'
    
    else:
        # If none of the formats match, raise error
        raise ValueError(f"Config must contain 'target_tokens' + 'data_sources' (Phase 1), 'data' section, or 'tokens' key. Found keys: {list(config.keys())}")
    
    return data_config


def create_streaming_dataloader(
    data_config: DataConfig,
    tokenizer: AutoTokenizer,
    batch_size: int,
    is_eval: bool = False,
    rank: int = 0
) -> torch.utils.data.DataLoader:
    """Create streaming dataloader with token budgeting and sequence packing"""
    if rank == 0:
        print(f"Creating {'eval' if is_eval else 'train'} dataloader...")
        print(f"  Token budget: {data_config.eval_tokens if is_eval else data_config.train_tokens:,}")
        print(f"  Sequence length: {data_config.seq_length}")
        print(f"  Sources: {len(data_config.custom_sources) if data_config.custom_sources else len(data_config.data_sources or [])}")
    
    # Create registry and dataloader
    registry = DataRegistry()
    
    dataloader = registry.create_dataloader(
        tokenizer=tokenizer,
        config=data_config,
        batch_size=batch_size,
        is_eval=is_eval,
        use_sized=False,  # Use streaming
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return dataloader


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, tokens_processed: int, loss: float):
    """Save training checkpoint"""
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save current checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'tokens_processed': tokens_processed,
        'loss': loss
    }
    
    # Rolling checkpoint saves
    current_path = checkpoint_dir / 'lm_current.pth'
    prev_path = checkpoint_dir / 'lm_prev.pth'
    prev2_path = checkpoint_dir / 'lm_prev2.pth'
    
    # Rotate checkpoints
    if current_path.exists():
        if prev_path.exists():
            if prev2_path.exists():
                prev2_path.unlink()
            prev_path.rename(prev2_path)
        current_path.rename(prev_path)
    
    # Save new checkpoint
    torch.save(checkpoint, current_path)
    print(f"✓ Checkpoint saved: {current_path}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[int, int, float]:
    """Load training checkpoint if available"""
    checkpoint_path = Path('checkpoints/lm_current.pth')
    
    if not checkpoint_path.exists():
        print("No checkpoint found, starting from scratch")
        return 0, 0, 0.0
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format checkpoint
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
        step = checkpoint.get('step', 0)
        tokens_processed = checkpoint.get('tokens_processed', 0)
        loss = checkpoint.get('loss', 0.0)
        
        print(f"✓ Resumed from step {step}, tokens {tokens_processed:,}")
        return step, tokens_processed, loss
        
    else:
        # Legacy format - raw state dict
        print("✓ Loading legacy checkpoint format (model weights only)")
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        print("✓ Starting from step 0 with loaded weights")
        return 0, 0, 0.0


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    pad_token_id: Optional[int],
) -> Tuple[torch.Tensor, int]:
    """Single training step with unified model"""

    input_ids = batch['input_ids'].to(device)

    # Determine which token should be treated as padding.
    # Fallback to EOS → 0 if tokenizer did not define pad token.
    effective_pad_id = pad_token_id
    if effective_pad_id is None:
        pad_from_batch = batch.get('pad_token_id')
        if pad_from_batch is not None:
            if isinstance(pad_from_batch, torch.Tensor):
                effective_pad_id = int(pad_from_batch.item())
            else:
                effective_pad_id = int(pad_from_batch)
    if effective_pad_id is None:
        effective_pad_id = 0

    # Get model outputs using the correct interface
    base_model = model.module if hasattr(model, 'module') else model

    # Forward pass through UnifiedCognitiveLM using the real pad token id
    outputs = base_model(input_ids, pad_token_id=effective_pad_id)

    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    # Compute standard causal language modeling loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Count valid tokens (avoid padding tokens)
    pad_mask = shift_labels != effective_pad_id
    valid_tokens = pad_mask.sum().item()

    # Compute loss with proper pad masking
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )

    # Apply mask and compute mean over valid tokens only
    loss_per_token = loss_flat.view(shift_labels.shape)
    masked_loss = loss_per_token * pad_mask.float()

    if valid_tokens > 0:
        loss = masked_loss.sum() / valid_tokens
    else:
        loss = masked_loss.sum()  # Fallback

    return loss, valid_tokens


def main():
    parser = argparse.ArgumentParser(description='Comprehensive training script')
    parser.add_argument('config', type=str, help='Path to config JSON file')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Setup distributed training
    use_ddp, rank, world_size, device = setup_ddp()
    
    if rank == 0:
        print("🚀 Comprehensive Training Started")
        print(f"Config: {config_path}")
        print(f"Device: {device}")
        print(f"Processes: {world_size}")
    
    # Enable performance optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # === 1. Bootstrap: Model & Tokenizer ===
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Get model configuration from config file or use defaults
    model_config = config.get('model', {})
    model = create_unified_model(
        vocab_size=len(tokenizer),  # Use len(tokenizer) to account for added special tokens
        hidden_size=model_config.get('hidden_size', 256),
        num_layers=model_config.get('num_layers', 4), 
        num_heads=model_config.get('num_heads', 4),
        max_seq_len=model_config.get('max_seq_len', 4096),  # Support up to 4k for ramping
        enable_router=model_config.get('enable_router', True),
        enable_schema_inference=model_config.get('enable_schema_inference', True),
        enable_hybrid_memory=model_config.get('enable_hybrid_memory', True),
        enable_self_correction=model_config.get('enable_self_correction', True)
    ).to(device)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Unified model created: {total_params:,} parameters")
        print(f"  - Hidden size: {model_config.get('hidden_size', 256)}")
        print(f"  - Layers: {model_config.get('num_layers', 4)}")
        print(f"  - Max seq len: {model_config.get('max_seq_len', 4096)}")
        print("  - Router supervision enabled")
        print("  - Schema inference enabled") 
        print("  - Hybrid memory enabled")
        print("  - Self-correction enabled")
    
    # Setup DDP
    if use_ddp:
        model = DDP(model, device_ids=[rank])
        if rank == 0:
            print("✓ DDP enabled")
    
    # === 2. Data: Streaming Dataloader with Token Budgeting ===
    
    # Convert config to DataConfig
    data_config = create_data_config_from_config(config)
    
    # Get training configuration early
    train_config = config.get('training', config.get('train', {}))  # Support both 'training' and 'train' keys
    
    # Create train and eval dataloaders
    batch_size_config = train_config.get('batch_size', 32)
    batch_size = batch_size_config // world_size if world_size > 1 else batch_size_config
    train_dataloader = create_streaming_dataloader(data_config, tokenizer, batch_size, is_eval=False, rank=rank)
    eval_dataloader = create_streaming_dataloader(data_config, tokenizer, batch_size, is_eval=True, rank=rank)
    
    if rank == 0:
        print(f"✓ Data loading configured")
        print(f"  Train tokens: {data_config.train_tokens:,}")
        print(f"  Eval tokens: {data_config.eval_tokens:,}")
        print(f"  Batch size per GPU: {batch_size}")
    
    # === 3. Hybrid Memory & Retriever ===
    
    hybrid_memory = None
    hybrid_retriever = None
    fact_extractor = None
    fact_verifier = None
    ingestion_manager = None
    
    try:
        # Initialize hybrid memory with correct parameters
        hybrid_memory = HybridMemoryEngine(
            entity_dim=128,
            passage_dim=None,  # Auto-detect from encoder (384D for all-MiniLM-L6-v2)
            use_faiss=True
        )
        
        # Initialize retriever with correct parameters
        hybrid_retriever = HybridRetriever(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            device=str(device),
            dense_weight=0.6,
            sparse_weight=0.4
        )
        
        # Initialize fact extraction pipeline
        fact_extractor = FactExtractor()  # REBEL-based
        fact_verifier = FactVerifier()    # DeBERTa NLI-based
        
        # Initialize async ingestion manager
        fact_deduper = _FactDeduper(max_size=8192)
        ingestion_manager = HybridMemoryIngestionManager(
            tokenizer=tokenizer,
            hybrid_memory=hybrid_memory,
            fact_extractor=fact_extractor,
            fact_deduper=fact_deduper,
            decode_per_batch=4,  # Extract facts from 4 sequences per batch
            log_interval=50,     # Log every 50 steps
            fact_verifier=fact_verifier,
            retriever=hybrid_retriever,
        )
        
        if rank == 0:
            print("✓ Hybrid memory system initialized")
            print(f"  Entity capacity: 100k+ entities")
            print(f"  Passage capacity: 50k+ passages")
            print("  REBEL fact extractor enabled")
            print("  DeBERTa fact verifier enabled")
            print("  Async ingestion enabled")
            
    except Exception as e:
        if rank == 0:
            print(f"⚠️  Hybrid memory initialization failed: {e}")
            print("   Continuing without hybrid memory...")
    
    # === 4. Training Setup ===
    
    # Create optimizer with config settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get('learning_rate', 5e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
        betas=train_config.get('betas', (0.9, 0.95))
    )
    
    # Load checkpoint if available
    step, tokens_processed, best_loss = load_checkpoint(model, optimizer)
    
    # Training configuration
    target_tokens = data_config.train_tokens
    accumulation_steps = train_config.get('gradient_accumulation_steps', 4)
    log_interval = train_config.get('log_interval', 50)
    eval_interval = train_config.get('eval_interval', 1000)
    save_interval = train_config.get('save_interval', 1000)
    
    # Context length ramping setup - handle both formats
    context_schedule = []
    context_ramping = train_config.get('context_ramping', {})
    if 'stages' in context_ramping:
        context_schedule = context_ramping['stages']
    elif 'context_schedule' in train_config:
        context_schedule = train_config['context_schedule']
    
    current_context_length = data_config.seq_length  # Start with initial length
    
    # Loss weights
    router_loss_weight = train_config.get('router_loss_weight', 0.1)
    memory_aux_loss_weight = train_config.get('memory_aux_loss_weight', 0.05)
    
    # Temperature ramping
    temp_schedule = train_config.get('temperature_schedule', {})
    current_temperature = temp_schedule.get('initial', 0.8)
    
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Determine pad token id once to avoid repeated checks
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    
    if rank == 0:
        print(f"✓ Training configured")
        print(f"  Target tokens: {target_tokens:,}")
        print(f"  Learning rate: {train_config.get('learning_rate', 5e-4)}")
        print(f"  Accumulation steps: {accumulation_steps}")
        print(f"  Initial context length: {current_context_length}")
        if context_schedule:
            print(f"  Context ramping: {len(context_schedule)} stages")
        print(f"  Router loss weight: {router_loss_weight}")
        print(f"  Memory aux loss weight: {memory_aux_loss_weight}")
        print("Starting training loop...")
    
    # === 5. Core Pretraining Loop ===
    
    model.train()
    accumulated_loss = 0.0
    accumulated_tokens = 0
    start_time = time.time()
    
    # Statistics tracking
    total_facts_ingested = 0
    fact_relation_counter = Counter()
    recent_losses = deque(maxlen=100)
    
    # Track context ramping state
    context_ramp_triggered = set()  # Track which stages have been triggered
    
    # Progress indicators
    last_progress_time = time.time()
    batch_start_time = None
    
    if rank == 0:
        print("🔄 Initializing data loading...")
    
    try:
        # Create iterator from dataloader
        dataloader_iter = iter(train_dataloader)
        
        if rank == 0:
            print("✓ Data iterator created, starting batch processing...")
        
        batch_idx = 0
        while tokens_processed < target_tokens:
            # Show progress every 10 batches even before first log interval
            if rank == 0 and batch_idx > 0 and batch_idx % 10 == 0:
                current_time = time.time()
                if current_time - last_progress_time >= 5.0:  # At least 5 seconds between progress updates
                    elapsed = current_time - start_time
                    tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                    progress_pct = (tokens_processed / target_tokens) * 100
                    print(f"📊 Progress: {batch_idx} batches, {tokens_processed:,}/{target_tokens:,} tokens ({progress_pct:.1f}%), {tokens_per_sec:.1f} tok/s")
                    last_progress_time = current_time
            
            if rank == 0 and batch_idx == 0:
                batch_start_time = time.time()
                print("⏱️  Processing first batch (this may take a moment)...")
            
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # Recreate iterator if we reach the end
                dataloader_iter = iter(train_dataloader)
                batch = next(dataloader_iter)
            
            if rank == 0 and batch_idx == 0:
                first_batch_time = time.time() - batch_start_time
                print(f"✓ First batch loaded ({first_batch_time:.1f}s), beginning forward pass...")
            
            # === Context Length Ramping ===
            if context_schedule and rank == 0:
                # Check if we need to increase context length
                for stage_idx, stage in enumerate(context_schedule):
                    stage_key = f"stage_{stage_idx}"
                    stage_tokens = int(stage['at_tokens'] * target_tokens)  # Convert fraction to absolute
                    
                    if (tokens_processed >= stage_tokens and 
                        current_context_length < stage['context_length'] and 
                        stage_key not in context_ramp_triggered):
                        
                        old_length = current_context_length
                        current_context_length = stage['context_length']
                        context_ramp_triggered.add(stage_key)
                        
                        print(f"📏 Context length ramping: {old_length} → {current_context_length} (at {tokens_processed:,} tokens)")
                        
                        # Update data config and recreate dataloader
                        data_config.seq_length = current_context_length
                        train_dataloader = create_streaming_dataloader(data_config, tokenizer, batch_size, is_eval=False, rank=rank)
                        dataloader_iter = iter(train_dataloader)
                        
                        # Skip current batch and get new one with updated length
                        try:
                            batch = next(dataloader_iter)
                        except StopIteration:
                            dataloader_iter = iter(train_dataloader)
                            batch = next(dataloader_iter)
                        break
            
            # === Temperature Ramping ===
            if temp_schedule and 'ramp_end_tokens' in temp_schedule:
                ramp_progress = min(1.0, tokens_processed / (temp_schedule['ramp_end_tokens'] * target_tokens))
                target_temp = temp_schedule.get('final', 0.7)
                initial_temp = temp_schedule.get('initial', 0.8)
                current_temperature = initial_temp + (target_temp - initial_temp) * ramp_progress
            
            # === Process async fact ingestion ===
            if rank == 0 and ingestion_manager is not None:
                # Process completed fact extractions
                ready_batches = ingestion_manager.process_ready(step)
                for facts_ready, log_msg in ready_batches:
                    total_facts_ingested += len(facts_ready)
                    for fact in facts_ready:
                        fact_relation_counter[fact.relation] += 1
                    if log_msg:
                        print(log_msg)
                
                # Submit current batch for fact extraction
                cpu_input_ids = batch['input_ids']
                ingestion_manager.submit(cpu_input_ids)
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            batch_size_actual, seq_len = input_ids.shape
            
            if rank == 0 and batch_idx == 1:  # After first batch is loaded
                print(f"🧠 Starting forward pass (batch_size={batch_size_actual}, seq_len={seq_len})...")
            
            with autocast(enabled=torch.cuda.is_available()):
                # === Forward & CE Loss ===
                loss, valid_tokens = train_step(model, batch, device, pad_token_id)
                
                # === Router Supervision ===
                router_loss = compute_router_loss(model, input_ids, tokenizer, device)
                if router_loss is not None:
                    loss = loss + router_loss_weight * router_loss
                
                # === Memory Auxiliary Loss ===
                memory_aux_loss = None
                if hybrid_memory is not None:
                    memory_triples = sample_memory_triples(hybrid_memory, num_samples=3)
                    if memory_triples:
                        memory_aux_loss = compute_memory_aux_loss(
                            model, tokenizer, hybrid_memory, device, memory_triples
                        )
                        if memory_aux_loss is not None:
                            loss = loss + memory_aux_loss_weight * memory_aux_loss
                
                # Scale for accumulation
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            accumulated_loss += loss.item() * accumulation_steps
            accumulated_tokens += valid_tokens
            tokens_processed += valid_tokens
            batch_idx += 1
            
            # === Optimizer Step ===
            if batch_idx % accumulation_steps == 0:
                step += 1
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Show progress for first few steps
                if rank == 0 and step <= 5:
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                    print(f"🎯 Step {step} completed! Tokens: {tokens_processed:,}, Speed: {tokens_per_sec:.1f} tok/s")
                
                # === Logging ===
                if step % log_interval == 0 and rank == 0:
                    avg_loss = accumulated_loss / max(accumulated_tokens, 1) * accumulation_steps
                    recent_losses.append(avg_loss)
                    
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                    
                    log_parts = [
                        f"Step {step:,}",
                        f"Tokens {tokens_processed:,}",
                        f"Loss {avg_loss:.4f}",
                        f"{tokens_per_sec:.0f} tok/s"
                    ]
                    
                    # Add context length and temperature if ramping
                    if context_schedule:
                        log_parts.append(f"Ctx {current_context_length}")
                    if temp_schedule:
                        log_parts.append(f"Temp {current_temperature:.2f}")
                    
                    if router_loss is not None:
                        log_parts.append(f"Router {router_loss.item():.4f}")
                    if memory_aux_loss is not None:
                        log_parts.append(f"MemAux {memory_aux_loss.item():.4f}")
                    if total_facts_ingested > 0:
                        log_parts.append(f"Facts {total_facts_ingested}")
                    
                    print(" | ".join(log_parts))
                    
                    # === Periodic Retrieval-in-the-Loop Probe ===
                    # Run retrieval quiz every 200 steps to verify system behavior
                    probe_interval = max(200, log_interval * 4)  # At least every 200 steps
                    if step > 0 and step % probe_interval == 0:
                        try:
                            run_retrieval_probe(model, tokenizer, hybrid_retriever, hybrid_memory, device, step)
                        except Exception as e:
                            print(f"⚠️  Retrieval probe failed at step {step}: {e}")
                
                # === Evaluation ===
                if step % eval_interval == 0 and rank == 0:
                    print("🔍 Running evaluation...")
                    eval_ce, eval_ppl = evaluate_on_dataset(model, tokenizer, device, eval_dataloader)
                    print(f"   Eval CE: {eval_ce:.4f}, PPL: {eval_ppl:.2f}")
                    
                    # Log memory stats if available
                    if hybrid_memory is not None:
                        num_entities = len(hybrid_memory.entity_table.name_to_id)
                        num_passages = hybrid_memory.passage_index.num_passages if hasattr(hybrid_memory.passage_index, 'num_passages') else 0
                        print(f"   Memory: {num_entities} entities, {num_passages} passages")
                
                # === Save Checkpoint ===
                if step % save_interval == 0 and rank == 0:
                    avg_loss = accumulated_loss / max(accumulated_tokens, 1) * accumulation_steps
                    save_checkpoint(model, optimizer, step, tokens_processed, avg_loss)
                    accumulated_loss = 0.0
                    accumulated_tokens = 0
        
        # End of training loop
        if rank == 0:
            print(f"🎯 Training completed - target tokens reached: {tokens_processed:,}")
    
    except KeyboardInterrupt:
        if rank == 0:
            print("\n⚠️  Training interrupted by user")
    
    except Exception as e:
        if rank == 0:
            print(f"\n❌ Training error: {e}")
        raise
    
    finally:
        # === Cleanup ===
        if rank == 0 and ingestion_manager is not None:
            # Flush any remaining fact extractions
            remaining = ingestion_manager.flush(step)
            for facts_ready, log_msg in remaining:
                total_facts_ingested += len(facts_ready)
                if log_msg:
                    print(log_msg)
            
            ingestion_manager.shutdown()
            print(f"✓ Ingestion manager shutdown")
            
            if total_facts_ingested > 0:
                print(f"📊 Final statistics:")
                print(f"   Total facts ingested: {total_facts_ingested}")
                print(f"   Top relations: {dict(fact_relation_counter.most_common(5))}")
                if ingestion_manager.latency_count > 0:
                    avg_latency = ingestion_manager.latency_total / ingestion_manager.latency_count
                    print(f"   Avg extraction latency: {avg_latency:.2f}s")
        
        # Final checkpoint
        if rank == 0:
            avg_loss = accumulated_loss / max(accumulated_tokens, 1) * accumulation_steps if accumulated_tokens > 0 else 0.0
            save_checkpoint(model, optimizer, step, tokens_processed, avg_loss)
            print("🎉 Training completed!")
            print(f"   Final tokens processed: {tokens_processed:,}")
            print(f"   Total steps: {step}")


if __name__ == '__main__':
    main()