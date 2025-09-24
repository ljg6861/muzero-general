#!/usr/bin/env python3
"""
Wikipedia Data Processing for Cognitive LLM
===========================================
Phase 1: Foundation Knowledge Training data pipeline with cognitive enhancement.
Processes Wikipedia data for concept formation, causal reasoning, and knowledge discovery.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
import os
import re
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
import unicodedata

# Wikipedia API imports
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    print("Warning: wikipedia package not available, using fallback API")
    WIKIPEDIA_AVAILABLE = False

try:
    import wikipediaapi
    WIKIPEDIA_API_AVAILABLE = True
except ImportError:
    print("Warning: wikipedia-api package not available, using fallback")
    WIKIPEDIA_API_AVAILABLE = False
from collections import defaultdict, Counter
import random
from tqdm import tqdm
import concurrent.futures
from urllib.parse import quote
import time

# For text processing
import nltk
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Using basic tokenization.")


@dataclass 
class WikiDataConfig:
    """Configuration for Wikipedia data processing."""
    # Data source
    wiki_dump_path: Optional[str] = None  # Path to Wikipedia dump file
    wiki_api_base: str = "https://en.wikipedia.org/api/rest_v1"
    use_api: bool = True  # Use Wikipedia API instead of dump
    
    # Processing parameters
    max_articles: int = 10000
    min_article_length: int = 500
    max_article_length: int = 10000
    max_seq_length: int = 1024
    overlap_length: int = 128
    
    # Tokenization
    tokenizer_name: str = "gpt2"  # HuggingFace tokenizer
    vocab_size: int = 50000
    
    # Cognitive enhancement
    extract_concepts: bool = True
    extract_causal_relations: bool = True
    build_knowledge_graph: bool = True
    
    # Caching
    cache_dir: str = "wiki_cache"
    processed_data_dir: str = "processed_wiki"
    
    # Categories to focus on for diverse knowledge
    target_categories: List[str] = None
    
    def __post_init__(self):
        if self.target_categories is None:
            self.target_categories = [
                "Science", "Technology", "Mathematics", "Physics", "Chemistry",
                "Biology", "Medicine", "History", "Philosophy", "Psychology",
                "Economics", "Politics", "Geography", "Art", "Literature",
                "Computer science", "Engineering", "Astronomy", "Geology"
            ]


class SimpleTokenizer:
    """Simple tokenizer for when transformers is not available."""
    
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3
        }
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.next_token_id = len(self.special_tokens)
        
    def _preprocess_text(self, text):
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Basic tokenization on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        token_counts = Counter()
        
        for text in tqdm(texts, desc="Building vocabulary"):
            tokens = self._preprocess_text(text)
            token_counts.update(tokens)
        
        # Add most frequent tokens to vocabulary
        most_common = token_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        for token, count in most_common:
            if token not in self.vocab:
                self.vocab[token] = self.next_token_id
                self.inverse_vocab[self.next_token_id] = token
                self.next_token_id += 1
    
    def encode(self, text, max_length=None):
        """Encode text to token IDs."""
        tokens = self._preprocess_text(text)
        token_ids = []
        
        for token in tokens:
            token_id = self.vocab.get(token, self.special_tokens['<unk>'])
            # Ensure token ID is within vocabulary bounds
            if token_id >= self.vocab_size:
                token_id = self.special_tokens['<unk>']
            token_ids.append(token_id)
        
        if max_length:
            token_ids = token_ids[:max_length]
        
        # Ensure all token IDs are within bounds
        token_ids = [min(tid, self.vocab_size - 1) for tid in token_ids]
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs to text."""
        tokens = [self.inverse_vocab.get(tid, '<unk>') for tid in token_ids]
        return ' '.join(tokens)
    
    def save(self, path):
        """Save tokenizer."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'vocab_size': self.vocab_size,
                'next_token_id': self.next_token_id
            }, f)
    
    def load(self, path):
        """Load tokenizer."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.inverse_vocab = data['inverse_vocab']
            self.vocab_size = data['vocab_size']
            self.next_token_id = data['next_token_id']


class WikipediaAPI:
    """Enhanced Wikipedia API client for fetching articles."""
    
    def __init__(self, config: WikiDataConfig):
        self.config = config
        
        # Initialize primary API client
        if WIKIPEDIA_API_AVAILABLE:
            self.wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                user_agent='CognitiveLLM/1.0 (Educational Research)'
            )
            self.api_type = 'wikipediaapi'
        elif WIKIPEDIA_AVAILABLE:
            wikipedia.set_lang("en")
            wikipedia.set_rate_limiting(True)
            self.api_type = 'wikipedia'
        else:
            # Fallback to REST API
            self.base_url = config.wiki_api_base
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'CognitiveLLM/1.0 (Educational Research)'
            })
            self.api_type = 'rest'
    
    def search_articles(self, query: str, limit: int = 10) -> List[str]:
        """Search for articles by query."""
        if self.api_type == 'wikipediaapi':
            # Use wikipediaapi search
            try:
                search_results = []
                # Simple search by getting page and checking if it exists
                page = self.wiki.page(query)
                if page.exists():
                    search_results.append(page.title)
                    
                # Add some related articles from links
                for link_title in list(page.links.keys())[:limit-1]:
                    search_results.append(link_title)
                    
                return search_results[:limit]
            except Exception as e:
                print(f"Error searching with wikipediaapi: {e}")
                return []
                
        elif self.api_type == 'wikipedia':
            # Use wikipedia library search
            try:
                return wikipedia.search(query, results=limit)
            except Exception as e:
                print(f"Error searching with wikipedia library: {e}")
                return []
                
        else:
            # Fallback to REST API
            return self._rest_search_articles(query, limit)
    
    def get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """Get article content by title."""
        if self.api_type == 'wikipediaapi':
            return self._get_content_wikipediaapi(title)
        elif self.api_type == 'wikipedia':
            return self._get_content_wikipedia(title)
        else:
            return self._get_content_rest(title)
    
    def _get_content_wikipediaapi(self, title: str) -> Optional[Dict[str, Any]]:
        """Get content using wikipediaapi library."""
        try:
            page = self.wiki.page(title)
            if not page.exists():
                return None
                
            return {
                'title': page.title,
                'summary': page.summary[:500] if page.summary else '',
                'content': page.text,
                'url': page.fullurl,
                'categories': list(page.categories.keys())[:10]
            }
        except Exception as e:
            print(f"Error fetching article '{title}' with wikipediaapi: {e}")
            return None
    
    def _get_content_wikipedia(self, title: str) -> Optional[Dict[str, Any]]:
        """Get content using wikipedia library."""
        try:
            page = wikipedia.page(title)
            
            return {
                'title': page.title,
                'summary': page.summary[:500] if page.summary else '',
                'content': page.content,
                'url': page.url,
                'categories': getattr(page, 'categories', [])[:10]
            }
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first suggestion
            try:
                page = wikipedia.page(e.options[0])
                return {
                    'title': page.title,
                    'summary': page.summary[:500] if page.summary else '',
                    'content': page.content,
                    'url': page.url,
                    'categories': getattr(page, 'categories', [])[:10]
                }
            except:
                return None
        except wikipedia.exceptions.PageError:
            print(f"Page '{title}' not found")
            return None
        except Exception as e:
            print(f"Error fetching article '{title}' with wikipedia library: {e}")
            return None
    
    def _rest_search_articles(self, query: str, limit: int = 10) -> List[str]:
        """Search for articles using REST API (fallback)."""
        url = f"{self.base_url}/page/search/{quote(query)}"
        
        try:
            response = self.session.get(url, params={'limit': limit})
            response.raise_for_status()
            
            data = response.json()
            return [page['title'] for page in data.get('pages', [])]
            
        except Exception as e:
            print(f"Error searching articles for '{query}': {e}")
            return []
    
    def _get_content_rest(self, title: str) -> Optional[Dict[str, Any]]:
        """Get article content using REST API (fallback)."""
        url = f"{self.base_url}/page/summary/{quote(title)}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            summary_data = response.json()
            
            # Get full content
            content_url = f"{self.base_url}/page/html/{quote(title)}"
            content_response = self.session.get(content_url)
            content_response.raise_for_status()
            
            # Extract text from HTML (simplified)
            html_content = content_response.text
            text_content = self._extract_text_from_html(html_content)
            
            return {
                'title': title,
                'summary': summary_data.get('extract', ''),
                'content': text_content,
                'url': summary_data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                'categories': self._extract_categories(html_content)
            }
            
        except Exception as e:
            print(f"Error fetching article '{title}': {e}")
            return None
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract text content from HTML (simplified)."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Decode HTML entities
        text = unicodedata.normalize('NFKD', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_categories(self, html: str) -> List[str]:
        """Extract categories from HTML (simplified)."""
        # This is a simplified implementation
        category_pattern = r'Category:([^"]+)'
        categories = re.findall(category_pattern, html)
        return categories[:10]  # Limit to 10 categories
    
    def get_random_articles(self, count: int) -> List[Dict[str, Any]]:
        """Get random articles."""
        articles = []
        
        if self.api_type == 'wikipedia':
            try:
                for _ in range(count):
                    title = wikipedia.random()
                    article = self.get_article_content(title)
                    if article:
                        articles.append(article)
            except Exception as e:
                print(f"Error getting random articles: {e}")
        else:
            # Fallback: get articles from predefined categories
            for category in self.config.target_categories:
                if len(articles) >= count:
                    break
                    
                search_results = self.search_articles(category, limit=count//len(self.config.target_categories) + 1)
                for title in search_results:
                    if len(articles) >= count:
                        break
                    article = self.get_article_content(title)
                    if article:
                        articles.append(article)
        
        return articles[:count]
        
        for _ in range(count):
            try:
                url = f"{self.base_url}/page/random/summary"
                response = self.session.get(url)
                response.raise_for_status()
                
                data = response.json()
                title = data.get('title')
                
                if title:
                    article = self.get_article_content(title)
                    if article and len(article['content']) >= self.config.min_article_length:
                        articles.append(article)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error getting random article: {e}")
                continue
        
        return articles


class ConceptExtractor:
    """Extract concepts and relationships from text for cognitive enhancement."""
    
    def __init__(self):
        # Simple patterns for concept extraction
        self.concept_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Proper nouns
            r'\b(\w+(?:\s+\w+)*)\s+is\s+a\s+(\w+(?:\s+\w+)*)\b',  # "X is a Y" patterns
            r'\b(\w+(?:\s+\w+)*)\s+(?:causes|leads\s+to|results\s+in)\s+(\w+(?:\s+\w+)*)\b',  # Causal patterns
        ]
        
        self.causal_keywords = [
            'because', 'since', 'due to', 'as a result', 'therefore', 'thus',
            'causes', 'leads to', 'results in', 'brings about', 'triggers',
            'affects', 'influences', 'impacts'
        ]
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text."""
        concepts = set()
        
        # Extract proper nouns and important phrases
        for pattern in self.concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    concepts.update(match)
                else:
                    concepts.add(match)
        
        # Filter and clean concepts
        cleaned_concepts = []
        for concept in concepts:
            concept = concept.strip()
            if len(concept) > 2 and len(concept) < 50:
                cleaned_concepts.append(concept)
        
        return list(cleaned_concepts)[:20]  # Limit to top 20
    
    def extract_causal_relations(self, text: str) -> List[Tuple[str, str]]:
        """Extract causal relationships from text."""
        relations = []
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Look for causal keywords
            for keyword in self.causal_keywords:
                if keyword in sentence.lower():
                    # Simple extraction around causal keywords
                    parts = sentence.lower().split(keyword)
                    if len(parts) == 2:
                        cause = parts[0].strip()[-50:]  # Last 50 chars before keyword
                        effect = parts[1].strip()[:50]  # First 50 chars after keyword
                        
                        if cause and effect:
                            relations.append((cause, effect))
        
        return relations[:10]  # Limit to top 10


class WikipediaDataset(Dataset):
    """Dataset for Wikipedia articles with cognitive enhancement."""
    
    def __init__(self, config: WikiDataConfig):
        self.config = config
        self.articles = []
        self.sequences = []
        
        # Setup directories
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.processed_data_dir, exist_ok=True)
        
        # Initialize tokenizer
        if HAS_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except:
                print("Failed to load HuggingFace tokenizer, using simple tokenizer")
                self.tokenizer = SimpleTokenizer(config.vocab_size)
        else:
            self.tokenizer = SimpleTokenizer(config.vocab_size)
        
        # Initialize concept extractor
        self.concept_extractor = ConceptExtractor()
        
        # Initialize Wikipedia API
        self.wiki_api = WikipediaAPI(config)
        
        # Load or create dataset
        self._load_or_create_dataset()
    
    def _load_or_create_dataset(self):
        """Load existing dataset or create new one."""
        cache_path = os.path.join(self.config.cache_dir, 'processed_articles.pkl')
        
        if os.path.exists(cache_path):
            print("Loading cached dataset...")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.articles = data['articles']
                self.sequences = data['sequences']
            print(f"Loaded {len(self.articles)} articles, {len(self.sequences)} sequences")
        else:
            print("Creating new dataset...")
            self._create_dataset()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'articles': self.articles,
                    'sequences': self.sequences,
                    'config': self.config
                }, f)
            print(f"Saved dataset to cache: {len(self.articles)} articles, {len(self.sequences)} sequences")
    
    def _create_dataset(self):
        """Create dataset from Wikipedia."""
        print("Fetching Wikipedia articles...")
        
        # Collect articles
        all_articles = []
        
        # Get articles from different categories for diversity
        for category in self.config.target_categories:
            print(f"Fetching articles from category: {category}")
            
            # Search for articles in this category
            article_titles = self.wiki_api.search_articles(category, limit=20)
            
            for title in article_titles[:5]:  # Limit to 5 per category for demo
                article = self.wiki_api.get_article_content(title)
                if article and len(article['content']) >= self.config.min_article_length:
                    all_articles.append(article)
                
                if len(all_articles) >= self.config.max_articles:
                    break
            
            if len(all_articles) >= self.config.max_articles:
                break
            
            # Rate limiting
            time.sleep(0.5)
        
        # Add some random articles for diversity
        random_count = min(100, self.config.max_articles - len(all_articles))
        if random_count > 0:
            print(f"Fetching {random_count} random articles...")
            from tqdm import tqdm
            random_articles = []
            
            with tqdm(total=random_count, desc="Random articles", unit="article") as pbar:
                for i in range(random_count):
                    try:
                        if self.wiki_api.api_type == 'wikipedia':
                            import wikipedia
                            title = wikipedia.random()
                            article = self.wiki_api.get_article_content(title)
                        else:
                            # For other API types, get articles from categories
                            import random as rand
                            category = rand.choice(self.config.target_categories)
                            titles = self.wiki_api.search_articles(category, limit=5)
                            if titles:
                                title = rand.choice(titles)
                                article = self.wiki_api.get_article_content(title)
                            else:
                                continue
                        
                        if article and len(article['content']) >= self.config.min_article_length:
                            random_articles.append(article)
                            pbar.set_postfix({'collected': len(random_articles)})
                        
                        pbar.update(1)
                        
                        if len(random_articles) >= random_count:
                            break
                            
                    except Exception as e:
                        pbar.set_postfix({'error': str(e)[:30]})
                        continue
            
            all_articles.extend(random_articles)
        
        self.articles = all_articles[:self.config.max_articles]
        
        print(f"Processing {len(self.articles)} articles...")
        
        # Build tokenizer vocabulary if using simple tokenizer
        if isinstance(self.tokenizer, SimpleTokenizer):
            texts = [article['content'] for article in self.articles]
            self.tokenizer.build_vocab(texts)
            
            # Save tokenizer
            tokenizer_path = os.path.join(self.config.cache_dir, 'tokenizer.pkl')
            self.tokenizer.save(tokenizer_path)
        
        # Process articles into sequences
        self._process_articles()
    
    def _process_articles(self):
        """Process articles into training sequences."""
        self.sequences = []
        
        for article in tqdm(self.articles, desc="Processing articles"):
            content = article['content']
            
            # Extract cognitive features
            concepts = []
            causal_relations = []
            
            if self.config.extract_concepts:
                concepts = self.concept_extractor.extract_concepts(content)
            
            if self.config.extract_causal_relations:
                causal_relations = self.concept_extractor.extract_causal_relations(content)
            
            # Tokenize content
            if hasattr(self.tokenizer, 'encode'):
                if HAS_TRANSFORMERS and hasattr(self.tokenizer, 'model_max_length'):
                    tokens = self.tokenizer.encode(content, max_length=self.config.max_article_length, truncation=True)
                else:
                    tokens = self.tokenizer.encode(content, max_length=self.config.max_article_length)
            else:
                tokens = self.tokenizer(content, max_length=self.config.max_article_length, truncation=True)['input_ids']
            
            # Ensure all token IDs are within vocabulary bounds
            vocab_size = getattr(self.config, 'vocab_size', 32000)
            tokens = [min(max(0, token), vocab_size - 1) for token in tokens]
            
            # Calculate step size for sequence splitting
            step_size = max(1, self.config.max_seq_length - self.config.overlap_length)
            
            # Split into sequences with overlap
            for i in range(0, len(tokens), step_size):
                seq_tokens = tokens[i:i + self.config.max_seq_length]
                
                if len(seq_tokens) >= 50:  # Minimum sequence length
                    # Ensure sequence tokens are also bounds-checked
                    seq_tokens = [min(max(0, token), vocab_size - 1) for token in seq_tokens]
                    self.sequences.append({
                        'input_ids': seq_tokens,
                        'article_title': article['title'],
                        'concepts': concepts,
                        'causal_relations': causal_relations,
                        'categories': article.get('categories', [])
                    })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Get original sequence input_ids
        input_ids = sequence['input_ids']
        original_length = len(input_ids)
        
        # Ensure all token IDs are within vocabulary bounds
        vocab_size = getattr(self.config, 'vocab_size', 32000)
        input_ids = [min(max(0, token), vocab_size - 1) for token in input_ids]
        
        # Truncate if too long
        if len(input_ids) > self.config.max_seq_length:
            input_ids = input_ids[:self.config.max_seq_length]
            original_length = self.config.max_seq_length
        
        # Pad if too short
        if len(input_ids) < self.config.max_seq_length:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
            # Ensure pad token is also within bounds
            pad_token_id = min(max(0, pad_token_id), vocab_size - 1)
            padding = [pad_token_id] * (self.config.max_seq_length - len(input_ids))
            input_ids = input_ids + padding
        
        # Create attention mask based on actual content length (before padding)
        attention_mask = [1] * original_length + [0] * (self.config.max_seq_length - original_length)
        
        # Ensure both tensors are exactly max_seq_length
        assert len(input_ids) == self.config.max_seq_length, f"input_ids length {len(input_ids)} != max_seq_length {self.config.max_seq_length}"
        assert len(attention_mask) == self.config.max_seq_length, f"attention_mask length {len(attention_mask)} != max_seq_length {self.config.max_seq_length}"
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'concepts': sequence['concepts'],
            'causal_relations': sequence['causal_relations'],
            'categories': sequence['categories']
        }


def collate_fn(batch):
    """Custom collate function to ensure consistent tensor sizes."""
    # Extract all the tensors
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    
    # Check that all tensors have the same size
    seq_length = input_ids[0].size(0)
    for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        if ids.size(0) != seq_length:
            print(f"Warning: Sequence {i} has length {ids.size(0)}, expected {seq_length}")
            # Pad or truncate to match
            if ids.size(0) < seq_length:
                # Pad
                padding = torch.zeros(seq_length - ids.size(0), dtype=ids.dtype)
                ids = torch.cat([ids, padding])
                mask_padding = torch.zeros(seq_length - mask.size(0), dtype=mask.dtype)
                mask = torch.cat([mask, mask_padding])
            else:
                # Truncate
                ids = ids[:seq_length]
                mask = mask[:seq_length]
            input_ids[i] = ids
            attention_masks[i] = mask
    
    # Stack tensors
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'concepts': [item['concepts'] for item in batch],
        'causal_relations': [item['causal_relations'] for item in batch],
        'categories': [item['categories'] for item in batch]
    }


def create_wikipedia_dataloader(config: WikiDataConfig = None, batch_size: int = 4, shuffle: bool = True):
    """Create DataLoader for Wikipedia dataset."""
    if config is None:
        config = WikiDataConfig()
    
    dataset = WikipediaDataset(config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with tokenizers
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn  # Use custom collate function
    )
    
    return dataloader, dataset


if __name__ == "__main__":
    # Test the Wikipedia data processing
    print("Testing Wikipedia Data Processing...")
    
    # Create config for testing
    config = WikiDataConfig(
        max_articles=10,  # Small number for testing
        min_article_length=200,
        max_seq_length=512,
        target_categories=["Science", "Technology"]
    )
    
    # Create dataset and dataloader
    dataloader, dataset = create_wikipedia_dataloader(config, batch_size=2)
    
    print(f"Dataset created with {len(dataset)} sequences")
    
    # Test a few batches
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i + 1}:")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Concepts: {batch['concepts']}")
        print(f"Causal relations: {batch['causal_relations']}")
        
        if i >= 2:  # Just test first few batches
            break
    
    print("\nWikipedia data processing system ready!")
    print("Use create_wikipedia_dataloader() to create datasets for training.")