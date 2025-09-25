#!/usr/bin/env python3
"""
Test script for the retri    # A    # Add some test facts
    print("Adding test facts to memory...")
    hybrid_memory.ingest_fact("France", "capital", "Paris")
    hybrid_memory.ingest_fact("Alexander Graham Bell", "invented", "telephone")  
    hybrid_memory.ingest_fact("Eiffel Tower", "located_in", "Paris")
    hybrid_memory.ingest_fact("World War II", "time_period", "1939-1945")
    hybrid_memory.ingest_fact("light", "speed", "299792458 m/s")
    hybrid_memory.ingest_fact("Earth", "continents", "7")test facts
    print("Adding test facts to memory...")
    hybrid_memory.ingest_fact("France", "capital", "Paris")
    hybrid_memory.ingest_fact("Alexander Graham Bell", "invented", "telephone")  
    hybrid_memory.ingest_fact("Eiffel Tower", "located_in", "Paris")
    hybrid_memory.ingest_fact("World War II", "time_period", "1939-1945")
    hybrid_memory.ingest_fact("light", "speed", "299792458 m/s")
    hybrid_memory.ingest_fact("Earth", "continents", "7")the-loop probe functionality
"""

import sys
import os
import torch
from transformers import AutoTokenizer
from unified_model import UnifiedCognitiveLM
from core.models.hybrid_memory import HybridMemoryEngine
from core.models.router_retriever import HybridRetriever

# Add the project directory to the path
sys.path.append('/home/lucas/muzero-general')

def test_retrieval_probe():
    """Test the retrieval probe functionality"""
    print("🧪 Testing retrieval probe functionality...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Initialize model  
    print("Initializing model...")
    model = UnifiedCognitiveLM(
        vocab_size=50257,
        max_seq_len=512,
        hidden_size=768,
        num_heads=12,
        num_layers=6,
        enable_meta_reasoning=True,
        enable_schema_inference=True,
        enable_hybrid_memory=True,
        enable_self_correction=True,
        enable_router=True
    ).to(device)
    
    # 3. Initialize hybrid memory
    print("Initializing hybrid memory...")
    hybrid_memory = HybridMemoryEngine(
        entity_dim=256,
        passage_dim=384,
        use_faiss=True
    )
    
    # Add some test facts
    print("Adding test facts to memory...")
    hybrid_memory.ingest_fact("France", "capital", "Paris")
    hybrid_memory.ingest_fact("Alexander Graham Bell", "invented", "telephone")  
    hybrid_memory.ingest_fact("Eiffel Tower", "located_in", "Paris")
    hybrid_memory.ingest_fact("World War II", "time_period", "1939-1945")
    hybrid_memory.ingest_fact("light", "speed", "299792458 m/s")
    hybrid_memory.ingest_fact("Earth", "continents", "7")
    
    # 4. Initialize hybrid retriever (minimal setup)
    print("Initializing hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device=str(device),
        dense_weight=0.6,
        sparse_weight=0.4
    )
    
    # Add some test passages
    passages = [
        "Paris is the capital and largest city of France.",
        "Alexander Graham Bell was a Scottish-American inventor who is credited with inventing the telephone.",
        "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.",
        "World War II was fought from 1939 to 1945 between the Axis and Allied powers.",
        "The speed of light in a vacuum is exactly 299,792,458 meters per second.",
        "Earth has seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."
    ]
    
    for i, passage in enumerate(passages):
        hybrid_retriever.add_document(f"doc_{i}", passage)
    
    print(f"✓ Added {len(passages)} passages to retriever")
    
    # 5. Import and test the retrieval probe
    print("\n" + "="*50)
    print("🎯 TESTING RETRIEVAL PROBE")
    print("="*50)
    
    # Import the function from train.py
    sys.path.append('/home/lucas/muzero-general')
    from train import run_retrieval_probe
    
    # Run the probe
    run_retrieval_probe(
        model=model,
        tokenizer=tokenizer,
        hybrid_retriever=hybrid_retriever,
        hybrid_memory=hybrid_memory,
        device=device,
        step=100
    )
    
    print("✅ Retrieval probe test completed successfully!")

if __name__ == '__main__':
    test_retrieval_probe()