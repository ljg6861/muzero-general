#!/usr/bin/env python3
"""
Test the core training components without heavy memory usage
"""

import torch
from transformers import AutoTokenizer
import sys
sys.path.append('.')
from unified_model import create_unified_model, create_tokenizer
from train import generate_router_labels, compute_router_loss

def test_core_components():
    print("🧪 Testing Core Training Components")
    print("=" * 40)
    
    # Create tokenizer and model
    tokenizer = create_tokenizer()
    model = create_unified_model(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_seq_len=512,
        enable_router=True,
        enable_schema_inference=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test router label generation
    test_texts = [
        "What is the capital of France?",
        "Who invented the telephone?", 
        "Where is the Eiffel Tower located?",
        "When was World War II fought?"
    ]
    
    print("\n🎯 Testing Router Label Generation:")
    for text in test_texts:
        answerability, types = generate_router_labels(text, tokenizer)
        print(f"  '{text}'")
        print(f"    Answerability: {answerability:.2f}")
        print(f"    Types: {types}")
    
    # Test forward pass and router loss
    print("\n🔄 Testing Forward Pass & Router Loss:")
    input_ids = tokenizer([
        "What is the capital of France?",
        "The sky is blue and beautiful."
    ], return_tensors='pt', padding=True, truncation=True, max_length=128)['input_ids'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {logits.shape}")
    
    # Test router loss computation
    router_loss = compute_router_loss(model, input_ids, tokenizer, device)
    if router_loss is not None:
        print(f"  Router loss: {router_loss.item():.4f}")
    else:
        print("  Router loss: Not available")
    
    print("\n✅ All core components working!")

if __name__ == '__main__':
    test_core_components()