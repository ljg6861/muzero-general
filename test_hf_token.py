#!/usr/bin/env python3
"""
Test HuggingFace Token Integration
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_token_setup():
    """Test if HF token is properly loaded"""
    print("🔑 Testing HuggingFace Token Setup...")
    
    # Check if token is loaded
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        print("❌ No HF_TOKEN found in environment")
        print("📝 Please add your token to .env file:")
        print("   HF_TOKEN=your_actual_token_here")
        return False
    elif hf_token == 'your_token_here':
        print("❌ HF_TOKEN is still set to placeholder value")
        print("📝 Please replace with your actual token in .env file")
        return False
    else:
        # Mask token for security (show first 4 and last 4 characters)
        masked_token = hf_token[:4] + "*" * (len(hf_token) - 8) + hf_token[-4:]
        print(f"✅ HF_TOKEN loaded: {masked_token}")
        return True

def test_dataset_with_token():
    """Test dataset loading with token"""
    if not test_token_setup():
        return
        
    print("\n📊 Testing dataset loading with token...")
    
    try:
        from datasets import load_dataset
        
        # Test with a simple, reliable dataset
        hf_token = os.getenv('HF_TOKEN')
        
        print("Loading IMDB dataset...")
        dataset = load_dataset(
            'imdb', 
            split='train', 
            streaming=True,
            token=hf_token if hf_token != 'your_token_here' else None
        )
        
        # Try to get first item
        first_item = next(iter(dataset))
        print(f"✅ Dataset loaded successfully!")
        print(f"   Sample text: {first_item['text'][:100]}...")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")

if __name__ == '__main__':
    test_dataset_with_token()