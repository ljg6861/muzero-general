"""
Simple Text Data Loader for Phase A
==================================
Fallback data loader that uses built-in text sources when external datasets fail.
Creates sufficient training data for Phase A baseline LM competence.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import re
import random
from typing import List, Iterator, Optional
from transformers import PreTrainedTokenizer


class SimpleTextDataset(Dataset):
    """Simple text dataset using built-in text sources."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        seq_length: int = 1024,
        num_samples: int = 10000,
        min_length: int = 50
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.min_length = min_length
        
        # Generate diverse text samples
        self.texts = self._generate_text_samples(num_samples)
        
        # Tokenize all texts
        self.tokenized_texts = []
        for text in self.texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) >= min_length:
                self.tokenized_texts.append(tokens)
        
        print(f"✓ Simple dataset created: {len(self.tokenized_texts)} samples, "
              f"~{sum(len(t) for t in self.tokenized_texts):,} tokens")
    
    def _generate_text_samples(self, num_samples: int) -> List[str]:
        """Generate diverse text samples for training."""
        
        # Base templates for different text types
        templates = {
            'narrative': [
                "Once upon a time, there was a {character} who lived in {place}. "
                "Every day, {character} would {activity} and dream of {goal}. "
                "One day, something unexpected happened that changed everything. "
                "The {character} discovered {discovery} which led to an amazing adventure.",
                
                "In the year {year}, {character} made a decision that would change history. "
                "They believed that {belief} and were willing to sacrifice everything for it. "
                "Through determination and courage, they overcame {obstacle} and achieved {achievement}.",
            ],
            
            'informative': [
                "The concept of {concept} has evolved significantly over the past {timeframe}. "
                "Researchers have found that {finding} which has important implications for {field}. "
                "This discovery suggests that {implication} and opens new possibilities for {application}.",
                
                "Understanding {topic} requires knowledge of several key principles. "
                "First, {principle1}. Second, {principle2}. Third, {principle3}. "
                "These principles work together to create {outcome} which benefits {beneficiary}.",
            ],
            
            'dialogue': [
                'Assistant: Hello! How can I help you today?\n'
                'User: {question}\n'
                'Assistant: {response}\n'
                'User: {followup}\n'
                'Assistant: {final_response}',
                
                'Person A: {greeting}\n'
                'Person B: {response}\n'
                'Person A: {continuation}\n'
                'Person B: {conclusion}',
            ],
            
            'technical': [
                "The {system} operates using {mechanism} to achieve {purpose}. "
                "Key components include {component1}, {component2}, and {component3}. "
                "When {condition}, the system {behavior} resulting in {outcome}.",
                
                "Algorithm {name} solves the problem of {problem} by {approach}. "
                "The time complexity is {complexity} and space requirements are {space}. "
                "This makes it suitable for {application} where {constraint} is important.",
            ]
        }
        
        # Word lists for filling templates
        words = {
            'character': ['scientist', 'explorer', 'artist', 'teacher', 'inventor', 'leader', 'student', 'researcher'],
            'place': ['a bustling city', 'a quiet village', 'a mysterious forest', 'a modern laboratory', 'a distant planet'],
            'activity': ['study the stars', 'paint landscapes', 'solve puzzles', 'help others', 'build inventions'],
            'goal': ['making a discovery', 'creating something beautiful', 'helping humanity', 'understanding the universe'],
            'year': ['2020', '2021', '2022', '2023', '2024'],
            'belief': ['science could solve any problem', 'art could change the world', 'education was the key to progress'],
            'obstacle': ['skeptical colleagues', 'limited resources', 'technical challenges', 'social barriers'],
            'achievement': ['revolutionary breakthrough', 'lasting positive change', 'new understanding', 'better future'],
            'concept': ['artificial intelligence', 'quantum computing', 'renewable energy', 'space exploration'],
            'timeframe': ['decade', 'century', 'few years', 'recent decades'],
            'finding': ['efficiency can be improved dramatically', 'new patterns emerge under specific conditions'],
            'field': ['computer science', 'physics', 'biology', 'engineering', 'medicine'],
            'topic': ['machine learning', 'sustainable development', 'human psychology', 'economic systems'],
            'question': ['What is the best way to learn programming?', 'How does artificial intelligence work?'],
            'response': ['Great question! Programming is best learned through practice and projects.', 'AI systems learn patterns from data to make predictions.'],
            'system': ['neural network', 'recommendation engine', 'search algorithm', 'optimization system'],
            'mechanism': ['gradient descent', 'collaborative filtering', 'tree traversal', 'genetic algorithms'],
            'name': ['QuickSort', 'Dijkstra', 'BackPropagation', 'PageRank'],
            'problem': ['sorting large datasets', 'finding shortest paths', 'learning from examples', 'ranking web pages'],
        }
        
        # Generate texts
        texts = []
        categories = list(templates.keys())
        
        for _ in range(num_samples):
            category = random.choice(categories)
            template = random.choice(templates[category])
            
            # Fill template with random words
            filled_text = template
            for key, values in words.items():
                if f'{{{key}}}' in filled_text:
                    filled_text = filled_text.replace(f'{{{key}}}', random.choice(values))
            
            # Add some variation in length
            if random.random() < 0.3:  # 30% chance to extend
                extension = self._generate_extension(category)
                filled_text += " " + extension
            
            texts.append(filled_text)
        
        return texts
    
    def _generate_extension(self, category: str) -> str:
        """Generate additional content to extend texts."""
        
        extensions = {
            'narrative': [
                "The journey was not easy, but it taught valuable lessons about perseverance and hope.",
                "Years later, people would remember this story and be inspired to pursue their own dreams.",
                "This experience changed not only the character but everyone they encountered along the way."
            ],
            'informative': [
                "Further research is needed to fully understand all the implications of this discovery.",
                "The practical applications of this knowledge continue to expand as technology advances.",
                "Scientists around the world are building upon this foundation to make new breakthroughs."
            ],
            'dialogue': [
                "This conversation continued for hours, exploring many fascinating topics.",
                "Both participants learned something new from this exchange of ideas.",
                "The discussion highlighted the importance of clear communication and understanding."
            ],
            'technical': [
                "Implementation details may vary depending on specific requirements and constraints.",
                "Performance optimization techniques can further improve efficiency in practical applications.",
                "This approach has been successfully applied to solve real-world problems across various domains."
            ]
        }
        
        return random.choice(extensions.get(category, extensions['informative']))
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        
        # Create input/target pairs for language modeling
        if len(tokens) <= self.seq_length:
            # Pad if too short
            input_ids = tokens + [self.tokenizer.pad_token_id] * (self.seq_length - len(tokens))
            labels = input_ids.copy()
            
            # Don't compute loss on padding tokens
            for i in range(len(tokens), len(labels)):
                labels[i] = -100
        else:
            # Truncate if too long
            start_idx = random.randint(0, len(tokens) - self.seq_length)
            input_ids = tokens[start_idx:start_idx + self.seq_length]
            labels = input_ids.copy()
        
        # For causal LM, labels are input_ids shifted by 1
        # But we'll use the same tokens for simplicity in this demo
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long)
        }


def create_simple_data_loader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    seq_length: int = 1024,
    num_samples: int = 50000,
    num_workers: int = 0
) -> DataLoader:
    """Create a simple data loader for Phase A training."""
    
    print("Creating Simple Text Data Loader")
    print("=" * 50)
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Number of samples: {num_samples:,}")
    print(f"Expected tokens: ~{num_samples * seq_length // 2:,}")
    
    dataset = SimpleTextDataset(
        tokenizer=tokenizer,
        seq_length=seq_length,
        num_samples=num_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✓ Data loader created with {len(dataset)} samples")
    return dataloader


# Example usage for testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Test the simple data loader
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataloader = create_simple_data_loader(
        tokenizer=tokenizer,
        batch_size=2,
        seq_length=512,
        num_samples=100
    )
    
    # Test a few batches
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Just test first 3 batches
            break
        
        print(f"\nBatch {i+1}:")
        print(f"  Input shape: {batch['input_ids'].shape}")
        print(f"  Sample text: {tokenizer.decode(batch['input_ids'][0][:100], skip_special_tokens=True)}")
    
    print("\n✓ Simple data loader test completed successfully!")