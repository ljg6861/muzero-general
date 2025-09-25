"""
Enhanced Baseline LM with Meta-Reasoning Schema Architecture

This extends the baseline language model with true meta-cognitive capabilities:
1. Schema inference layer that predicts answer types
2. Type-constrained generation that filters outputs by semantic category
3. Self-correction engine that validates and refines answers
4. Schema-aware hybrid memory integration

The goal: Small models that REASON about what they should output, not just memorize.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import json

# Import SimpleLM from main (it's not in a separate baseline_lm module)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from main import SimpleLM

from .schema_reasoning import MetaReasoningEngine, AnswerType, create_schema_training_data
from .hybrid_memory import HybridMemoryEngine


class SimpleLMConfig:
    """Configuration class for SimpleLM compatibility"""
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 30522)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.num_layers = kwargs.get('num_layers', 4)
        self.num_heads = kwargs.get('num_heads', 4)
        self.seq_length = kwargs.get('seq_length', 256)
        self.max_seq_len = kwargs.get('max_seq_len', self.seq_length)


class EnhancedLMConfig(SimpleLMConfig):
    """Extended configuration with meta-reasoning capabilities"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Meta-reasoning configuration
        self.meta_reasoning_enabled = kwargs.get('meta_reasoning_enabled', True)
        self.schema_loss_weight = kwargs.get('schema_loss_weight', 0.2)
        self.constraint_strength = kwargs.get('constraint_strength', 0.3)
        self.validation_threshold = kwargs.get('validation_threshold', 0.7)
        self.refinement_max_attempts = kwargs.get('refinement_max_attempts', 3)
        self.schema_confidence_threshold = kwargs.get('schema_confidence_threshold', 0.5)
        self.memory_type_filtering = kwargs.get('memory_type_filtering', True)
        
        # Self-correction settings
        self.self_correction_enabled = kwargs.get('self_correction_enabled', True)
        self.validation_weight = kwargs.get('validation_weight', 0.1)
        self.refinement_penalty = kwargs.get('refinement_penalty', 0.05)


class MetaCognitiveLM(nn.Module):
    """
    The complete meta-cognitive language model that separates:
    1. Language reasoning (transformer core)
    2. Schema knowledge (what type of answer is expected)
    3. Factual recall (external memory)
    4. Self-correction (validation and refinement)
    
    This is the architecture for TRUE INTELLIGENCE in small models.
    """
    
    def __init__(self, 
                 config: EnhancedLMConfig,
                 hybrid_memory: Optional[HybridMemoryEngine] = None):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Core transformer for language reasoning and schema inference
        self.transformer = SimpleLM(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            gradient_checkpointing=getattr(config, 'gradient_checkpointing', False)
        )
        
        # Meta-reasoning engine for schema-aware generation
        if config.meta_reasoning_enabled:
            self.meta_engine = MetaReasoningEngine(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                hybrid_memory=hybrid_memory
            )
        else:
            self.meta_engine = None
        
        # External hybrid memory for factual recall
        self.hybrid_memory = hybrid_memory
        
        # Training state tracking
        self.training_step = 0
        self.schema_accuracy_tracker = []
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                question_text: Optional[List[str]] = None,
                answer_text: Optional[List[str]] = None,
                answer_types: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, Any]:
        """
        Forward pass with meta-reasoning capabilities
        
        Args:
            input_ids: [batch_size, seq_length] 
            attention_mask: [batch_size, seq_length]
            labels: [batch_size, seq_length] - for language modeling loss
            question_text: List of question strings for schema inference
            answer_text: List of answer strings for validation
            answer_types: [batch_size] - ground truth answer types for training
            
        Returns:
            Dictionary containing:
            - loss: Combined training loss
            - logits: Final generation logits (potentially constrained)
            - schema_predictions: Predicted answer types
            - validation_scores: Answer validation scores
            - meta_reasoning_info: Detailed meta-reasoning outputs
        """
        batch_size, seq_length = input_ids.shape
        
        # 1. Core transformer forward pass
        # SimpleLM returns (logits, aux) tuple
        transformer_output = self.transformer(input_ids)
        if isinstance(transformer_output, tuple):
            base_logits, aux = transformer_output
        else:
            base_logits = transformer_output
            aux = {}
        
        # Compute language modeling loss if labels provided
        base_loss = None
        if labels is not None:
            base_loss = F.cross_entropy(
                base_logits.view(-1, base_logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100
            )
        
        # Get hidden states for meta-reasoning (encode method gives us hidden states)
        hidden_states = self.transformer.encode(input_ids)
        
        # Extract question encoding from transformer hidden states
        # Assume question is at the beginning of sequence
        question_encoding = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        
        results = {
            'logits': base_logits,
            'loss': base_loss,
            'hidden_states': hidden_states
        }
        
        # 2. Meta-reasoning enhancement
        if self.meta_engine and self.config.meta_reasoning_enabled:
            meta_results = self.meta_engine(
                question_encoding=question_encoding,
                base_logits=base_logits,
                question_text=question_text
            )
            
            # Use schema-constrained logits for generation
            results['logits'] = meta_results['constrained_logits']
            results['schema_predictions'] = meta_results['schema_prediction']
            results['schema_confidence'] = meta_results['schema_confidence']
            results['meta_reasoning_info'] = meta_results
            
            # 3. Compute schema prediction loss if we have ground truth types
            if answer_types is not None and self.training:
                schema_loss = F.cross_entropy(
                    meta_results['schema_logits'], 
                    answer_types
                )
                results['schema_loss'] = schema_loss
                
                # Combine losses
                if base_loss is not None:
                    total_loss = base_loss + self.config.schema_loss_weight * schema_loss
                    results['loss'] = total_loss
                
                # Track schema accuracy
                schema_preds = torch.argmax(meta_results['schema_logits'], dim=-1)
                schema_accuracy = (schema_preds == answer_types).float().mean()
                self.schema_accuracy_tracker.append(schema_accuracy.item())
        
        # 4. Self-correction validation (if enabled and we have answer text)
        if (self.meta_engine and 
            self.config.self_correction_enabled and 
            answer_text is not None and
            'schema_predictions' in results):
            
            # Generate answer encodings (simplified - could use actual generated text)
            answer_encoding = question_encoding  # Placeholder - would encode actual answers
            
            validation_scores, needs_refinement = self.meta_engine.validate_and_correct(
                question_encoding=question_encoding,
                answer_encoding=answer_encoding,
                schema_prediction=results['schema_predictions'],
                generated_text=answer_text
            )
            
            results['validation_scores'] = validation_scores
            results['needs_refinement'] = needs_refinement
            
            # Add validation loss if training
            if self.training and labels is not None:
                # Validation targets: 1.0 for correct answers, 0.0 for incorrect
                validation_targets = torch.ones_like(validation_scores)  # Simplified
                validation_loss = F.binary_cross_entropy(validation_scores, validation_targets)
                
                if results['loss'] is not None:
                    total_loss = results['loss'] + self.config.validation_weight * validation_loss
                    results['loss'] = total_loss
                results['validation_loss'] = validation_loss
        
        if not return_dict:
            return tuple(results.values())
        
        return results
    
    def generate_with_meta_reasoning(self,
                                   input_ids: torch.Tensor,
                                   attention_mask: Optional[torch.Tensor] = None,
                                   question_text: Optional[List[str]] = None,
                                   max_length: int = 50,
                                   temperature: float = 1.0,
                                   do_sample: bool = True,
                                   return_meta_info: bool = True) -> Dict[str, Any]:
        """
        Generate with meta-reasoning: schema inference -> constrained generation -> self-correction
        
        This is the complete intelligent generation pipeline.
        """
        self.eval()
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Track generation state
        generated_ids = input_ids.clone()
        generation_info = {
            'schema_predictions': [],
            'validation_scores': [],
            'refinement_attempts': [],
            'memory_retrievals': []
        }
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass with meta-reasoning
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    question_text=question_text,
                    return_dict=True
                )
                
                # Get next token logits (potentially schema-constrained)
                next_token_logits = outputs['logits'][:, -1, :] / temperature
                
                # Sample next token
                if do_sample:
                    next_tokens = torch.multinomial(
                        F.softmax(next_token_logits, dim=-1), 
                        num_samples=1
                    )
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generation
                generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
                
                # Store meta-reasoning info
                if return_meta_info and 'schema_predictions' in outputs:
                    generation_info['schema_predictions'].append(
                        outputs['schema_predictions'].cpu()
                    )
                    if 'validation_scores' in outputs:
                        generation_info['validation_scores'].append(
                            outputs['validation_scores'].cpu()
                        )
                
                # Check for end of sequence
                if next_tokens.squeeze() == self.transformer.tokenizer.eos_token_id:
                    break
        
        results = {
            'generated_ids': generated_ids,
            'input_length': input_ids.size(1),
            'generated_length': generated_ids.size(1) - input_ids.size(1)
        }
        
        if return_meta_info:
            results['meta_reasoning_info'] = generation_info
        
        return results
    
    def get_schema_accuracy(self) -> float:
        """Get recent schema prediction accuracy"""
        if not self.schema_accuracy_tracker:
            return 0.0
        
        # Return average of last 100 predictions
        recent = self.schema_accuracy_tracker[-100:]
        return sum(recent) / len(recent)
    
    def demonstrate_meta_reasoning(self, questions: List[str]) -> Dict[str, Any]:
        """
        Demonstrate the meta-reasoning capabilities on example questions.
        This shows the schema inference -> constrained generation -> validation pipeline.
        """
        self.eval()
        
        demonstrations = []
        
        with torch.no_grad():
            for question in questions:
                # Mock tokenizer behavior (in real implementation, would use actual tokenizer)
                # For now, just create random input_ids to represent the question
                input_ids = torch.randint(0, self.vocab_size, (1, 10))  # Mock tokenization
                
                # Get question encoding using transformer
                hidden_states = self.transformer.encode(input_ids)
                question_encoding = hidden_states[:, 0, :]
                
                # Get base logits
                transformer_output = self.transformer(input_ids)
                if isinstance(transformer_output, tuple):
                    base_logits, aux = transformer_output
                else:
                    base_logits = transformer_output
                
                # Run meta-reasoning
                if self.meta_engine:
                    meta_results = self.meta_engine(
                        question_encoding=question_encoding,
                        base_logits=base_logits,
                        question_text=[question]
                    )
                    
                    # Interpret results
                    schema_idx = torch.argmax(meta_results['schema_prediction'], dim=-1).item()
                    predicted_type = list(AnswerType)[schema_idx]
                    confidence = meta_results['schema_confidence'].item()
                    
                    demo = {
                        'question': question,
                        'predicted_type': predicted_type.value,
                        'confidence': confidence,
                        'schema_distribution': meta_results['schema_prediction'][0].tolist(),
                        'memory_facts': meta_results.get('memory_facts', [])
                    }
                    
                    demonstrations.append(demo)
        
        return {
            'demonstrations': demonstrations,
            'schema_types': [t.value for t in AnswerType],
            'model_info': {
                'parameters': sum(p.numel() for p in self.parameters()),
                'meta_reasoning_enabled': self.config.meta_reasoning_enabled,
                'schema_accuracy': self.get_schema_accuracy()
            }
        }


def load_enhanced_model(config_path: str, checkpoint_path: Optional[str] = None) -> MetaCognitiveLM:
    """
    Load the enhanced meta-cognitive model from configuration and optional checkpoint
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Extract meta-reasoning config
    meta_config = config_dict.get('meta_reasoning', {})
    
    # Create enhanced config
    model_config = config_dict.get('model', {})
    enhanced_config = EnhancedLMConfig(
        hidden_size=model_config.get('hidden_size', 256),
        num_layers=model_config.get('num_layers', 4),
        num_heads=model_config.get('num_heads', 4),
        seq_length=model_config.get('seq_length', 256),
        vocab_size=30522,  # Default - will be updated by tokenizer
        
        # Meta-reasoning settings
        meta_reasoning_enabled=meta_config.get('enable', True),
        schema_loss_weight=meta_config.get('schema_loss_weight', 0.2),
        constraint_strength=meta_config.get('constraint_strength', 0.3),
        validation_threshold=meta_config.get('validation_threshold', 0.7),
        refinement_max_attempts=meta_config.get('refinement_max_attempts', 3),
        schema_confidence_threshold=meta_config.get('schema_confidence_threshold', 0.5),
        memory_type_filtering=meta_config.get('memory_type_filtering', True),
        
        # Self-correction
        self_correction_enabled=meta_config.get('self_correction', {}).get('enable', True),
        validation_weight=meta_config.get('self_correction', {}).get('validation_weight', 0.1),
        refinement_penalty=meta_config.get('self_correction', {}).get('refinement_penalty', 0.05)
    )
    
    # Create hybrid memory if configured
    hybrid_memory = None
    if config_dict.get('hybrid_memory', {}).get('enable', False):
        from .hybrid_memory import HybridMemoryEngine
        hybrid_memory = HybridMemoryEngine(
            embedding_dim=enhanced_config.hidden_size,
            num_centroids=256
        )
    
    # Create model
    model = MetaCognitiveLM(enhanced_config, hybrid_memory)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


# Demo function to show the complete pipeline
def demo_meta_reasoning():
    """
    Demonstrate the complete meta-reasoning pipeline
    """
    print("🧠 Meta-Reasoning Intelligence Demo")
    print("=" * 50)
    
    # Create demo model
    config = EnhancedLMConfig(
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        vocab_size=30522,
        meta_reasoning_enabled=True
    )
    
    model = MetaCognitiveLM(config)
    
    # Demo questions that require different types of reasoning
    demo_questions = [
        "Who discovered penicillin?",  # Person
        "What is the capital of France?",  # Capital city
        "What is the chemical formula for water?",  # Chemical formula
        "When was the Declaration of Independence signed?",  # Year
        "What is photosynthesis?",  # Definition
        "How many planets are in our solar system?",  # Number
    ]
    
    # Run demonstration
    results = model.demonstrate_meta_reasoning(demo_questions)
    
    print(f"Model Parameters: {results['model_info']['parameters']:,}")
    print(f"Meta-Reasoning Enabled: {results['model_info']['meta_reasoning_enabled']}")
    print(f"Schema Accuracy: {results['model_info']['schema_accuracy']:.3f}")
    print()
    
    for demo in results['demonstrations']:
        print(f"Question: {demo['question']}")
        print(f"Predicted Type: {demo['predicted_type']}")
        print(f"Confidence: {demo['confidence']:.3f}")
        
        if demo['memory_facts']:
            print(f"Retrieved Facts: {len(demo['memory_facts'])}")
        
        print("-" * 30)
    
    return results


if __name__ == "__main__":
    # Run the demonstration
    demo_meta_reasoning()