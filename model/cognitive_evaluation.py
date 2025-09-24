#!/usr/bin/env python3
"""
Cognitive LLM Evaluation and Monitoring System
==============================================
Comprehensive metrics for tracking both language modeling performance 
and cognitive reasoning quality during Phase 1 training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from cognitive_llm import CognitiveLLM


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Language modeling metrics
    compute_perplexity: bool = True
    compute_bleu: bool = True
    compute_diversity: bool = True
    
    # Cognitive metrics
    evaluate_concept_formation: bool = True
    evaluate_causal_reasoning: bool = True
    evaluate_knowledge_consistency: bool = True
    evaluate_reasoning_quality: bool = True
    
    # Evaluation data
    eval_batch_size: int = 8
    num_eval_samples: int = 1000
    generation_max_length: int = 100
    
    # Output
    save_detailed_results: bool = True
    generate_plots: bool = True
    save_attention_maps: bool = False


class LanguageModelingMetrics:
    """Traditional language modeling evaluation metrics."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def compute_perplexity(self, model: CognitiveLLM, dataloader) -> float:
        """Compute perplexity on evaluation data."""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                try:
                    input_ids = batch['input_ids'].to(next(model.parameters()).device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(input_ids.device)
                    
                    # Clear cache before forward pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                    
                    if outputs['loss'] is not None:
                        # Count actual tokens (excluding padding)
                        if attention_mask is not None:
                            num_tokens = attention_mask.sum().item()
                        else:
                            num_tokens = input_ids.numel()
                        
                        total_loss += outputs['loss'].item() * num_tokens
                        total_tokens += num_tokens
                
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                        print(f"   ⚠️  GPU memory error, skipping batch: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(min(avg_loss, 100))  # Clip to prevent overflow
        
        return perplexity
    
    def compute_bleu_score(self, model: CognitiveLLM, test_prompts: List[str], references: List[str]) -> float:
        """Compute BLEU score for text generation quality."""
        model.eval()
        device = next(model.parameters()).device
        
        generated_texts = []
        
        with torch.no_grad():
            for prompt in tqdm(test_prompts, desc="Generating for BLEU"):
                # Simple tokenization (would need proper tokenizer in practice)
                input_tokens = prompt.split()[:50]  # Limit prompt length
                input_ids = torch.tensor([hash(token) % model.config.vocab_size for token in input_tokens]).unsqueeze(0).to(device)
                
                # Generate continuation
                generated = model.generate(input_ids, max_length=self.config.generation_max_length)
                generated_text = ' '.join([str(token.item()) for token in generated[0]])
                generated_texts.append(generated_text)
        
        # Simplified BLEU computation (would use proper BLEU implementation in practice)
        bleu_scores = []
        for gen, ref in zip(generated_texts, references):
            gen_words = set(gen.split())
            ref_words = set(ref.split())
            
            if len(ref_words) > 0:
                precision = len(gen_words & ref_words) / len(gen_words) if len(gen_words) > 0 else 0
                recall = len(gen_words & ref_words) / len(ref_words)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                bleu_scores.append(f1)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def compute_diversity_metrics(self, model: CognitiveLLM, test_prompts: List[str]) -> Dict[str, float]:
        """Compute diversity metrics for generated text."""
        model.eval()
        device = next(model.parameters()).device
        
        all_generated = []
        
        with torch.no_grad():
            for prompt in tqdm(test_prompts, desc="Generating for diversity"):
                # Generate multiple continuations for each prompt
                input_tokens = prompt.split()[:50]
                input_ids = torch.tensor([hash(token) % model.config.vocab_size for token in input_tokens]).unsqueeze(0).to(device)
                
                for _ in range(3):  # Generate 3 variations
                    generated = model.generate(input_ids, max_length=self.config.generation_max_length, temperature=0.8)
                    generated_text = ' '.join([str(token.item()) for token in generated[0]])
                    all_generated.append(generated_text)
        
        # Calculate diversity metrics
        all_tokens = []
        all_bigrams = []
        
        for text in all_generated:
            tokens = text.split()
            all_tokens.extend(tokens)
            all_bigrams.extend([(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])
        
        # Distinct-1 and Distinct-2
        distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
        distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'vocab_size': len(set(all_tokens)),
            'total_tokens': len(all_tokens)
        }


class CognitiveMetrics:
    """Metrics for evaluating cognitive reasoning capabilities."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate_concept_formation(self, model: CognitiveLLM, dataloader) -> Dict[str, float]:
        """Evaluate concept formation quality."""
        model.eval()
        concept_consistency_scores = []
        concept_diversity_scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating concept formation"):
                input_ids = batch['input_ids'].to(next(model.parameters()).device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(input_ids.device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                hidden_states = outputs['hidden_states']
                
                # Analyze concept representations in hidden states
                batch_size, seq_len, hidden_size = hidden_states.shape
                
                # Concept consistency: measure stability of representations
                if seq_len > 10:
                    # Compare representations at different positions
                    early_repr = hidden_states[:, :seq_len//3, :].mean(dim=1)
                    late_repr = hidden_states[:, 2*seq_len//3:, :].mean(dim=1)
                    
                    # Cosine similarity between early and late representations
                    consistency = F.cosine_similarity(early_repr, late_repr, dim=1).mean().item()
                    concept_consistency_scores.append(consistency)
                
                # Concept diversity: measure how different concepts are represented
                concept_repr = hidden_states.mean(dim=1)  # Average representation per sample
                if len(concept_repr) > 1:
                    # Pairwise distances between concept representations
                    pairwise_dist = torch.pdist(concept_repr).mean().item()
                    concept_diversity_scores.append(pairwise_dist)
        
        return {
            'concept_consistency': np.mean(concept_consistency_scores) if concept_consistency_scores else 0.0,
            'concept_diversity': np.mean(concept_diversity_scores) if concept_diversity_scores else 0.0,
            'concept_formation_score': (np.mean(concept_consistency_scores) + np.mean(concept_diversity_scores)) / 2 if (concept_consistency_scores and concept_diversity_scores) else 0.0
        }
    
    def evaluate_causal_reasoning(self, model: CognitiveLLM, dataloader) -> Dict[str, float]:
        """Evaluate causal reasoning through attention patterns."""
        model.eval()
        causal_scores = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating causal reasoning"):
                input_ids = batch['input_ids'].to(next(model.parameters()).device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(input_ids.device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                attention_probs = outputs['attention_probs']
                
                # Analyze causal structure in attention patterns
                for layer_attention in attention_probs:
                    batch_size, num_heads, seq_len, _ = layer_attention.shape
                    
                    # Check if attention respects causal ordering
                    # Future tokens should have minimal attention
                    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(layer_attention.device)
                    
                    # Attention to future positions (should be minimal)
                    future_attention = (layer_attention * causal_mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean()
                    
                    # Causal score: lower future attention = better causal reasoning
                    causal_score = 1.0 - future_attention.item()
                    causal_scores.append(max(0.0, causal_score))
        
        return {
            'causal_adherence': np.mean(causal_scores) if causal_scores else 0.0,
            'causal_reasoning_score': np.mean(causal_scores) if causal_scores else 0.0
        }
    
    def evaluate_knowledge_consistency(self, model: CognitiveLLM, test_questions: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate consistency of knowledge across different phrasings."""
        model.eval()
        device = next(model.parameters()).device
        consistency_scores = []
        
        with torch.no_grad():
            for question_set in tqdm(test_questions, desc="Evaluating knowledge consistency"):
                question1 = question_set['question1']
                question2 = question_set['question2']  # Same question, different phrasing
                
                # Generate answers for both questions
                answers = []
                for question in [question1, question2]:
                    input_tokens = question.split()[:50]
                    input_ids = torch.tensor([hash(token) % model.config.vocab_size for token in input_tokens]).unsqueeze(0).to(device)
                    
                    generated = model.generate(input_ids, max_length=50, temperature=0.1)  # Low temp for consistency
                    answer = ' '.join([str(token.item()) for token in generated[0]])
                    answers.append(answer)
                
                # Measure similarity between answers
                answer1_tokens = set(answers[0].split())
                answer2_tokens = set(answers[1].split())
                
                if len(answer1_tokens | answer2_tokens) > 0:
                    jaccard_similarity = len(answer1_tokens & answer2_tokens) / len(answer1_tokens | answer2_tokens)
                    consistency_scores.append(jaccard_similarity)
        
        return {
            'knowledge_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
            'consistency_variance': np.var(consistency_scores) if consistency_scores else 0.0
        }
    
    def evaluate_reasoning_quality(self, model: CognitiveLLM, reasoning_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate quality of multi-step reasoning."""
        model.eval()
        device = next(model.parameters()).device
        reasoning_scores = []
        
        with torch.no_grad():
            for task in tqdm(reasoning_tasks, desc="Evaluating reasoning quality"):
                premise = task['premise']
                expected_steps = task.get('expected_steps', [])
                
                # Generate reasoning chain
                input_tokens = premise.split()[:100]
                input_ids = torch.tensor([hash(token) % model.config.vocab_size for token in input_tokens]).unsqueeze(0).to(device)
                
                # Generate with attention to reasoning steps
                generated = model.generate(input_ids, max_length=200, temperature=0.3)
                reasoning_chain = ' '.join([str(token.item()) for token in generated[0]])
                
                # Simplified reasoning quality assessment
                # Count logical connectors and step indicators
                logical_indicators = ['therefore', 'because', 'since', 'thus', 'hence', 'consequently']
                step_indicators = ['first', 'second', 'third', 'next', 'then', 'finally']
                
                logical_count = sum(1 for indicator in logical_indicators if indicator in reasoning_chain.lower())
                step_count = sum(1 for indicator in step_indicators if indicator in reasoning_chain.lower())
                
                # Reasoning score based on presence of logical structure
                reasoning_score = min(1.0, (logical_count + step_count) / 10.0)
                reasoning_scores.append(reasoning_score)
        
        return {
            'reasoning_structure_score': np.mean(reasoning_scores) if reasoning_scores else 0.0,
            'reasoning_consistency': 1.0 - np.var(reasoning_scores) if reasoning_scores else 0.0
        }


class CognitiveLLMEvaluator:
    """Main evaluator for Cognitive LLM combining all metrics."""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.language_metrics = LanguageModelingMetrics(self.config)
        self.cognitive_metrics = CognitiveMetrics(self.config)
        
        # Test data for evaluation
        self.test_prompts = [
            "The theory of relativity states that",
            "Machine learning algorithms can be used to",
            "The process of photosynthesis involves",
            "Economic inflation occurs when",
            "Neural networks are computational models that"
        ]
        
        self.test_questions = [
            {
                'question1': "What is the speed of light?",
                'question2': "How fast does light travel?"
            },
            {
                'question1': "What causes gravity?",
                'question2': "Why do objects fall?"
            }
        ]
        
        self.reasoning_tasks = [
            {
                'premise': "All birds can fly. Penguins are birds.",
                'expected_steps': ["Penguins are birds", "All birds can fly", "Therefore penguins can fly"]
            },
            {
                'premise': "If it rains, the ground gets wet. The ground is wet.",
                'expected_steps': ["Ground is wet", "If rain then wet ground", "Therefore it might have rained"]
            }
        ]
    
    def evaluate_model(self, model: CognitiveLLM, eval_dataloader, save_path: str = None) -> Dict[str, Any]:
        """Comprehensive evaluation of the Cognitive LLM."""
        print("Starting comprehensive model evaluation...")
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_config': {
                'vocab_size': model.config.vocab_size,
                'hidden_size': model.config.hidden_size,
                'num_layers': model.config.num_layers,
                'num_attention_heads': model.config.num_attention_heads
            }
        }
        
        # Language modeling metrics
        if self.config.compute_perplexity:
            print("Computing perplexity...")
            perplexity = self.language_metrics.compute_perplexity(model, eval_dataloader)
            evaluation_results['perplexity'] = perplexity
            print(f"Perplexity: {perplexity:.2f}")
        
        if self.config.compute_bleu:
            print("Computing BLEU score...")
            references = ["light travels at constant speed", "algorithms process data automatically"]  # Simplified
            bleu_score = self.language_metrics.compute_bleu_score(model, self.test_prompts[:2], references)
            evaluation_results['bleu_score'] = bleu_score
            print(f"BLEU Score: {bleu_score:.3f}")
        
        if self.config.compute_diversity:
            print("Computing diversity metrics...")
            diversity_metrics = self.language_metrics.compute_diversity_metrics(model, self.test_prompts)
            evaluation_results['diversity_metrics'] = diversity_metrics
            print(f"Distinct-1: {diversity_metrics['distinct_1']:.3f}, Distinct-2: {diversity_metrics['distinct_2']:.3f}")
        
        # Cognitive metrics
        if self.config.evaluate_concept_formation:
            print("Evaluating concept formation...")
            concept_metrics = self.cognitive_metrics.evaluate_concept_formation(model, eval_dataloader)
            evaluation_results['concept_formation'] = concept_metrics
            print(f"Concept Formation Score: {concept_metrics['concept_formation_score']:.3f}")
        
        if self.config.evaluate_causal_reasoning:
            print("Evaluating causal reasoning...")
            causal_metrics = self.cognitive_metrics.evaluate_causal_reasoning(model, eval_dataloader)
            evaluation_results['causal_reasoning'] = causal_metrics
            print(f"Causal Reasoning Score: {causal_metrics['causal_reasoning_score']:.3f}")
        
        if self.config.evaluate_knowledge_consistency:
            print("Evaluating knowledge consistency...")
            consistency_metrics = self.cognitive_metrics.evaluate_knowledge_consistency(model, self.test_questions)
            evaluation_results['knowledge_consistency'] = consistency_metrics
            print(f"Knowledge Consistency: {consistency_metrics['knowledge_consistency']:.3f}")
        
        if self.config.evaluate_reasoning_quality:
            print("Evaluating reasoning quality...")
            reasoning_metrics = self.cognitive_metrics.evaluate_reasoning_quality(model, self.reasoning_tasks)
            evaluation_results['reasoning_quality'] = reasoning_metrics
            print(f"Reasoning Quality: {reasoning_metrics['reasoning_structure_score']:.3f}")
        
        # Calculate overall cognitive score
        cognitive_scores = []
        if 'concept_formation' in evaluation_results:
            cognitive_scores.append(evaluation_results['concept_formation']['concept_formation_score'])
        if 'causal_reasoning' in evaluation_results:
            cognitive_scores.append(evaluation_results['causal_reasoning']['causal_reasoning_score'])
        if 'knowledge_consistency' in evaluation_results:
            cognitive_scores.append(evaluation_results['knowledge_consistency']['knowledge_consistency'])
        if 'reasoning_quality' in evaluation_results:
            cognitive_scores.append(evaluation_results['reasoning_quality']['reasoning_structure_score'])
        
        if cognitive_scores:
            overall_cognitive_score = np.mean(cognitive_scores)
            evaluation_results['overall_cognitive_score'] = overall_cognitive_score
            print(f"Overall Cognitive Score: {overall_cognitive_score:.3f}")
        
        # Save results
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"Evaluation results saved to {save_path}")
        
        # Generate plots if requested
        if self.config.generate_plots and save_path:
            self._generate_evaluation_plots(evaluation_results, os.path.dirname(save_path))
        
        return evaluation_results
    
    def _generate_evaluation_plots(self, results: Dict[str, Any], save_dir: str):
        """Generate visualization plots for evaluation results."""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # Cognitive metrics radar plot
        if all(key in results for key in ['concept_formation', 'causal_reasoning', 'knowledge_consistency', 'reasoning_quality']):
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            metrics = ['Concept\nFormation', 'Causal\nReasoning', 'Knowledge\nConsistency', 'Reasoning\nQuality']
            values = [
                results['concept_formation']['concept_formation_score'],
                results['causal_reasoning']['causal_reasoning_score'],
                results['knowledge_consistency']['knowledge_consistency'],
                results['reasoning_quality']['reasoning_structure_score']
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            ax.plot(angles, values, 'o-', linewidth=2, label='Cognitive LLM')
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Cognitive Capabilities Assessment', size=16, weight='bold', pad=20)
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cognitive_metrics_radar.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Performance summary bar plot
        summary_metrics = {}
        if 'perplexity' in results:
            # Normalize perplexity (lower is better, so invert)
            summary_metrics['Language\nModeling'] = max(0, 1 - (results['perplexity'] - 1) / 100)
        if 'overall_cognitive_score' in results:
            summary_metrics['Cognitive\nReasoning'] = results['overall_cognitive_score']
        if 'diversity_metrics' in results:
            summary_metrics['Text\nDiversity'] = results['diversity_metrics']['distinct_1']
        
        if summary_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_pos = range(len(summary_metrics))
            values = list(summary_metrics.values())
            labels = list(summary_metrics.keys())
            
            bars = ax.bar(x_pos, values, color=['#3498db', '#e74c3c', '#2ecc71'])
            
            ax.set_xlabel('Evaluation Categories', size=12, weight='bold')
            ax.set_ylabel('Performance Score', size=12, weight='bold')
            ax.set_title('Cognitive LLM Performance Summary', size=16, weight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', weight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Evaluation plots saved to {save_dir}")


def create_evaluation_config():
    """Create default evaluation configuration."""
    return EvaluationConfig(
        compute_perplexity=True,
        compute_bleu=True,
        compute_diversity=True,
        evaluate_concept_formation=True,
        evaluate_causal_reasoning=True,
        evaluate_knowledge_consistency=True,
        evaluate_reasoning_quality=True,
        eval_batch_size=4,
        num_eval_samples=200,
        generation_max_length=50,
        save_detailed_results=True,
        generate_plots=True
    )


if __name__ == "__main__":
    # Test the evaluation system
    from cognitive_llm import CognitiveLLM, create_default_config
    from torch.utils.data import DataLoader, TensorDataset
    
    print("Testing Cognitive LLM Evaluation System...")
    
    # Create dummy model
    model_config = create_default_config()
    model = CognitiveLLM(model_config)
    
    # Create dummy evaluation data
    dummy_data = torch.randint(0, model_config.vocab_size, (100, model_config.max_seq_length))
    dummy_attention = torch.ones(100, model_config.max_seq_length)
    eval_dataset = TensorDataset(dummy_data, dummy_attention)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)
    
    # Create evaluator
    eval_config = create_evaluation_config()
    evaluator = CognitiveLLMEvaluator(eval_config)
    
    # Run evaluation
    results = evaluator.evaluate_model(
        model, 
        eval_dataloader, 
        save_path="test_evaluation/results.json"
    )
    
    print("\nEvaluation completed!")
    print(f"Results: {json.dumps(results, indent=2)}")
    print("\nCognitive LLM evaluation system ready for Phase 1 training monitoring!")