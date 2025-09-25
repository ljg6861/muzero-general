"""
Meta-Reasoning Schema Architecture for True Hybrid Intelligence

This module implements:
1. Schema Inference: Predicts what TYPE of answer is expected
2. Type-Constrained Generation: Filters outputs to match expected schemas
3. Self-Correction: Validates answers and triggers refinement
4. Schema-Aware Memory: Stores type relationships alongside facts

The goal is to separate "schema knowledge" (knowing Paris is a city) 
from "factual recall" (remembering Paris is France's capital).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import re
from dataclasses import dataclass


class AnswerType(Enum):
    """Semantic categories for answer types - the core of schema reasoning"""
    PERSON = "person"
    PLACE = "place" 
    CAPITAL_CITY = "capital_city"
    COUNTRY = "country"
    NUMBER = "number"
    DATE = "date"
    YEAR = "year"
    CHEMICAL_FORMULA = "chemical_formula"
    DEFINITION = "definition"
    PROCESS = "process"
    ORGANIZATION = "organization"
    BOOK = "book"
    MOVIE = "movie"
    LANGUAGE = "language"
    CURRENCY = "currency"
    MEASUREMENT = "measurement"
    COLOR = "color"
    UNKNOWN = "unknown"


@dataclass
class SchemaConstraint:
    """Constraints that define what a valid answer looks like"""
    answer_type: AnswerType
    max_tokens: int = 50
    must_contain_patterns: List[str] = None  # Regex patterns
    forbidden_patterns: List[str] = None
    capitalization: Optional[str] = None  # 'title', 'upper', 'lower', None
    
    def __post_init__(self):
        if self.must_contain_patterns is None:
            self.must_contain_patterns = []
        if self.forbidden_patterns is None:
            self.forbidden_patterns = []


class SchemaInferenceHead(nn.Module):
    """
    Predicts the expected answer type from a question.
    This is the "meta-reasoning" layer that knows schema patterns.
    """
    
    def __init__(self, hidden_size: int, num_types: int = len(AnswerType)):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_types = num_types
        
        # Multi-layer schema classifier
        self.schema_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_types)
        )
        
        # Confidence estimator - how sure are we about the schema?
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Question pattern encoder for better schema detection
        self.pattern_embeddings = self._create_pattern_embeddings()
        
    def _create_pattern_embeddings(self) -> Dict[str, torch.Tensor]:
        """Create embeddings for common question patterns"""
        patterns = {
            "who": ["who is", "who was", "who discovered", "who wrote", "who invented"],
            "what": ["what is", "what was", "what are", "what were"],
            "where": ["where is", "where was", "where are", "where were"],
            "when": ["when is", "when was", "when did", "when will"],
            "how": ["how much", "how many", "how long", "how far"],
            "capital": ["capital of", "capital city"],
            "definition": ["define", "definition of", "meaning of"],
            "chemical": ["chemical formula", "formula for", "molecular formula"]
        }
        
        # This would ideally be learned, but we can start with simple rules
        return patterns
    
    def forward(self, question_encoding: torch.Tensor, question_text: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            question_encoding: [batch_size, hidden_size] - encoded question representation
            question_text: List of actual question strings for pattern matching
            
        Returns:
            schema_logits: [batch_size, num_types] - predicted answer type distribution
            confidence: [batch_size, 1] - confidence in schema prediction
        """
        batch_size = question_encoding.size(0)
        
        # Add pattern-based features if we have question text
        if question_text is not None:
            pattern_features = self._extract_pattern_features(question_text, batch_size)
            enhanced_encoding = question_encoding + pattern_features
        else:
            enhanced_encoding = question_encoding
        
        schema_logits = self.schema_layers(enhanced_encoding)
        confidence = self.confidence_head(enhanced_encoding)
        
        return schema_logits, confidence
    
    def _extract_pattern_features(self, questions: List[str], batch_size: int) -> torch.Tensor:
        """Extract rule-based pattern features to help schema detection"""
        features = torch.zeros(batch_size, self.hidden_size, device=next(self.parameters()).device)
        
        for i, question in enumerate(questions):
            q_lower = question.lower()
            
            # Simple pattern matching - this could be much more sophisticated
            if any(pattern in q_lower for pattern in ["who is", "who was", "who discovered", "who wrote"]):
                features[i, 0] = 1.0  # Person indicator
            elif any(pattern in q_lower for pattern in ["capital of", "capital city"]):
                features[i, 1] = 1.0  # Capital city indicator
            elif any(pattern in q_lower for pattern in ["what is", "define"]):
                features[i, 2] = 1.0  # Definition indicator
            elif any(pattern in q_lower for pattern in ["chemical formula", "formula for"]):
                features[i, 3] = 1.0  # Chemical formula indicator
            elif any(pattern in q_lower for pattern in ["when", "year"]):
                features[i, 4] = 1.0  # Date/year indicator
                
        return features


class TypeConstraintDecoder(nn.Module):
    """
    Constrains generation to match the predicted schema type.
    This ensures the model outputs the RIGHT KIND of answer.
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, num_types: int = len(AnswerType)):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_types = num_types
        
        # Type-specific vocabulary weights
        self.type_vocab_weights = nn.Parameter(torch.ones(num_types, vocab_size))
        
        # Constraint templates for different answer types
        self.constraints = self._create_constraint_templates()
        
    def _create_constraint_templates(self) -> Dict[AnswerType, SchemaConstraint]:
        """Define constraints for each answer type"""
        return {
            AnswerType.PERSON: SchemaConstraint(
                answer_type=AnswerType.PERSON,
                max_tokens=10,
                capitalization='title',
                forbidden_patterns=[r'\d+', r'[a-z]{20,}']  # No numbers, no very long lowercase
            ),
            AnswerType.CAPITAL_CITY: SchemaConstraint(
                answer_type=AnswerType.CAPITAL_CITY,
                max_tokens=5,
                capitalization='title',
                forbidden_patterns=[r'\d+']
            ),
            AnswerType.CHEMICAL_FORMULA: SchemaConstraint(
                answer_type=AnswerType.CHEMICAL_FORMULA,
                max_tokens=10,
                must_contain_patterns=[r'[A-Z][a-z]?\d*'],  # Chemical element pattern
                forbidden_patterns=[r'\s']  # No spaces in formulas
            ),
            AnswerType.YEAR: SchemaConstraint(
                answer_type=AnswerType.YEAR,
                max_tokens=2,
                must_contain_patterns=[r'^\d{4}$'],  # Exactly 4 digits
            ),
            AnswerType.DEFINITION: SchemaConstraint(
                answer_type=AnswerType.DEFINITION,
                max_tokens=100,
                must_contain_patterns=[r'(is|are|refers to|means)']
            )
        }
    
    def forward(self, logits: torch.Tensor, schema_type: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """
        Constrain logits based on predicted schema type
        
        Args:
            logits: [batch_size, seq_len, vocab_size] or [batch_size, vocab_size] - original language model logits
            schema_type: [batch_size, num_types] - schema type probabilities
            confidence: [batch_size, 1] - confidence in schema prediction
            
        Returns:
            constrained_logits: Same shape as input logits - schema-constrained logits
        """
        original_shape = logits.shape
        
        # Handle both 2D and 3D logits
        if len(original_shape) == 3:
            batch_size, seq_len, vocab_size = original_shape
            logits_2d = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
            schema_type_expanded = schema_type.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, schema_type.size(-1))
            confidence_expanded = confidence.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, 1)
        else:
            batch_size, vocab_size = original_shape
            logits_2d = logits
            schema_type_expanded = schema_type
            confidence_expanded = confidence
        
        # Compute type-specific vocabulary weights
        # [batch_size(*seq_len), vocab_size] = [batch_size(*seq_len), num_types] @ [num_types, vocab_size]
        type_weights = torch.matmul(schema_type_expanded, self.type_vocab_weights)
        
        # Apply constraints with confidence weighting
        constraint_strength = confidence_expanded.expand(-1, vocab_size)
        constrained_logits_2d = logits_2d + constraint_strength * torch.log(type_weights + 1e-8)
        
        # Reshape back to original shape
        constrained_logits = constrained_logits_2d.view(original_shape)
        
        return constrained_logits


class SelfCorrectionEngine(nn.Module):
    """
    Validates generated answers against schema expectations.
    If answer doesn't match schema, triggers refinement.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Answer validation network
        self.validator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # question + answer encodings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Validation score 0-1
        )
        
        # Refinement trigger threshold
        self.validation_threshold = 0.7
        
    def forward(self, 
                question_encoding: torch.Tensor,
                answer_encoding: torch.Tensor,
                predicted_schema: torch.Tensor,
                generated_text: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate answer against schema expectations
        
        Args:
            question_encoding: [batch_size, hidden_size]
            answer_encoding: [batch_size, hidden_size] 
            predicted_schema: [batch_size, num_types]
            generated_text: List of generated answer strings
            
        Returns:
            validation_score: [batch_size] - how well answer matches schema
            needs_refinement: [batch_size] - boolean mask for answers needing refinement
        """
        # Combine question and answer representations
        combined = torch.cat([question_encoding, answer_encoding], dim=-1)
        validation_score = self.validator(combined).squeeze(-1)
        
        # Add rule-based validation if we have text
        if generated_text is not None:
            rule_scores = self._rule_based_validation(generated_text, predicted_schema)
            validation_score = 0.7 * validation_score + 0.3 * rule_scores
        
        needs_refinement = validation_score < self.validation_threshold
        
        return validation_score, needs_refinement
    
    def _rule_based_validation(self, answers: List[str], schema_dist: torch.Tensor) -> torch.Tensor:
        """Apply rule-based validation checks"""
        batch_size = len(answers)
        scores = torch.ones(batch_size, device=schema_dist.device)
        
        schema_types = torch.argmax(schema_dist, dim=-1)
        
        for i, (answer, schema_idx) in enumerate(zip(answers, schema_types)):
            answer_type = list(AnswerType)[schema_idx.item()]
            
            # Simple validation rules - could be much more sophisticated
            if answer_type == AnswerType.PERSON:
                if not answer.istitle() or len(answer.split()) > 4:
                    scores[i] *= 0.5
            elif answer_type == AnswerType.CHEMICAL_FORMULA:
                if not re.match(r'^[A-Z][a-z]?\d*([A-Z][a-z]?\d*)*$', answer.strip()):
                    scores[i] *= 0.2
            elif answer_type == AnswerType.YEAR:
                if not re.match(r'^\d{4}$', answer.strip()):
                    scores[i] *= 0.1
        
        return scores


class SchemaAwareHybridMemory:
    """
    Extension of HybridMemory that stores and retrieves schema information
    alongside facts. This enables type-aware fact retrieval.
    """
    
    def __init__(self, base_memory, schema_dim: int = 64):
        self.base_memory = base_memory
        self.schema_dim = schema_dim
        
        # Schema embeddings for each answer type
        self.type_embeddings = nn.Embedding(len(AnswerType), schema_dim)
        
        # Enhanced entity storage with type information
        self.entity_types: Dict[str, AnswerType] = {}
        self.type_constraints: Dict[AnswerType, Set[str]] = {}
        
    def ingest_typed_fact(self, 
                         subject: str, 
                         relation: str, 
                         object_val: str,
                         object_type: AnswerType,
                         passage: str):
        """Store fact with explicit type information"""
        # Store in base memory
        self.base_memory.ingest_fact(subject, relation, object_val, passage)
        
        # Store type information
        self.entity_types[object_val] = object_type
        
        if object_type not in self.type_constraints:
            self.type_constraints[object_type] = set()
        self.type_constraints[object_type].add(object_val)
    
    def retrieve_by_schema(self, 
                          query: str, 
                          expected_type: AnswerType, 
                          k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve facts filtered by expected answer type"""
        # Get candidates from base memory
        candidates = self.base_memory.query(query, k=k*2)  # Get more to filter
        
        # Filter by type constraints
        typed_candidates = []
        for fact, score in candidates:
            # Extract potential answer from fact
            potential_answer = self._extract_answer_from_fact(fact)
            
            if potential_answer in self.entity_types:
                if self.entity_types[potential_answer] == expected_type:
                    typed_candidates.append((fact, score))
        
        return typed_candidates[:k]
    
    def _extract_answer_from_fact(self, fact: str) -> str:
        """Extract the likely answer entity from a fact string"""
        # Simple extraction - could be much more sophisticated
        # Look for patterns like "Paris is the capital" -> "Paris"
        words = fact.split()
        if len(words) > 0:
            return words[0]  # First word as potential answer
        return ""


# Integration class that combines all components
class MetaReasoningEngine(nn.Module):
    """
    The complete meta-reasoning system that orchestrates:
    1. Schema inference from questions
    2. Type-constrained generation 
    3. Self-correction and refinement
    4. Schema-aware memory retrieval
    """
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_size: int,
                 hybrid_memory=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Core components
        self.schema_head = SchemaInferenceHead(hidden_size)
        self.constraint_decoder = TypeConstraintDecoder(vocab_size, hidden_size)
        self.correction_engine = SelfCorrectionEngine(hidden_size)
        
        # Schema-aware memory
        if hybrid_memory:
            self.memory = SchemaAwareHybridMemory(hybrid_memory)
        else:
            self.memory = None
    
    def forward(self, 
                question_encoding: torch.Tensor,
                base_logits: torch.Tensor,
                question_text: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Complete meta-reasoning forward pass
        
        Returns dict with:
        - constrained_logits: Schema-constrained generation logits
        - schema_prediction: Predicted answer type distribution  
        - schema_confidence: Confidence in schema prediction
        - memory_facts: Retrieved schema-relevant facts (if memory available)
        """
        # 1. Infer expected answer schema
        schema_logits, confidence = self.schema_head(question_encoding, question_text)
        schema_dist = F.softmax(schema_logits, dim=-1)
        
        # 2. Constrain generation to match schema
        constrained_logits = self.constraint_decoder(base_logits, schema_dist, confidence)
        
        results = {
            'constrained_logits': constrained_logits,
            'schema_prediction': schema_dist,
            'schema_confidence': confidence,
            'schema_logits': schema_logits
        }
        
        # 3. Retrieve schema-relevant facts if memory available
        if self.memory and question_text:
            memory_facts = []
            predicted_types = torch.argmax(schema_dist, dim=-1)
            
            for i, (question, type_idx) in enumerate(zip(question_text, predicted_types)):
                answer_type = list(AnswerType)[type_idx.item()]
                facts = self.memory.retrieve_by_schema(question, answer_type, k=3)
                memory_facts.append(facts)
            
            results['memory_facts'] = memory_facts
        
        return results
    
    def validate_and_correct(self,
                           question_encoding: torch.Tensor,
                           answer_encoding: torch.Tensor, 
                           schema_prediction: torch.Tensor,
                           generated_text: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate generated answers and identify which need correction
        """
        return self.correction_engine(
            question_encoding, 
            answer_encoding, 
            schema_prediction, 
            generated_text
        )


def create_schema_training_data():
    """
    Create training examples for schema inference.
    This is crucial - the model needs to learn question->answer_type mappings.
    """
    schema_examples = [
        # Person questions
        ("Who discovered penicillin?", AnswerType.PERSON),
        ("Who wrote Romeo and Juliet?", AnswerType.PERSON),
        ("Who invented the telephone?", AnswerType.PERSON),
        
        # Capital questions  
        ("What is the capital of France?", AnswerType.CAPITAL_CITY),
        ("What is the capital of Japan?", AnswerType.CAPITAL_CITY),
        
        # Chemical formulas
        ("What is the chemical formula for water?", AnswerType.CHEMICAL_FORMULA),
        ("What is the formula for salt?", AnswerType.CHEMICAL_FORMULA),
        
        # Years/dates
        ("When was the Declaration of Independence signed?", AnswerType.YEAR),
        ("What year did World War II end?", AnswerType.YEAR),
        
        # Definitions
        ("What is photosynthesis?", AnswerType.DEFINITION),
        ("Define democracy.", AnswerType.DEFINITION),
        
        # Numbers
        ("How many planets are in our solar system?", AnswerType.NUMBER),
        ("What is the population of Tokyo?", AnswerType.NUMBER),
    ]
    
    return schema_examples


# Configuration additions for meta-reasoning
META_REASONING_CONFIG = {
    "meta_reasoning": {
        "enable": True,
        "schema_loss_weight": 0.2,
        "constraint_strength": 0.3,
        "validation_threshold": 0.7,
        "refinement_max_attempts": 3,
        "schema_confidence_threshold": 0.5,
        "memory_type_filtering": True,
        "self_correction": {
            "enable": True,
            "validation_weight": 0.1,
            "refinement_penalty": 0.05
        }
    }
}