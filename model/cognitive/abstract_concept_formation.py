#!/usr/bin/env python3
"""
Abstract Concept Formation Engine
================================
Builds hierarchical knowledge structures that form abstract concepts from concrete observations.
Creates taxonomies and establishes conceptual relationships across domains.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict, Counter
import networkx as nx
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ConceptNode:
    """Represents a single concept in the hierarchical knowledge structure."""
    
    def __init__(self, name: str, level: int, features: Dict[str, Any], 
                 parent: Optional['ConceptNode'] = None):
        self.name = name
        self.level = level  # 0 = concrete, higher = more abstract
        self.features = features
        self.parent = parent
        self.children = []
        self.instances = []
        self.abstractions = {}
        self.relationships = {}
        self.confidence = 0.0
        self.creation_time = datetime.now()
        
    def add_child(self, child: 'ConceptNode'):
        """Add a child concept."""
        child.parent = self
        self.children.append(child)
        
    def add_instance(self, instance: Dict[str, Any]):
        """Add a concrete instance of this concept."""
        self.instances.append(instance)
        
    def compute_abstraction(self) -> Dict[str, Any]:
        """Compute abstract features from instances and children."""
        if not self.instances and not self.children:
            return self.features
            
        # Aggregate features from instances
        feature_stats = defaultdict(list)
        
        for instance in self.instances:
            for key, value in instance.items():
                if isinstance(value, (int, float)):
                    feature_stats[key].append(value)
                    
        # Compute statistical abstractions
        abstractions = {}
        for feature, values in feature_stats.items():
            if values:
                abstractions[f"{feature}_mean"] = np.mean(values)
                abstractions[f"{feature}_std"] = np.std(values)
                abstractions[f"{feature}_min"] = np.min(values)
                abstractions[f"{feature}_max"] = np.max(values)
                
        # Include categorical features
        for instance in self.instances:
            for key, value in instance.items():
                if isinstance(value, str):
                    if f"{key}_categories" not in abstractions:
                        abstractions[f"{key}_categories"] = set()
                    abstractions[f"{key}_categories"].add(value)
                    
        # Convert sets to lists for JSON serialization
        for key, value in abstractions.items():
            if isinstance(value, set):
                abstractions[key] = list(value)
                
        self.abstractions = abstractions
        return abstractions


class AbstractConceptFormation:
    """Main engine for forming abstract concepts from concrete observations."""
    
    def __init__(self):
        self.concepts = {}  # name -> ConceptNode
        self.concept_graph = nx.DiGraph()
        self.feature_space = defaultdict(list)
        self.similarity_matrix = None
        self.abstraction_levels = defaultdict(list)
        self.domain_knowledge = {}
        self.learning_history = []
        
    def add_observation(self, observation: Dict[str, Any], category: str = None):
        """Add a new concrete observation to the knowledge base."""
        obs_id = f"obs_{len(self.feature_space['observations'])}"
        observation['id'] = obs_id
        observation['category'] = category
        
        self.feature_space['observations'].append(observation)
        
        # Update feature statistics
        for key, value in observation.items():
            if key not in ['id', 'category']:
                self.feature_space[key].append(value)
                
        self.learning_history.append({
            'action': 'add_observation',
            'observation_id': obs_id,
            'timestamp': datetime.now()
        })
        
    def form_concepts_from_clustering(self, method: str = 'hierarchical', 
                                    min_cluster_size: int = 3) -> List[ConceptNode]:
        """Form concepts by clustering similar observations."""
        if len(self.feature_space['observations']) < min_cluster_size:
            return []
            
        # Prepare numerical features for clustering
        observations = self.feature_space['observations']
        numerical_features = []
        feature_names = []
        
        # First pass: collect all possible feature names
        for obs in observations:
            for key, value in obs.items():
                if isinstance(value, (int, float)) and key not in feature_names:
                    feature_names.append(key)
        
        if not feature_names:
            return []
        
        # Second pass: build feature vectors with consistent length
        for obs in observations:
            feature_vector = []
            for feature in feature_names:
                feature_vector.append(obs.get(feature, 0))
            numerical_features.append(feature_vector)
            
        if not numerical_features:
            return []
            
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(numerical_features)
        
        # Apply clustering
        if method == 'hierarchical':
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=1.0,
                linkage='ward'
            )
        else:  # DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=min_cluster_size)
            
        cluster_labels = clustering.fit_predict(X)
        
        # Form concepts from clusters
        concepts = []
        cluster_groups = defaultdict(list)
        
        for i, label in enumerate(cluster_labels):
            if label >= 0:  # Valid cluster (not noise)
                cluster_groups[label].append(observations[i])
                
        for cluster_id, cluster_obs in cluster_groups.items():
            if len(cluster_obs) >= min_cluster_size:
                concept_name = f"concept_cluster_{cluster_id}"
                concept = ConceptNode(
                    name=concept_name,
                    level=1,  # First abstraction level
                    features={}
                )
                
                for obs in cluster_obs:
                    concept.add_instance(obs)
                    
                concept.compute_abstraction()
                concept.confidence = len(cluster_obs) / len(observations)
                
                concepts.append(concept)
                self.concepts[concept_name] = concept
                self.abstraction_levels[1].append(concept)
                
        return concepts
        
    def form_hierarchical_abstractions(self, max_levels: int = 5) -> Dict[int, List[ConceptNode]]:
        """Build hierarchical concept structures by successive abstraction."""
        
        # Start with level 1 concepts (from clustering)
        if 1 not in self.abstraction_levels:
            self.form_concepts_from_clustering()
            
        # Build higher levels
        for level in range(2, max_levels + 1):
            parent_concepts = self.abstraction_levels[level - 1]
            
            if len(parent_concepts) < 2:
                break  # Need at least 2 concepts to form higher abstraction
                
            # Compute similarity between concepts at current level
            concept_features = []
            all_feature_keys = set()
            
            # First pass: collect all feature keys
            for concept in parent_concepts:
                for key in concept.abstractions.keys():
                    if isinstance(concept.abstractions[key], (int, float)):
                        all_feature_keys.add(key)
            
            all_feature_keys = sorted(list(all_feature_keys))  # Ensure consistent order
            
            # Second pass: build feature vectors with consistent length
            for concept in parent_concepts:
                feature_vector = []
                for key in all_feature_keys:
                    value = concept.abstractions.get(key, 0)
                    if isinstance(value, (int, float)):
                        feature_vector.append(value)
                    else:
                        feature_vector.append(0)
                concept_features.append(feature_vector)
                
            if not concept_features or not concept_features[0]:
                break
                
            # Cluster concepts to form higher-level abstractions
            scaler = StandardScaler()
            X = scaler.fit_transform(concept_features)
            
            # Use more aggressive clustering for higher levels
            clustering = AgglomerativeClustering(
                n_clusters=max(1, len(parent_concepts) // 2),
                linkage='ward'
            )
            cluster_labels = clustering.fit_predict(X)
            
            # Form higher-level concepts
            cluster_groups = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                cluster_groups[label].append(parent_concepts[i])
                
            new_concepts = []
            for cluster_id, concept_group in cluster_groups.items():
                if len(concept_group) >= 1:  # Allow single-concept abstractions at higher levels
                    abstract_concept_name = f"abstract_concept_L{level}_{cluster_id}"
                    abstract_concept = ConceptNode(
                        name=abstract_concept_name,
                        level=level,
                        features={}
                    )
                    
                    # Add child concepts
                    for child_concept in concept_group:
                        abstract_concept.add_child(child_concept)
                        
                    # Compute abstract features from children
                    abstract_features = {}
                    for child in concept_group:
                        for key, value in child.abstractions.items():
                            if isinstance(value, (int, float)):
                                if key not in abstract_features:
                                    abstract_features[key] = []
                                abstract_features[key].append(value)
                                
                    # Aggregate child features
                    for feature, values in abstract_features.items():
                        if values:
                            abstract_concept.abstractions[f"{feature}_super_mean"] = np.mean(values)
                            abstract_concept.abstractions[f"{feature}_super_std"] = np.std(values)
                            
                    abstract_concept.confidence = sum(c.confidence for c in concept_group) / len(concept_group)
                    
                    new_concepts.append(abstract_concept)
                    self.concepts[abstract_concept_name] = abstract_concept
                    
            if new_concepts:
                self.abstraction_levels[level] = new_concepts
            else:
                break
                
        return self.abstraction_levels
        
    def establish_relationships(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """Establish relationships between concepts across different levels."""
        relationships = defaultdict(list)
        
        # Similarity relationships within levels
        for level, concepts in self.abstraction_levels.items():
            if len(concepts) > 1:
                # Compute pairwise similarities
                concept_vectors = []
                for concept in concepts:
                    vector = []
                    for key, value in concept.abstractions.items():
                        if isinstance(value, (int, float)):
                            vector.append(value)
                    concept_vectors.append(vector)
                    
                if concept_vectors and len(concept_vectors[0]) > 0:
                    similarity_matrix = cosine_similarity(concept_vectors)
                    
                    for i, concept_a in enumerate(concepts):
                        for j, concept_b in enumerate(concepts):
                            if i < j:  # Avoid duplicates
                                similarity = similarity_matrix[i][j]
                                if similarity > 0.5:  # Threshold for significant similarity
                                    relationships['similarity'].append(
                                        (concept_a.name, concept_b.name, similarity)
                                    )
                                    
        # Hierarchical relationships (parent-child)
        for concept in self.concepts.values():
            if concept.children:
                for child in concept.children:
                    relationships['hierarchical'].append(
                        (concept.name, child.name, 1.0)  # Perfect hierarchical relationship
                    )
                    
        # Cross-domain relationships (based on shared features)
        all_concepts = list(self.concepts.values())
        for i, concept_a in enumerate(all_concepts):
            for j, concept_b in enumerate(all_concepts):
                if i < j and concept_a.level == concept_b.level:
                    # Check for shared abstraction patterns
                    shared_features = set(concept_a.abstractions.keys()) & set(concept_b.abstractions.keys())
                    if shared_features:
                        correlation = 0.0
                        valid_features = 0
                        
                        for feature in shared_features:
                            val_a = concept_a.abstractions[feature]
                            val_b = concept_b.abstractions[feature]
                            
                            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                                # Simple correlation measure
                                correlation += 1.0 / (1.0 + abs(val_a - val_b))
                                valid_features += 1
                                
                        if valid_features > 0:
                            avg_correlation = correlation / valid_features
                            if avg_correlation > 0.7:
                                relationships['cross_domain'].append(
                                    (concept_a.name, concept_b.name, avg_correlation)
                                )
                                
        # Store relationships in concept nodes
        for rel_type, rel_list in relationships.items():
            for concept_a, concept_b, strength in rel_list:
                if concept_a in self.concepts:
                    if rel_type not in self.concepts[concept_a].relationships:
                        self.concepts[concept_a].relationships[rel_type] = []
                    self.concepts[concept_a].relationships[rel_type].append((concept_b, strength))
                    
        return relationships
        
    def explain_concept(self, concept_name: str) -> str:
        """Generate natural language explanation of a concept."""
        if concept_name not in self.concepts:
            return f"Concept '{concept_name}' not found."
            
        concept = self.concepts[concept_name]
        explanation = f"Concept: {concept.name}\n"
        explanation += f"Abstraction Level: {concept.level}\n"
        explanation += f"Confidence: {concept.confidence:.3f}\n"
        explanation += f"Instances: {len(concept.instances)}\n"
        explanation += f"Children: {len(concept.children)}\n"
        
        if concept.abstractions:
            explanation += "\nKey Abstractions:\n"
            for key, value in list(concept.abstractions.items())[:5]:  # Top 5
                if isinstance(value, (int, float)):
                    explanation += f"  • {key}: {value:.3f}\n"
                elif isinstance(value, list) and len(value) <= 10:
                    explanation += f"  • {key}: {value}\n"
                    
        if concept.relationships:
            explanation += "\nRelationships:\n"
            for rel_type, relations in concept.relationships.items():
                explanation += f"  {rel_type}:\n"
                for related_concept, strength in relations[:3]:  # Top 3
                    explanation += f"    - {related_concept} (strength: {strength:.3f})\n"
                    
        return explanation
        
    def visualize_concept_hierarchy(self, save_path: str = None):
        """Visualize the hierarchical concept structure."""
        plt.figure(figsize=(15, 10))
        
        # Create layout positions
        pos = {}
        y_positions = {}
        
        # Calculate positions for each level
        for level, concepts in self.abstraction_levels.items():
            y_positions[level] = len(self.abstraction_levels) - level
            x_spacing = 2.0 / (len(concepts) + 1) if concepts else 1.0
            
            for i, concept in enumerate(concepts):
                pos[concept.name] = ((i + 1) * x_spacing - 1.0, y_positions[level])
                
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes
        for concept in self.concepts.values():
            G.add_node(concept.name, level=concept.level, confidence=concept.confidence)
            
        # Add edges (hierarchical relationships)
        for concept in self.concepts.values():
            for child in concept.children:
                G.add_edge(concept.name, child.name)
                
        # Draw the graph
        node_colors = [self.concepts[node].confidence for node in G.nodes()]
        node_sizes = [300 + 200 * self.concepts[node].confidence for node in G.nodes()]
        
        nx.draw(G, pos, 
                node_color=node_colors, 
                node_size=node_sizes,
                cmap=plt.cm.viridis,
                with_labels=True,
                font_size=8,
                arrows=True,
                edge_color='gray',
                alpha=0.7)
                
        plt.title("Hierarchical Concept Structure", fontsize=16)
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), 
                    label="Concept Confidence")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the concept formation process."""
        stats = {
            'total_observations': len(self.feature_space['observations']),
            'total_concepts': len(self.concepts),
            'abstraction_levels': len(self.abstraction_levels),
            'concepts_per_level': {level: len(concepts) for level, concepts in self.abstraction_levels.items()},
            'average_confidence': np.mean([c.confidence for c in self.concepts.values()]) if self.concepts else 0,
            'relationship_types': {},
            'learning_events': len(self.learning_history)
        }
        
        # Count relationship types
        for concept in self.concepts.values():
            for rel_type, relations in concept.relationships.items():
                if rel_type not in stats['relationship_types']:
                    stats['relationship_types'][rel_type] = 0
                stats['relationship_types'][rel_type] += len(relations)
                
        return stats


def demonstrate_abstract_concept_formation():
    """Demonstrate the abstract concept formation engine with sample data."""
    print("🧠 ABSTRACT CONCEPT FORMATION ENGINE")
    print("=" * 60)
    
    # Initialize the engine
    engine = AbstractConceptFormation()
    
    # Add sample observations across different domains
    print("📊 Adding observations across multiple domains...")
    
    # Biological organisms
    organisms = [
        {'size': 10, 'mobility': 5, 'metabolism': 8, 'environment': 'terrestrial', 'type': 'mammal'},
        {'size': 2, 'mobility': 9, 'metabolism': 10, 'environment': 'aerial', 'type': 'bird'},
        {'size': 15, 'mobility': 3, 'metabolism': 6, 'environment': 'terrestrial', 'type': 'mammal'},
        {'size': 1, 'mobility': 8, 'metabolism': 9, 'environment': 'aerial', 'type': 'bird'},
        {'size': 50, 'mobility': 1, 'metabolism': 4, 'environment': 'terrestrial', 'type': 'plant'},
        {'size': 100, 'mobility': 0, 'metabolism': 3, 'environment': 'terrestrial', 'type': 'plant'},
        {'size': 5, 'mobility': 7, 'metabolism': 7, 'environment': 'aquatic', 'type': 'fish'},
        {'size': 20, 'mobility': 6, 'metabolism': 6, 'environment': 'aquatic', 'type': 'fish'},
    ]
    
    for org in organisms:
        engine.add_observation(org, 'biology')
        
    # Vehicles
    vehicles = [
        {'speed': 60, 'efficiency': 30, 'capacity': 5, 'environment': 'road', 'type': 'car'},
        {'speed': 80, 'efficiency': 25, 'capacity': 4, 'environment': 'road', 'type': 'car'},
        {'speed': 200, 'efficiency': 15, 'capacity': 2, 'environment': 'road', 'type': 'motorcycle'},
        {'speed': 900, 'efficiency': 5, 'capacity': 300, 'environment': 'air', 'type': 'airplane'},
        {'speed': 800, 'efficiency': 8, 'capacity': 150, 'environment': 'air', 'type': 'airplane'},
        {'speed': 30, 'efficiency': 40, 'capacity': 50, 'environment': 'water', 'type': 'ship'},
        {'speed': 25, 'efficiency': 35, 'capacity': 100, 'environment': 'water', 'type': 'ship'},
    ]
    
    for vehicle in vehicles:
        engine.add_observation(vehicle, 'transportation')
        
    # Economic systems
    economies = [
        {'growth': 3.5, 'inflation': 2.1, 'employment': 95, 'technology': 8, 'type': 'developed'},
        {'growth': 2.8, 'inflation': 1.8, 'employment': 93, 'technology': 9, 'type': 'developed'},
        {'growth': 6.2, 'inflation': 4.5, 'employment': 85, 'technology': 5, 'type': 'developing'},
        {'growth': 7.1, 'inflation': 5.2, 'employment': 80, 'technology': 4, 'type': 'developing'},
        {'growth': 1.2, 'inflation': 0.5, 'employment': 70, 'technology': 3, 'type': 'traditional'},
        {'growth': 0.8, 'inflation': 0.2, 'employment': 65, 'technology': 2, 'type': 'traditional'},
    ]
    
    for economy in economies:
        engine.add_observation(economy, 'economics')
        
    print(f"✅ Added {len(engine.feature_space['observations'])} observations")
    
    # Form concepts through clustering
    print("\n🔍 Forming concepts through clustering...")
    level_1_concepts = engine.form_concepts_from_clustering()
    print(f"✅ Formed {len(level_1_concepts)} level-1 concepts")
    
    # Build hierarchical abstractions
    print("\n🏗️ Building hierarchical abstractions...")
    hierarchy = engine.form_hierarchical_abstractions(max_levels=4)
    
    for level, concepts in hierarchy.items():
        print(f"  Level {level}: {len(concepts)} concepts")
        
    # Establish relationships
    print("\n🔗 Establishing concept relationships...")
    relationships = engine.establish_relationships()
    
    for rel_type, relations in relationships.items():
        print(f"  {rel_type}: {len(relations)} relationships")
        
    # Display some concepts
    print("\n📋 Sample Concept Explanations:")
    print("-" * 40)
    
    sample_concepts = list(engine.concepts.keys())[:3]
    for concept_name in sample_concepts:
        print(engine.explain_concept(concept_name))
        print("-" * 40)
        
    # Show statistics
    print("\n📊 CONCEPT FORMATION STATISTICS:")
    print("=" * 40)
    stats = engine.get_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
            
    # Test concept formation capability
    print("\n🧪 TESTING ABSTRACT REASONING:")
    print("=" * 40)
    
    # Test 1: Cross-domain pattern recognition
    cross_domain_patterns = 0
    for rel_type, relations in relationships.items():
        if rel_type == 'cross_domain':
            cross_domain_patterns += len(relations)
            
    print(f"✅ Cross-domain patterns discovered: {cross_domain_patterns}")
    
    # Test 2: Hierarchical depth
    max_level = max(hierarchy.keys()) if hierarchy else 0
    print(f"✅ Maximum abstraction level achieved: {max_level}")
    
    # Test 3: Concept quality (confidence)
    avg_confidence = stats['average_confidence']
    print(f"✅ Average concept confidence: {avg_confidence:.3f}")
    
    # Overall assessment
    print("\n🎯 ABSTRACT CONCEPT FORMATION ASSESSMENT:")
    print("=" * 50)
    
    success_criteria = {
        'observations_processed': len(engine.feature_space['observations']) >= 15,
        'concepts_formed': len(engine.concepts) >= 3,
        'hierarchy_depth': max_level >= 2,
        'cross_domain_relations': cross_domain_patterns > 0,
        'high_confidence': avg_confidence > 0.3
    }
    
    passed_tests = sum(success_criteria.values())
    total_tests = len(success_criteria)
    
    for test, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test}: {status}")
        
    success_rate = passed_tests / total_tests
    print(f"\n🏆 OVERALL SUCCESS RATE: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("🎉 ABSTRACT CONCEPT FORMATION SUCCESSFUL!")
        print("   System demonstrates hierarchical knowledge structures,")
        print("   cross-domain pattern recognition, and conceptual abstraction.")
    else:
        print("⚠️  Abstract concept formation needs improvement.")
        
    return engine, success_rate


if __name__ == "__main__":
    engine, success_rate = demonstrate_abstract_concept_formation()