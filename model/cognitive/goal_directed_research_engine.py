#!/usr/bin/env python3
"""
Goal-Directed Research Engine
============================
Creates autonomous scientific curiosity system that generates research questions,
designs experiments, and pursues knowledge discovery independently.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict, Counter, deque
import networkx as nx
import json
import random
import itertools
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class ResearchHypothesis:
    """Represents a scientific hypothesis with evidence and testing results."""
    
    def __init__(self, hypothesis_id: str, description: str, variables: List[str],
                 predicted_relationship: str, confidence: float = 0.5):
        self.hypothesis_id = hypothesis_id
        self.description = description
        self.variables = variables
        self.predicted_relationship = predicted_relationship  # e.g., "positive", "negative", "complex"
        self.confidence = confidence
        self.evidence = []
        self.test_results = []
        self.status = "untested"  # untested, supported, refuted, inconclusive
        self.creation_time = datetime.now()
        self.last_tested = None
        self.importance_score = 0.0
        
    def add_evidence(self, evidence_type: str, data: Any, strength: float):
        """Add supporting or contradicting evidence."""
        evidence_item = {
            'type': evidence_type,
            'data': data,
            'strength': strength,
            'timestamp': datetime.now()
        }
        self.evidence.append(evidence_item)
        
    def test_hypothesis(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test the hypothesis against new data."""
        test_result = {
            'test_id': f"test_{len(self.test_results)}",
            'data': test_data,
            'timestamp': datetime.now(),
            'outcome': None,
            'p_value': None,
            'effect_size': None,
            'support_level': 0.0
        }
        
        # Perform statistical tests based on hypothesis type
        if len(self.variables) >= 2:
            var1_data = test_data.get(self.variables[0], [])
            var2_data = test_data.get(self.variables[1], [])
            
            if len(var1_data) > 3 and len(var2_data) > 3:
                # Correlation test
                try:
                    correlation, p_value = stats.pearsonr(var1_data, var2_data)
                    test_result['correlation'] = correlation
                    test_result['p_value'] = p_value
                    test_result['effect_size'] = abs(correlation)
                    
                    # Determine if hypothesis is supported
                    if self.predicted_relationship == "positive" and correlation > 0.3 and p_value < 0.05:
                        test_result['support_level'] = min(1.0, abs(correlation) * 2)
                        test_result['outcome'] = "supported"
                    elif self.predicted_relationship == "negative" and correlation < -0.3 and p_value < 0.05:
                        test_result['support_level'] = min(1.0, abs(correlation) * 2)
                        test_result['outcome'] = "supported"
                    elif p_value < 0.05:
                        test_result['support_level'] = 0.5
                        test_result['outcome'] = "partially_supported"
                    else:
                        test_result['support_level'] = 0.0
                        test_result['outcome'] = "not_supported"
                        
                except Exception as e:
                    test_result['outcome'] = "test_failed"
                    test_result['error'] = str(e)
                    
        self.test_results.append(test_result)
        self.last_tested = datetime.now()
        
        # Update overall status
        if self.test_results:
            recent_support = np.mean([r.get('support_level', 0) for r in self.test_results[-3:]])
            if recent_support > 0.7:
                self.status = "supported"
            elif recent_support > 0.3:
                self.status = "inconclusive"
            else:
                self.status = "refuted"
                
        return test_result


class ExperimentDesign:
    """Represents an experimental design for testing hypotheses."""
    
    def __init__(self, experiment_id: str, objective: str, variables: List[str]):
        self.experiment_id = experiment_id
        self.objective = objective
        self.variables = variables
        self.design_type = "observational"  # observational, controlled, factorial
        self.sample_size = 0
        self.data_requirements = {}
        self.analysis_plan = []
        self.expected_outcomes = []
        self.feasibility_score = 0.0
        self.creation_time = datetime.now()
        
    def design_experiment(self, available_data: Dict[str, Any]) -> Dict[str, Any]:
        """Design an experiment based on available data and objectives."""
        design = {
            'experiment_id': self.experiment_id,
            'type': 'observational',  # Default to observational for available data
            'variables_to_collect': self.variables,
            'sample_size_recommendation': max(30, len(self.variables) * 10),
            'analysis_methods': [],
            'controls_needed': [],
            'potential_confounds': []
        }
        
        # Determine analysis methods
        if len(self.variables) == 2:
            design['analysis_methods'] = ['correlation', 'regression', 'visualization']
        elif len(self.variables) > 2:
            design['analysis_methods'] = ['multiple_regression', 'pca', 'clustering']
        else:
            design['analysis_methods'] = ['descriptive_statistics', 'distribution_analysis']
            
        # Identify potential confounds
        for var in self.variables:
            if var in available_data:
                # Simple heuristic for identifying related variables
                related_vars = [k for k in available_data.keys() 
                              if k != var and any(word in k.lower() for word in var.lower().split('_'))]
                design['potential_confounds'].extend(related_vars)
                
        # Protect against division by zero in feasibility calculation
        self.feasibility_score = min(1.0, len([v for v in self.variables if v in available_data]) / len(self.variables)) if len(self.variables) > 0 else 0.0
        
        return design


class KnowledgeGraph:
    """Represents accumulated scientific knowledge as a graph structure."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.concepts = {}
        self.relationships = {}
        self.discoveries = []
        self.evidence_strength = defaultdict(float)
        
    def add_concept(self, concept_id: str, properties: Dict[str, Any]):
        """Add a new concept to the knowledge graph."""
        self.concepts[concept_id] = properties
        self.graph.add_node(concept_id, **properties)
        
    def add_relationship(self, concept1: str, concept2: str, relationship_type: str, 
                        strength: float, evidence: Any = None):
        """Add or update a relationship between concepts."""
        rel_id = f"{concept1}->{concept2}:{relationship_type}"
        
        self.relationships[rel_id] = {
            'source': concept1,
            'target': concept2,
            'type': relationship_type,
            'strength': strength,
            'evidence': evidence,
            'timestamp': datetime.now()
        }
        
        self.graph.add_edge(concept1, concept2, 
                           relationship=relationship_type, 
                           strength=strength,
                           evidence=evidence)
        self.evidence_strength[rel_id] = strength
        
    def find_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Identify gaps in current knowledge that could be research targets."""
        gaps = []
        
        # Find disconnected concepts
        if len(self.graph.nodes()) > 1:
            connected_components = list(nx.weakly_connected_components(self.graph))
            if len(connected_components) > 1:
                for i, component1 in enumerate(connected_components):
                    for j, component2 in enumerate(connected_components):
                        if i < j:
                            gaps.append({
                                'type': 'disconnected_components',
                                'components': [list(component1), list(component2)],
                                'potential_bridges': self._suggest_bridges(component1, component2)
                            })
                            
        # Find concepts with few connections
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree < 2:  # Isolated or barely connected
                gaps.append({
                    'type': 'isolated_concept',
                    'concept': node,
                    'current_connections': degree,
                    'suggested_explorations': self._suggest_connections(node)
                })
                
        # Find weak relationships that need more evidence
        weak_relationships = []
        for rel_id, rel_data in self.relationships.items():
            if rel_data['strength'] < 0.5:
                weak_relationships.append(rel_id)
                
        if weak_relationships:
            gaps.append({
                'type': 'weak_evidence',
                'relationships': weak_relationships,
                'research_priority': 'high'
            })
            
        return gaps
        
    def _suggest_bridges(self, component1: Set[str], component2: Set[str]) -> List[str]:
        """Suggest potential bridges between disconnected components."""
        bridges = []
        
        # Simple heuristic: look for similar concept names
        for c1 in component1:
            for c2 in component2:
                if any(word in c2.lower() for word in c1.lower().split('_')):
                    bridges.append(f"{c1} <-> {c2}")
                    
        return bridges[:3]  # Return top 3 suggestions
        
    def _suggest_connections(self, concept: str) -> List[str]:
        """Suggest potential connections for an isolated concept."""
        suggestions = []
        
        # Look for concepts with similar properties
        if concept in self.concepts:
            concept_props = self.concepts[concept]
            
            for other_concept, other_props in self.concepts.items():
                if other_concept != concept:
                    # Simple similarity based on shared property keys
                    shared_props = set(concept_props.keys()) & set(other_props.keys())
                    if len(shared_props) > 0:
                        suggestions.append(other_concept)
                        
        return suggestions[:3]


class AutonomousResearcher:
    """Main engine for autonomous scientific research and discovery."""
    
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.hypotheses = {}
        self.experiments = {}
        self.research_questions = deque()
        self.research_priorities = defaultdict(float)
        self.discovery_history = []
        self.curiosity_drivers = {
            'novelty_seeking': 0.7,
            'gap_filling': 0.8,
            'pattern_discovery': 0.9,
            'anomaly_investigation': 0.6
        }
        self.research_state = {
            'active_investigations': [],
            'completed_studies': [],
            'pending_verifications': []
        }
        
    def observe_data(self, data: Dict[str, Any], domain: str = "general"):
        """Process new observational data and extract insights."""
        observation_id = f"obs_{len(self.discovery_history)}"
        
        # Extract variables and patterns
        variables = list(data.keys())
        patterns = self._detect_patterns(data)
        anomalies = self._detect_anomalies(data)
        
        # Add concepts to knowledge graph
        for var in variables:
            if var not in self.knowledge_graph.concepts:
                self.knowledge_graph.add_concept(var, {
                    'domain': domain,
                    'data_type': type(data[var]).__name__,
                    'observations': 1
                })
            else:
                self.knowledge_graph.concepts[var]['observations'] += 1
                
        # Add patterns as relationships
        for pattern in patterns:
            self.knowledge_graph.add_relationship(
                pattern['var1'], pattern['var2'],
                pattern['type'], pattern['strength'],
                evidence=pattern['evidence']
            )
            
        # Generate research questions from observations
        new_questions = self._generate_research_questions(data, patterns, anomalies)
        self.research_questions.extend(new_questions)
        
        # Record discovery
        discovery = {
            'id': observation_id,
            'type': 'observation',
            'data': data,
            'domain': domain,
            'patterns_found': len(patterns),
            'anomalies_found': len(anomalies),
            'questions_generated': len(new_questions),
            'timestamp': datetime.now()
        }
        
        self.discovery_history.append(discovery)
        
        return {
            'observation_id': observation_id,
            'patterns': patterns,
            'anomalies': anomalies,
            'research_questions': new_questions
        }
        
    def _detect_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect patterns in the observed data."""
        patterns = []
        
        # Get numerical variables
        numerical_vars = {k: v for k, v in data.items() 
                         if isinstance(v, (list, np.ndarray)) and 
                         len(v) > 0 and isinstance(v[0], (int, float))}
        
        # Pairwise correlation analysis
        var_names = list(numerical_vars.keys())
        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                if i < j:
                    try:
                        corr, p_value = stats.pearsonr(numerical_vars[var1], numerical_vars[var2])
                        
                        if abs(corr) > 0.3 and p_value < 0.1:  # Loose threshold for exploration
                            pattern = {
                                'var1': var1,
                                'var2': var2,
                                'type': 'positive_correlation' if corr > 0 else 'negative_correlation',
                                'strength': abs(corr),
                                'evidence': {
                                    'correlation': corr,
                                    'p_value': p_value,
                                    'sample_size': len(numerical_vars[var1])
                                }
                            }
                            patterns.append(pattern)
                    except:
                        continue  # Skip if correlation can't be computed
                        
        return patterns
        
    def _detect_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies that might warrant investigation."""
        anomalies = []
        
        for var_name, var_data in data.items():
            if isinstance(var_data, (list, np.ndarray)) and len(var_data) > 5:
                if all(isinstance(x, (int, float)) for x in var_data):
                    # Statistical outlier detection
                    z_scores = np.abs(stats.zscore(var_data))
                    outlier_indices = np.where(z_scores > 2.5)[0]
                    
                    if len(outlier_indices) > 0:
                        anomaly = {
                            'variable': var_name,
                            'type': 'statistical_outlier',
                            'indices': outlier_indices.tolist(),
                            'values': [var_data[i] for i in outlier_indices],
                            'severity': np.max(z_scores)
                        }
                        anomalies.append(anomaly)
                        
        return anomalies
        
    def _generate_research_questions(self, data: Dict[str, Any], patterns: List[Dict[str, Any]], 
                                   anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate research questions based on observations."""
        questions = []
        
        # Questions from patterns
        for pattern in patterns:
            if pattern['strength'] > 0.5:
                questions.append(
                    f"What mechanism explains the {pattern['type']} between "
                    f"{pattern['var1']} and {pattern['var2']}?"
                )
            elif pattern['strength'] > 0.3:
                questions.append(
                    f"Is the relationship between {pattern['var1']} and {pattern['var2']} "
                    f"causal or spurious?"
                )
                
        # Questions from anomalies
        for anomaly in anomalies:
            questions.append(
                f"What causes the extreme values observed in {anomaly['variable']}?"
            )
            
        # Knowledge gap questions
        gaps = self.knowledge_graph.find_knowledge_gaps()
        for gap in gaps:
            if gap['type'] == 'isolated_concept':
                questions.append(
                    f"How does {gap['concept']} relate to other system components?"
                )
            elif gap['type'] == 'disconnected_components':
                questions.append(
                    "What connections exist between the observed component groups?"
                )
                
        return questions
        
    def generate_hypothesis(self, research_question: str, context_data: Dict[str, Any] = None) -> ResearchHypothesis:
        """Generate a testable hypothesis from a research question."""
        hypothesis_id = f"hyp_{len(self.hypotheses)}"
        
        # Extract variables from the research question (simple heuristic)
        variables = []
        if context_data:
            for var in context_data.keys():
                if var.lower() in research_question.lower():
                    variables.append(var)
                    
        # Determine predicted relationship based on question content
        predicted_relationship = "positive"
        if any(word in research_question.lower() for word in ['negative', 'inverse', 'opposite']):
            predicted_relationship = "negative"
        elif any(word in research_question.lower() for word in ['complex', 'nonlinear', 'interaction']):
            predicted_relationship = "complex"
            
        # Generate hypothesis description
        if len(variables) >= 2:
            description = f"There is a {predicted_relationship} relationship between {variables[0]} and {variables[1]}"
        else:
            description = f"Investigation of: {research_question}"
            
        hypothesis = ResearchHypothesis(
            hypothesis_id, description, variables, 
            predicted_relationship, confidence=0.6
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        return hypothesis
        
    def design_experiment(self, hypothesis: ResearchHypothesis, available_data: Dict[str, Any]) -> ExperimentDesign:
        """Design an experiment to test a hypothesis."""
        experiment_id = f"exp_{len(self.experiments)}"
        
        experiment = ExperimentDesign(
            experiment_id, 
            f"Test hypothesis: {hypothesis.description}",
            hypothesis.variables
        )
        
        design = experiment.design_experiment(available_data)
        
        self.experiments[experiment_id] = experiment
        
        return experiment
        
    def conduct_research_cycle(self, data: Dict[str, Any], domain: str = "general") -> Dict[str, Any]:
        """Conduct a complete research cycle: observe, hypothesize, experiment, conclude."""
        
        cycle_results = {
            'cycle_id': f"cycle_{len(self.research_state['completed_studies'])}",
            'domain': domain,
            'observations': None,
            'hypotheses_generated': [],
            'experiments_designed': [],
            'conclusions': [],
            'new_questions': []
        }
        
        # Step 1: Observe and extract patterns
        observations = self.observe_data(data, domain)
        cycle_results['observations'] = observations
        
        # Step 2: Generate hypotheses from research questions
        questions_to_test = list(self.research_questions)[:3]  # Test top 3 questions
        
        for question in questions_to_test:
            hypothesis = self.generate_hypothesis(question, data)
            cycle_results['hypotheses_generated'].append(hypothesis.hypothesis_id)
            
            # Step 3: Design experiment
            experiment = self.design_experiment(hypothesis, data)
            cycle_results['experiments_designed'].append(experiment.experiment_id)
            
            # Step 4: Test hypothesis (if feasible)
            if experiment.feasibility_score > 0.5:
                test_result = hypothesis.test_hypothesis(data)
                
                # Step 5: Draw conclusions
                if test_result['outcome'] == 'supported':
                    conclusion = f"Evidence supports: {hypothesis.description}"
                    
                    # Add to knowledge graph
                    if len(hypothesis.variables) >= 2:
                        self.knowledge_graph.add_relationship(
                            hypothesis.variables[0], hypothesis.variables[1],
                            hypothesis.predicted_relationship,
                            test_result.get('support_level', 0.5),
                            evidence=test_result
                        )
                        
                elif test_result['outcome'] == 'not_supported':
                    conclusion = f"Evidence does not support: {hypothesis.description}"
                else:
                    conclusion = f"Inconclusive evidence for: {hypothesis.description}"
                    
                cycle_results['conclusions'].append(conclusion)
                
            # Remove tested question
            if question in self.research_questions:
                self.research_questions.remove(question)
                
        # Step 6: Generate new research questions
        new_questions = self._prioritize_research_directions()
        cycle_results['new_questions'] = new_questions[:5]  # Top 5 priorities
        
        # Update research state
        self.research_state['completed_studies'].append(cycle_results)
        
        return cycle_results
        
    def _prioritize_research_directions(self) -> List[str]:
        """Prioritize future research directions based on current knowledge."""
        
        # Find knowledge gaps
        gaps = self.knowledge_graph.find_knowledge_gaps()
        
        # Generate questions targeting gaps
        priority_questions = []
        
        for gap in gaps:
            if gap['type'] == 'weak_evidence':
                for rel_id in gap['relationships'][:2]:  # Top 2 weak relationships
                    rel_data = self.knowledge_graph.relationships[rel_id]
                    priority_questions.append(
                        f"What additional evidence supports the relationship between "
                        f"{rel_data['source']} and {rel_data['target']}?"
                    )
                    
            elif gap['type'] == 'isolated_concept':
                priority_questions.append(
                    f"How does {gap['concept']} interact with the broader system?"
                )
                
        # Add curiosity-driven questions
        if len(self.knowledge_graph.concepts) > 3:
            concepts = list(self.knowledge_graph.concepts.keys())
            for i in range(min(3, len(concepts))):
                for j in range(i+1, min(i+3, len(concepts))):
                    if not self.knowledge_graph.graph.has_edge(concepts[i], concepts[j]):
                        priority_questions.append(
                            f"Is there a relationship between {concepts[i]} and {concepts[j]}?"
                        )
                        
        return priority_questions
        
    def get_research_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of research activities and discoveries."""
        
        summary = {
            'total_observations': len(self.discovery_history),
            'hypotheses_tested': len([h for h in self.hypotheses.values() if h.test_results]),
            'supported_hypotheses': len([h for h in self.hypotheses.values() if h.status == 'supported']),
            'knowledge_graph_size': len(self.knowledge_graph.concepts),
            'relationships_discovered': len(self.knowledge_graph.relationships),
            'research_cycles_completed': len(self.research_state['completed_studies']),
            'active_questions': len(self.research_questions),
            'key_discoveries': [],
            'research_efficiency': 0.0
        }
        
        # Identify key discoveries
        supported_hypotheses = [h for h in self.hypotheses.values() if h.status == 'supported']
        for hypothesis in supported_hypotheses[:5]:  # Top 5
            summary['key_discoveries'].append({
                'hypothesis': hypothesis.description,
                'evidence_strength': np.mean([r.get('support_level', 0) for r in hypothesis.test_results]),
                'variables': hypothesis.variables
            })
            
        # Calculate research efficiency with zero-division protection
        if summary['hypotheses_tested'] > 0:
            summary['research_efficiency'] = summary['supported_hypotheses'] / summary['hypotheses_tested']
        else:
            summary['research_efficiency'] = 0.0
            
        return summary
        
    def visualize_knowledge_graph(self, save_path: str = None):
        """Visualize the current knowledge graph."""
        
        plt.figure(figsize=(12, 8))
        
        if len(self.knowledge_graph.graph.nodes()) > 0:
            # Layout
            pos = nx.spring_layout(self.knowledge_graph.graph, k=1, iterations=50)
            
            # Node sizes based on number of connections
            node_sizes = [300 + 200 * self.knowledge_graph.graph.degree(node) 
                         for node in self.knowledge_graph.graph.nodes()]
            
            # Edge colors based on relationship strength
            edge_colors = []
            for edge in self.knowledge_graph.graph.edges(data=True):
                strength = edge[2].get('strength', 0.5)
                edge_colors.append(strength)
                
            # Draw the graph
            nx.draw_networkx_nodes(self.knowledge_graph.graph, pos, 
                                 node_size=node_sizes, 
                                 node_color='lightblue',
                                 alpha=0.7)
            
            if edge_colors:
                nx.draw_networkx_edges(self.knowledge_graph.graph, pos,
                                     edge_color=edge_colors,
                                     edge_cmap=plt.cm.viridis,
                                     width=2, alpha=0.6)
            
            nx.draw_networkx_labels(self.knowledge_graph.graph, pos, font_size=10)
            
            plt.title("Scientific Knowledge Graph", fontsize=16)
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()


def demonstrate_goal_directed_research():
    """Demonstrate the goal-directed research engine."""
    print("🔬 GOAL-DIRECTED RESEARCH ENGINE")
    print("=" * 60)
    
    # Initialize the autonomous researcher
    researcher = AutonomousResearcher()
    
    # Simulate research across multiple domains
    print("📊 Conducting autonomous research across multiple domains...")
    
    # Domain 1: Biological ecosystem data
    ecosystem_data = {
        'temperature': np.random.normal(25, 5, 50),
        'humidity': np.random.normal(60, 15, 50),
        'biodiversity_index': None,
        'population_density': np.random.exponential(2, 50)
    }
    
    # Create realistic relationships
    ecosystem_data['biodiversity_index'] = (
        0.3 * ecosystem_data['temperature'] + 
        0.2 * ecosystem_data['humidity'] + 
        np.random.normal(0, 2, 50)
    )
    
    # Domain 2: Economic system data
    economic_data = {
        'gdp_growth': np.random.normal(3, 1.5, 40),
        'unemployment': np.random.normal(5, 2, 40),
        'inflation': np.random.normal(2, 1, 40),
        'investment_rate': None
    }
    
    # Create relationships
    economic_data['investment_rate'] = (
        2 * economic_data['gdp_growth'] - 
        0.5 * economic_data['unemployment'] + 
        np.random.normal(0, 1, 40)
    )
    
    # Domain 3: Physics experiment data
    physics_data = {
        'force': np.random.uniform(1, 10, 30),
        'mass': np.random.uniform(0.5, 5, 30),
        'acceleration': None,
        'friction': np.random.uniform(0.1, 0.5, 30)
    }
    
    # F = ma relationship (with friction)
    physics_data['acceleration'] = (
        physics_data['force'] / physics_data['mass'] - 
        physics_data['friction'] * 9.8
    )
    
    domains_data = [
        (ecosystem_data, "biology"),
        (economic_data, "economics"), 
        (physics_data, "physics")
    ]
    
    # Conduct research cycles
    all_cycle_results = []
    
    for i, (data, domain) in enumerate(domains_data):
        print(f"\n🔍 Research Cycle {i+1}: {domain.title()}")
        print("-" * 40)
        
        cycle_results = researcher.conduct_research_cycle(data, domain)
        all_cycle_results.append(cycle_results)
        
        print(f"  Patterns found: {len(cycle_results['observations']['patterns'])}")
        print(f"  Hypotheses generated: {len(cycle_results['hypotheses_generated'])}")
        print(f"  Experiments designed: {len(cycle_results['experiments_designed'])}")
        print(f"  Conclusions drawn: {len(cycle_results['conclusions'])}")
        
        # Show some conclusions
        for conclusion in cycle_results['conclusions'][:2]:
            print(f"    • {conclusion}")
            
    # Cross-domain research cycle
    print(f"\n🌐 Cross-Domain Analysis Cycle")
    print("-" * 40)
    
    # Combine data for cross-domain analysis
    combined_data = {}
    for data, domain in domains_data:
        for key, values in data.items():
            if values is not None:
                combined_data[f"{domain}_{key}"] = values
                
    cross_domain_results = researcher.conduct_research_cycle(combined_data, "cross_domain")
    all_cycle_results.append(cross_domain_results)
    
    print(f"  Cross-domain patterns: {len(cross_domain_results['observations']['patterns'])}")
    print(f"  New hypotheses: {len(cross_domain_results['hypotheses_generated'])}")
    
    # Research summary
    print(f"\n📋 RESEARCH SUMMARY:")
    print("=" * 40)
    
    summary = researcher.get_research_summary()
    
    for key, value in summary.items():
        if key == 'key_discoveries':
            print(f"{key}:")
            for i, discovery in enumerate(value, 1):
                print(f"  {i}. {discovery['hypothesis']}")
                print(f"     Evidence strength: {discovery['evidence_strength']:.3f}")
        elif not isinstance(value, (list, dict)):
            print(f"{key}: {value}")
            
    # Evaluate autonomous research capability
    print(f"\n🧪 EVALUATING RESEARCH CAPABILITIES:")
    print("=" * 50)
    
    # Test 1: Discovery rate
    discovery_rate = summary['supported_hypotheses'] / max(1, summary['hypotheses_tested'])
    
    # Test 2: Cross-domain integration
    cross_domain_concepts = len([c for c in researcher.knowledge_graph.concepts.keys() 
                                if '_' in c])  # Concepts with domain prefixes
    integration_score = cross_domain_concepts / max(1, summary['knowledge_graph_size'])
    
    # Test 3: Research question generation
    question_generation_rate = summary['active_questions'] / max(1, summary['total_observations'])
    
    # Test 4: Knowledge graph growth
    knowledge_growth = summary['relationships_discovered'] / max(1, summary['knowledge_graph_size'])
    
    # Test 5: Hypothesis quality (average evidence strength)
    avg_evidence_strength = 0
    if summary['key_discoveries']:
        avg_evidence_strength = np.mean([d['evidence_strength'] for d in summary['key_discoveries']])
        
    print(f"✅ Discovery success rate: {discovery_rate:.3f}")
    print(f"✅ Cross-domain integration: {integration_score:.3f}")
    print(f"✅ Question generation rate: {question_generation_rate:.3f}")
    print(f"✅ Knowledge graph growth: {knowledge_growth:.3f}")
    print(f"✅ Evidence strength: {avg_evidence_strength:.3f}")
    
    # Overall assessment
    print(f"\n🎯 GOAL-DIRECTED RESEARCH ASSESSMENT:")
    print("=" * 50)
    
    success_criteria = {
        'discovery_rate': discovery_rate > 0.3,
        'cross_domain_integration': integration_score > 0.2,
        'question_generation': question_generation_rate > 0.5,
        'knowledge_growth': knowledge_growth > 0.5,
        'evidence_quality': avg_evidence_strength > 0.4,
        'research_cycles': summary['research_cycles_completed'] >= 3
    }
    
    passed_tests = sum(success_criteria.values())
    total_tests = len(success_criteria)
    
    for test, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test}: {status}")
        
    # Protect against division by zero
    success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
    print(f"\n🏆 OVERALL SUCCESS RATE: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("🎉 GOAL-DIRECTED RESEARCH SUCCESSFUL!")
        print("   System demonstrates autonomous scientific curiosity,")
        print("   hypothesis generation, and knowledge discovery.")
    else:
        print("⚠️  Goal-directed research capabilities need improvement.")
        
    return researcher, success_rate


if __name__ == "__main__":
    researcher, success_rate = demonstrate_goal_directed_research()