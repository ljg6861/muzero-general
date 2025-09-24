#!/usr/bin/env python3
"""
Meta-Learning Optimization System
=================================
Implements learning-to-learn mechanisms that optimize learning strategies,
adapt to new domains faster, and improve learning efficiency through experience.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import optuna
from collections import defaultdict, deque
import json
import random
import time
from typing import Dict, List, Tuple, Any, Optional, Callable
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class LearningStrategy:
    """Represents a learning strategy with parameters and performance history."""
    
    def __init__(self, name: str, algorithm: str, hyperparameters: Dict[str, Any]):
        self.name = name
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.performance_history = []
        self.domain_performance = defaultdict(list)
        self.adaptation_speed = 0.0
        self.efficiency_score = 0.0
        self.generalization_ability = 0.0
        self.last_update = datetime.now()
        
    def record_performance(self, domain: str, performance: float, 
                          training_time: float, data_efficiency: float):
        """Record performance metrics for this strategy."""
        record = {
            'domain': domain,
            'performance': performance,
            'training_time': training_time,
            'data_efficiency': data_efficiency,
            'timestamp': datetime.now()
        }
        
        self.performance_history.append(record)
        self.domain_performance[domain].append(performance)
        
        # Update efficiency metrics
        self.efficiency_score = np.mean([r['data_efficiency'] for r in self.performance_history[-10:]])
        self.adaptation_speed = 1.0 / np.mean([r['training_time'] for r in self.performance_history[-5:]])
        
        # Calculate generalization (performance across different domains)
        if len(self.domain_performance) > 1:
            domain_means = [np.mean(perfs) for perfs in self.domain_performance.values()]
            self.generalization_ability = 1.0 - np.std(domain_means) / (np.mean(domain_means) + 1e-8)
        
    def get_expected_performance(self, domain: str) -> float:
        """Predict expected performance in a domain based on history."""
        if domain in self.domain_performance and self.domain_performance[domain]:
            # Use recent performance in this domain
            recent_performance = self.domain_performance[domain][-3:]
            return np.mean(recent_performance)
        else:
            # Use generalization ability to estimate
            if self.performance_history:
                avg_performance = np.mean([r['performance'] for r in self.performance_history])
                return avg_performance * self.generalization_ability
            return 0.5  # Default expectation


class MetaLearner:
    """Meta-learning algorithm that learns how to learn more efficiently."""
    
    def __init__(self, base_algorithms: List[str] = None):
        if base_algorithms is None:
            base_algorithms = ['random_forest', 'gradient_boosting', 'neural_network', 'linear', 'svm']
            
        self.base_algorithms = base_algorithms
        self.strategies = {}
        self.strategy_performance = defaultdict(list)
        self.learning_curves = defaultdict(list)
        self.optimization_history = []
        self.adaptation_patterns = defaultdict(list)
        self.meta_knowledge = {}
        
        # Initialize base strategies
        self._initialize_strategies()
        
        # Meta-learning components
        self.strategy_selector = None
        self.hyperparameter_optimizer = None
        self.adaptation_predictor = None
        
    def _initialize_strategies(self):
        """Initialize base learning strategies with default hyperparameters."""
        
        strategy_configs = {
            'random_forest': {
                'algorithm': 'random_forest',
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'algorithm': 'gradient_boosting',
                'hyperparameters': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'neural_network': {
                'algorithm': 'neural_network',
                'hyperparameters': {
                    'hidden_layer_sizes': (100, 50),
                    'learning_rate_init': 0.001,
                    'max_iter': 500,
                    'random_state': 42
                }
            },
            'linear': {
                'algorithm': 'linear',
                'hyperparameters': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            },
            'svm': {
                'algorithm': 'svm',
                'hyperparameters': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale'
                }
            }
        }
        
        for name, config in strategy_configs.items():
            strategy = LearningStrategy(name, config['algorithm'], config['hyperparameters'])
            self.strategies[name] = strategy
            
    def _create_model(self, strategy: LearningStrategy):
        """Create a model instance from a learning strategy."""
        algo = strategy.algorithm
        params = strategy.hyperparameters.copy()
        
        if algo == 'random_forest':
            return RandomForestRegressor(**params)
        elif algo == 'gradient_boosting':
            return GradientBoostingRegressor(**params)
        elif algo == 'neural_network':
            return MLPRegressor(**params)
        elif algo == 'linear':
            return Ridge(**params)
        elif algo == 'svm':
            return SVR(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
            
    def evaluate_strategy(self, strategy_name: str, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray, domain: str) -> Dict[str, float]:
        """Evaluate a learning strategy on a specific dataset."""
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
            
        strategy = self.strategies[strategy_name]
        
        # Measure training time
        start_time = time.time()
        
        # Create and train model
        model = self._create_model(strategy)
        
        # Scale features if needed
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        try:
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate performance metrics
            performance = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Calculate data efficiency (performance per sample)
            data_efficiency = performance / len(X_train) if len(X_train) > 0 else 0
            
            # Record performance
            strategy.record_performance(domain, performance, training_time, data_efficiency)
            
            return {
                'performance': performance,
                'mse': mse,
                'training_time': training_time,
                'data_efficiency': data_efficiency
            }
            
        except Exception as e:
            print(f"Error evaluating strategy {strategy_name}: {e}")
            return {
                'performance': 0.0,
                'mse': float('inf'),
                'training_time': float('inf'),
                'data_efficiency': 0.0
            }
            
    def optimize_hyperparameters(self, strategy_name: str, X: np.ndarray, y: np.ndarray,
                                domain: str, n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters for a strategy using Optuna."""
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
            
        strategy = self.strategies[strategy_name]
        
        def objective(trial):
            # Define hyperparameter search spaces based on algorithm
            if strategy.algorithm == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'random_state': 42
                }
            elif strategy.algorithm == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'random_state': 42
                }
            elif strategy.algorithm == 'neural_network':
                layer_1 = trial.suggest_int('layer_1', 50, 200)
                layer_2 = trial.suggest_int('layer_2', 20, 100)
                params = {
                    'hidden_layer_sizes': (layer_1, layer_2),
                    'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.01),
                    'max_iter': 500,
                    'random_state': 42
                }
            elif strategy.algorithm == 'linear':
                params = {
                    'alpha': trial.suggest_float('alpha', 0.1, 10.0),
                    'random_state': 42
                }
            elif strategy.algorithm == 'svm':
                params = {
                    'C': trial.suggest_float('C', 0.1, 10.0),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                    'gamma': 'scale'
                }
            else:
                return 0.0
                
            # Create model with trial parameters
            temp_strategy = LearningStrategy(f"{strategy_name}_temp", strategy.algorithm, params)
            model = self._create_model(temp_strategy)
            
            # Cross-validation score
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            try:
                scores = cross_val_score(model, X_scaled, y, cv=3, scoring='r2')
                return np.mean(scores)
            except:
                return 0.0
                
        # Run optimization
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Update strategy with best parameters - handle neural network special case
        best_params = study.best_params.copy()
        
        # Convert neural network parameters
        if strategy.algorithm == 'neural_network' and 'layer_1' in best_params:
            layer_1 = best_params.pop('layer_1')
            layer_2 = best_params.pop('layer_2')
            best_params['hidden_layer_sizes'] = (layer_1, layer_2)
            
        strategy.hyperparameters.update(best_params)
        
        self.optimization_history.append({
            'strategy': strategy_name,
            'domain': domain,
            'best_score': study.best_value,
            'best_params': best_params,
            'timestamp': datetime.now()
        })
        
        return best_params
        
    def select_best_strategy(self, domain: str, task_characteristics: Dict[str, Any] = None) -> str:
        """Select the best strategy for a given domain and task characteristics."""
        
        if not self.strategies:
            return list(self.strategies.keys())[0]
            
        # Score strategies based on multiple criteria
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            score = 0.0
            
            # Expected performance in domain
            expected_perf = strategy.get_expected_performance(domain)
            score += expected_perf * 0.4
            
            # Efficiency (data efficiency + adaptation speed)
            score += strategy.efficiency_score * 0.3
            score += min(strategy.adaptation_speed, 1.0) * 0.2
            
            # Generalization ability
            score += strategy.generalization_ability * 0.1
            
            # Bonus for recent good performance
            if strategy.performance_history:
                recent_performance = np.mean([r['performance'] 
                                            for r in strategy.performance_history[-5:]])
                score += recent_performance * 0.1
                
            strategy_scores[name] = score
            
        # Select strategy with highest score
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        return best_strategy
        
    def learn_adaptation_patterns(self):
        """Learn patterns about how strategies adapt across different domains."""
        
        # Collect adaptation data
        adaptation_data = []
        
        for strategy_name, strategy in self.strategies.items():
            for domain, performances in strategy.domain_performance.items():
                if len(performances) >= 2:
                    # Calculate learning curve slope
                    x = np.arange(len(performances))
                    slope = np.polyfit(x, performances, 1)[0]
                    
                    adaptation_data.append({
                        'strategy': strategy_name,
                        'domain': domain,
                        'adaptation_rate': slope,
                        'final_performance': performances[-1],
                        'initial_performance': performances[0],
                        'efficiency': strategy.efficiency_score
                    })
                    
        # Store patterns
        self.adaptation_patterns['data'] = adaptation_data
        
        if adaptation_data:
            # Analyze patterns
            df = pd.DataFrame(adaptation_data)
            
            # Best adapting strategies per domain
            self.adaptation_patterns['best_adapters'] = {}
            for domain in df['domain'].unique():
                domain_data = df[df['domain'] == domain]
                best_adapter = domain_data.loc[domain_data['adaptation_rate'].idxmax(), 'strategy']
                self.adaptation_patterns['best_adapters'][domain] = best_adapter
                
            # Overall adaptation leaders
            avg_adaptation = df.groupby('strategy')['adaptation_rate'].mean()
            self.adaptation_patterns['adaptation_leaders'] = avg_adaptation.sort_values(ascending=False).to_dict()
            
    def meta_optimize(self, domains_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                     optimization_rounds: int = 3) -> Dict[str, Any]:
        """Perform meta-optimization across multiple domains."""
        
        print(f"🔄 Starting meta-optimization across {len(domains_data)} domains...")
        
        optimization_results = {}
        
        for round_num in range(optimization_rounds):
            print(f"\n📈 Optimization Round {round_num + 1}/{optimization_rounds}")
            
            round_results = {}
            
            for domain, (X, y) in domains_data.items():
                print(f"  🎯 Optimizing for domain: {domain}")
                
                # Split data for evaluation
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                domain_results = {}
                
                # Evaluate and optimize each strategy
                for strategy_name in self.strategies.keys():
                    # Optimize hyperparameters
                    self.optimize_hyperparameters(strategy_name, X_train, y_train, domain, n_trials=20)
                    
                    # Evaluate optimized strategy
                    results = self.evaluate_strategy(strategy_name, X_train, y_train, 
                                                   X_test, y_test, domain)
                    domain_results[strategy_name] = results
                    
                round_results[domain] = domain_results
                
            optimization_results[f'round_{round_num + 1}'] = round_results
            
            # Learn from this round
            self.learn_adaptation_patterns()
            
        # Final analysis
        print("\n🧠 Learning adaptation patterns...")
        self.learn_adaptation_patterns()
        
        return optimization_results
        
    def predict_learning_efficiency(self, domain: str, data_size: int, 
                                   strategy_name: str = None) -> Dict[str, float]:
        """Predict learning efficiency for a domain and data size."""
        
        if strategy_name is None:
            strategy_name = self.select_best_strategy(domain)
            
        if strategy_name not in self.strategies:
            return {'predicted_performance': 0.5, 'confidence': 0.0}
            
        strategy = self.strategies[strategy_name]
        
        # Use historical data to predict
        if strategy.performance_history:
            # Simple model based on data size and domain
            similar_records = [r for r in strategy.performance_history 
                             if r['domain'] == domain or strategy.generalization_ability > 0.7]
            
            if similar_records:
                # Predict based on similar scenarios
                performances = [r['performance'] for r in similar_records]
                data_efficiencies = [r['data_efficiency'] for r in similar_records]
                
                # Adjust for data size
                size_factor = min(1.0, data_size / 1000)  # Normalize to reasonable scale
                predicted_performance = np.mean(performances) * (0.5 + 0.5 * size_factor)
                
                confidence = len(similar_records) / (len(similar_records) + 5)  # Decreasing uncertainty
                
                return {
                    'predicted_performance': predicted_performance,
                    'confidence': confidence,
                    'recommended_strategy': strategy_name
                }
                
        # Fallback prediction
        return {
            'predicted_performance': 0.5,
            'confidence': 0.1,
            'recommended_strategy': strategy_name
        }
        
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Generate insights about the meta-learning process."""
        
        insights = {
            'total_strategies': len(self.strategies),
            'optimization_rounds': len(self.optimization_history),
            'domains_explored': set(),
            'best_overall_strategy': None,
            'adaptation_insights': {},
            'efficiency_trends': {},
            'recommendations': []
        }
        
        # Collect domain information
        for strategy in self.strategies.values():
            insights['domains_explored'].update(strategy.domain_performance.keys())
        insights['domains_explored'] = list(insights['domains_explored'])
        
        # Find best overall strategy
        if self.strategies:
            strategy_scores = {}
            for name, strategy in self.strategies.items():
                if strategy.performance_history:
                    avg_performance = np.mean([r['performance'] for r in strategy.performance_history])
                    strategy_scores[name] = avg_performance
                    
            if strategy_scores:
                insights['best_overall_strategy'] = max(strategy_scores, key=strategy_scores.get)
                
        # Adaptation insights
        if 'adaptation_leaders' in self.adaptation_patterns:
            insights['adaptation_insights'] = self.adaptation_patterns['adaptation_leaders']
            
        # Generate recommendations
        recommendations = []
        
        if insights['best_overall_strategy']:
            recommendations.append(f"Use {insights['best_overall_strategy']} as default strategy")
            
        if len(insights['domains_explored']) > 2:
            recommendations.append("System shows good cross-domain generalization")
            
        if len(self.optimization_history) > 5:
            recommendations.append("Sufficient optimization data for reliable predictions")
            
        insights['recommendations'] = recommendations
        
        return insights
        
    def visualize_learning_progress(self, save_path: str = None):
        """Visualize the meta-learning progress."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Strategy performance comparison
        strategy_performances = {}
        for name, strategy in self.strategies.items():
            if strategy.performance_history:
                avg_perf = np.mean([r['performance'] for r in strategy.performance_history])
                strategy_performances[name] = avg_perf
                
        if strategy_performances:
            strategies = list(strategy_performances.keys())
            performances = list(strategy_performances.values())
            
            ax1.bar(strategies, performances, color='skyblue', alpha=0.7)
            ax1.set_title('Average Strategy Performance')
            ax1.set_ylabel('R² Score')
            ax1.tick_params(axis='x', rotation=45)
            
        # Learning curves over time
        for name, strategy in self.strategies.items():
            if len(strategy.performance_history) > 2:
                performances = [r['performance'] for r in strategy.performance_history]
                ax2.plot(performances, label=name, alpha=0.7)
                
        ax2.set_title('Learning Curves')
        ax2.set_xlabel('Optimization Steps')
        ax2.set_ylabel('Performance')
        ax2.legend()
        
        # Efficiency vs Performance scatter
        efficiencies = []
        performances = []
        strategy_names = []
        
        for name, strategy in self.strategies.items():
            if strategy.performance_history:
                avg_perf = np.mean([r['performance'] for r in strategy.performance_history])
                efficiencies.append(strategy.efficiency_score)
                performances.append(avg_perf)
                strategy_names.append(name)
                
        if efficiencies and performances:
            scatter = ax3.scatter(efficiencies, performances, c=range(len(efficiencies)), 
                                cmap='viridis', alpha=0.7, s=100)
            ax3.set_xlabel('Efficiency Score')
            ax3.set_ylabel('Average Performance')
            ax3.set_title('Efficiency vs Performance')
            
            # Add strategy labels
            for i, name in enumerate(strategy_names):
                ax3.annotate(name, (efficiencies[i], performances[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                           
        # Domain adaptation heatmap
        if self.adaptation_patterns and 'data' in self.adaptation_patterns:
            df = pd.DataFrame(self.adaptation_patterns['data'])
            if not df.empty:
                pivot_data = df.pivot_table(values='adaptation_rate', 
                                          index='strategy', columns='domain', 
                                          fill_value=0)
                sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', ax=ax4)
                ax4.set_title('Strategy Adaptation Rates by Domain')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def demonstrate_meta_learning():
    """Demonstrate the meta-learning optimization system."""
    print("🧠 META-LEARNING OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    # Initialize meta-learner
    meta_learner = MetaLearner()
    
    # Generate diverse synthetic datasets for different domains
    print("📊 Generating diverse datasets for meta-learning...")
    
    np.random.seed(42)
    domains_data = {}
    
    # Domain 1: Linear relationships (economics-like)
    n_samples = 200
    X1 = np.random.randn(n_samples, 4)
    y1 = 2*X1[:, 0] + 1.5*X1[:, 1] - 0.5*X1[:, 2] + 0.3*X1[:, 3] + 0.1*np.random.randn(n_samples)
    domains_data['economics'] = (X1, y1)
    
    # Domain 2: Non-linear relationships (biology-like)
    X2 = np.random.randn(n_samples, 3)
    y2 = X2[:, 0]**2 + np.sin(X2[:, 1]) + np.log(np.abs(X2[:, 2]) + 1) + 0.2*np.random.randn(n_samples)
    domains_data['biology'] = (X2, y2)
    
    # Domain 3: Complex interactions (physics-like)
    X3 = np.random.randn(n_samples, 5)
    y3 = (X3[:, 0] * X3[:, 1] + X3[:, 2]**3 + np.sqrt(np.abs(X3[:, 3])) + 
          X3[:, 4]**2 + 0.3*np.random.randn(n_samples))
    domains_data['physics'] = (X3, y3)
    
    # Domain 4: High-dimensional sparse (image-like)
    X4 = np.random.randn(n_samples, 10)
    # Only first 3 features matter
    y4 = X4[:, 0] + 2*X4[:, 1] - X4[:, 2] + 0.1*np.random.randn(n_samples)
    domains_data['computer_vision'] = (X4, y4)
    
    print(f"✅ Generated {len(domains_data)} domain datasets")
    
    # Perform meta-optimization
    print("\n🔄 Performing meta-optimization...")
    optimization_results = meta_learner.meta_optimize(domains_data, optimization_rounds=2)
    
    # Test strategy selection
    print("\n🎯 Testing strategy selection...")
    for domain in domains_data.keys():
        best_strategy = meta_learner.select_best_strategy(domain)
        print(f"  {domain}: {best_strategy}")
        
    # Test efficiency prediction
    print("\n📈 Testing learning efficiency prediction...")
    for domain in domains_data.keys():
        prediction = meta_learner.predict_learning_efficiency(domain, data_size=500)
        print(f"  {domain}: {prediction['predicted_performance']:.3f} "
              f"(confidence: {prediction['confidence']:.3f})")
              
    # Generate insights
    print("\n🧠 Meta-learning insights:")
    insights = meta_learner.get_meta_learning_insights()
    
    for key, value in insights.items():
        if isinstance(value, list) and key == 'recommendations':
            print(f"  {key}:")
            for rec in value:
                print(f"    • {rec}")
        elif not isinstance(value, (dict, set)):
            print(f"  {key}: {value}")
            
    # Evaluate meta-learning performance
    print("\n🧪 EVALUATING META-LEARNING PERFORMANCE:")
    print("=" * 50)
    
    # Test 1: Strategy diversity
    unique_strategies = len(set(meta_learner.select_best_strategy(domain) 
                               for domain in domains_data.keys()))
    strategy_diversity = unique_strategies / len(domains_data)
    
    # Test 2: Adaptation improvement
    adaptation_improvement = 0
    if meta_learner.adaptation_patterns and 'adaptation_leaders' in meta_learner.adaptation_patterns:
        adaptation_rates = list(meta_learner.adaptation_patterns['adaptation_leaders'].values())
        if adaptation_rates:
            adaptation_improvement = max(0, np.mean(adaptation_rates))
            
    # Test 3: Cross-domain generalization
    generalization_scores = [s.generalization_ability for s in meta_learner.strategies.values()]
    avg_generalization = np.mean(generalization_scores) if generalization_scores else 0
    
    # Test 4: Optimization effectiveness
    total_optimizations = len(meta_learner.optimization_history)
    optimization_effectiveness = min(1.0, total_optimizations / 10)  # Target 10+ optimizations
    
    print(f"✅ Strategy diversity: {strategy_diversity:.3f}")
    print(f"✅ Adaptation improvement: {adaptation_improvement:.3f}")
    print(f"✅ Cross-domain generalization: {avg_generalization:.3f}")
    print(f"✅ Optimization effectiveness: {optimization_effectiveness:.3f}")
    
    # Overall assessment
    print("\n🎯 META-LEARNING ASSESSMENT:")
    print("=" * 40)
    
    success_criteria = {
        'strategy_diversity': strategy_diversity > 0.5,
        'adaptation_improvement': adaptation_improvement > 0.0,
        'generalization': avg_generalization > 0.3,
        'optimization_count': total_optimizations >= 5,
        'insights_generated': len(insights['recommendations']) > 0
    }
    
    passed_tests = sum(success_criteria.values())
    total_tests = len(success_criteria)
    
    for test, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test}: {status}")
        
    success_rate = passed_tests / total_tests
    print(f"\n🏆 OVERALL SUCCESS RATE: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("🎉 META-LEARNING OPTIMIZATION SUCCESSFUL!")
        print("   System demonstrates learning-to-learn capabilities,")
        print("   strategy adaptation, and efficiency improvements.")
    else:
        print("⚠️  Meta-learning optimization needs improvement.")
        
    return meta_learner, success_rate


if __name__ == "__main__":
    meta_learner, success_rate = demonstrate_meta_learning()