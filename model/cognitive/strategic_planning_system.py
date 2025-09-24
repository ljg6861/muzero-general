#!/usr/bin/env python3
"""
Strategic Planning System
========================
Implements multi-step problem decomposition capabilities with hierarchical planning,
goal prioritization, and adaptive strategy selection for complex problem solving.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set, Callable
from enum import Enum
import heapq
import json
import random
import itertools
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


class PlanningHorizon(Enum):
    """Time horizons for different planning levels."""
    IMMEDIATE = "immediate"      # 1-5 steps
    SHORT_TERM = "short_term"    # 6-20 steps
    MEDIUM_TERM = "medium_term"  # 21-100 steps
    LONG_TERM = "long_term"      # 100+ steps


class TaskComplexity(Enum):
    """Complexity levels for tasks and subproblems."""
    TRIVIAL = "trivial"          # Single operation
    SIMPLE = "simple"            # 2-5 operations
    MODERATE = "moderate"        # 6-20 operations
    COMPLEX = "complex"          # 21-100 operations
    VERY_COMPLEX = "very_complex"  # 100+ operations


class PlanExecutionStatus(Enum):
    """Status of plan execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ADAPTED = "adapted"


@dataclass
class Task:
    """Represents a single task or action in a plan."""
    task_id: str
    name: str
    description: str
    complexity: TaskComplexity
    prerequisites: List[str] = field(default_factory=list)
    estimated_effort: float = 1.0
    estimated_duration: float = 1.0
    success_probability: float = 0.8
    resources_required: Dict[str, float] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    execution_status: PlanExecutionStatus = PlanExecutionStatus.NOT_STARTED
    actual_effort: Optional[float] = None
    actual_duration: Optional[float] = None
    completion_time: Optional[datetime] = None


@dataclass
class Goal:
    """Represents a goal with multiple achievement criteria."""
    goal_id: str
    name: str
    description: str
    priority: float  # 0.0 to 1.0
    importance: float  # 0.0 to 1.0
    urgency: float  # 0.0 to 1.0
    success_criteria: List[str]
    deadline: Optional[datetime] = None
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    achievement_level: float = 0.0  # 0.0 to 1.0
    status: PlanExecutionStatus = PlanExecutionStatus.NOT_STARTED


@dataclass
class Strategy:
    """Represents a problem-solving strategy."""
    strategy_id: str
    name: str
    description: str
    applicable_domains: List[str]
    effectiveness_score: float = 0.5
    complexity_handling: TaskComplexity = TaskComplexity.MODERATE
    resource_efficiency: float = 0.5
    adaptation_capability: float = 0.5
    success_history: List[float] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)


@dataclass
class Plan:
    """Represents a hierarchical plan with tasks, dependencies, and strategies."""
    plan_id: str
    name: str
    description: str
    main_goal: str
    tasks: List[Task]
    dependencies: nx.DiGraph
    strategies: List[Strategy]
    planning_horizon: PlanningHorizon
    estimated_completion_time: float
    success_probability: float
    resource_requirements: Dict[str, float]
    contingency_plans: List[str] = field(default_factory=list)
    monitoring_checkpoints: List[str] = field(default_factory=list)
    adaptation_triggers: List[str] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class StrategicPlanner:
    """Main engine for hierarchical strategic planning and problem decomposition."""
    
    def __init__(self):
        self.goals = {}
        self.tasks = {}
        self.strategies = {}
        self.plans = {}
        self.execution_history = []
        self.performance_metrics = defaultdict(list)
        
        # Planning knowledge base
        self.decomposition_patterns = {}
        self.strategy_effectiveness = defaultdict(float)
        self.resource_utilization = defaultdict(float)
        
        # Initialize with common strategies
        self._initialize_base_strategies()
        
    def _initialize_base_strategies(self):
        """Initialize common problem-solving strategies."""
        
        base_strategies = [
            Strategy(
                strategy_id="divide_and_conquer",
                name="Divide and Conquer",
                description="Break complex problems into smaller, manageable subproblems",
                applicable_domains=["algorithms", "project_management", "research", "engineering"],
                effectiveness_score=0.85,
                complexity_handling=TaskComplexity.VERY_COMPLEX,
                resource_efficiency=0.8,
                adaptation_capability=0.7
            ),
            Strategy(
                strategy_id="iterative_refinement",
                name="Iterative Refinement",
                description="Start with simple solution and progressively improve",
                applicable_domains=["design", "research", "software", "optimization"],
                effectiveness_score=0.75,
                complexity_handling=TaskComplexity.COMPLEX,
                resource_efficiency=0.7,
                adaptation_capability=0.9
            ),
            Strategy(
                strategy_id="parallel_processing",
                name="Parallel Processing",
                description="Execute multiple independent tasks simultaneously",
                applicable_domains=["computation", "manufacturing", "research", "logistics"],
                effectiveness_score=0.8,
                complexity_handling=TaskComplexity.COMPLEX,
                resource_efficiency=0.6,
                adaptation_capability=0.5
            ),
            Strategy(
                strategy_id="dependency_optimization",
                name="Dependency Optimization",
                description="Optimize task ordering based on dependencies and critical paths",
                applicable_domains=["project_management", "manufacturing", "logistics", "research"],
                effectiveness_score=0.9,
                complexity_handling=TaskComplexity.VERY_COMPLEX,
                resource_efficiency=0.9,
                adaptation_capability=0.6
            ),
            Strategy(
                strategy_id="resource_balancing",
                name="Resource Balancing",
                description="Balance resource allocation across competing objectives",
                applicable_domains=["economics", "project_management", "optimization", "logistics"],
                effectiveness_score=0.7,
                complexity_handling=TaskComplexity.COMPLEX,
                resource_efficiency=0.95,
                adaptation_capability=0.8
            )
        ]
        
        for strategy in base_strategies:
            self.strategies[strategy.strategy_id] = strategy
            
    def decompose_problem(self, problem_description: str, target_complexity: TaskComplexity = TaskComplexity.SIMPLE) -> List[Task]:
        """Decompose a complex problem into manageable tasks."""
        
        # Analyze problem characteristics
        problem_analysis = self._analyze_problem(problem_description)
        
        # Select appropriate decomposition strategy
        decomposition_strategy = self._select_decomposition_strategy(problem_analysis)
        
        # Generate task decomposition
        tasks = self._generate_task_decomposition(
            problem_description, 
            problem_analysis, 
            decomposition_strategy,
            target_complexity
        )
        
        # Store tasks
        for task in tasks:
            self.tasks[task.task_id] = task
            
        return tasks
        
    def _analyze_problem(self, problem_description: str) -> Dict[str, Any]:
        """Analyze problem characteristics to guide decomposition."""
        
        # Simple heuristics for problem analysis
        analysis = {
            'description': problem_description,
            'estimated_complexity': TaskComplexity.MODERATE,
            'domain': 'general',
            'key_challenges': [],
            'resource_requirements': {},
            'constraints': [],
            'success_criteria': []
        }
        
        # Complexity estimation based on keywords
        complexity_indicators = {
            'trivial': ['simple', 'basic', 'easy', 'straightforward'],
            'simple': ['implement', 'create', 'build', 'develop'],
            'moderate': ['design', 'optimize', 'analyze', 'integrate'],
            'complex': ['research', 'discover', 'invent', 'innovate'],
            'very_complex': ['revolutionize', 'transform', 'breakthrough', 'paradigm']
        }
        
        desc_lower = problem_description.lower()
        for complexity, keywords in complexity_indicators.items():
            if any(keyword in desc_lower for keyword in keywords):
                analysis['estimated_complexity'] = TaskComplexity(complexity)
                break
                
        # Domain detection
        domain_keywords = {
            'research': ['research', 'study', 'investigate', 'discover', 'analyze'],
            'engineering': ['build', 'design', 'construct', 'engineer', 'develop'],
            'software': ['code', 'program', 'software', 'algorithm', 'application'],
            'management': ['manage', 'organize', 'coordinate', 'plan', 'execute'],
            'optimization': ['optimize', 'improve', 'enhance', 'maximize', 'minimize']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                analysis['domain'] = domain
                break
                
        # Extract key challenges (simplified)
        challenge_patterns = [
            'challenge', 'difficulty', 'problem', 'obstacle', 'constraint'
        ]
        
        for pattern in challenge_patterns:
            if pattern in desc_lower:
                analysis['key_challenges'].append(f"Address {pattern} in {problem_description}")
                
        return analysis
        
    def _select_decomposition_strategy(self, problem_analysis: Dict[str, Any]) -> Strategy:
        """Select the best decomposition strategy for the problem."""
        
        domain = problem_analysis['domain']
        complexity = problem_analysis['estimated_complexity']
        
        # Score strategies based on applicability
        strategy_scores = {}
        
        for strategy_id, strategy in self.strategies.items():
            score = 0.0
            
            # Domain applicability
            if domain in strategy.applicable_domains or 'general' in strategy.applicable_domains:
                score += 0.4
                
            # Complexity handling
            if strategy.complexity_handling.value == complexity.value:
                score += 0.3
            elif abs(list(TaskComplexity).index(strategy.complexity_handling) - 
                    list(TaskComplexity).index(complexity)) <= 1:
                score += 0.2
                
            # Effectiveness and efficiency
            score += strategy.effectiveness_score * 0.2
            score += strategy.resource_efficiency * 0.1
            
            strategy_scores[strategy_id] = score
            
        # Select strategy with highest score
        best_strategy_id = max(strategy_scores, key=strategy_scores.get)
        return self.strategies[best_strategy_id]
        
    def _generate_task_decomposition(self, problem_description: str, analysis: Dict[str, Any], 
                                   strategy: Strategy, target_complexity: TaskComplexity) -> List[Task]:
        """Generate task decomposition using the selected strategy."""
        
        tasks = []
        
        if strategy.strategy_id == "divide_and_conquer":
            tasks = self._divide_and_conquer_decomposition(problem_description, analysis, target_complexity)
        elif strategy.strategy_id == "iterative_refinement":
            tasks = self._iterative_refinement_decomposition(problem_description, analysis, target_complexity)
        elif strategy.strategy_id == "parallel_processing":
            tasks = self._parallel_processing_decomposition(problem_description, analysis, target_complexity)
        else:
            # Default sequential decomposition
            tasks = self._sequential_decomposition(problem_description, analysis, target_complexity)
            
        return tasks
        
    def _divide_and_conquer_decomposition(self, problem: str, analysis: Dict[str, Any], 
                                        target_complexity: TaskComplexity) -> List[Task]:
        """Implement divide and conquer decomposition strategy."""
        
        tasks = []
        base_id = f"task_{len(self.tasks)}"
        
        # Break into analysis, design, implementation, and validation phases
        phases = [
            ("analyze", "Analyze and understand the problem requirements"),
            ("design", "Design solution architecture and approach"),
            ("implement", "Implement the designed solution"),
            ("validate", "Test and validate the solution")
        ]
        
        for i, (phase_name, phase_desc) in enumerate(phases):
            task_id = f"{base_id}_{phase_name}"
            
            task = Task(
                task_id=task_id,
                name=f"{phase_name.title()} Phase",
                description=f"{phase_desc} for: {problem}",
                complexity=target_complexity,
                estimated_effort=1.0 + 0.5 * i,
                estimated_duration=1.0 + 0.3 * i,
                success_probability=0.8 - 0.05 * i,
                resources_required={'time': 1.0, 'effort': 1.0}
            )
            
            # Add dependencies
            if i > 0:
                task.dependencies = [f"{base_id}_{phases[i-1][0]}"]
                task.prerequisites = [phases[i-1][1]]
                
            tasks.append(task)
            
        # Add integration task if more than 2 phases
        if len(phases) > 2:
            integration_task = Task(
                task_id=f"{base_id}_integrate",
                name="Integration Phase",
                description=f"Integrate all components for: {problem}",
                complexity=target_complexity,
                estimated_effort=2.0,
                estimated_duration=1.5,
                success_probability=0.75,
                dependencies=[f"{base_id}_implement"],
                resources_required={'time': 1.5, 'effort': 2.0}
            )
            tasks.append(integration_task)
            
        return tasks
        
    def _iterative_refinement_decomposition(self, problem: str, analysis: Dict[str, Any], 
                                          target_complexity: TaskComplexity) -> List[Task]:
        """Implement iterative refinement decomposition strategy."""
        
        tasks = []
        base_id = f"task_{len(self.tasks)}"
        
        # Create iterations from prototype to final solution
        iterations = [
            ("prototype", "Create basic prototype", 0.5, 0.9),
            ("alpha", "Develop alpha version with core features", 1.0, 0.8),
            ("beta", "Refine beta version with full features", 1.5, 0.85),
            ("final", "Finalize production-ready solution", 2.0, 0.9)
        ]
        
        for i, (iteration_name, iteration_desc, effort_multiplier, success_prob) in enumerate(iterations):
            task_id = f"{base_id}_{iteration_name}"
            
            task = Task(
                task_id=task_id,
                name=f"{iteration_name.title()} Version",
                description=f"{iteration_desc} for: {problem}",
                complexity=target_complexity,
                estimated_effort=effort_multiplier,
                estimated_duration=effort_multiplier * 0.8,
                success_probability=success_prob,
                resources_required={'time': effort_multiplier, 'effort': effort_multiplier}
            )
            
            # Add dependencies to previous iteration
            if i > 0:
                task.dependencies = [f"{base_id}_{iterations[i-1][0]}"]
                task.prerequisites = [f"Complete {iterations[i-1][0]} version"]
                
            tasks.append(task)
            
        return tasks
        
    def _parallel_processing_decomposition(self, problem: str, analysis: Dict[str, Any], 
                                         target_complexity: TaskComplexity) -> List[Task]:
        """Implement parallel processing decomposition strategy."""
        
        tasks = []
        base_id = f"task_{len(self.tasks)}"
        
        # Create parallel components that can be worked on simultaneously
        components = [
            ("component_a", "Develop component A"),
            ("component_b", "Develop component B"), 
            ("component_c", "Develop component C"),
            ("integration", "Integrate all components")
        ]
        
        for i, (comp_name, comp_desc) in enumerate(components):
            task_id = f"{base_id}_{comp_name}"
            
            # Integration task depends on all components
            if comp_name == "integration":
                dependencies = [f"{base_id}_{components[j][0]}" for j in range(i)]
                effort_multiplier = 1.5
                success_prob = 0.7
            else:
                dependencies = []
                effort_multiplier = 1.0
                success_prob = 0.85
                
            task = Task(
                task_id=task_id,
                name=comp_name.replace('_', ' ').title(),
                description=f"{comp_desc} for: {problem}",
                complexity=target_complexity,
                estimated_effort=effort_multiplier,
                estimated_duration=effort_multiplier,
                success_probability=success_prob,
                dependencies=dependencies,
                resources_required={'time': effort_multiplier, 'effort': effort_multiplier}
            )
            
            tasks.append(task)
            
        return tasks
        
    def _sequential_decomposition(self, problem: str, analysis: Dict[str, Any], 
                                target_complexity: TaskComplexity) -> List[Task]:
        """Implement simple sequential decomposition."""
        
        tasks = []
        base_id = f"task_{len(self.tasks)}"
        
        # Simple sequential steps
        steps = [
            ("step_1", "Complete first step"),
            ("step_2", "Complete second step"),
            ("step_3", "Complete final step")
        ]
        
        for i, (step_name, step_desc) in enumerate(steps):
            task_id = f"{base_id}_{step_name}"
            
            task = Task(
                task_id=task_id,
                name=step_name.replace('_', ' ').title(),
                description=f"{step_desc} for: {problem}",
                complexity=target_complexity,
                estimated_effort=1.0,
                estimated_duration=1.0,
                success_probability=0.8,
                dependencies=[f"{base_id}_{steps[i-1][0]}"] if i > 0 else [],
                resources_required={'time': 1.0, 'effort': 1.0}
            )
            
            tasks.append(task)
            
        return tasks
        
    def create_hierarchical_plan(self, main_goal_description: str, time_horizon: PlanningHorizon = PlanningHorizon.MEDIUM_TERM) -> Plan:
        """Create a hierarchical plan for achieving a complex goal."""
        
        plan_id = f"plan_{len(self.plans)}"
        
        # Create main goal
        main_goal = Goal(
            goal_id=f"goal_{len(self.goals)}",
            name="Main Goal",
            description=main_goal_description,
            priority=1.0,
            importance=1.0,
            urgency=0.7,
            success_criteria=[f"Successfully complete: {main_goal_description}"]
        )
        
        self.goals[main_goal.goal_id] = main_goal
        
        # Decompose main goal into tasks
        tasks = self.decompose_problem(main_goal_description, TaskComplexity.MODERATE)
        
        # Create dependency graph
        dependency_graph = nx.DiGraph()
        
        for task in tasks:
            dependency_graph.add_node(task.task_id, task=task)
            
        for task in tasks:
            for dep in task.dependencies:
                if dep in [t.task_id for t in tasks]:
                    dependency_graph.add_edge(dep, task.task_id)
                    
        # Select strategies for the plan
        applicable_strategies = [s for s in self.strategies.values() 
                               if s.complexity_handling in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX]]
        
        # Calculate plan metrics
        total_effort = sum(task.estimated_effort for task in tasks)
        total_duration = self._calculate_critical_path_duration(tasks, dependency_graph)
        overall_success_prob = np.prod([task.success_probability for task in tasks])
        
        # Create the plan
        plan = Plan(
            plan_id=plan_id,
            name=f"Plan for {main_goal_description}",
            description=f"Hierarchical plan to achieve: {main_goal_description}",
            main_goal=main_goal.goal_id,
            tasks=tasks,
            dependencies=dependency_graph,
            strategies=applicable_strategies,
            planning_horizon=time_horizon,
            estimated_completion_time=total_duration,
            success_probability=overall_success_prob,
            resource_requirements={'total_effort': total_effort, 'total_time': total_duration}
        )
        
        self.plans[plan_id] = plan
        
        return plan
        
    def _calculate_critical_path_duration(self, tasks: List[Task], dependency_graph: nx.DiGraph) -> float:
        """Calculate the critical path duration through the task network."""
        
        # Simple critical path calculation
        task_dict = {task.task_id: task for task in tasks}
        
        # Topological sort to process tasks in dependency order
        try:
            topo_order = list(nx.topological_sort(dependency_graph))
        except nx.NetworkXError:
            # If there are cycles, use original order
            topo_order = [task.task_id for task in tasks]
            
        # Calculate earliest start times
        earliest_start = {}
        
        for task_id in topo_order:
            if task_id not in task_dict:
                continue
                
            task = task_dict[task_id]
            
            # Start time is maximum of predecessor finish times
            if not task.dependencies:
                earliest_start[task_id] = 0.0
            else:
                max_predecessor_finish = 0.0
                for dep_id in task.dependencies:
                    if dep_id in earliest_start and dep_id in task_dict:
                        dep_finish = earliest_start[dep_id] + task_dict[dep_id].estimated_duration
                        max_predecessor_finish = max(max_predecessor_finish, dep_finish)
                earliest_start[task_id] = max_predecessor_finish
                
        # Critical path duration is maximum finish time
        max_finish_time = 0.0
        for task_id, start_time in earliest_start.items():
            if task_id in task_dict:
                finish_time = start_time + task_dict[task_id].estimated_duration
                max_finish_time = max(max_finish_time, finish_time)
                
        return max_finish_time
        
    def optimize_plan(self, plan_id: str) -> Plan:
        """Optimize an existing plan for better performance."""
        
        if plan_id not in self.plans:
            raise ValueError(f"Plan {plan_id} not found")
            
        plan = self.plans[plan_id]
        
        # Optimization strategies
        optimizations = []
        
        # 1. Parallelize independent tasks
        parallel_opportunities = self._find_parallelization_opportunities(plan)
        optimizations.extend(parallel_opportunities)
        
        # 2. Optimize resource allocation
        resource_optimizations = self._optimize_resource_allocation(plan)
        optimizations.extend(resource_optimizations)
        
        # 3. Reorder tasks for efficiency
        ordering_optimizations = self._optimize_task_ordering(plan)
        optimizations.extend(ordering_optimizations)
        
        # Apply optimizations
        for optimization in optimizations:
            self._apply_optimization(plan, optimization)
            
        # Recalculate plan metrics
        plan.estimated_completion_time = self._calculate_critical_path_duration(plan.tasks, plan.dependencies)
        plan.last_updated = datetime.now()
        
        return plan
        
    def _find_parallelization_opportunities(self, plan: Plan) -> List[Dict[str, Any]]:
        """Find opportunities to parallelize independent tasks."""
        
        opportunities = []
        
        # Find tasks with no dependencies that can be run in parallel
        independent_tasks = []
        for task in plan.tasks:
            if not task.dependencies:
                independent_tasks.append(task)
                
        if len(independent_tasks) > 1:
            opportunities.append({
                'type': 'parallelize_independent',
                'tasks': independent_tasks,
                'estimated_speedup': 0.6 * len(independent_tasks)
            })
            
        # Find tasks that only depend on completed tasks
        completed_task_ids = {task.task_id for task in plan.tasks 
                            if task.execution_status == PlanExecutionStatus.COMPLETED}
        
        parallel_ready = []
        for task in plan.tasks:
            if (task.execution_status == PlanExecutionStatus.NOT_STARTED and 
                all(dep in completed_task_ids for dep in task.dependencies)):
                parallel_ready.append(task)
                
        if len(parallel_ready) > 1:
            opportunities.append({
                'type': 'parallelize_ready',
                'tasks': parallel_ready,
                'estimated_speedup': 0.4 * len(parallel_ready)
            })
            
        return opportunities
        
    def _optimize_resource_allocation(self, plan: Plan) -> List[Dict[str, Any]]:
        """Optimize resource allocation across tasks."""
        
        optimizations = []
        
        # Find resource bottlenecks
        resource_usage = defaultdict(float)
        for task in plan.tasks:
            for resource, amount in task.resources_required.items():
                resource_usage[resource] += amount
                
        # Suggest resource balancing
        if resource_usage:
            max_resource = max(resource_usage.values())
            min_resource = min(resource_usage.values())
            
            if max_resource > 2 * min_resource:  # Significant imbalance
                optimizations.append({
                    'type': 'balance_resources',
                    'imbalanced_resources': dict(resource_usage),
                    'estimated_improvement': 0.2
                })
                
        return optimizations
        
    def _optimize_task_ordering(self, plan: Plan) -> List[Dict[str, Any]]:
        """Optimize task ordering for efficiency."""
        
        optimizations = []
        
        # Suggest prioritizing high-impact, low-effort tasks
        task_priorities = []
        for task in plan.tasks:
            # Simple priority score: success_probability / effort
            priority_score = task.success_probability / max(task.estimated_effort, 0.1)
            task_priorities.append((task, priority_score))
            
        # Sort by priority
        task_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Check if reordering would improve efficiency
        current_order = [task.task_id for task in plan.tasks]
        optimal_order = [task.task_id for task, _ in task_priorities]
        
        if current_order != optimal_order:
            optimizations.append({
                'type': 'reorder_tasks',
                'current_order': current_order,
                'optimal_order': optimal_order,
                'estimated_improvement': 0.15
            })
            
        return optimizations
        
    def _apply_optimization(self, plan: Plan, optimization: Dict[str, Any]):
        """Apply a specific optimization to the plan."""
        
        if optimization['type'] == 'parallelize_independent':
            # Mark tasks as parallelizable (implementation detail)
            for task in optimization['tasks']:
                if 'parallel_group' not in task.constraints:
                    task.constraints.append('parallel_group_1')
                    
        elif optimization['type'] == 'reorder_tasks':
            # Reorder tasks according to optimal order
            optimal_order = optimization['optimal_order']
            task_dict = {task.task_id: task for task in plan.tasks}
            plan.tasks = [task_dict[task_id] for task_id in optimal_order if task_id in task_dict]
            
        # Record optimization in plan
        if not hasattr(plan, 'optimizations_applied'):
            plan.optimizations_applied = []
        plan.optimizations_applied.append(optimization)
        
    def execute_plan(self, plan_id: str, simulation: bool = True) -> Dict[str, Any]:
        """Execute a plan and track progress."""
        
        if plan_id not in self.plans:
            raise ValueError(f"Plan {plan_id} not found")
            
        plan = self.plans[plan_id]
        execution_results = {
            'plan_id': plan_id,
            'start_time': datetime.now(),
            'task_results': {},
            'overall_success': False,
            'completion_time': None,
            'total_effort_used': 0.0,
            'challenges_encountered': [],
            'adaptations_made': []
        }
        
        # Execute tasks in dependency order
        completed_tasks = set()
        
        while len(completed_tasks) < len(plan.tasks):
            # Find ready tasks (dependencies completed)
            ready_tasks = []
            for task in plan.tasks:
                if (task.task_id not in completed_tasks and 
                    all(dep in completed_tasks for dep in task.dependencies)):
                    ready_tasks.append(task)
                    
            if not ready_tasks:
                # No ready tasks - might be circular dependency
                execution_results['challenges_encountered'].append("Circular dependency detected")
                break
                
            # Execute ready tasks
            for task in ready_tasks:
                task_result = self._execute_task(task, simulation=simulation)
                execution_results['task_results'][task.task_id] = task_result
                execution_results['total_effort_used'] += task_result.get('actual_effort', task.estimated_effort)
                
                if task_result['success']:
                    completed_tasks.add(task.task_id)
                    task.execution_status = PlanExecutionStatus.COMPLETED
                    task.completion_time = datetime.now()
                else:
                    task.execution_status = PlanExecutionStatus.FAILED
                    execution_results['challenges_encountered'].append(f"Task {task.task_id} failed")
                    
        # Calculate overall success
        successful_tasks = sum(1 for result in execution_results['task_results'].values() if result['success'])
        execution_results['overall_success'] = successful_tasks == len(plan.tasks)
        execution_results['completion_time'] = datetime.now()
        
        # Record execution history
        self.execution_history.append(execution_results)
        
        return execution_results
        
    def _execute_task(self, task: Task, simulation: bool = True) -> Dict[str, Any]:
        """Execute a single task (simulated or real)."""
        
        task_result = {
            'task_id': task.task_id,
            'start_time': datetime.now(),
            'success': False,
            'actual_effort': task.estimated_effort,
            'actual_duration': task.estimated_duration,
            'output': None,
            'issues_encountered': []
        }
        
        if simulation:
            # Simulate task execution with some randomness
            effort_variance = random.uniform(0.8, 1.2)
            duration_variance = random.uniform(0.9, 1.1)
            
            task_result['actual_effort'] = task.estimated_effort * effort_variance
            task_result['actual_duration'] = task.estimated_duration * duration_variance
            
            # Determine success based on success probability
            success_roll = random.random()
            task_result['success'] = success_roll < task.success_probability
            
            if not task_result['success']:
                task_result['issues_encountered'].append("Simulated failure based on probability")
            else:
                task_result['output'] = f"Completed: {task.description}"
                
        else:
            # Real execution would go here
            task_result['success'] = True
            task_result['output'] = f"Executed: {task.description}"
            
        return task_result
        
    def adapt_plan_during_execution(self, plan_id: str, new_constraints: List[str] = None, 
                                  new_objectives: List[str] = None) -> Plan:
        """Adapt a plan during execution based on new information."""
        
        if plan_id not in self.plans:
            raise ValueError(f"Plan {plan_id} not found")
            
        plan = self.plans[plan_id]
        
        # Record adaptation trigger
        adaptation_record = {
            'timestamp': datetime.now(),
            'trigger': 'manual_adaptation',
            'new_constraints': new_constraints or [],
            'new_objectives': new_objectives or [],
            'changes_made': []
        }
        
        # Adapt to new constraints
        if new_constraints:
            for constraint in new_constraints:
                # Add constraint to all uncompleted tasks
                for task in plan.tasks:
                    if task.execution_status == PlanExecutionStatus.NOT_STARTED:
                        task.constraints.append(constraint)
                        adaptation_record['changes_made'].append(f"Added constraint '{constraint}' to {task.task_id}")
                        
        # Adapt to new objectives
        if new_objectives:
            # Create new tasks for new objectives
            for i, objective in enumerate(new_objectives):
                new_tasks = self.decompose_problem(objective, TaskComplexity.SIMPLE)
                for new_task in new_tasks:
                    plan.tasks.append(new_task)
                    plan.dependencies.add_node(new_task.task_id, task=new_task)
                    adaptation_record['changes_made'].append(f"Added task {new_task.task_id} for objective '{objective}'")
                    
        # Re-optimize the adapted plan
        self.optimize_plan(plan_id)
        
        # Record adaptation
        if not hasattr(plan, 'adaptations_made'):
            plan.adaptations_made = []
        plan.adaptations_made.append(adaptation_record)
        
        plan.last_updated = datetime.now()
        
        return plan
        
    def get_planning_analytics(self) -> Dict[str, Any]:
        """Generate analytics about planning performance."""
        
        analytics = {
            'total_plans_created': len(self.plans),
            'total_tasks_created': len(self.tasks),
            'total_executions': len(self.execution_history),
            'success_rates': {},
            'efficiency_metrics': {},
            'strategy_effectiveness': {},
            'common_failure_modes': [],
            'planning_insights': []
        }
        
        if self.execution_history:
            # Success rates
            successful_executions = sum(1 for exec_result in self.execution_history if exec_result['overall_success'])
            analytics['success_rates']['overall'] = successful_executions / len(self.execution_history)
            
            # Efficiency metrics
            estimated_vs_actual_effort = []
            estimated_vs_actual_duration = []
            
            for execution in self.execution_history:
                plan = self.plans[execution['plan_id']]
                estimated_effort = sum(task.estimated_effort for task in plan.tasks)
                actual_effort = execution['total_effort_used']
                
                if estimated_effort > 0:
                    estimated_vs_actual_effort.append(actual_effort / estimated_effort)
                    
            if estimated_vs_actual_effort:
                analytics['efficiency_metrics']['effort_accuracy'] = np.mean(estimated_vs_actual_effort)
                analytics['efficiency_metrics']['effort_variance'] = np.std(estimated_vs_actual_effort)
                
        # Strategy effectiveness
        for strategy_id, strategy in self.strategies.items():
            if strategy.success_history:
                analytics['strategy_effectiveness'][strategy_id] = {
                    'average_success_rate': np.mean(strategy.success_history),
                    'usage_count': len(strategy.success_history),
                    'effectiveness_score': strategy.effectiveness_score
                }
                
        # Generate insights
        insights = []
        
        if analytics['success_rates'].get('overall', 0) > 0.8:
            insights.append("High overall planning success rate indicates effective decomposition strategies")
        elif analytics['success_rates'].get('overall', 0) < 0.6:
            insights.append("Low success rate suggests need for better risk assessment and contingency planning")
            
        if analytics['efficiency_metrics'].get('effort_accuracy', 1.0) > 1.2:
            insights.append("Effort estimates tend to be optimistic - consider adding buffer time")
        elif analytics['efficiency_metrics'].get('effort_accuracy', 1.0) < 0.8:
            insights.append("Effort estimates tend to be pessimistic - consider tighter planning")
            
        analytics['planning_insights'] = insights
        
        return analytics


def demonstrate_strategic_planning():
    """Demonstrate the strategic planning system capabilities."""
    print("🎯 STRATEGIC PLANNING SYSTEM")
    print("=" * 60)
    
    # Initialize the strategic planner
    planner = StrategicPlanner()
    
    # Test problem decomposition
    print("📋 Testing Problem Decomposition:")
    print("-" * 40)
    
    test_problems = [
        "Develop a machine learning system for medical diagnosis",
        "Design and build a sustainable energy monitoring platform",
        "Research and implement quantum computing optimization algorithms"
    ]
    
    all_decompositions = []
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n🎯 Problem {i}: {problem}")
        
        tasks = planner.decompose_problem(problem, TaskComplexity.MODERATE)
        all_decompositions.append((problem, tasks))
        
        print(f"  Generated {len(tasks)} tasks:")
        for task in tasks[:3]:  # Show first 3 tasks
            print(f"    • {task.name}: {task.description}")
            print(f"      Effort: {task.estimated_effort:.1f}, Duration: {task.estimated_duration:.1f}")
            print(f"      Dependencies: {len(task.dependencies)}")
            
    # Test hierarchical planning
    print(f"\n🏗️ Testing Hierarchical Planning:")
    print("-" * 40)
    
    complex_goal = "Create an autonomous AI research assistant that can conduct scientific literature review and generate research hypotheses"
    
    plan = planner.create_hierarchical_plan(complex_goal, PlanningHorizon.LONG_TERM)
    
    print(f"📋 Created plan: {plan.name}")
    print(f"  Total tasks: {len(plan.tasks)}")
    print(f"  Estimated completion time: {plan.estimated_completion_time:.1f} units")
    print(f"  Success probability: {plan.success_probability:.3f}")
    print(f"  Planning horizon: {plan.planning_horizon.value}")
    
    # Show task dependencies
    print(f"\n  Task Dependencies:")
    for task in plan.tasks[:5]:  # Show first 5 tasks
        deps = ', '.join(task.dependencies) if task.dependencies else 'None'
        print(f"    {task.name} → Dependencies: {deps}")
        
    # Test plan optimization
    print(f"\n⚡ Testing Plan Optimization:")
    print("-" * 40)
    
    original_completion_time = plan.estimated_completion_time
    optimized_plan = planner.optimize_plan(plan.plan_id)
    
    print(f"  Original completion time: {original_completion_time:.1f}")
    print(f"  Optimized completion time: {optimized_plan.estimated_completion_time:.1f}")
    print(f"  Improvement: {(original_completion_time - optimized_plan.estimated_completion_time):.1f} units")
    
    if hasattr(optimized_plan, 'optimizations_applied'):
        print(f"  Optimizations applied: {len(optimized_plan.optimizations_applied)}")
        for opt in optimized_plan.optimizations_applied[:3]:
            print(f"    • {opt['type']}")
            
    # Test plan execution (simulated)
    print(f"\n🚀 Testing Plan Execution (Simulated):")
    print("-" * 40)
    
    execution_result = planner.execute_plan(plan.plan_id, simulation=True)
    
    print(f"  Plan execution started: {execution_result['start_time'].strftime('%H:%M:%S')}")
    print(f"  Overall success: {execution_result['overall_success']}")
    print(f"  Tasks completed: {len([r for r in execution_result['task_results'].values() if r['success']])}/{len(execution_result['task_results'])}")
    print(f"  Total effort used: {execution_result['total_effort_used']:.1f}")
    print(f"  Challenges encountered: {len(execution_result['challenges_encountered'])}")
    
    # Test plan adaptation
    print(f"\n🔄 Testing Plan Adaptation:")
    print("-" * 40)
    
    new_constraints = ["Must comply with new safety regulations", "Budget reduced by 20%"]
    new_objectives = ["Add real-time monitoring capability"]
    
    adapted_plan = planner.adapt_plan_during_execution(
        plan.plan_id, 
        new_constraints=new_constraints,
        new_objectives=new_objectives
    )
    
    print(f"  Plan adapted with {len(new_constraints)} new constraints and {len(new_objectives)} new objectives")
    print(f"  New task count: {len(adapted_plan.tasks)}")
    print(f"  Adaptations recorded: {len(getattr(adapted_plan, 'adaptations_made', []))}")
    
    if hasattr(adapted_plan, 'adaptations_made'):
        for adaptation in adapted_plan.adaptations_made[-1:]:  # Show latest adaptation
            print(f"    Changes made: {len(adaptation['changes_made'])}")
            
    # Generate analytics
    print(f"\n📊 Planning Analytics:")
    print("-" * 40)
    
    analytics = planner.get_planning_analytics()
    
    for key, value in analytics.items():
        if isinstance(value, dict):
            if value:  # Only show non-empty dicts
                print(f"  {key}:")
                for sub_key, sub_value in list(value.items())[:3]:  # Show first 3 items
                    if isinstance(sub_value, float):
                        print(f"    {sub_key}: {sub_value:.3f}")
                    else:
                        print(f"    {sub_key}: {sub_value}")
        elif isinstance(value, list):
            if value:  # Only show non-empty lists
                print(f"  {key}: {len(value)} items")
                for item in value[:2]:  # Show first 2 items
                    print(f"    • {item}")
        else:
            print(f"  {key}: {value}")
            
    # Evaluate strategic planning capabilities
    print(f"\n🧪 EVALUATING STRATEGIC PLANNING:")
    print("=" * 50)
    
    # Test 1: Problem decomposition effectiveness
    avg_tasks_per_problem = np.mean([len(tasks) for _, tasks in all_decompositions])
    decomposition_score = min(1.0, avg_tasks_per_problem / 5.0)  # Target 5+ tasks per problem
    
    # Test 2: Plan complexity handling
    complex_tasks = len([task for task in plan.tasks if task.complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]])
    complexity_score = min(1.0, complex_tasks / max(1, len(plan.tasks)))
    
    # Test 3: Optimization effectiveness
    optimization_improvement = (original_completion_time - optimized_plan.estimated_completion_time) / original_completion_time
    optimization_score = min(1.0, optimization_improvement * 5)  # Scale improvement
    
    # Test 4: Execution simulation reliability
    execution_score = 1.0 if execution_result['overall_success'] else 0.5
    
    # Test 5: Adaptation capability
    adaptation_score = 1.0 if hasattr(adapted_plan, 'adaptations_made') and adapted_plan.adaptations_made else 0.0
    
    print(f"✅ Problem decomposition: {decomposition_score:.3f}")
    print(f"✅ Complexity handling: {complexity_score:.3f}")
    print(f"✅ Optimization effectiveness: {optimization_score:.3f}")
    print(f"✅ Execution reliability: {execution_score:.3f}")
    print(f"✅ Adaptation capability: {adaptation_score:.3f}")
    
    # Overall assessment
    print(f"\n🎯 STRATEGIC PLANNING ASSESSMENT:")
    print("=" * 50)
    
    success_criteria = {
        'decomposition_quality': decomposition_score > 0.6,
        'complexity_handling': complexity_score > 0.3,
        'optimization_effectiveness': optimization_score > 0.2,
        'execution_simulation': execution_score > 0.5,
        'adaptation_capability': adaptation_score > 0.5,
        'plan_creation': len(planner.plans) > 0,
        'strategy_diversity': len(planner.strategies) >= 3
    }
    
    passed_tests = sum(success_criteria.values())
    total_tests = len(success_criteria)
    
    for test, passed in success_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test}: {status}")
        
    success_rate = passed_tests / total_tests
    print(f"\n🏆 OVERALL SUCCESS RATE: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("🎉 STRATEGIC PLANNING SUCCESSFUL!")
        print("   System demonstrates hierarchical planning,")
        print("   problem decomposition, and adaptive strategy selection.")
    else:
        print("⚠️  Strategic planning capabilities need improvement.")
        
    return planner, success_rate


if __name__ == "__main__":
    planner, success_rate = demonstrate_strategic_planning()