import datetime
import pathlib
import random
import re
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

from .abstract_game import AbstractGame


class MathTokenizer:
    """Tokenizer for mathematical expressions and reasoning steps."""
    
    def __init__(self):
        # Define mathematical vocabulary
        self.special_tokens = {
            '<PAD>': 0, '<START>': 1, '<END>': 2, '<STEP>': 3, '<EQ>': 4
        }
        
        # Numbers (0-9)
        self.numbers = {str(i): i + 5 for i in range(10)}
        
        # Basic operations
        self.operations = {
            '+': 15, '-': 16, '*': 17, '/': 18, '=': 19, '(': 20, ')': 21
        }
        
        # Variables and constants
        self.variables = {
            'x': 22, 'y': 23, 'z': 24, 'a': 25, 'b': 26, 'c': 27
        }
        
        # Mathematical functions
        self.functions = {
            'sin': 28, 'cos': 29, 'log': 30, 'exp': 31, 'sqrt': 32
        }
        
        # Combine all vocabularies
        self.vocab = {}
        self.vocab.update(self.special_tokens)
        self.vocab.update(self.numbers)
        self.vocab.update(self.operations)
        self.vocab.update(self.variables)
        self.vocab.update(self.functions)
        
        self.vocab_size = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
    def tokenize(self, expression: str) -> List[int]:
        """Convert mathematical expression to token ids."""
        tokens = []
        expression = expression.replace(' ', '')  # Remove spaces
        
        i = 0
        while i < len(expression):
            # Check for multi-character tokens (functions)
            found = False
            for length in [4, 3, 2]:  # Check longer tokens first
                if i + length <= len(expression):
                    substr = expression[i:i+length]
                    if substr in self.vocab:
                        tokens.append(self.vocab[substr])
                        i += length
                        found = True
                        break
            
            if not found:
                # Single character token
                char = expression[i]
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    # Unknown token, skip or handle as needed
                    pass
                i += 1
                
        return tokens
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token ids back to mathematical expression."""
        expression = ""
        for token in tokens:
            if token in self.inverse_vocab:
                expression += self.inverse_vocab[token]
        return expression


class MathProblem:
    """Represents a mathematical problem with steps and solution."""
    
    def __init__(self, expression: str, target: float, steps: List[str] = None):
        self.expression = expression
        self.target = target
        self.steps = steps or []
        self.current_step = 0
        
    def evaluate(self, expr: str) -> Optional[float]:
        """Safely evaluate mathematical expression."""
        try:
            # Simple evaluation for basic arithmetic
            # In practice, you'd want a more robust mathematical parser
            return eval(expr)
        except:
            return None


class Game(AbstractGame):
    """Mathematical reasoning game for MuZero."""
    
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.tokenizer = MathTokenizer()
        self.max_sequence_length = 50
        self.max_steps = 10
        
        self.reset()
        
    def step(self, action):
        """Apply action (choose next mathematical step)."""
        if self.done:
            return self.get_observation(), 0, True
            
        # Action represents choosing a mathematical operation or step
        reward = self._apply_action(action)
        self.current_step += 1
        
        # Check if problem is solved or max steps reached
        if self.current_step >= self.max_steps or self._is_solved():
            self.done = True
            if self._is_solved():
                reward += 10  # Bonus for solving
        
        return self.get_observation(), reward, self.done
    
    def _apply_action(self, action):
        """Apply mathematical reasoning action."""
        # This is a simplified version - in practice you'd have more sophisticated
        # action space for mathematical reasoning steps
        
        if action == 0:  # Simplify expression
            reward = self._try_simplify()
        elif action == 1:  # Apply distributive property
            reward = self._try_distribute()
        elif action == 2:  # Combine like terms
            reward = self._try_combine_terms()
        elif action == 3:  # Solve for variable
            reward = self._try_solve()
        else:
            reward = -0.1  # Invalid action penalty
            
        return reward
    
    def _try_simplify(self):
        """Try to simplify current expression."""
        # Placeholder - implement actual simplification logic
        return 0.1
    
    def _try_distribute(self):
        """Try to apply distributive property."""
        # Placeholder - implement actual distribution logic
        return 0.1
    
    def _try_combine_terms(self):
        """Try to combine like terms."""
        # Placeholder - implement actual term combination logic
        return 0.1
    
    def _try_solve(self):
        """Try to solve for the variable."""
        # Placeholder - implement actual solving logic
        return 0.1
    
    def _is_solved(self):
        """Check if the mathematical problem is solved."""
        # Placeholder - implement actual solution checking
        return False
    
    def legal_actions(self):
        """Return legal actions at current state."""
        if self.done:
            return []
        return [0, 1, 2, 3]  # Four basic mathematical reasoning actions
    
    def reset(self):
        """Reset game with new mathematical problem."""
        self.problem = self._generate_problem()
        self.current_step = 0
        self.done = False
        self.solution_history = []
        
        return self.get_observation()
    
    def _generate_problem(self):
        """Generate a random mathematical problem."""
        # Generate simple arithmetic problems for now
        problem_types = [
            self._generate_linear_equation,
            self._generate_quadratic_equation,
            self._generate_arithmetic_sequence
        ]
        
        problem_generator = random.choice(problem_types)
        return problem_generator()
    
    def _generate_linear_equation(self):
        """Generate simple linear equation like 2x + 3 = 7."""
        a = random.randint(1, 5)
        b = random.randint(1, 10)
        c = random.randint(1, 20)
        
        # ax + b = c, solution is x = (c-b)/a
        expression = f"{a}*x+{b}={c}"
        target = (c - b) / a
        
        return MathProblem(expression, target)
    
    def _generate_quadratic_equation(self):
        """Generate simple quadratic equation."""
        a = random.randint(1, 3)
        b = random.randint(-5, 5)
        c = random.randint(-10, 10)
        
        expression = f"{a}*x*x+{b}*x+{c}=0"
        # For simplicity, we'll just store the coefficients
        target = 0  # We're solving for roots
        
        return MathProblem(expression, target)
    
    def _generate_arithmetic_sequence(self):
        """Generate arithmetic sequence problem."""
        first = random.randint(1, 10)
        diff = random.randint(1, 5)
        n = random.randint(5, 10)
        
        # Find nth term of arithmetic sequence
        expression = f"a_1={first},d={diff},find_a_{n}"
        target = first + (n - 1) * diff
        
        return MathProblem(expression, target)
    
    def get_observation(self):
        """Get current state observation."""
        # Convert current problem state to tokenized sequence
        tokens = self.tokenizer.tokenize(self.problem.expression)
        
        # Pad or truncate to fixed length
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[:self.max_sequence_length]
        else:
            tokens.extend([0] * (self.max_sequence_length - len(tokens)))
        
        # Add step information
        step_info = [self.current_step, int(self.done)]
        
        # Combine tokens and step info
        observation = np.array(tokens + step_info, dtype=np.float32)
        
        return observation.reshape(1, 1, -1)  # MuZero expects 3D observations
    
    def render(self):
        """Display current mathematical problem state."""
        print(f"Problem: {self.problem.expression}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Target: {self.problem.target}")
        print(f"Done: {self.done}")
        print("---")
    
    def human_to_action(self):
        """Get action from human input."""
        print("Available actions:")
        print("0: Simplify expression")
        print("1: Apply distributive property")
        print("2: Combine like terms")
        print("3: Solve for variable")
        
        while True:
            try:
                action = int(input("Enter action (0-3): "))
                if action in self.legal_actions():
                    return action
                else:
                    print("Invalid action. Try again.")
            except ValueError:
                print("Please enter a number.")
    
    def action_to_string(self, action_number):
        """Convert action number to string description."""
        actions = {
            0: "Simplify expression",
            1: "Apply distributive property", 
            2: "Combine like terms",
            3: "Solve for variable"
        }
        return actions.get(action_number, "Unknown action")


class MuZeroConfig:
    """Configuration for mathematical reasoning MuZero."""
    
    def __init__(self):
        # Seed and GPU settings
        self.seed = 0
        self.max_num_gpus = None
        
        # Game settings
        self.observation_shape = (1, 1, 52)  # 50 tokens + 2 step info
        self.action_space = list(range(4))  # 4 mathematical reasoning actions
        self.players = list(range(1))
        self.stacked_observations = 0
        
        # Evaluation
        self.muzero_player = 0
        self.opponent = None
        
        # Self-play settings
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 10  # Maximum reasoning steps
        self.num_simulations = 50
        self.discount = 0.99
        self.temperature_threshold = None
        
        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        
        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # Network settings - we'll use transformer architecture
        self.network = "transformer"  # New transformer network
        self.support_size = 10
        
        # Transformer-specific settings
        self.vocab_size = 33  # From tokenizer
        self.embedding_dim = 256
        self.num_heads = 8
        self.num_layers = 6
        self.hidden_dim = 512
        self.dropout = 0.1
        self.max_sequence_length = 50
        
        # Training settings
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 10000
        self.batch_size = 64
        self.checkpoint_interval = 100
        self.value_loss_weight = 1
        self.train_on_gpu = False  # Force CPU for testing
        
        # Optimizer settings
        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # Learning rate schedule
        self.lr_init = 0.001
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 1000
        
        # Replay buffer settings
        self.replay_buffer_size = 1000
        self.num_unroll_steps = 5
        self.td_steps = 10
        self.PER = True
        
        # Reanalyze settings
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False