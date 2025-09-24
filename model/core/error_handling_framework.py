#!/usr/bin/env python3
"""
Comprehensive Error Handling Framework
=====================================
Provides robust error handling, logging, and graceful degradation
for all cognitive architecture components.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
from enum import Enum
import json


class ErrorSeverity(Enum):
    """Error severity levels for categorizing issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentStatus(Enum):
    """Component operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CognitiveError(Exception):
    """Base exception for cognitive architecture errors."""
    
    def __init__(self, message: str, component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 original_error: Exception = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.component = component
        self.severity = severity
        self.original_error = original_error
        self.context = context or {}
        self.timestamp = datetime.now()


class ErrorHandler:
    """Centralized error handling system for cognitive architecture."""
    
    def __init__(self, log_level: int = logging.INFO):
        self.error_log = []
        self.component_status = {}
        self.error_counts = {}
        self.recovery_attempts = {}
        
        # Setup logging
        self.logger = logging.getLogger('CognitiveArchitecture')
        self.logger.setLevel(log_level)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def log_error(self, error: CognitiveError):
        """Log an error and update component status."""
        
        # Record error
        error_record = {
            'timestamp': error.timestamp,
            'component': error.component,
            'severity': error.severity.value,
            'message': error.message,
            'original_error': str(error.original_error) if error.original_error else None,
            'context': error.context
        }
        
        self.error_log.append(error_record)
        
        # Update error counts
        component = error.component
        if component not in self.error_counts:
            self.error_counts[component] = {
                'low': 0, 'medium': 0, 'high': 0, 'critical': 0
            }
        self.error_counts[component][error.severity.value] += 1
        
        # Update component status
        self._update_component_status(component, error.severity)
        
        # Log to standard logger
        log_level = self._severity_to_log_level(error.severity)
        self.logger.log(log_level, f"[{component}] {error.message}")
        
        if error.original_error:
            self.logger.debug(f"Original error: {error.original_error}")
            
    def _severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Convert error severity to logging level."""
        mapping = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.WARNING)
        
    def _update_component_status(self, component: str, severity: ErrorSeverity):
        """Update component operational status based on error severity."""
        
        if severity == ErrorSeverity.CRITICAL:
            self.component_status[component] = ComponentStatus.FAILED
        elif severity == ErrorSeverity.HIGH:
            # Multiple high severity errors -> failed
            high_errors = self.error_counts.get(component, {}).get('high', 0)
            if high_errors >= 3:
                self.component_status[component] = ComponentStatus.FAILED
            else:
                self.component_status[component] = ComponentStatus.DEGRADED
        elif severity in [ErrorSeverity.MEDIUM, ErrorSeverity.LOW]:
            # Only degrade if not already failed
            current_status = self.component_status.get(component, ComponentStatus.HEALTHY)
            if current_status != ComponentStatus.FAILED:
                medium_errors = self.error_counts.get(component, {}).get('medium', 0)
                if medium_errors >= 5:
                    self.component_status[component] = ComponentStatus.DEGRADED
                else:
                    self.component_status[component] = ComponentStatus.HEALTHY
                    
    def is_component_healthy(self, component: str) -> bool:
        """Check if a component is healthy."""
        return self.component_status.get(component, ComponentStatus.HEALTHY) == ComponentStatus.HEALTHY
        
    def get_component_status(self, component: str) -> ComponentStatus:
        """Get current status of a component."""
        return self.component_status.get(component, ComponentStatus.HEALTHY)
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors and component status."""
        
        total_errors = len(self.error_log)
        recent_errors = [e for e in self.error_log 
                        if (datetime.now() - e['timestamp']).total_seconds() < 3600]  # Last hour
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for error in self.error_log:
            severity_counts[error['severity']] += 1
            
        return {
            'total_errors': total_errors,
            'recent_errors': len(recent_errors),
            'severity_distribution': severity_counts,
            'component_status': {comp: status.value for comp, status in self.component_status.items()},
            'error_counts_by_component': self.error_counts,
            'last_error': self.error_log[-1] if self.error_log else None
        }


# Global error handler instance
global_error_handler = ErrorHandler()


def safe_execute(component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 fallback_value: Any = None, log_errors: bool = True):
    """Decorator for safe execution of functions with error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    error = CognitiveError(
                        message=f"Error in {func.__name__}: {str(e)}",
                        component=component,
                        severity=severity,
                        original_error=e,
                        context={
                            'function': func.__name__,
                            'args_count': len(args),
                            'kwargs_keys': list(kwargs.keys())
                        }
                    )
                    global_error_handler.log_error(error)
                
                return fallback_value
        return wrapper
    return decorator


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safely perform division with zero-division protection."""
    try:
        if denominator == 0:
            return fallback
        return numerator / denominator
    except (TypeError, ValueError):
        return fallback


def safe_array_operation(operation: Callable, array_data: Any, fallback: Any = None) -> Any:
    """Safely perform array operations with shape validation."""
    try:
        if array_data is None:
            return fallback
            
        # Validate array-like structure
        if hasattr(array_data, '__len__') and len(array_data) == 0:
            return fallback
            
        return operation(array_data)
    except (ValueError, IndexError, TypeError) as e:
        error = CognitiveError(
            message=f"Array operation failed: {str(e)}",
            component="array_operations",
            severity=ErrorSeverity.LOW,
            original_error=e,
            context={'operation': operation.__name__ if hasattr(operation, '__name__') else str(operation)}
        )
        global_error_handler.log_error(error)
        return fallback


def safe_import(module_name: str, fallback_class: type = None):
    """Safely import modules with graceful fallback."""
    try:
        module = __import__(module_name)
        return module
    except ImportError as e:
        error = CognitiveError(
            message=f"Failed to import {module_name}: {str(e)}",
            component="import_system",
            severity=ErrorSeverity.MEDIUM,
            original_error=e,
            context={'module': module_name}
        )
        global_error_handler.log_error(error)
        
        if fallback_class:
            return fallback_class
        return None


class GracefulDegradation:
    """Manages graceful degradation of system capabilities."""
    
    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.degradation_strategies = {}
        self.component_dependencies = {}
        
    def register_degradation_strategy(self, component: str, strategy: Callable):
        """Register a degradation strategy for a component."""
        self.degradation_strategies[component] = strategy
        
    def register_dependency(self, component: str, dependencies: List[str]):
        """Register component dependencies for cascading degradation."""
        self.component_dependencies[component] = dependencies
        
    def apply_degradation(self, component: str):
        """Apply degradation strategy for a failed component."""
        
        # Apply component-specific degradation
        if component in self.degradation_strategies:
            try:
                self.degradation_strategies[component]()
                self.error_handler.logger.info(f"Applied degradation strategy for {component}")
            except Exception as e:
                error = CognitiveError(
                    message=f"Degradation strategy failed for {component}: {str(e)}",
                    component="degradation_system",
                    severity=ErrorSeverity.HIGH,
                    original_error=e
                )
                self.error_handler.log_error(error)
                
        # Check for cascading effects
        self._check_cascading_degradation(component)
        
    def _check_cascading_degradation(self, failed_component: str):
        """Check if other components need degradation due to dependency failure."""
        
        for component, dependencies in self.component_dependencies.items():
            if failed_component in dependencies:
                current_status = self.error_handler.get_component_status(component)
                if current_status == ComponentStatus.HEALTHY:
                    self.error_handler.component_status[component] = ComponentStatus.DEGRADED
                    self.error_handler.logger.warning(
                        f"Component {component} degraded due to dependency failure: {failed_component}"
                    )


def validate_input(data: Any, expected_type: type = None, 
                  expected_shape: Tuple[int, ...] = None,
                  min_value: float = None, max_value: float = None) -> bool:
    """Validate input data with comprehensive checks."""
    
    try:
        # Type validation
        if expected_type and not isinstance(data, expected_type):
            return False
            
        # Shape validation for array-like data
        if expected_shape and hasattr(data, 'shape'):
            if data.shape != expected_shape:
                return False
                
        # Value range validation
        if min_value is not None or max_value is not None:
            if hasattr(data, '__iter__') and not isinstance(data, str):
                # Check all values in iterable
                for value in data:
                    if min_value is not None and value < min_value:
                        return False
                    if max_value is not None and value > max_value:
                        return False
            else:
                # Check single value
                if min_value is not None and data < min_value:
                    return False
                if max_value is not None and data > max_value:
                    return False
                    
        return True
        
    except Exception:
        return False


def create_error_recovery_checkpoint(component: str, state: Dict[str, Any]):
    """Create a checkpoint for error recovery."""
    
    checkpoint = {
        'component': component,
        'state': state,
        'timestamp': datetime.now(),
        'errors_at_checkpoint': len(global_error_handler.error_log)
    }
    
    # Store checkpoint (in practice, this might be persisted)
    checkpoint_id = f"{component}_{int(time.time())}"
    
    global_error_handler.logger.info(f"Created recovery checkpoint: {checkpoint_id}")
    
    return checkpoint_id, checkpoint


def demonstrate_error_handling():
    """Demonstrate the error handling framework."""
    
    print("🛡️  ERROR HANDLING FRAMEWORK DEMONSTRATION")
    print("=" * 60)
    
    # Test safe execution decorator
    @safe_execute(component="test_component", severity=ErrorSeverity.MEDIUM, fallback_value="fallback")
    def test_function_with_error():
        raise ValueError("This is a test error")
        
    @safe_execute(component="test_component", severity=ErrorSeverity.LOW, fallback_value=42)
    def test_function_success():
        return 100
        
    print("🧪 Testing safe execution decorator...")
    result1 = test_function_with_error()
    result2 = test_function_success()
    print(f"   Error function result: {result1}")
    print(f"   Success function result: {result2}")
    
    # Test safe division
    print("\n🧪 Testing safe division...")
    print(f"   10 / 2 = {safe_divide(10, 2)}")
    print(f"   10 / 0 = {safe_divide(10, 0, fallback=999)}")
    
    # Test input validation
    print("\n🧪 Testing input validation...")
    print(f"   Valid int: {validate_input(42, expected_type=int)}")
    print(f"   Invalid type: {validate_input('hello', expected_type=int)}")
    print(f"   Valid range: {validate_input(5, min_value=0, max_value=10)}")
    print(f"   Invalid range: {validate_input(15, min_value=0, max_value=10)}")
    
    # Show error summary
    print("\n📊 Error Summary:")
    summary = global_error_handler.get_error_summary()
    print(f"   Total errors: {summary['total_errors']}")
    print(f"   Component status: {summary['component_status']}")
    print(f"   Severity distribution: {summary['severity_distribution']}")
    
    print("\n✅ Error handling framework demonstration complete!")
    
    return global_error_handler


if __name__ == "__main__":
    error_handler = demonstrate_error_handling()