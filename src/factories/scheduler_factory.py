from typing import Dict, Type, List, Optional
from strategies.base import SchedulerStrategy
from strategies.brute_force import BruteForceScheduler
from strategies.optimized import OptimizedScheduler
from strategies.ml_strategy import MLEnhancedScheduler, EnsembleMLScheduler, ReinforcementLearningScheduler


class SchedulerFactory:
    """
    Factory class for creating scheduling strategy instances.
    
    This class implements the Factory pattern and follows the Open/Closed Principle
    by allowing new strategies to be registered without modifying existing code.
    """
    
    def __init__(self):
        """Initialize the factory with default strategies."""
        self._strategies: Dict[str, Type[SchedulerStrategy]] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register all built-in scheduling strategies."""
        # Register brute force strategy
        self.register_strategy("brute_force", BruteForceScheduler)
        
        # Register optimized strategies with different heuristics
        for heuristic in ["shortest_processing_time", "longest_processing_time", 
                         "balanced", "random_search", "greedy_makespan"]:
            strategy_name = f"optimized_{heuristic}"
            # Create a lambda that captures the heuristic value
            self.register_strategy(
                strategy_name, 
                lambda h=heuristic: OptimizedScheduler(h)
            )
        
        # Register ML-enhanced strategies
        ml_strategies = [
            "predictive_optimization", "quality_focused", 
            "cost_focused", "energy_focused"
        ]
        for ml_strategy in ml_strategies:
            strategy_name = f"ml_{ml_strategy}"
            self.register_strategy(
                strategy_name,
                lambda s=ml_strategy: MLEnhancedScheduler(base_strategy=s)
            )
        
        # Register ensemble and RL strategies
        self.register_strategy("ensemble_ml", EnsembleMLScheduler)
        self.register_strategy("reinforcement_learning", ReinforcementLearningScheduler)
    
    def register_strategy(self, name: str, strategy_class_or_factory):
        """
        Register a new scheduling strategy.
        
        Args:
            name: Unique name for the strategy
            strategy_class_or_factory: Strategy class or factory function
            
        Raises:
            ValueError: If strategy name already exists
        """
        if name in self._strategies:
            raise ValueError(f"Strategy '{name}' is already registered")
        
        self._strategies[name] = strategy_class_or_factory
    
    def unregister_strategy(self, name: str):
        """
        Unregister a scheduling strategy.
        
        Args:
            name: Name of the strategy to remove
            
        Raises:
            KeyError: If strategy name doesn't exist
        """
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' is not registered")
        
        del self._strategies[name]
    
    def create_strategy(self, name: str, **kwargs) -> SchedulerStrategy:
        """
        Create an instance of the specified scheduling strategy.
        
        Args:
            name: Name of the strategy to create
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            SchedulerStrategy instance
            
        Raises:
            KeyError: If strategy name is not registered
            TypeError: If strategy creation fails
        """
        if name not in self._strategies:
            available = list(self.get_available_strategies().keys())
            raise KeyError(f"Strategy '{name}' not found. Available strategies: {available}")
        
        strategy_factory = self._strategies[name]
        
        try:
            # Handle both class constructors and factory functions
            if callable(strategy_factory):
                if hasattr(strategy_factory, '__name__') and strategy_factory.__name__ == '<lambda>':
                    # It's a lambda factory function
                    return strategy_factory()
                else:
                    # It's a class constructor
                    return strategy_factory(**kwargs)
            else:
                raise TypeError(f"Invalid strategy factory for '{name}'")
                
        except Exception as e:
            raise TypeError(f"Failed to create strategy '{name}': {e}")
    
    def get_available_strategies(self) -> Dict[str, str]:
        """
        Get information about all available strategies.
        
        Returns:
            Dictionary mapping strategy names to descriptions
        """
        strategies_info = {}
        
        for name in self._strategies:
            try:
                # Create a temporary instance to get info
                strategy = self.create_strategy(name)
                info = strategy.get_strategy_info()
                strategies_info[name] = info.get('name', name)
            except Exception:
                strategies_info[name] = f"Strategy: {name}"
        
        return strategies_info
    
    def get_strategy_recommendations(self, num_jobs: int, 
                                   performance_requirements: Optional[Dict] = None) -> List[str]:
        """
        Get recommended strategies based on problem characteristics.
        
        Args:
            num_jobs: Number of jobs to schedule
            performance_requirements: Optional performance requirements
            
        Returns:
            List of recommended strategy names, ordered by suitability
        """
        recommendations = []
        
        # Performance requirements defaults
        if performance_requirements is None:
            performance_requirements = {}
        
        max_time = performance_requirements.get('max_execution_time', float('inf'))
        requires_optimal = performance_requirements.get('requires_optimal', False)
        
        # Brute force recommendations
        if num_jobs <= 8 and (requires_optimal or max_time > 60):
            recommendations.append("brute_force")
        elif num_jobs <= 10 and requires_optimal:
            recommendations.append("brute_force")
        
        # Optimized strategy recommendations
        if num_jobs <= 20 or max_time < 10:
            recommendations.extend([
                "optimized_shortest_processing_time",
                "optimized_longest_processing_time", 
                "optimized_balanced"
            ])
        
        if max_time > 30:
            recommendations.extend([
                "optimized_greedy_makespan",
                "optimized_random_search"
            ])
        
        # Default fallback
        if not recommendations:
            recommendations = [
                "optimized_balanced",
                "optimized_shortest_processing_time"
            ]
        
        return recommendations
    
    def compare_strategies(self, strategy_names: List[str], num_jobs: int) -> Dict[str, Dict]:
        """
        Compare multiple strategies for a given problem size.
        
        Args:
            strategy_names: List of strategy names to compare
            num_jobs: Number of jobs for comparison
            
        Returns:
            Dictionary with comparison information for each strategy
        """
        comparison = {}
        
        for name in strategy_names:
            try:
                strategy = self.create_strategy(name)
                info = strategy.get_strategy_info()
                
                comparison[name] = {
                    'info': info,
                    'suitable_for_size': self._is_suitable_for_size(info, num_jobs),
                    'estimated_performance': self._estimate_performance(info, num_jobs)
                }
                
            except Exception as e:
                comparison[name] = {
                    'error': str(e),
                    'suitable_for_size': False,
                    'estimated_performance': 'unknown'
                }
        
        return comparison
    
    def _is_suitable_for_size(self, strategy_info: Dict, num_jobs: int) -> bool:
        """Check if a strategy is suitable for the given problem size."""
        strategy_type = strategy_info.get('type', '')
        
        if strategy_type == 'exhaustive_search':
            # Brute force becomes impractical quickly
            return num_jobs <= 10
        elif strategy_type == 'heuristic_optimization':
            # Heuristics can handle larger problems
            return True
        else:
            # Conservative default
            return num_jobs <= 50
    
    def _estimate_performance(self, strategy_info: Dict, num_jobs: int) -> str:
        """Estimate performance category for the strategy and problem size."""
        complexity = strategy_info.get('time_complexity', '')
        strategy_type = strategy_info.get('type', '')
        
        if 'n!' in complexity:
            if num_jobs <= 8:
                return 'fast'
            elif num_jobs <= 10:
                return 'moderate'
            else:
                return 'very_slow'
        elif 'n²' in complexity:
            if num_jobs <= 50:
                return 'fast'
            elif num_jobs <= 200:
                return 'moderate'
            else:
                return 'slow'
        elif 'n log n' in complexity or strategy_type == 'heuristic_optimization':
            if num_jobs <= 1000:
                return 'fast'
            else:
                return 'moderate'
        else:
            return 'unknown'
    
    def create_best_strategy_for_problem(self, num_jobs: int, 
                                       performance_requirements: Optional[Dict] = None) -> SchedulerStrategy:
        """
        Create the best strategy for a specific problem.
        
        Args:
            num_jobs: Number of jobs to schedule
            performance_requirements: Optional performance requirements
            
        Returns:
            SchedulerStrategy instance best suited for the problem
        """
        recommendations = self.get_strategy_recommendations(num_jobs, performance_requirements)
        
        if not recommendations:
            # Fallback to balanced heuristic
            return self.create_strategy("optimized_balanced")
        
        # Return the top recommendation
        return self.create_strategy(recommendations[0])
    
    def get_factory_info(self) -> Dict[str, any]:
        """
        Get information about the factory and registered strategies.
        
        Returns:
            Dictionary with factory information
        """
        return {
            'total_strategies': len(self._strategies),
            'available_strategies': list(self._strategies.keys()),
            'strategy_types': self._get_strategy_types(),
            'supported_features': [
                'Dynamic strategy registration',
                'Automatic strategy recommendation', 
                'Performance-based strategy selection',
                'Strategy comparison and analysis'
            ]
        }
    
    def _get_strategy_types(self) -> Dict[str, int]:
        """Get count of strategies by type."""
        types = {}
        
        for name in self._strategies:
            try:
                strategy = self.create_strategy(name)
                info = strategy.get_strategy_info()
                strategy_type = info.get('type', 'unknown')
                types[strategy_type] = types.get(strategy_type, 0) + 1
            except Exception:
                types['unknown'] = types.get('unknown', 0) + 1
        
        return types
    
    def __str__(self) -> str:
        return f"SchedulerFactory({len(self._strategies)} strategies)"
    
    def __repr__(self) -> str:
        strategies = list(self._strategies.keys())
        return f"SchedulerFactory(strategies={strategies})"


# Global factory instance for easy access
default_factory = SchedulerFactory()


def create_scheduler(strategy_name: str, **kwargs) -> SchedulerStrategy:
    """
    Convenience function to create a scheduler using the default factory.
    
    Args:
        strategy_name: Name of the strategy to create
        **kwargs: Additional parameters for strategy initialization
        
    Returns:
        SchedulerStrategy instance
    """
    return default_factory.create_strategy(strategy_name, **kwargs)


def get_available_schedulers() -> Dict[str, str]:
    """
    Convenience function to get available schedulers from the default factory.
    
    Returns:
        Dictionary mapping strategy names to descriptions
    """
    return default_factory.get_available_strategies()


def recommend_scheduler(num_jobs: int, performance_requirements: Optional[Dict] = None) -> SchedulerStrategy:
    """
    Convenience function to get the best scheduler for a problem.
    
    Args:
        num_jobs: Number of jobs to schedule
        performance_requirements: Optional performance requirements
        
    Returns:
        SchedulerStrategy instance best suited for the problem
    """
    return default_factory.create_best_strategy_for_problem(num_jobs, performance_requirements)