import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from factories.scheduler_factory import (
    SchedulerFactory, create_scheduler, get_available_schedulers, recommend_scheduler
)
from strategies.base import SchedulerStrategy


class MockStrategy(SchedulerStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name="MockStrategy"):
        self.name = name
    
    def find_optimal_schedule(self, jobs, num_machines, **kwargs):
        """Mock implementation."""
        from models.schedule import Schedule
        return Schedule(
            job_sequence=[0, 1, 2],
            makespan=10,
            num_jobs=3,
            num_machines=num_machines
        )
    
    def get_strategy_info(self):
        """Mock strategy info."""
        return {
            'name': self.name,
            'type': 'mock',
            'time_complexity': 'O(1)',
            'description': 'Mock strategy for testing'
        }


class TestSchedulerFactory(unittest.TestCase):
    """Test cases for SchedulerFactory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = SchedulerFactory()
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        strategies = self.factory.get_available_strategies()
        
        # Should have default strategies
        self.assertGreater(len(strategies), 0)
        
        # Should have brute force
        self.assertIn('brute_force', strategies)
        
        # Should have optimized strategies
        self.assertIn('optimized_balanced', strategies)
        self.assertIn('optimized_shortest_processing_time', strategies)
    
    def test_create_strategy_brute_force(self):
        """Test creating brute force strategy."""
        strategy = self.factory.create_strategy('brute_force')
        
        self.assertIsNotNone(strategy)
        info = strategy.get_strategy_info()
        self.assertEqual(info['type'], 'exhaustive_search')
    
    def test_create_strategy_optimized(self):
        """Test creating optimized strategy."""
        strategy = self.factory.create_strategy('optimized_balanced')
        
        self.assertIsNotNone(strategy)
        info = strategy.get_strategy_info()
        self.assertEqual(info['type'], 'heuristic_optimization')
    
    def test_create_unknown_strategy(self):
        """Test creating unknown strategy raises error."""
        with self.assertRaises(KeyError) as cm:
            self.factory.create_strategy('nonexistent_strategy')
        self.assertIn('not found', str(cm.exception))
    
    def test_register_custom_strategy(self):
        """Test registering custom strategy."""
        # Register mock strategy
        self.factory.register_strategy('mock', MockStrategy)
        
        # Should be available
        strategies = self.factory.get_available_strategies()
        self.assertIn('mock', strategies)
        
        # Should be creatable
        strategy = self.factory.create_strategy('mock')
        self.assertIsInstance(strategy, MockStrategy)
    
    def test_register_duplicate_strategy(self):
        """Test registering duplicate strategy raises error."""
        self.factory.register_strategy('mock', MockStrategy)
        
        with self.assertRaises(ValueError) as cm:
            self.factory.register_strategy('mock', MockStrategy)
        self.assertIn('already registered', str(cm.exception))
    
    def test_unregister_strategy(self):
        """Test unregistering strategy."""
        # Register and then unregister
        self.factory.register_strategy('mock', MockStrategy)
        self.assertIn('mock', self.factory.get_available_strategies())
        
        self.factory.unregister_strategy('mock')
        self.assertNotIn('mock', self.factory.get_available_strategies())
    
    def test_unregister_nonexistent_strategy(self):
        """Test unregistering nonexistent strategy raises error."""
        with self.assertRaises(KeyError):
            self.factory.unregister_strategy('nonexistent')
    
    def test_get_strategy_recommendations(self):
        """Test getting strategy recommendations."""
        # Small problem
        recommendations = self.factory.get_strategy_recommendations(5)
        self.assertGreater(len(recommendations), 0)
        self.assertIn('brute_force', recommendations)  # Should recommend brute force for small problems
        
        # Large problem
        recommendations = self.factory.get_strategy_recommendations(50)
        self.assertGreater(len(recommendations), 0)
        # Should not recommend brute force for large problems
        self.assertNotIn('brute_force', recommendations)
    
    def test_compare_strategies(self):
        """Test comparing strategies."""
        strategy_names = ['brute_force', 'optimized_balanced']
        comparison = self.factory.compare_strategies(strategy_names, 5)
        
        self.assertEqual(len(comparison), 2)
        
        for name in strategy_names:
            self.assertIn(name, comparison)
            self.assertIn('info', comparison[name])
            self.assertIn('suitable_for_size', comparison[name])
    
    def test_create_best_strategy_for_problem(self):
        """Test creating best strategy for problem."""
        # Small problem
        strategy = self.factory.create_best_strategy_for_problem(5)
        self.assertIsNotNone(strategy)
        
        # Large problem  
        strategy = self.factory.create_best_strategy_for_problem(100)
        self.assertIsNotNone(strategy)
    
    def test_get_factory_info(self):
        """Test getting factory information."""
        info = self.factory.get_factory_info()
        
        self.assertIn('total_strategies', info)
        self.assertIn('available_strategies', info)
        self.assertIn('strategy_types', info)
        self.assertIn('supported_features', info)
        
        self.assertGreater(info['total_strategies'], 0)
        self.assertIsInstance(info['available_strategies'], list)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_create_scheduler(self):
        """Test create_scheduler convenience function."""
        scheduler = create_scheduler('brute_force')
        self.assertIsNotNone(scheduler)
    
    def test_get_available_schedulers(self):
        """Test get_available_schedulers convenience function."""
        schedulers = get_available_schedulers()
        self.assertGreater(len(schedulers), 0)
        self.assertIn('brute_force', schedulers)
    
    def test_recommend_scheduler(self):
        """Test recommend_scheduler convenience function."""
        scheduler = recommend_scheduler(5)
        self.assertIsNotNone(scheduler)


if __name__ == '__main__':
    unittest.main()