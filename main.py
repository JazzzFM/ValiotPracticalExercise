#!/usr/bin/env python3
"""
Advanced Manufacturing Scheduler - Main Application

This is a refactored and optimized version of the original slow_scheduler.py,
designed with SOLID principles and modern software engineering practices.

Author: AI Assistant (Claude)
Date: 2024
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.job import Job
from src.config.settings import ConfigurationManager, load_default_configuration
from src.factories.scheduler_factory import (
    default_factory, create_scheduler, recommend_scheduler, get_available_schedulers
)
from src.utils.performance import PerformanceProfiler, ComparisonResult


def create_jobs_from_original_data() -> List[Job]:
    """Create Job instances from the original slow_scheduler.py data."""
    original_times = [
        [5, 3, 4],  # job 0
        [2, 6, 1],
        [4, 2, 5],
        [3, 4, 2],
        [1, 5, 3],
        [6, 1, 4],
        [2, 3, 5],
        [4, 5, 1],
        [3, 1, 6],
        [5, 4, 2]   # job 9
    ]
    
    return [Job(id=i, processing_times=times) for i, times in enumerate(original_times)]


def run_single_strategy(strategy_name: str, jobs: List[Job], num_machines: int,
                       verbose: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Run a single scheduling strategy on the given problem.
    
    Args:
        strategy_name: Name of the strategy to use
        jobs: List of jobs to schedule
        num_machines: Number of machines available
        verbose: Whether to print detailed output
        **kwargs: Additional arguments for the strategy
        
    Returns:
        Dictionary with results and timing information
    """
    if verbose:
        print(f"Running {strategy_name} strategy...")
        print(f"Problem: {len(jobs)} jobs, {num_machines} machines")
    
    try:
        # Create strategy
        strategy = create_scheduler(strategy_name, **kwargs)
        
        # Run optimization with timing
        start_time = time.time()
        result = strategy.find_optimal_schedule(jobs, num_machines, **kwargs)
        execution_time = time.time() - start_time
        
        if verbose:
            print(f"\nResults:")
            print(f"  Strategy: {strategy.name}")
            print(f"  Optimal sequence: {result.job_sequence}")
            print(f"  Makespan: {result.makespan} minutes")
            print(f"  Average utilization: {result.average_utilization:.2%}")
            print(f"  Execution time: {execution_time:.3f} seconds")
            
            # Print detailed schedule
            print(f"\nDetailed Schedule:")
            for line in result.get_schedule_summary():
                print(f"  {line}")
        
        return {
            'strategy_name': strategy.name,
            'job_sequence': result.job_sequence,
            'makespan': result.makespan,
            'utilization': result.average_utilization,
            'execution_time': execution_time,
            'num_jobs': result.num_jobs,
            'num_machines': result.num_machines,
            'success': True
        }
        
    except Exception as e:
        if verbose:
            print(f"Error running {strategy_name}: {e}")
        
        return {
            'strategy_name': strategy_name,
            'error': str(e),
            'success': False
        }


def compare_strategies(jobs: List[Job], num_machines: int, 
                      strategy_names: List[str] = None,
                      num_runs: int = 3, export_results: bool = True) -> ComparisonResult:
    """
    Compare multiple scheduling strategies on the same problem.
    
    Args:
        jobs: List of jobs to schedule
        num_machines: Number of machines available
        strategy_names: List of strategy names to compare (None for auto-selection)
        num_runs: Number of benchmark runs per strategy
        export_results: Whether to export results to files
        
    Returns:
        ComparisonResult with detailed comparison data
    """
    print("="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    # Auto-select strategies if not provided
    if strategy_names is None:
        recommendations = default_factory.get_strategy_recommendations(
            len(jobs), {'max_execution_time': 60}
        )
        strategy_names = recommendations[:4]  # Top 4 recommendations
        
        print(f"Auto-selected strategies: {strategy_names}")
    
    # Prepare strategy configurations
    strategy_configs = []
    for name in strategy_names:
        try:
            strategy = create_scheduler(name)
            strategy_configs.append({
                'strategy': strategy,
                'name': name,
                'kwargs': {}
            })
        except Exception as e:
            print(f"Warning: Could not create strategy '{name}': {e}")
    
    if not strategy_configs:
        raise ValueError("No valid strategies to compare")
    
    # Run comparison
    profiler = PerformanceProfiler(enable_memory_profiling=True)
    results = profiler.compare_strategies(strategy_configs, jobs, num_machines, num_runs)
    
    # Print summary
    profiler.print_comparison_summary(results)
    
    # Print insights
    insights = profiler.get_performance_insights(results)
    print(f"\nPERFORMANCE INSIGHTS:")
    print(f"  Fastest: {insights['fastest_strategy']}")
    print(f"  Best Quality: {insights['best_quality_strategy']}")
    print(f"  Speed Difference: {insights['speedup_ratio']:.1f}x")
    print(f"  Quality Difference: {insights['quality_gap_ratio']:.2f}x")
    
    if insights['recommendations']:
        print(f"\nRecommendations:")
        for rec in insights['recommendations']:
            print(f"  • {rec}")
    
    # Export results if requested
    if export_results:
        timestamp = int(time.time())
        profiler.export_results(results, f"benchmark_results_{timestamp}.json", "json")
        profiler.export_results(results, f"benchmark_results_{timestamp}.csv", "csv")
    
    return results


def demonstrate_improvement():
    """Demonstrate the improvement over the original slow_scheduler.py."""
    print("="*80)
    print("IMPROVEMENT DEMONSTRATION")
    print("="*80)
    
    jobs = create_jobs_from_original_data()
    num_machines = 3
    
    print("Comparing with original brute force approach...")
    
    # Run original-style brute force (but optimized)
    original_result = run_single_strategy("brute_force", jobs, num_machines, 
                                        verbose=False, use_fixed_delays=True)
    
    # Run optimized strategy
    optimized_result = run_single_strategy("optimized_balanced", jobs, num_machines,
                                         verbose=False)
    
    # Compare results
    print(f"\nComparison Results:")
    print(f"{'Metric':<20} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
    print("-" * 70)
    
    if original_result['success'] and optimized_result['success']:
        speedup = original_result['execution_time'] / optimized_result['execution_time']
        quality_ratio = optimized_result['makespan'] / original_result['makespan']
        
        print(f"{'Execution Time':<20} {original_result['execution_time']:<15.3f} "
              f"{optimized_result['execution_time']:<15.3f} {speedup:<15.1f}x faster")
        print(f"{'Makespan':<20} {original_result['makespan']:<15} "
              f"{optimized_result['makespan']:<15} {quality_ratio:<15.2f}x ratio")
        print(f"{'Strategy':<20} {'Brute Force':<15} {'Heuristic':<15} {'Scalable':<15}")
    
    print(f"\nKey Improvements:")
    print(f"  • SOLID Principles: Modular, extensible design")
    print(f"  • Performance: Heuristic algorithms for large problems") 
    print(f"  • Maintainability: Clean separation of concerns")
    print(f"  • Extensibility: Easy to add new scheduling strategies")
    print(f"  • Configuration: External configuration management")
    print(f"  • Testing: Comprehensive unit test coverage")
    print(f"  • Documentation: Detailed documentation and examples")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Manufacturing Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run default strategy
  %(prog)s --strategy brute_force            # Run specific strategy  
  %(prog)s --compare                         # Compare multiple strategies
  %(prog)s --list-strategies                 # List available strategies
  %(prog)s --demonstrate                     # Show improvement over original
        """
    )
    
    parser.add_argument(
        '--strategy', '-s',
        help='Scheduling strategy to use',
        choices=list(get_available_schedulers().keys()),
        default='optimized_balanced'
    )
    
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Compare multiple strategies'
    )
    
    parser.add_argument(
        '--list-strategies', '-l',
        action='store_true',
        help='List all available strategies'
    )
    
    parser.add_argument(
        '--demonstrate', '-d',
        action='store_true',
        help='Demonstrate improvement over original implementation'
    )
    
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3,
        help='Number of benchmark runs for comparison (default: 3)'
    )
    
    parser.add_argument(
        '--config',
        help='Configuration file to use (default: use original problem data)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle list strategies command
    if args.list_strategies:
        print("Available Scheduling Strategies:")
        print("-" * 40)
        
        strategies = get_available_schedulers()
        for name, description in strategies.items():
            print(f"  {name:<30} - {description}")
        
        print(f"\nTotal: {len(strategies)} strategies available")
        return
    
    # Handle demonstrate command  
    if args.demonstrate:
        demonstrate_improvement()
        return
    
    # Load configuration and create jobs
    try:
        if args.config:
            config_manager = ConfigurationManager()
            settings = config_manager.load_from_file(args.config)
            jobs, num_machines = config_manager.get_job_data_for_scheduler()
        else:
            # Use original problem data
            jobs = create_jobs_from_original_data()
            num_machines = 3
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Handle comparison command
    if args.compare:
        try:
            compare_strategies(jobs, num_machines, num_runs=args.num_runs)
        except Exception as e:
            print(f"Error during comparison: {e}")
            sys.exit(1)
        return
    
    # Run single strategy
    try:
        result = run_single_strategy(args.strategy, jobs, num_machines, args.verbose)
        
        if not result['success']:
            print(f"Strategy execution failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error running strategy: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()