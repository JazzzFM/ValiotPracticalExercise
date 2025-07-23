import time
import tracemalloc
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import statistics
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Contains results from a single benchmark run."""
    strategy_name: str
    execution_time: float
    memory_peak: Optional[float]  # MB
    makespan: int
    job_sequence: List[int]
    num_jobs: int
    num_machines: int
    iterations: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'execution_time': self.execution_time,
            'memory_peak': self.memory_peak,
            'makespan': self.makespan,
            'job_sequence': self.job_sequence,
            'num_jobs': self.num_jobs,
            'num_machines': self.num_machines,
            'iterations': self.iterations
        }


@dataclass
class ComparisonResult:
    """Contains results from comparing multiple strategies."""
    problem_size: Dict[str, int]
    benchmark_results: List[BenchmarkResult]
    statistics: Dict[str, Dict[str, float]]
    best_strategy: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'problem_size': self.problem_size,
            'benchmark_results': [result.to_dict() for result in self.benchmark_results],
            'statistics': self.statistics,
            'best_strategy': self.best_strategy
        }


class PerformanceProfiler:
    """
    Performance profiling utility for scheduling strategies.
    
    Provides benchmarking, memory profiling, and comparison capabilities.
    """
    
    def __init__(self, enable_memory_profiling: bool = True):
        """
        Initialize the profiler.
        
        Args:
            enable_memory_profiling: Whether to enable memory usage tracking
        """
        self.enable_memory_profiling = enable_memory_profiling
        self.results_history: List[BenchmarkResult] = []
    
    def benchmark_strategy(self, strategy_func: Callable, strategy_name: str,
                          *args, **kwargs) -> BenchmarkResult:
        """
        Benchmark a single strategy execution.
        
        Args:
            strategy_func: Function to benchmark (should return ScheduleResult)
            strategy_name: Name of the strategy being benchmarked
            *args, **kwargs: Arguments to pass to strategy_func
            
        Returns:
            BenchmarkResult with timing and memory information
        """
        # Start memory profiling if enabled
        if self.enable_memory_profiling:
            tracemalloc.start()
        
        # Benchmark execution
        start_time = time.perf_counter()
        
        try:
            result = strategy_func(*args, **kwargs)
        except Exception as e:
            if self.enable_memory_profiling:
                tracemalloc.stop()
            raise RuntimeError(f"Benchmark failed for {strategy_name}: {e}")
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Get memory usage
        memory_peak = None
        if self.enable_memory_profiling:
            current, peak = tracemalloc.get_traced_memory()
            memory_peak = peak / 1024 / 1024  # Convert to MB
            tracemalloc.stop()
        
        # Extract information from result
        if hasattr(result, 'schedule_result'):
            # It's an OptimizationResult
            schedule_result = result.schedule_result
            iterations = result.iterations_performed
        else:
            # It's a ScheduleResult
            schedule_result = result
            iterations = None
        
        benchmark_result = BenchmarkResult(
            strategy_name=strategy_name,
            execution_time=execution_time,
            memory_peak=memory_peak,
            makespan=schedule_result.makespan,
            job_sequence=schedule_result.job_sequence,
            num_jobs=schedule_result.num_jobs,
            num_machines=schedule_result.num_machines,
            iterations=iterations
        )
        
        self.results_history.append(benchmark_result)
        return benchmark_result
    
    def benchmark_multiple_runs(self, strategy_func: Callable, strategy_name: str,
                              num_runs: int = 5, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark multiple runs of the same strategy.
        
        Args:
            strategy_func: Function to benchmark
            strategy_name: Name of the strategy
            num_runs: Number of benchmark runs
            *args, **kwargs: Arguments to pass to strategy_func
            
        Returns:
            Dictionary with statistical results
        """
        results = []
        
        print(f"Running {num_runs} benchmark iterations for {strategy_name}...")
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...")
            result = self.benchmark_strategy(strategy_func, f"{strategy_name}_run_{i+1}", 
                                           *args, **kwargs)
            results.append(result)
        
        # Calculate statistics
        execution_times = [r.execution_time for r in results]
        makespans = [r.makespan for r in results]
        memory_peaks = [r.memory_peak for r in results if r.memory_peak is not None]
        
        statistics_dict = {
            'execution_time': {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'min': min(execution_times),
                'max': max(execution_times)
            },
            'makespan': {
                'mean': statistics.mean(makespans),
                'median': statistics.median(makespans),
                'stdev': statistics.stdev(makespans) if len(makespans) > 1 else 0,
                'min': min(makespans),
                'max': max(makespans)
            }
        }
        
        if memory_peaks:
            statistics_dict['memory_peak_mb'] = {
                'mean': statistics.mean(memory_peaks),
                'median': statistics.median(memory_peaks),
                'stdev': statistics.stdev(memory_peaks) if len(memory_peaks) > 1 else 0,
                'min': min(memory_peaks),
                'max': max(memory_peaks)
            }
        
        return {
            'strategy_name': strategy_name,
            'num_runs': num_runs,
            'individual_results': [r.to_dict() for r in results],
            'statistics': statistics_dict
        }
    
    def compare_strategies(self, strategy_configs: List[Dict[str, Any]], 
                          jobs, num_machines, num_runs: int = 3) -> ComparisonResult:
        """
        Compare multiple strategies on the same problem.
        
        Args:
            strategy_configs: List of dicts with 'strategy', 'name', and optional 'kwargs'
            jobs: List of Job instances
            num_machines: Number of machines
            num_runs: Number of runs per strategy
            
        Returns:
            ComparisonResult with detailed comparison
        """
        print(f"Comparing {len(strategy_configs)} strategies on {len(jobs)} jobs, {num_machines} machines")
        
        all_results = []
        strategy_statistics = {}
        
        for config in strategy_configs:
            strategy = config['strategy']
            name = config['name']
            kwargs = config.get('kwargs', {})
            
            print(f"\nBenchmarking {name}...")
            
            # Run multiple benchmarks for this strategy
            runs_data = self.benchmark_multiple_runs(
                lambda: strategy.find_optimal_schedule(jobs, num_machines, **kwargs),
                name, 
                num_runs
            )
            
            # Store the best result from all runs
            best_result = min(runs_data['individual_results'], 
                            key=lambda x: x['makespan'])
            best_benchmark = BenchmarkResult(**best_result)
            all_results.append(best_benchmark)
            
            # Store statistics
            strategy_statistics[name] = runs_data['statistics']
        
        # Determine best strategy
        best_result = min(all_results, key=lambda x: x.makespan)
        best_strategy = best_result.strategy_name
        
        return ComparisonResult(
            problem_size={'num_jobs': len(jobs), 'num_machines': num_machines},
            benchmark_results=all_results,
            statistics=strategy_statistics,
            best_strategy=best_strategy
        )
    
    def export_results(self, results: ComparisonResult, filename: str, 
                      format: str = "json"):
        """
        Export comparison results to file.
        
        Args:
            results: ComparisonResult to export
            filename: Output filename
            format: Export format ("json", "csv")
        """
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
        
        elif format.lower() == "csv":
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'strategy_name', 'execution_time', 'memory_peak_mb',
                    'makespan', 'num_jobs', 'num_machines', 'iterations'
                ])
                
                # Write data
                for result in results.benchmark_results:
                    writer.writerow([
                        result.strategy_name,
                        result.execution_time,
                        result.memory_peak,
                        result.makespan,
                        result.num_jobs,
                        result.num_machines,
                        result.iterations
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"Results exported to: {output_path}")
    
    def print_comparison_summary(self, results: ComparisonResult):
        """
        Print a formatted summary of comparison results.
        
        Args:
            results: ComparisonResult to summarize
        """
        print("\n" + "="*80)
        print(f"PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        print(f"Problem Size: {results.problem_size['num_jobs']} jobs, "
              f"{results.problem_size['num_machines']} machines")
        print(f"Best Strategy: {results.best_strategy}")
        print()
        
        # Sort results by makespan for display
        sorted_results = sorted(results.benchmark_results, key=lambda x: x.makespan)
        
        print(f"{'Strategy':<25} {'Time (s)':<10} {'Memory (MB)':<12} {'Makespan':<10} {'Quality':<10}")
        print("-" * 80)
        
        best_makespan = sorted_results[0].makespan
        
        for result in sorted_results:
            quality_ratio = result.makespan / best_makespan
            quality_str = f"{quality_ratio:.2f}x"
            
            memory_str = f"{result.memory_peak:.1f}" if result.memory_peak else "N/A"
            
            print(f"{result.strategy_name:<25} {result.execution_time:<10.3f} "
                  f"{memory_str:<12} {result.makespan:<10} {quality_str:<10}")
        
        print("\n" + "="*80)
    
    def get_performance_insights(self, results: ComparisonResult) -> Dict[str, Any]:
        """
        Generate performance insights from comparison results.
        
        Args:
            results: ComparisonResult to analyze
            
        Returns:
            Dictionary with insights and recommendations
        """
        sorted_by_time = sorted(results.benchmark_results, key=lambda x: x.execution_time)
        sorted_by_quality = sorted(results.benchmark_results, key=lambda x: x.makespan)
        
        fastest = sorted_by_time[0]
        slowest = sorted_by_time[-1]
        best_quality = sorted_by_quality[0]
        worst_quality = sorted_by_quality[-1]
        
        speedup_ratio = slowest.execution_time / fastest.execution_time
        quality_gap = worst_quality.makespan / best_quality.makespan
        
        insights = {
            'fastest_strategy': fastest.strategy_name,
            'slowest_strategy': slowest.strategy_name,
            'best_quality_strategy': best_quality.strategy_name,
            'worst_quality_strategy': worst_quality.strategy_name,
            'speedup_ratio': speedup_ratio,
            'quality_gap_ratio': quality_gap,
            'recommendations': []
        }
        
        # Generate recommendations
        if speedup_ratio > 10:
            insights['recommendations'].append(
                f"Large performance difference detected. {fastest.strategy_name} is "
                f"{speedup_ratio:.1f}x faster than {slowest.strategy_name}."
            )
        
        if quality_gap > 1.1:
            insights['recommendations'].append(
                f"Quality varies significantly. {best_quality.strategy_name} produces "
                f"{((quality_gap - 1) * 100):.1f}% better solutions than {worst_quality.strategy_name}."
            )
        
        if fastest.strategy_name == best_quality.strategy_name:
            insights['recommendations'].append(
                f"{fastest.strategy_name} offers the best combination of speed and quality."
            )
        
        return insights
    
    def clear_history(self):
        """Clear the results history."""
        self.results_history.clear()
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all benchmarked results in history.
        
        Returns:
            Dictionary with historical performance data
        """
        if not self.results_history:
            return {'total_runs': 0, 'strategies_tested': 0}
        
        strategies = set(result.strategy_name for result in self.results_history)
        execution_times = [result.execution_time for result in self.results_history]
        
        return {
            'total_runs': len(self.results_history),
            'strategies_tested': len(strategies),
            'total_execution_time': sum(execution_times),
            'average_execution_time': statistics.mean(execution_times),
            'strategies': list(strategies)
        }