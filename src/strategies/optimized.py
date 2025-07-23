import time
import random
from typing import List, Dict, Optional, Tuple
from strategies.base import SchedulerStrategy, OptimizationResult
from models.job import Job
from models.schedule import ScheduleResult


class OptimizedScheduler(SchedulerStrategy):
    """
    Optimized scheduling strategy using heuristic approaches.
    
    This strategy combines multiple heuristics to efficiently find good solutions
    without evaluating all possible permutations. Suitable for larger job sets.
    """
    
    def __init__(self, heuristic: str = "shortest_processing_time"):
        """
        Initialize optimized scheduler with specified heuristic.
        
        Args:
            heuristic: The heuristic to use ("shortest_processing_time", 
                      "longest_processing_time", "balanced", "random_search")
        """
        super().__init__(f"Optimized Heuristic ({heuristic})")
        self.heuristic = heuristic
        self.available_heuristics = {
            "shortest_processing_time": self._shortest_processing_time_heuristic,
            "longest_processing_time": self._longest_processing_time_heuristic,
            "balanced": self._balanced_heuristic,
            "random_search": self._random_search_heuristic,
            "greedy_makespan": self._greedy_makespan_heuristic
        }
        
        if heuristic not in self.available_heuristics:
            raise ValueError(f"Unknown heuristic: {heuristic}. Available: {list(self.available_heuristics.keys())}")
    
    def find_optimal_schedule(self, jobs: List[Job], num_machines: int,
                            max_iterations: int = 1000,
                            improvement_threshold: float = 0.01,
                            time_limit: float = 60.0,
                            **kwargs) -> ScheduleResult:
        """
        Find optimal schedule using the selected heuristic.
        
        Args:
            jobs: List of jobs to schedule
            num_machines: Number of available machines
            max_iterations: Maximum iterations for iterative heuristics
            improvement_threshold: Minimum improvement ratio to continue
            time_limit: Maximum time to spend searching (seconds)
            **kwargs: Additional heuristic-specific parameters
            
        Returns:
            ScheduleResult with the best schedule found
        """
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(jobs, num_machines)
        
        # Create schedule manager
        schedule_manager = self.create_schedule_manager(jobs, num_machines)
        
        # Apply selected heuristic
        heuristic_func = self.available_heuristics[self.heuristic]
        
        print(f"Running {self.name} with {len(jobs)} jobs...")
        
        best_result = heuristic_func(
            jobs, schedule_manager, max_iterations, 
            improvement_threshold, time_limit, **kwargs
        )
        
        execution_time = time.time() - start_time
        print(f"Optimization completed in {execution_time:.3f} seconds")
        print(f"Best makespan found: {best_result.makespan}")
        
        return best_result
    
    def _shortest_processing_time_heuristic(self, jobs: List[Job], schedule_manager,
                                          max_iterations: int, improvement_threshold: float,
                                          time_limit: float, **kwargs) -> ScheduleResult:
        """Sort jobs by shortest total processing time first."""
        job_times = [(job.id, job.get_total_processing_time()) for job in jobs]
        job_times.sort(key=lambda x: x[1])  # Sort by processing time
        
        job_sequence = [job_id for job_id, _ in job_times]
        return schedule_manager.execute_sequence(job_sequence)
    
    def _longest_processing_time_heuristic(self, jobs: List[Job], schedule_manager,
                                         max_iterations: int, improvement_threshold: float,
                                         time_limit: float, **kwargs) -> ScheduleResult:
        """Sort jobs by longest total processing time first."""
        job_times = [(job.id, job.get_total_processing_time()) for job in jobs]
        job_times.sort(key=lambda x: x[1], reverse=True)  # Sort by processing time (descending)
        
        job_sequence = [job_id for job_id, _ in job_times]
        return schedule_manager.execute_sequence(job_sequence)
    
    def _balanced_heuristic(self, jobs: List[Job], schedule_manager,
                          max_iterations: int, improvement_threshold: float,
                          time_limit: float, **kwargs) -> ScheduleResult:
        """Balance job ordering considering both total time and machine distribution."""
        job_metrics = []
        
        for job in jobs:
            total_time = job.get_total_processing_time()
            time_variance = self._calculate_variance(job.processing_times)
            # Lower variance means more balanced across machines
            balance_score = total_time / (1 + time_variance)  # Favor balanced jobs
            job_metrics.append((job.id, balance_score))
        
        # Sort by balance score (higher is better for this metric)
        job_metrics.sort(key=lambda x: x[1], reverse=True)
        job_sequence = [job_id for job_id, _ in job_metrics]
        
        return schedule_manager.execute_sequence(job_sequence)
    
    def _random_search_heuristic(self, jobs: List[Job], schedule_manager,
                               max_iterations: int, improvement_threshold: float,
                               time_limit: float, **kwargs) -> ScheduleResult:
        """Use random search with local improvements."""
        start_time = time.time()
        job_ids = [job.id for job in jobs]
        
        # Start with a random sequence
        best_sequence = job_ids.copy()
        random.shuffle(best_sequence)
        best_result = schedule_manager.execute_sequence(best_sequence)
        best_makespan = best_result.makespan
        
        iterations = 0
        improvements = 0
        
        while (iterations < max_iterations and 
               time.time() - start_time < time_limit):
            
            # Generate neighbor by swapping two random jobs
            new_sequence = best_sequence.copy()
            i, j = random.sample(range(len(new_sequence)), 2)
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
            
            try:
                result = schedule_manager.execute_sequence(new_sequence)
                
                # Accept improvement or occasionally accept worse solution (simulated annealing)
                improvement_ratio = (best_makespan - result.makespan) / best_makespan
                temperature = max(0.1, 1.0 - (iterations / max_iterations))
                accept_worse = random.random() < temperature * 0.1
                
                if result.makespan < best_makespan or accept_worse:
                    if result.makespan < best_makespan:
                        improvements += 1
                        print(f"Iteration {iterations}: New best makespan {result.makespan} "
                              f"(improvement: {improvement_ratio:.3%})")
                    
                    best_sequence = new_sequence
                    best_result = result
                    best_makespan = result.makespan
                
            except Exception as e:
                print(f"Error in iteration {iterations}: {e}")
            
            iterations += 1
            
            # Progress reporting
            if iterations % max(1, max_iterations // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {iterations}/{max_iterations} iterations, "
                      f"{improvements} improvements, {elapsed:.1f}s elapsed")
        
        print(f"Random search completed: {iterations} iterations, {improvements} improvements")
        return best_result
    
    def _greedy_makespan_heuristic(self, jobs: List[Job], schedule_manager,
                                 max_iterations: int, improvement_threshold: float,
                                 time_limit: float, **kwargs) -> ScheduleResult:
        """Greedy approach: always select job that minimizes current makespan."""
        remaining_jobs = set(job.id for job in jobs)
        sequence = []
        
        # Generate initial IoT delays for consistency
        iot_delays = {}
        for job in jobs:
            iot_delays[job.id] = job.generate_iot_delays()
        
        while remaining_jobs:
            best_job = None
            best_makespan = float('inf')
            
            # Try each remaining job and pick the one that gives best makespan
            for job_id in remaining_jobs:
                test_sequence = sequence + [job_id]
                try:
                    result = schedule_manager.execute_sequence(test_sequence, iot_delays)
                    if result.makespan < best_makespan:
                        best_makespan = result.makespan
                        best_job = job_id
                except Exception:
                    continue
            
            if best_job is None:
                # Fallback: pick any remaining job
                best_job = next(iter(remaining_jobs))
            
            sequence.append(best_job)
            remaining_jobs.remove(best_job)
        
        return schedule_manager.execute_sequence(sequence, iot_delays)
    
    def _calculate_variance(self, values: List[int]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def find_optimal_schedule_with_metadata(self, jobs: List[Job], num_machines: int,
                                          **kwargs) -> OptimizationResult:
        """
        Find optimal schedule and return detailed optimization results.
        
        Args:
            jobs: List of jobs to schedule
            num_machines: Number of available machines
            **kwargs: Additional parameters passed to find_optimal_schedule
            
        Returns:
            OptimizationResult with detailed optimization information
        """
        start_time = time.time()
        
        schedule_result = self.find_optimal_schedule(jobs, num_machines, **kwargs)
        
        execution_time = time.time() - start_time
        
        # Estimate iterations based on heuristic type
        if self.heuristic == "random_search":
            iterations = kwargs.get('max_iterations', 1000)
        else:
            iterations = len(jobs)  # Deterministic heuristics
        
        convergence_info = {
            'algorithm': self.heuristic,
            'guaranteed_optimal': False,
            'heuristic_type': self._get_heuristic_type(),
            'search_space_coverage': self._estimate_coverage(len(jobs), iterations)
        }
        
        return OptimizationResult(
            schedule_result=schedule_result,
            strategy_name=self.name,
            execution_time=execution_time,
            iterations_performed=iterations,
            convergence_info=convergence_info
        )
    
    def _get_heuristic_type(self) -> str:
        """Get the type of heuristic being used."""
        if self.heuristic in ["shortest_processing_time", "longest_processing_time"]:
            return "constructive"
        elif self.heuristic == "random_search":
            return "metaheuristic"
        elif self.heuristic == "greedy_makespan":
            return "greedy"
        else:
            return "hybrid"
    
    def _estimate_coverage(self, num_jobs: int, iterations: int) -> float:
        """Estimate what fraction of search space was covered."""
        total_permutations = 1
        for i in range(1, num_jobs + 1):
            total_permutations *= i
            if total_permutations > 1e15:  # Avoid overflow
                total_permutations = 1e15
                break
        
        return min(1.0, iterations / total_permutations)
    
    def get_strategy_info(self) -> Dict[str, any]:
        """
        Get information about the optimized strategy.
        
        Returns:
            Dictionary containing strategy metadata
        """
        return {
            'name': self.name,
            'heuristic': self.heuristic,
            'type': 'heuristic_optimization',
            'time_complexity': self._get_time_complexity(),
            'space_complexity': 'O(n)',
            'guarantees_optimal': False,
            'suitable_for_jobs': 'Any number',
            'available_heuristics': list(self.available_heuristics.keys()),
            'characteristics': self._get_heuristic_characteristics()
        }
    
    def _get_time_complexity(self) -> str:
        """Get time complexity for current heuristic."""
        complexities = {
            "shortest_processing_time": "O(n log n)",
            "longest_processing_time": "O(n log n)",
            "balanced": "O(n log n)",
            "random_search": "O(k * n)",  # k = iterations
            "greedy_makespan": "O(n² * m)"  # n = jobs, m = machines
        }
        return complexities.get(self.heuristic, "O(n log n)")
    
    def _get_heuristic_characteristics(self) -> List[str]:
        """Get characteristics of current heuristic."""
        characteristics = {
            "shortest_processing_time": [
                "Prioritizes jobs with shorter total processing times",
                "Fast and deterministic",
                "Good for minimizing average completion time"
            ],
            "longest_processing_time": [
                "Prioritizes jobs with longer total processing times",
                "Can help balance machine utilization",
                "Often effective for makespan minimization"
            ],
            "balanced": [
                "Considers both processing time and machine balance",
                "Tries to avoid jobs that create bottlenecks",
                "Good compromise between different objectives"
            ],
            "random_search": [
                "Uses random sampling with local improvements",
                "Can escape local optima",
                "Quality improves with more iterations"
            ],
            "greedy_makespan": [
                "Makes locally optimal choices at each step",
                "Directly optimizes makespan at each decision",
                "More computationally intensive but often effective"
            ]
        }
        return characteristics.get(self.heuristic, ["Heuristic-based optimization"])
    
    def __str__(self) -> str:
        return f"OptimizedScheduler(heuristic='{self.heuristic}')"