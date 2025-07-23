import itertools
import time
from typing import List, Dict, Optional
from strategies.base import SchedulerStrategy, OptimizationResult
from models.job import Job
from models.schedule import ScheduleResult


class BruteForceScheduler(SchedulerStrategy):
    """
    Brute force scheduling strategy that evaluates all possible job permutations.
    
    This strategy guarantees finding the optimal solution but has O(n!) time complexity,
    making it impractical for large numbers of jobs.
    """
    
    def __init__(self):
        super().__init__("Brute Force Exhaustive Search")
        self.max_jobs_warning = 10
        self.max_jobs_limit = 12
    
    def find_optimal_schedule(self, jobs: List[Job], num_machines: int, 
                            max_jobs: Optional[int] = None,
                            use_fixed_delays: bool = False,
                            iot_delays: Optional[Dict[int, List[int]]] = None,
                            **kwargs) -> ScheduleResult:
        """
        Find optimal schedule using brute force approach.
        
        Args:
            jobs: List of jobs to schedule
            num_machines: Number of available machines
            max_jobs: Maximum number of jobs to process (safety limit)
            use_fixed_delays: If True, use same IoT delays for all permutations
            iot_delays: Specific IoT delays to use (if use_fixed_delays=True)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ScheduleResult with the optimal schedule
            
        Raises:
            ValueError: If too many jobs for brute force approach
        """
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(jobs, num_machines)
        
        # Safety checks
        num_jobs = len(jobs)
        effective_limit = max_jobs or self.max_jobs_limit
        
        if num_jobs > effective_limit:
            raise ValueError(
                f"Too many jobs ({num_jobs}) for brute force approach. "
                f"Limit is {effective_limit}. Use a different strategy for larger problems."
            )
        
        if num_jobs > self.max_jobs_warning:
            print(f"Warning: {num_jobs} jobs will require {self._factorial(num_jobs):,} evaluations. This may take a while...")
        
        # Create schedule manager
        schedule_manager = self.create_schedule_manager(jobs, num_machines)
        
        # Generate fixed IoT delays if requested
        if use_fixed_delays and iot_delays is None:
            iot_delays = {}
            for job in jobs:
                iot_delays[job.id] = job.generate_iot_delays()
        
        # Find optimal sequence
        job_ids = [job.id for job in jobs]
        best_result = None
        best_makespan = float('inf')
        iterations = 0
        
        print(f"Evaluating {self._factorial(num_jobs):,} possible job sequences...")
        
        for i, permutation in enumerate(itertools.permutations(job_ids)):
            iterations += 1
            
            # Use fixed delays or generate new ones for each permutation
            current_delays = iot_delays if use_fixed_delays else None
            
            try:
                result = schedule_manager.execute_sequence(list(permutation), current_delays)
                
                if result.makespan < best_makespan:
                    best_makespan = result.makespan
                    best_result = result
                    print(f"New best found: sequence {list(permutation)} with makespan {best_makespan}")
                
                # Progress reporting for large searches
                if iterations % max(1, self._factorial(num_jobs) // 10) == 0:
                    progress = (iterations / self._factorial(num_jobs)) * 100
                    print(f"Progress: {progress:.1f}% ({iterations:,}/{self._factorial(num_jobs):,})")
                    
            except Exception as e:
                print(f"Error evaluating permutation {permutation}: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("No valid schedule found")
        
        execution_time = time.time() - start_time
        print(f"Brute force completed in {execution_time:.3f} seconds")
        print(f"Evaluated {iterations:,} permutations")
        print(f"Optimal makespan: {best_makespan}")
        
        return best_result
    
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
        iterations = self._factorial(len(jobs))
        
        convergence_info = {
            'algorithm': 'exhaustive_search',
            'guaranteed_optimal': True,
            'total_permutations': iterations,
            'search_space_coverage': 1.0
        }
        
        return OptimizationResult(
            schedule_result=schedule_result,
            strategy_name=self.name,
            execution_time=execution_time,
            iterations_performed=iterations,
            convergence_info=convergence_info
        )
    
    def get_strategy_info(self) -> Dict[str, any]:
        """
        Get information about the brute force strategy.
        
        Returns:
            Dictionary containing strategy metadata
        """
        return {
            'name': self.name,
            'type': 'exhaustive_search',
            'time_complexity': 'O(n!)',
            'space_complexity': 'O(n)',
            'guarantees_optimal': True,
            'suitable_for_jobs': f'<= {self.max_jobs_limit}',
            'warning_threshold': self.max_jobs_warning,
            'characteristics': [
                'Evaluates all possible job permutations',
                'Guarantees finding the global optimum',
                'Becomes impractical for large job counts',
                'Good baseline for comparing other algorithms'
            ]
        }
    
    def estimate_runtime(self, num_jobs: int, time_per_evaluation: float = 1e-4) -> Dict[str, any]:
        """
        Estimate runtime for a given number of jobs.
        
        Args:
            num_jobs: Number of jobs to schedule
            time_per_evaluation: Estimated time per permutation evaluation (seconds)
            
        Returns:
            Dictionary with runtime estimates
        """
        if num_jobs <= 0:
            return {'error': 'Number of jobs must be positive'}
        
        total_evaluations = self._factorial(num_jobs)
        estimated_time = total_evaluations * time_per_evaluation
        
        return {
            'num_jobs': num_jobs,
            'total_evaluations': total_evaluations,
            'estimated_time_seconds': estimated_time,
            'estimated_time_minutes': estimated_time / 60,
            'estimated_time_hours': estimated_time / 3600,
            'feasible': estimated_time < 3600,  # Less than 1 hour
            'recommendation': self._get_feasibility_recommendation(estimated_time)
        }
    
    def _factorial(self, n: int) -> int:
        """Calculate factorial of n."""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    
    def _get_feasibility_recommendation(self, estimated_time: float) -> str:
        """Get recommendation based on estimated runtime."""
        if estimated_time < 1:
            return "Very fast - suitable for real-time use"
        elif estimated_time < 60:
            return "Fast - suitable for interactive use"
        elif estimated_time < 3600:
            return "Moderate - acceptable for batch processing"
        elif estimated_time < 86400:
            return "Slow - consider overnight processing"
        else:
            return "Very slow - consider alternative algorithms"
    
    def validate_inputs(self, jobs: List[Job], num_machines: int):
        """
        Extended validation for brute force strategy.
        
        Args:
            jobs: List of jobs to validate
            num_machines: Number of machines to validate
            
        Raises:
            ValueError: If inputs are invalid or unsuitable for brute force
        """
        super().validate_inputs(jobs, num_machines)
        
        num_jobs = len(jobs)
        if num_jobs > self.max_jobs_limit:
            raise ValueError(
                f"Brute force strategy cannot handle {num_jobs} jobs. "
                f"Maximum supported: {self.max_jobs_limit}. "
                f"Consider using an optimized strategy instead."
            )
    
    def __str__(self) -> str:
        return f"BruteForceScheduler(max_jobs={self.max_jobs_limit})"