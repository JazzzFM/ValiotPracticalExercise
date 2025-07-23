from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from models.job import Job
from models.schedule import Schedule, ScheduleResult


class SchedulerStrategy(ABC):
    """
    Abstract base class for scheduling strategies.
    
    This interface follows the Open/Closed Principle - new scheduling algorithms
    can be added by implementing this interface without modifying existing code.
    """
    
    def __init__(self, name: str):
        """
        Initialize the strategy with a descriptive name.
        
        Args:
            name: Human-readable name for this strategy
        """
        self.name = name
    
    @abstractmethod
    def find_optimal_schedule(self, jobs: List[Job], num_machines: int, 
                            **kwargs) -> ScheduleResult:
        """
        Find the optimal job schedule using this strategy.
        
        Args:
            jobs: List of jobs to schedule
            num_machines: Number of available machines
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            ScheduleResult containing the optimal schedule found
            
        Raises:
            ValueError: If inputs are invalid
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    def get_strategy_info(self) -> Dict[str, any]:
        """
        Get information about this strategy.
        
        Returns:
            Dictionary containing strategy metadata
        """
        pass
    
    def validate_inputs(self, jobs: List[Job], num_machines: int):
        """
        Validates common input parameters.
        
        Args:
            jobs: List of jobs to validate
            num_machines: Number of machines to validate
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not jobs:
            raise ValueError("At least one job must be provided")
        
        if num_machines <= 0:
            raise ValueError("Number of machines must be positive")
        
        # Check that all jobs have consistent machine requirements
        expected_machines = jobs[0].num_machines
        for job in jobs:
            if job.num_machines != expected_machines:
                raise ValueError("All jobs must require the same number of machines")
        
        if expected_machines != num_machines:
            raise ValueError(f"Jobs require {expected_machines} machines, but {num_machines} were provided")
    
    def create_schedule_manager(self, jobs: List[Job], num_machines: int) -> Schedule:
        """
        Creates a Schedule instance for the given jobs and machines.
        
        Args:
            jobs: List of jobs to schedule
            num_machines: Number of available machines
            
        Returns:
            Schedule instance ready for job sequencing
        """
        self.validate_inputs(jobs, num_machines)
        return Schedule(jobs, num_machines)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class OptimizationResult:
    """
    Contains results and metadata from an optimization run.
    """
    
    def __init__(self, schedule_result: ScheduleResult, strategy_name: str,
                 execution_time: float, iterations_performed: int = None,
                 convergence_info: Dict = None):
        """
        Initialize optimization result.
        
        Args:
            schedule_result: The best schedule found
            strategy_name: Name of the strategy used
            execution_time: Time taken to find the solution
            iterations_performed: Number of iterations/evaluations performed
            convergence_info: Additional convergence information
        """
        self.schedule_result = schedule_result
        self.strategy_name = strategy_name
        self.execution_time = execution_time
        self.iterations_performed = iterations_performed
        self.convergence_info = convergence_info or {}
    
    @property
    def makespan(self) -> int:
        """Returns the makespan of the optimal schedule."""
        return self.schedule_result.makespan
    
    @property
    def job_sequence(self) -> List[int]:
        """Returns the optimal job sequence."""
        return self.schedule_result.job_sequence
    
    def get_performance_summary(self) -> Dict[str, any]:
        """
        Returns a summary of the optimization performance.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'strategy': self.strategy_name,
            'makespan': self.makespan,
            'execution_time': self.execution_time,
            'iterations': self.iterations_performed,
            'job_sequence': self.job_sequence,
            'num_jobs': self.schedule_result.num_jobs,
            'num_machines': self.schedule_result.num_machines,
            'average_utilization': self.schedule_result.average_utilization
        }
    
    def __str__(self) -> str:
        return f"OptimizationResult({self.strategy_name}, makespan={self.makespan}, time={self.execution_time:.3f}s)"