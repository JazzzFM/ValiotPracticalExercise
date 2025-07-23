from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass(frozen=True)
class Job:
    """
    Represents a manufacturing job with processing times for each machine.
    
    This class follows the Single Responsibility Principle by only handling
    job-related data and operations.
    """
    id: int
    processing_times: Tuple[int, ...]
    
    def __init__(self, id: int, processing_times: List[int]):
        """Initialize job with id and processing times."""
        if not processing_times:
            raise ValueError("Job must have at least one processing time")
        if any(time < 0 for time in processing_times):
            raise ValueError("Processing times must be non-negative")
            
        object.__setattr__(self, 'id', id)
        object.__setattr__(self, 'processing_times', tuple(processing_times))
    
    def __post_init__(self):
        if not self.processing_times:
            raise ValueError("Job must have at least one processing time")
        if any(time < 0 for time in self.processing_times):
            raise ValueError("Processing times must be non-negative")
    
    @property
    def num_machines(self) -> int:
        """Returns the number of machines this job requires."""
        return len(self.processing_times)
    
    def get_processing_time(self, machine_id: int) -> int:
        """
        Gets the processing time for a specific machine.
        
        Args:
            machine_id: The ID of the machine (0-indexed)
            
        Returns:
            Processing time in minutes
            
        Raises:
            IndexError: If machine_id is invalid
        """
        if machine_id < 0 or machine_id >= len(self.processing_times):
            raise IndexError(f"Machine ID {machine_id} is invalid. Valid range: 0-{len(self.processing_times)-1}")
        return self.processing_times[machine_id]
    
    def get_total_processing_time(self) -> int:
        """Returns the sum of all processing times across machines."""
        return sum(self.processing_times)
    
    def generate_iot_delays(self, max_delay: int = 5) -> List[int]:
        """
        Generates random IoT sensor delays for each machine.
        
        Args:
            max_delay: Maximum delay in minutes (default: 5)
            
        Returns:
            List of delays for each machine
        """
        return [random.randint(0, max_delay) for _ in range(len(self.processing_times))]
    
    def get_effective_processing_times(self, iot_delays: List[int] = None) -> List[int]:
        """
        Gets processing times including IoT delays.
        
        Args:
            iot_delays: Optional list of delays. If None, generates random delays.
            
        Returns:
            List of effective processing times (base + delay)
        """
        if iot_delays is None:
            iot_delays = self.generate_iot_delays()
        
        if len(iot_delays) != len(self.processing_times):
            raise ValueError("IoT delays must match number of machines")
            
        return [base + delay for base, delay in zip(self.processing_times, iot_delays)]
    
    def __str__(self) -> str:
        return f"Job {self.id}: {list(self.processing_times)}"
    
    def __repr__(self) -> str:
        return f"Job(id={self.id}, processing_times={self.processing_times})"