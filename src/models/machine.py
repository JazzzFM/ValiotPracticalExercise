from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Task:
    """Represents a task assigned to a machine."""
    job_id: int
    start_time: int
    end_time: int
    processing_time: int
    
    @property
    def duration(self) -> int:
        """Returns the actual duration of the task."""
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        return f"Task(job={self.job_id}, {self.start_time}-{self.end_time})"


@dataclass
class Machine:
    """
    Represents a manufacturing machine that can process jobs.
    
    This class follows the Single Responsibility Principle by only handling
    machine-specific operations and scheduling.
    """
    id: int
    name: Optional[str] = None
    
    def __post_init__(self):
        self.schedule: List[Task] = []
        self._available_time = 0
        
        if self.name is None:
            self.name = f"Machine_{self.id}"
    
    @property
    def available_time(self) -> int:
        """Returns the earliest time this machine becomes available."""
        return self._available_time
    
    @property
    def is_idle(self) -> bool:
        """Returns True if the machine is currently idle."""
        return self._available_time == 0 or len(self.schedule) == 0
    
    @property
    def completion_time(self) -> int:
        """Returns the time when all scheduled tasks will be completed."""
        return self._available_time
    
    def can_start_at(self, start_time: int) -> bool:
        """
        Checks if the machine can start a new task at the given time.
        
        Args:
            start_time: The proposed start time
            
        Returns:
            True if the machine is available at that time
        """
        return start_time >= self._available_time
    
    def schedule_task(self, job_id: int, processing_time: int, earliest_start: int = 0) -> Task:
        """
        Schedules a new task on this machine.
        
        Args:
            job_id: ID of the job to schedule
            processing_time: Time required to process the job
            earliest_start: Earliest time the task can start
            
        Returns:
            The scheduled Task object
        """
        start_time = max(self._available_time, earliest_start)
        end_time = start_time + processing_time
        
        task = Task(
            job_id=job_id,
            start_time=start_time,
            end_time=end_time,
            processing_time=processing_time
        )
        
        self.schedule.append(task)
        self._available_time = end_time
        
        return task
    
    def get_task_for_job(self, job_id: int) -> Optional[Task]:
        """
        Retrieves the task for a specific job.
        
        Args:
            job_id: ID of the job to find
            
        Returns:
            Task object if found, None otherwise
        """
        for task in self.schedule:
            if task.job_id == job_id:
                return task
        return None
    
    def reset(self):
        """Resets the machine schedule and availability."""
        self.schedule.clear()
        self._available_time = 0
    
    def get_utilization(self, total_time: int) -> float:
        """
        Calculates machine utilization as a percentage.
        
        Args:
            total_time: Total time period to calculate utilization over
            
        Returns:
            Utilization percentage (0.0 to 1.0)
        """
        if total_time <= 0:
            return 0.0
        
        busy_time = sum(task.processing_time for task in self.schedule)
        return min(busy_time / total_time, 1.0)
    
    def get_schedule_summary(self) -> List[str]:
        """
        Returns a human-readable summary of the machine's schedule.
        
        Returns:
            List of strings describing each scheduled task
        """
        if not self.schedule:
            return [f"{self.name}: Idle"]
        
        summary = [f"{self.name} Schedule:"]
        for task in self.schedule:
            summary.append(f"  Job {task.job_id}: {task.start_time}-{task.end_time} ({task.processing_time}min)")
        
        return summary
    
    def __str__(self) -> str:
        return f"{self.name} (available at {self._available_time})"
    
    def __repr__(self) -> str:
        return f"Machine(id={self.id}, name='{self.name}', available_time={self._available_time})"