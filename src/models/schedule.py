from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from models.job import Job
from models.machine import Machine, Task


@dataclass
class ScheduleResult:
    """
    Contains the results of a scheduling operation.
    
    This class encapsulates all relevant information about a completed schedule,
    following the Single Responsibility Principle.
    """
    job_sequence: List[int]
    makespan: int
    machines: List[Machine]
    total_processing_time: int
    average_utilization: float
    iot_delays: Optional[Dict[int, List[int]]] = None
    
    @property
    def num_jobs(self) -> int:
        """Returns the number of jobs in the schedule."""
        return len(self.job_sequence)
    
    @property
    def num_machines(self) -> int:
        """Returns the number of machines used."""
        return len(self.machines)
    
    def get_machine_utilization(self, machine_id: int) -> float:
        """
        Gets utilization for a specific machine.
        
        Args:
            machine_id: ID of the machine
            
        Returns:
            Utilization percentage (0.0 to 1.0)
        """
        if machine_id < 0 or machine_id >= len(self.machines):
            raise IndexError(f"Machine ID {machine_id} is invalid")
        
        return self.machines[machine_id].get_utilization(self.makespan)
    
    def get_schedule_summary(self) -> List[str]:
        """
        Returns a comprehensive summary of the schedule.
        
        Returns:
            List of strings describing the schedule
        """
        summary = [
            f"Schedule Summary:",
            f"  Job Sequence: {self.job_sequence}",
            f"  Makespan: {self.makespan} minutes",
            f"  Total Processing Time: {self.total_processing_time} minutes",
            f"  Average Utilization: {self.average_utilization:.2%}",
            f"  Number of Jobs: {self.num_jobs}",
            f"  Number of Machines: {self.num_machines}",
            ""
        ]
        
        for machine in self.machines:
            summary.extend(machine.get_schedule_summary())
            summary.append("")
        
        return summary
    
    def __str__(self) -> str:
        return f"Schedule(sequence={self.job_sequence}, makespan={self.makespan})"


class Schedule:
    """
    Manages the scheduling of jobs on machines.
    
    This class is responsible for executing job sequences and calculating
    performance metrics, following the Single Responsibility Principle.
    """
    
    def __init__(self, jobs: List[Job], num_machines: int):
        """
        Initializes the schedule manager.
        
        Args:
            jobs: List of jobs to be scheduled
            num_machines: Number of available machines
        """
        self.jobs = {job.id: job for job in jobs}
        self.num_machines = num_machines
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validates the input jobs and machine configuration."""
        if not self.jobs:
            raise ValueError("At least one job must be provided")
        
        if self.num_machines <= 0:
            raise ValueError("Number of machines must be positive")
        
        # Verify all jobs have the same number of machine requirements
        expected_machines = None
        for job in self.jobs.values():
            if expected_machines is None:
                expected_machines = job.num_machines
            elif job.num_machines != expected_machines:
                raise ValueError("All jobs must require the same number of machines")
        
        if expected_machines != self.num_machines:
            raise ValueError(f"Jobs require {expected_machines} machines, but {self.num_machines} were provided")
    
    def execute_sequence(self, job_sequence: List[int], 
                        iot_delays: Optional[Dict[int, List[int]]] = None) -> ScheduleResult:
        """
        Executes a job sequence and returns the scheduling result.
        
        Args:
            job_sequence: List of job IDs in execution order
            iot_delays: Optional dictionary mapping job IDs to their IoT delays
            
        Returns:
            ScheduleResult containing all scheduling information
        """
        if not job_sequence:
            raise ValueError("Job sequence cannot be empty")
        
        if len(set(job_sequence)) != len(self.jobs):
            raise ValueError("Job sequence must contain all jobs exactly once")
        
        # Initialize machines
        machines = [Machine(id=i) for i in range(self.num_machines)]
        
        # Generate IoT delays if not provided
        if iot_delays is None:
            iot_delays = {}
            for job_id in job_sequence:
                iot_delays[job_id] = self.jobs[job_id].generate_iot_delays()
        
        # Schedule jobs in sequence
        for job_id in job_sequence:
            job = self.jobs[job_id]
            job_delays = iot_delays.get(job_id, [0] * self.num_machines)
            effective_times = job.get_effective_processing_times(job_delays)
            
            # Schedule job on each machine in sequence
            previous_completion = 0
            for machine_id, processing_time in enumerate(effective_times):
                task = machines[machine_id].schedule_task(
                    job_id=job_id,
                    processing_time=processing_time,
                    earliest_start=previous_completion
                )
                previous_completion = task.end_time
        
        # Calculate metrics
        makespan = max(machine.completion_time for machine in machines)
        total_processing_time = sum(
            sum(job.processing_times) for job in self.jobs.values()
        )
        
        average_utilization = sum(
            machine.get_utilization(makespan) for machine in machines
        ) / len(machines)
        
        return ScheduleResult(
            job_sequence=job_sequence.copy(),
            makespan=makespan,
            machines=machines,
            total_processing_time=total_processing_time,
            average_utilization=average_utilization,
            iot_delays=iot_delays.copy()
        )
    
    def compare_sequences(self, sequences: List[List[int]], 
                         iot_delays: Optional[Dict[int, List[int]]] = None) -> List[ScheduleResult]:
        """
        Compares multiple job sequences and returns results sorted by makespan.
        
        Args:
            sequences: List of job sequences to compare
            iot_delays: Optional IoT delays to use for all sequences
            
        Returns:
            List of ScheduleResult objects sorted by makespan (best first)
        """
        results = []
        
        for sequence in sequences:
            try:
                result = self.execute_sequence(sequence, iot_delays)
                results.append(result)
            except Exception as e:
                # Log error but continue with other sequences
                print(f"Error executing sequence {sequence}: {e}")
                continue
        
        # Sort by makespan (ascending - best first)
        results.sort(key=lambda r: r.makespan)
        return results
    
    def get_job_info(self) -> Dict[int, Dict]:
        """
        Returns information about all jobs.
        
        Returns:
            Dictionary mapping job IDs to job information
        """
        return {
            job_id: {
                'processing_times': job.processing_times,
                'total_time': job.get_total_processing_time(),
                'num_machines': job.num_machines
            }
            for job_id, job in self.jobs.items()
        }
    
    def reset_machines(self, machines: List[Machine]):
        """
        Resets all machines to idle state.
        
        Args:
            machines: List of machines to reset
        """
        for machine in machines:
            machine.reset()
    
    def __str__(self) -> str:
        return f"Schedule({len(self.jobs)} jobs, {self.num_machines} machines)"
    
    def __repr__(self) -> str:
        return f"Schedule(jobs={list(self.jobs.keys())}, num_machines={self.num_machines})"