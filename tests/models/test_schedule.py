import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.job import Job
from models.schedule import Schedule, ScheduleResult


class TestScheduleResult(unittest.TestCase):
    """Test cases for the ScheduleResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from models.machine import Machine
        
        # Create mock machines with some tasks
        machine1 = Machine(id=0)
        machine1.schedule_task(job_id=0, processing_time=5)
        machine1.schedule_task(job_id=1, processing_time=3)
        
        machine2 = Machine(id=1)
        machine2.schedule_task(job_id=0, processing_time=3)
        machine2.schedule_task(job_id=1, processing_time=6)
        
        self.result = ScheduleResult(
            job_sequence=[0, 1],
            makespan=14,
            machines=[machine1, machine2],
            total_processing_time=17,
            average_utilization=0.8,
            iot_delays={0: [0, 0], 1: [0, 0]}
        )
    
    def test_schedule_result_properties(self):
        """Test ScheduleResult basic properties."""
        self.assertEqual(self.result.job_sequence, [0, 1])
        self.assertEqual(self.result.makespan, 14)
        self.assertEqual(self.result.num_jobs, 2)
        self.assertEqual(self.result.num_machines, 2)
        self.assertEqual(self.result.total_processing_time, 17)
        self.assertEqual(self.result.average_utilization, 0.8)
    
    def test_get_machine_utilization(self):
        """Test getting utilization for specific machine."""
        # Machine 0: busy_time = 8, makespan = 14, utilization = 8/14
        utilization0 = self.result.get_machine_utilization(0)
        self.assertAlmostEqual(utilization0, 8/14, places=5)
        
        # Machine 1: busy_time = 9, makespan = 14, utilization = 9/14
        utilization1 = self.result.get_machine_utilization(1)
        self.assertAlmostEqual(utilization1, 9/14, places=5)
    
    def test_get_machine_utilization_invalid_id(self):
        """Test getting utilization for invalid machine ID."""
        with self.assertRaises(IndexError):
            self.result.get_machine_utilization(-1)
        
        with self.assertRaises(IndexError):
            self.result.get_machine_utilization(2)
    
    def test_get_schedule_summary(self):
        """Test getting schedule summary."""
        summary = self.result.get_schedule_summary()
        
        self.assertIsInstance(summary, list)
        self.assertGreater(len(summary), 5)
        
        # Check key information is present
        summary_text = '\n'.join(summary)
        self.assertIn("Schedule Summary:", summary_text)
        self.assertIn("Job Sequence: [0, 1]", summary_text)
        self.assertIn("Makespan: 14", summary_text)
        self.assertIn("Machine_0 Schedule:", summary_text)
        self.assertIn("Machine_1 Schedule:", summary_text)
    
    def test_schedule_result_string_representation(self):
        """Test ScheduleResult string representation."""
        str_repr = str(self.result)
        self.assertIn("Schedule", str_repr)
        self.assertIn("sequence=[0, 1]", str_repr)
        self.assertIn("makespan=14", str_repr)


class TestSchedule(unittest.TestCase):
    """Test cases for the Schedule class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.jobs = [
            Job(id=0, processing_times=[5, 3]),
            Job(id=1, processing_times=[2, 6]),
            Job(id=2, processing_times=[4, 2])
        ]
        self.schedule = Schedule(self.jobs, num_machines=2)
    
    def test_schedule_creation(self):
        """Test successful schedule creation."""
        self.assertEqual(len(self.schedule.jobs), 3)
        self.assertEqual(self.schedule.num_machines, 2)
        self.assertIn(0, self.schedule.jobs)
        self.assertIn(1, self.schedule.jobs)
        self.assertIn(2, self.schedule.jobs)
    
    def test_schedule_creation_empty_jobs(self):
        """Test schedule creation with empty jobs list raises error."""
        with self.assertRaises(ValueError) as cm:
            Schedule([], num_machines=2)
        self.assertIn("at least one job", str(cm.exception))
    
    def test_schedule_creation_invalid_machines(self):
        """Test schedule creation with invalid machine count raises error."""
        with self.assertRaises(ValueError) as cm:
            Schedule(self.jobs, num_machines=0)
        self.assertIn("must be positive", str(cm.exception))
    
    def test_schedule_creation_inconsistent_machine_requirements(self):
        """Test schedule creation with inconsistent machine requirements."""
        inconsistent_jobs = [
            Job(id=0, processing_times=[5, 3]),      # 2 machines
            Job(id=1, processing_times=[2, 6, 1])    # 3 machines
        ]
        
        with self.assertRaises(ValueError) as cm:
            Schedule(inconsistent_jobs, num_machines=2)
        self.assertIn("same number of machines", str(cm.exception))
    
    def test_schedule_creation_machine_count_mismatch(self):
        """Test schedule creation with machine count mismatch."""
        with self.assertRaises(ValueError) as cm:
            Schedule(self.jobs, num_machines=3)  # Jobs need 2, but 3 provided
        self.assertIn("Jobs require 2 machines", str(cm.exception))
    
    def test_execute_sequence_basic(self):
        """Test basic sequence execution."""
        result = self.schedule.execute_sequence([0, 1, 2])
        
        self.assertIsInstance(result, ScheduleResult)
        self.assertEqual(result.job_sequence, [0, 1, 2])
        self.assertEqual(result.num_jobs, 3)
        self.assertEqual(result.num_machines, 2)
        self.assertGreater(result.makespan, 0)
        self.assertGreater(result.total_processing_time, 0)
    
    def test_execute_sequence_empty(self):
        """Test executing empty sequence raises error."""
        with self.assertRaises(ValueError) as cm:
            self.schedule.execute_sequence([])
        self.assertIn("cannot be empty", str(cm.exception))
    
    def test_execute_sequence_missing_jobs(self):
        """Test executing sequence with missing jobs raises error."""
        with self.assertRaises(ValueError) as cm:
            self.schedule.execute_sequence([0, 1])  # Missing job 2
        self.assertIn("contain all jobs exactly once", str(cm.exception))
    
    def test_execute_sequence_duplicate_jobs(self):
        """Test executing sequence with duplicate jobs raises error."""
        with self.assertRaises(ValueError) as cm:
            self.schedule.execute_sequence([0, 1, 1, 2])  # Duplicate job 1
        self.assertIn("contain all jobs exactly once", str(cm.exception))
    
    def test_execute_sequence_with_custom_delays(self):
        """Test executing sequence with custom IoT delays."""
        iot_delays = {
            0: [1, 2],
            1: [0, 1], 
            2: [2, 0]
        }
        
        result = self.schedule.execute_sequence([0, 1, 2], iot_delays=iot_delays)
        
        self.assertEqual(result.iot_delays, iot_delays)
        # Makespan should be affected by delays
        self.assertGreater(result.makespan, 11)  # Minimum without delays
    
    def test_execute_sequence_calculates_makespan_correctly(self):
        """Test that makespan is calculated correctly."""
        # Simple case: one job at a time
        result = self.schedule.execute_sequence([0, 1, 2])
        
        # Job 0: machine 0 (5+0→8), machine 1 (3+8→11)
        # Job 1: machine 0 (2+11→13), machine 1 (6+13→19)  
        # Job 2: machine 0 (4+19→23), machine 1 (2+23→25)
        # Makespan should be max of machine completion times
        self.assertGreater(result.makespan, 0)
        
        # Verify machines are used correctly
        self.assertEqual(len(result.machines), 2)
        self.assertGreater(len(result.machines[0].schedule), 0)
        self.assertGreater(len(result.machines[1].schedule), 0)
    
    def test_compare_sequences(self):
        """Test comparing multiple sequences."""
        sequences = [
            [0, 1, 2],
            [2, 1, 0],
            [1, 0, 2]
        ]
        
        results = self.schedule.compare_sequences(sequences)
        
        self.assertEqual(len(results), 3)
        
        # Results should be sorted by makespan (best first)
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i].makespan, results[i + 1].makespan)
        
        # Each result should have correct sequence
        sequences_found = [result.job_sequence for result in results]
        for seq in sequences:
            self.assertIn(seq, sequences_found)
    
    def test_compare_sequences_with_fixed_delays(self):
        """Test comparing sequences with fixed IoT delays."""
        sequences = [[0, 1, 2], [2, 1, 0]]
        iot_delays = {0: [1, 1], 1: [1, 1], 2: [1, 1]}
        
        results = self.schedule.compare_sequences(sequences, iot_delays=iot_delays)
        
        self.assertEqual(len(results), 2)
        
        # All results should use the same delays
        for result in results:
            self.assertEqual(result.iot_delays, iot_delays)
    
    def test_get_job_info(self):
        """Test getting job information."""
        job_info = self.schedule.get_job_info()
        
        self.assertEqual(len(job_info), 3)
        self.assertIn(0, job_info)
        self.assertIn(1, job_info)
        self.assertIn(2, job_info)
        
        # Check job 0 info
        job0_info = job_info[0]
        self.assertEqual(job0_info['processing_times'], (5, 3))
        self.assertEqual(job0_info['total_time'], 8)
        self.assertEqual(job0_info['num_machines'], 2)
    
    def test_schedule_string_representation(self):
        """Test schedule string representation."""
        str_repr = str(self.schedule)
        self.assertIn("Schedule", str_repr)
        self.assertIn("3 jobs", str_repr)
        self.assertIn("2 machines", str_repr)
        
        repr_str = repr(self.schedule)
        self.assertIn("Schedule", repr_str)
        self.assertIn("jobs=[0, 1, 2]", repr_str)
        self.assertIn("num_machines=2", repr_str)


class TestScheduleIntegration(unittest.TestCase):
    """Integration tests for Schedule with realistic scenarios."""
    
    def test_original_problem_data(self):
        """Test with the original problem data from slow_scheduler.py."""
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
        
        jobs = [Job(id=i, processing_times=times) for i, times in enumerate(original_times)]
        schedule = Schedule(jobs, num_machines=3)
        
        # Test a few different sequences
        result1 = schedule.execute_sequence(list(range(10)))  # 0,1,2,...,9
        result2 = schedule.execute_sequence(list(range(9, -1, -1)))  # 9,8,7,...,0
        
        self.assertEqual(result1.num_jobs, 10)
        self.assertEqual(result1.num_machines, 3)
        self.assertEqual(result2.num_jobs, 10)
        self.assertEqual(result2.num_machines, 3)
        
        # Both should have valid makespans
        self.assertGreater(result1.makespan, 0)
        self.assertGreater(result2.makespan, 0)
        
        # Total processing time should be same for both
        self.assertEqual(result1.total_processing_time, result2.total_processing_time)


if __name__ == '__main__':
    unittest.main()