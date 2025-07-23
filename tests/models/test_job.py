import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.job import Job


class TestJob(unittest.TestCase):
    """Test cases for the Job class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processing_times = [5, 3, 4]
        self.job = Job(id=0, processing_times=self.processing_times)
    
    def test_job_creation(self):
        """Test successful job creation."""
        self.assertEqual(self.job.id, 0)
        self.assertEqual(self.job.processing_times, (5, 3, 4))
        self.assertEqual(self.job.num_machines, 3)
    
    def test_job_creation_with_empty_times(self):
        """Test job creation with empty processing times raises error."""
        with self.assertRaises(ValueError) as cm:
            Job(id=1, processing_times=[])
        self.assertIn("at least one processing time", str(cm.exception))
    
    def test_job_creation_with_negative_times(self):
        """Test job creation with negative processing times raises error."""
        with self.assertRaises(ValueError) as cm:
            Job(id=1, processing_times=[5, -3, 4])
        self.assertIn("non-negative", str(cm.exception))
    
    def test_get_processing_time(self):
        """Test getting processing time for specific machine."""
        self.assertEqual(self.job.get_processing_time(0), 5)
        self.assertEqual(self.job.get_processing_time(1), 3)
        self.assertEqual(self.job.get_processing_time(2), 4)
    
    def test_get_processing_time_invalid_machine(self):
        """Test getting processing time for invalid machine raises error."""
        with self.assertRaises(IndexError):
            self.job.get_processing_time(-1)
        
        with self.assertRaises(IndexError):
            self.job.get_processing_time(3)
    
    def test_get_total_processing_time(self):
        """Test calculation of total processing time."""
        self.assertEqual(self.job.get_total_processing_time(), 12)  # 5 + 3 + 4
    
    def test_generate_iot_delays(self):
        """Test IoT delay generation."""
        delays = self.job.generate_iot_delays()
        self.assertEqual(len(delays), 3)
        for delay in delays:
            self.assertGreaterEqual(delay, 0)
            self.assertLessEqual(delay, 5)
    
    def test_generate_iot_delays_custom_max(self):
        """Test IoT delay generation with custom maximum."""
        delays = self.job.generate_iot_delays(max_delay=10)
        self.assertEqual(len(delays), 3)
        for delay in delays:
            self.assertGreaterEqual(delay, 0)
            self.assertLessEqual(delay, 10)
    
    def test_get_effective_processing_times(self):
        """Test calculation of effective processing times with delays."""
        iot_delays = [1, 2, 0]
        effective_times = self.job.get_effective_processing_times(iot_delays)
        
        self.assertEqual(effective_times, [6, 5, 4])  # [5+1, 3+2, 4+0]
    
    def test_get_effective_processing_times_auto_delays(self):
        """Test calculation of effective processing times with auto-generated delays."""
        effective_times = self.job.get_effective_processing_times()
        
        self.assertEqual(len(effective_times), 3)
        for i, effective_time in enumerate(effective_times):
            self.assertGreaterEqual(effective_time, self.processing_times[i])
            self.assertLessEqual(effective_time, self.processing_times[i] + 5)
    
    def test_get_effective_processing_times_mismatched_delays(self):
        """Test effective processing times with mismatched delay count raises error."""
        iot_delays = [1, 2]  # Wrong length
        
        with self.assertRaises(ValueError) as cm:
            self.job.get_effective_processing_times(iot_delays)
        self.assertIn("must match number of machines", str(cm.exception))
    
    def test_job_immutability(self):
        """Test that job is immutable (frozen dataclass)."""
        with self.assertRaises(AttributeError):
            self.job.id = 999
    
    def test_job_string_representation(self):
        """Test string representation of job."""
        str_repr = str(self.job)
        self.assertIn("Job 0", str_repr)
        self.assertIn("[5, 3, 4]", str_repr)
        
        repr_str = repr(self.job)
        self.assertIn("Job(id=0", repr_str)
        self.assertIn("processing_times=[5, 3, 4]", repr_str)
    
    def test_job_equality(self):
        """Test job equality comparison."""
        job2 = Job(id=0, processing_times=[5, 3, 4])
        job3 = Job(id=1, processing_times=[5, 3, 4])
        job4 = Job(id=0, processing_times=[1, 2, 3])
        
        self.assertEqual(self.job, job2)
        self.assertNotEqual(self.job, job3)
        self.assertNotEqual(self.job, job4)
    
    def test_job_hashable(self):
        """Test that job is hashable (can be used in sets/dicts)."""
        job_set = {self.job}
        job_dict = {self.job: "test"}
        
        self.assertIn(self.job, job_set)
        self.assertEqual(job_dict[self.job], "test")


if __name__ == '__main__':
    unittest.main()