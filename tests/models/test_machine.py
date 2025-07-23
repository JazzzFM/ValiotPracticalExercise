import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from models.machine import Machine, Task


class TestTask(unittest.TestCase):
    """Test cases for the Task class."""
    
    def test_task_creation(self):
        """Test successful task creation."""
        task = Task(job_id=1, start_time=10, end_time=20, processing_time=10)
        
        self.assertEqual(task.job_id, 1)
        self.assertEqual(task.start_time, 10)
        self.assertEqual(task.end_time, 20)
        self.assertEqual(task.processing_time, 10)
        self.assertEqual(task.duration, 10)
    
    def test_task_duration_calculation(self):
        """Test task duration calculation."""
        task = Task(job_id=1, start_time=5, end_time=15, processing_time=8)
        self.assertEqual(task.duration, 10)  # end - start
    
    def test_task_string_representation(self):
        """Test task string representation."""
        task = Task(job_id=1, start_time=10, end_time=20, processing_time=10)
        str_repr = str(task)
        
        self.assertIn("Task(job=1", str_repr)
        self.assertIn("10-20", str_repr)


class TestMachine(unittest.TestCase):
    """Test cases for the Machine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.machine = Machine(id=0)
    
    def test_machine_creation(self):
        """Test successful machine creation."""
        self.assertEqual(self.machine.id, 0)
        self.assertEqual(self.machine.name, "Machine_0")
        self.assertEqual(self.machine.available_time, 0)
        self.assertTrue(self.machine.is_idle)
        self.assertEqual(len(self.machine.schedule), 0)
    
    def test_machine_creation_with_name(self):
        """Test machine creation with custom name."""
        machine = Machine(id=1, name="Custom_Machine")
        self.assertEqual(machine.name, "Custom_Machine")
    
    def test_schedule_task(self):
        """Test scheduling a task on the machine."""
        task = self.machine.schedule_task(job_id=1, processing_time=10)
        
        self.assertEqual(task.job_id, 1)
        self.assertEqual(task.start_time, 0)
        self.assertEqual(task.end_time, 10)
        self.assertEqual(task.processing_time, 10)
        
        # Check machine state after scheduling
        self.assertEqual(self.machine.available_time, 10)
        self.assertEqual(self.machine.completion_time, 10)
        self.assertFalse(self.machine.is_idle)
        self.assertEqual(len(self.machine.schedule), 1)
    
    def test_schedule_multiple_tasks(self):
        """Test scheduling multiple tasks on the machine."""
        task1 = self.machine.schedule_task(job_id=1, processing_time=10)
        task2 = self.machine.schedule_task(job_id=2, processing_time=5)
        
        # First task
        self.assertEqual(task1.start_time, 0)
        self.assertEqual(task1.end_time, 10)
        
        # Second task should start when first ends
        self.assertEqual(task2.start_time, 10)
        self.assertEqual(task2.end_time, 15)
        
        # Machine should be available at end of last task
        self.assertEqual(self.machine.available_time, 15)
        self.assertEqual(len(self.machine.schedule), 2)
    
    def test_schedule_task_with_earliest_start(self):
        """Test scheduling task with earliest start time constraint."""
        # Schedule first task
        self.machine.schedule_task(job_id=1, processing_time=5)  # ends at 5
        
        # Schedule second task with earliest start at 10 (> available_time)
        task = self.machine.schedule_task(job_id=2, processing_time=3, earliest_start=10)
        
        self.assertEqual(task.start_time, 10)  # Should respect earliest_start
        self.assertEqual(task.end_time, 13)
        self.assertEqual(self.machine.available_time, 13)
    
    def test_schedule_task_earliest_start_before_available(self):
        """Test scheduling task with earliest start before machine available time."""
        # Schedule first task
        self.machine.schedule_task(job_id=1, processing_time=10)  # ends at 10
        
        # Try to schedule with earliest_start=5 (< available_time=10)
        task = self.machine.schedule_task(job_id=2, processing_time=3, earliest_start=5)
        
        self.assertEqual(task.start_time, 10)  # Should use available_time instead
        self.assertEqual(task.end_time, 13)
    
    def test_can_start_at(self):
        """Test checking if machine can start at specific time."""
        self.assertTrue(self.machine.can_start_at(0))
        self.assertTrue(self.machine.can_start_at(10))
        
        # Schedule a task
        self.machine.schedule_task(job_id=1, processing_time=10)  # ends at 10
        
        self.assertFalse(self.machine.can_start_at(5))   # Before available
        self.assertTrue(self.machine.can_start_at(10))   # At available time
        self.assertTrue(self.machine.can_start_at(15))   # After available
    
    def test_get_task_for_job(self):
        """Test retrieving task for specific job."""
        task1 = self.machine.schedule_task(job_id=1, processing_time=5)
        task2 = self.machine.schedule_task(job_id=2, processing_time=3)
        
        found_task1 = self.machine.get_task_for_job(1)
        found_task2 = self.machine.get_task_for_job(2)
        not_found = self.machine.get_task_for_job(999)
        
        self.assertEqual(found_task1, task1)
        self.assertEqual(found_task2, task2)
        self.assertIsNone(not_found)
    
    def test_reset(self):
        """Test resetting machine state."""
        # Schedule some tasks
        self.machine.schedule_task(job_id=1, processing_time=10)
        self.machine.schedule_task(job_id=2, processing_time=5)
        
        # Verify machine has tasks
        self.assertEqual(len(self.machine.schedule), 2)
        self.assertEqual(self.machine.available_time, 15)
        
        # Reset machine
        self.machine.reset()
        
        # Verify machine is reset
        self.assertEqual(len(self.machine.schedule), 0)
        self.assertEqual(self.machine.available_time, 0)
        self.assertTrue(self.machine.is_idle)
    
    def test_get_utilization(self):
        """Test machine utilization calculation."""
        # Empty machine
        self.assertEqual(self.machine.get_utilization(100), 0.0)
        
        # Schedule tasks with total processing time of 15
        self.machine.schedule_task(job_id=1, processing_time=10)
        self.machine.schedule_task(job_id=2, processing_time=5)
        
        # Utilization over 30 time units
        utilization = self.machine.get_utilization(30)
        self.assertEqual(utilization, 0.5)  # 15/30 = 0.5
        
        # Utilization over 15 time units (full utilization)
        utilization = self.machine.get_utilization(15)
        self.assertEqual(utilization, 1.0)
        
        # Utilization over shorter period (capped at 1.0)
        utilization = self.machine.get_utilization(10)
        self.assertEqual(utilization, 1.0)
        
        # Edge case: zero total time
        self.assertEqual(self.machine.get_utilization(0), 0.0)
    
    def test_get_schedule_summary(self):
        """Test getting schedule summary."""
        # Empty machine
        summary = self.machine.get_schedule_summary()
        self.assertEqual(len(summary), 1)
        self.assertIn("Idle", summary[0])
        
        # Machine with tasks
        self.machine.schedule_task(job_id=1, processing_time=10)
        self.machine.schedule_task(job_id=2, processing_time=5)
        
        summary = self.machine.get_schedule_summary()
        self.assertGreater(len(summary), 1)
        self.assertIn("Schedule:", summary[0])
        self.assertIn("Job 1", summary[1])
        self.assertIn("Job 2", summary[2])
    
    def test_machine_string_representation(self):
        """Test machine string representation."""
        str_repr = str(self.machine)
        self.assertIn("Machine_0", str_repr)
        self.assertIn("available at 0", str_repr)
        
        repr_str = repr(self.machine)
        self.assertIn("Machine(id=0", repr_str)
        self.assertIn("name='Machine_0'", repr_str)
        self.assertIn("available_time=0", repr_str)
    
    def test_machine_properties_after_scheduling(self):
        """Test machine properties change correctly after scheduling."""
        # Initial state
        self.assertTrue(self.machine.is_idle)
        self.assertEqual(self.machine.completion_time, 0)
        
        # After scheduling
        self.machine.schedule_task(job_id=1, processing_time=10)
        
        self.assertFalse(self.machine.is_idle)
        self.assertEqual(self.machine.completion_time, 10)
        self.assertEqual(self.machine.available_time, 10)


if __name__ == '__main__':
    unittest.main()