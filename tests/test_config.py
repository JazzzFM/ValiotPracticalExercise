import unittest
import tempfile
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.settings import (
    JobConfig, OptimizationConfig, PerformanceConfig, 
    SchedulerSettings, ConfigurationManager
)


class TestJobConfig(unittest.TestCase):
    """Test cases for JobConfig."""
    
    def test_valid_job_config(self):
        """Test creating valid job configuration."""
        config = JobConfig(processing_times=[5, 3, 4])
        self.assertEqual(config.processing_times, [5, 3, 4])
        self.assertEqual(config.max_iot_delay, 5)
        
        # Should not raise
        config.validate()
    
    def test_job_config_validation_empty_times(self):
        """Test job config validation with empty processing times."""
        config = JobConfig(processing_times=[])
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("at least one processing time", str(cm.exception))
    
    def test_job_config_validation_negative_times(self):
        """Test job config validation with negative processing times."""
        config = JobConfig(processing_times=[5, -3, 4])
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("non-negative", str(cm.exception))
    
    def test_job_config_validation_negative_delay(self):
        """Test job config validation with negative IoT delay."""
        config = JobConfig(processing_times=[5, 3, 4], max_iot_delay=-1)
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("non-negative", str(cm.exception))


class TestOptimizationConfig(unittest.TestCase):
    """Test cases for OptimizationConfig."""
    
    def test_valid_optimization_config(self):
        """Test creating valid optimization configuration."""
        config = OptimizationConfig()
        self.assertEqual(config.max_iterations, 1000)
        self.assertEqual(config.time_limit, 60.0)
        
        # Should not raise
        config.validate()
    
    def test_optimization_config_validation_invalid_iterations(self):
        """Test optimization config validation with invalid iterations."""
        config = OptimizationConfig(max_iterations=0)
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("positive", str(cm.exception))
    
    def test_optimization_config_validation_invalid_time_limit(self):
        """Test optimization config validation with invalid time limit."""
        config = OptimizationConfig(time_limit=-1.0)
        with self.assertRaises(ValueError) as cm:
            config.validate()
        self.assertIn("positive", str(cm.exception))


class TestConfigurationManager(unittest.TestCase):
    """Test cases for ConfigurationManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
    
    def test_load_from_original_data(self):
        """Test loading configuration from original data."""
        settings = self.config_manager.load_from_original_data()
        
        self.assertEqual(len(settings.jobs), 10)
        self.assertEqual(settings.num_machines, 3)
        self.assertEqual(settings.default_strategy, "optimized_balanced")
        
        # Should not raise
        settings.validate()
    
    def test_save_and_load_configuration(self):
        """Test saving and loading configuration to/from file."""
        # Create test configuration
        jobs = [JobConfig(processing_times=[5, 3, 4])]
        settings = SchedulerSettings(
            jobs=jobs,
            num_machines=3,
            default_strategy="brute_force"
        )
        
        # Save configuration
        filename = "test_config.json"
        self.config_manager.save_to_file(settings, filename)
        
        # Load configuration
        loaded_settings = self.config_manager.load_from_file(filename)
        
        self.assertEqual(len(loaded_settings.jobs), 1)
        self.assertEqual(loaded_settings.num_machines, 3)
        self.assertEqual(loaded_settings.default_strategy, "brute_force")
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent configuration file raises error."""
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_from_file("nonexistent.json")
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        # Create invalid JSON file
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("{ invalid json }")
        
        with self.assertRaises(ValueError) as cm:
            self.config_manager.load_from_file("invalid.json")
        self.assertIn("Invalid JSON", str(cm.exception))
    
    def test_validation_current_settings(self):
        """Test validation of current settings."""
        # Initially no settings loaded
        self.assertFalse(self.config_manager.validate_current_settings())
        
        # Load valid settings
        self.config_manager.load_from_original_data()
        self.assertTrue(self.config_manager.validate_current_settings())
    
    def test_get_job_data_for_scheduler(self):
        """Test getting job data in scheduler format."""
        settings = self.config_manager.load_from_original_data()
        jobs, num_machines = self.config_manager.get_job_data_for_scheduler()
        
        self.assertEqual(len(jobs), 10)
        self.assertEqual(num_machines, 3)
        
        # Check first job
        self.assertEqual(jobs[0].id, 0)
        self.assertEqual(jobs[0].processing_times, (5, 3, 4))
    
    def test_get_job_data_no_settings(self):
        """Test getting job data without loaded settings raises error."""
        with self.assertRaises(RuntimeError):
            self.config_manager.get_job_data_for_scheduler()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)


if __name__ == '__main__':
    unittest.main()