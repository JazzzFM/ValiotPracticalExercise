import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class JobConfig:
    """Configuration for individual jobs."""
    processing_times: List[int]
    max_iot_delay: int = 5
    
    def validate(self):
        """Validate job configuration."""
        if not self.processing_times:
            raise ValueError("Job must have at least one processing time")
        if any(time < 0 for time in self.processing_times):
            raise ValueError("Processing times must be non-negative")
        if self.max_iot_delay < 0:
            raise ValueError("Max IoT delay must be non-negative")


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    max_iterations: int = 1000
    improvement_threshold: float = 0.01
    time_limit: float = 60.0
    use_fixed_delays: bool = False
    random_seed: Optional[int] = None
    
    def validate(self):
        """Validate optimization configuration."""
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.improvement_threshold < 0:
            raise ValueError("Improvement threshold must be non-negative")
        if self.time_limit <= 0:
            raise ValueError("Time limit must be positive")


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and benchmarking."""
    enable_benchmarking: bool = True
    benchmark_iterations: int = 5
    memory_profiling: bool = False
    detailed_logging: bool = False
    export_results: bool = True
    export_format: str = "json"  # json, csv, xlsx
    
    def validate(self):
        """Validate performance configuration."""
        if self.benchmark_iterations <= 0:
            raise ValueError("Benchmark iterations must be positive")
        if self.export_format not in ["json", "csv", "xlsx"]:
            raise ValueError("Export format must be one of: json, csv, xlsx")


@dataclass
class SchedulerSettings:
    """Main configuration class for the scheduler system."""
    jobs: List[JobConfig]
    num_machines: int
    default_strategy: str = "optimized_balanced"
    optimization: OptimizationConfig = None
    performance: PerformanceConfig = None
    
    def __post_init__(self):
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
    
    def validate(self):
        """Validate all configuration settings."""
        if not self.jobs:
            raise ValueError("At least one job must be configured")
        if self.num_machines <= 0:
            raise ValueError("Number of machines must be positive")
        
        # Validate all jobs
        for i, job in enumerate(self.jobs):
            try:
                job.validate()
            except ValueError as e:
                raise ValueError(f"Job {i} configuration error: {e}")
        
        # Check machine consistency
        expected_machines = len(self.jobs[0].processing_times)
        for i, job in enumerate(self.jobs):
            if len(job.processing_times) != expected_machines:
                raise ValueError(f"Job {i} has {len(job.processing_times)} processing times, "
                               f"expected {expected_machines}")
        
        if expected_machines != self.num_machines:
            raise ValueError(f"Jobs require {expected_machines} machines, "
                           f"but configuration specifies {self.num_machines}")
        
        # Validate sub-configurations
        self.optimization.validate()
        self.performance.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchedulerSettings':
        """Create settings from dictionary."""
        # Convert job configurations
        jobs = []
        for job_data in data.get('jobs', []):
            jobs.append(JobConfig(**job_data))
        
        # Create optimization config
        opt_data = data.get('optimization', {})
        optimization = OptimizationConfig(**opt_data)
        
        # Create performance config
        perf_data = data.get('performance', {})
        performance = PerformanceConfig(**perf_data)
        
        # Create main settings
        return cls(
            jobs=jobs,
            num_machines=data.get('num_machines', 3),
            default_strategy=data.get('default_strategy', 'optimized_balanced'),
            optimization=optimization,
            performance=performance
        )


class ConfigurationManager:
    """
    Manages configuration loading, saving, and validation.
    
    This class follows the Dependency Inversion Principle by providing
    an abstraction for configuration management.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files. If None, uses current directory.
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.config_dir.mkdir(exist_ok=True)
        self._settings: Optional[SchedulerSettings] = None
    
    def load_from_file(self, filename: str) -> SchedulerSettings:
        """
        Load configuration from JSON file.
        
        Args:
            filename: Name of the configuration file
            
        Returns:
            SchedulerSettings instance
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            settings = SchedulerSettings.from_dict(data)
            settings.validate()
            
            self._settings = settings
            return settings
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def save_to_file(self, settings: SchedulerSettings, filename: str):
        """
        Save configuration to JSON file.
        
        Args:
            settings: SchedulerSettings to save
            filename: Name of the configuration file
        """
        settings.validate()
        
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w') as f:
                json.dump(settings.to_dict(), f, indent=2)
            
            self._settings = settings
            
        except Exception as e:
            raise ValueError(f"Error saving configuration: {e}")
    
    def load_from_original_data(self) -> SchedulerSettings:
        """
        Create configuration from the original slow_scheduler.py data.
        
        Returns:
            SchedulerSettings with original problem data
        """
        # Original data from slow_scheduler.py
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
        
        jobs = [JobConfig(processing_times=times) for times in original_times]
        
        settings = SchedulerSettings(
            jobs=jobs,
            num_machines=3,
            default_strategy="optimized_balanced",
            optimization=OptimizationConfig(
                max_iterations=1000,
                time_limit=60.0,
                use_fixed_delays=False
            ),
            performance=PerformanceConfig(
                enable_benchmarking=True,
                benchmark_iterations=5,
                export_results=True
            )
        )
        
        settings.validate()
        self._settings = settings
        return settings
    
    def load_from_environment(self) -> SchedulerSettings:
        """
        Load configuration from environment variables.
        
        Returns:
            SchedulerSettings from environment variables
        """
        # Get configuration file path from environment
        config_file = os.getenv('SCHEDULER_CONFIG_FILE', 'scheduler_config.json')
        
        if os.path.exists(config_file):
            return self.load_from_file(config_file)
        else:
            # Fall back to original data
            return self.load_from_original_data()
    
    def get_current_settings(self) -> Optional[SchedulerSettings]:
        """
        Get currently loaded settings.
        
        Returns:
            Current SchedulerSettings or None if not loaded
        """
        return self._settings
    
    def create_default_config_file(self, filename: str = "scheduler_config.json"):
        """
        Create a default configuration file with original problem data.
        
        Args:
            filename: Name of the configuration file to create
        """
        settings = self.load_from_original_data()
        self.save_to_file(settings, filename)
        
        print(f"Default configuration saved to: {self.config_dir / filename}")
    
    def validate_current_settings(self) -> bool:
        """
        Validate currently loaded settings.
        
        Returns:
            True if settings are valid, False otherwise
        """
        if self._settings is None:
            return False
        
        try:
            self._settings.validate()
            return True
        except ValueError:
            return False
    
    def get_job_data_for_scheduler(self) -> tuple:
        """
        Get job data in format suitable for scheduler classes.
        
        Returns:
            Tuple of (jobs_list, num_machines)
            
        Raises:
            RuntimeError: If no settings are loaded
        """
        if self._settings is None:
            raise RuntimeError("No configuration loaded. Call load_* method first.")
        
        # Import Job here to avoid circular imports
        try:
            from ..models.job import Job
        except ImportError:
            # Fallback for direct execution
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models.job import Job
        
        jobs = []
        for i, job_config in enumerate(self._settings.jobs):
            job = Job(id=i, processing_times=job_config.processing_times)
            jobs.append(job)
        
        return jobs, self._settings.num_machines
    
    def update_optimization_config(self, **kwargs):
        """
        Update optimization configuration parameters.
        
        Args:
            **kwargs: Optimization parameters to update
        """
        if self._settings is None:
            raise RuntimeError("No configuration loaded")
        
        for key, value in kwargs.items():
            if hasattr(self._settings.optimization, key):
                setattr(self._settings.optimization, key, value)
            else:
                raise ValueError(f"Unknown optimization parameter: {key}")
        
        self._settings.validate()
    
    def update_performance_config(self, **kwargs):
        """
        Update performance configuration parameters.
        
        Args:
            **kwargs: Performance parameters to update
        """
        if self._settings is None:
            raise RuntimeError("No configuration loaded")
        
        for key, value in kwargs.items():
            if hasattr(self._settings.performance, key):
                setattr(self._settings.performance, key, value)
            else:
                raise ValueError(f"Unknown performance parameter: {key}")
        
        self._settings.validate()
    
    def __str__(self) -> str:
        loaded = "loaded" if self._settings else "not loaded"
        return f"ConfigurationManager(config_dir={self.config_dir}, settings={loaded})"


# Global configuration manager instance
default_config_manager = ConfigurationManager()


def load_default_configuration() -> SchedulerSettings:
    """
    Load default configuration using the global manager.
    
    Returns:
        Default SchedulerSettings
    """
    return default_config_manager.load_from_original_data()


def get_configuration_manager() -> ConfigurationManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigurationManager instance
    """
    return default_config_manager