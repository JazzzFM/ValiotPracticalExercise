import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


@dataclass
class ManufacturingParams:
    """Parameters for realistic manufacturing data generation."""
    num_job_types: int = 50
    num_machines: int = 5
    seasonal_amplitude: float = 0.3
    noise_level: float = 0.1
    failure_rate: float = 0.02
    quality_defect_rate: float = 0.05
    energy_cost_variation: float = 0.2


class RealisticDataGenerator:
    """
    Generates realistic manufacturing scheduling datasets for ML training.
    
    This addresses the limitation of hardcoded, unrealistic data by creating
    diverse, time-series aware manufacturing scenarios.
    """
    
    def __init__(self, params: ManufacturingParams = None):
        """Initialize the data generator with manufacturing parameters."""
        self.params = params or ManufacturingParams()
        self.random_state = np.random.RandomState(42)
        self._setup_base_patterns()
    
    def _setup_base_patterns(self):
        """Setup base patterns for different job types and machines."""
        # Create base processing time patterns
        self.job_complexity_factors = self.random_state.exponential(1.0, self.params.num_job_types)
        self.machine_efficiency_factors = self.random_state.uniform(0.8, 1.2, self.params.num_machines)
        
        # Machine specialization matrix (some machines better for certain jobs)
        self.specialization_matrix = self.random_state.uniform(0.5, 1.5, 
                                                              (self.params.num_job_types, self.params.num_machines))
    
    def generate_historical_data(self, num_samples: int = 10000, 
                               time_span_days: int = 365) -> pd.DataFrame:
        """
        Generate historical manufacturing data with realistic patterns.
        
        Args:
            num_samples: Number of historical job records
            time_span_days: Time span to simulate
            
        Returns:
            DataFrame with realistic historical manufacturing data
        """
        # Generate timestamps with realistic distribution (more during work hours)
        start_date = datetime.now() - timedelta(days=time_span_days)
        timestamps = self._generate_realistic_timestamps(start_date, num_samples)
        
        data = []
        for i, timestamp in enumerate(timestamps):
            # Job characteristics
            job_type = self.random_state.randint(0, self.params.num_job_types)
            job_priority = self.random_state.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
            batch_size = max(1, int(self.random_state.exponential(10)))
            
            # Generate processing times with realistic patterns
            processing_times = self._generate_processing_times(job_type, timestamp)
            
            # Quality and performance metrics
            quality_score = self._generate_quality_score(job_type, processing_times)
            energy_consumption = self._calculate_energy_consumption(processing_times, timestamp)
            
            # Machine availability and maintenance
            machine_availability = self._get_machine_availability(timestamp)
            
            # Economic factors
            material_cost = self._calculate_material_cost(job_type, timestamp)
            rush_order = self.random_state.random() < 0.1  # 10% rush orders
            
            record = {
                'timestamp': timestamp,
                'job_id': f"JOB_{i:06d}",
                'job_type': job_type,
                'job_priority': job_priority,
                'batch_size': batch_size,
                'processing_times': processing_times,
                'actual_completion_time': self._simulate_actual_completion(processing_times),
                'quality_score': quality_score,
                'defect_rate': max(0, self.random_state.normal(self.params.quality_defect_rate, 0.02)),
                'energy_consumption': energy_consumption,
                'material_cost': material_cost,
                'machine_availability': machine_availability,
                'rush_order': rush_order,
                'seasonal_factor': self._get_seasonal_factor(timestamp),
                'day_of_week': timestamp.weekday(),
                'hour_of_day': timestamp.hour,
                'weather_impact': self._get_weather_impact(timestamp),  # Affects outdoor operations
            }
            data.append(record)
        
        return pd.DataFrame(data)
    
    def _generate_realistic_timestamps(self, start_date: datetime, num_samples: int) -> List[datetime]:
        """Generate timestamps with realistic work patterns."""
        timestamps = []
        current_date = start_date
        
        for _ in range(num_samples):
            # Higher probability during work hours (8 AM - 6 PM)
            if self.random_state.random() < 0.7:  # 70% during work hours
                hour = self.random_state.choice(range(8, 18))
            else:
                hour = self.random_state.choice(range(24))
            
            # Skip weekends with lower probability
            while current_date.weekday() >= 5 and self.random_state.random() < 0.8:
                current_date += timedelta(days=1)
            
            timestamp = current_date.replace(
                hour=hour,
                minute=self.random_state.randint(0, 60),
                second=self.random_state.randint(0, 60)
            )
            timestamps.append(timestamp)
            
            # Advance time realistically
            current_date += timedelta(
                hours=self.random_state.exponential(2),  # Average 2 hours between jobs
                minutes=self.random_state.randint(0, 120)
            )
        
        return sorted(timestamps)
    
    def _generate_processing_times(self, job_type: int, timestamp: datetime) -> List[float]:
        """Generate processing times with realistic variations."""
        base_times = []
        
        for machine_id in range(self.params.num_machines):
            # Base processing time
            base_time = (self.job_complexity_factors[job_type] * 
                        self.specialization_matrix[job_type, machine_id] /
                        self.machine_efficiency_factors[machine_id])
            
            # Add temporal variations
            seasonal_factor = self._get_seasonal_factor(timestamp)
            time_of_day_factor = self._get_time_of_day_factor(timestamp)
            
            # Add realistic noise
            noise = self.random_state.normal(1.0, self.params.noise_level)
            
            final_time = max(0.1, base_time * seasonal_factor * time_of_day_factor * noise * 10)
            base_times.append(round(final_time, 2))
        
        return base_times
    
    def _get_seasonal_factor(self, timestamp: datetime) -> float:
        """Calculate seasonal factor affecting processing times."""
        day_of_year = timestamp.timetuple().tm_yday
        # Peak efficiency in spring/fall, slower in summer/winter
        seasonal_cycle = np.sin(2 * np.pi * day_of_year / 365)
        return 1.0 + self.params.seasonal_amplitude * seasonal_cycle
    
    def _get_time_of_day_factor(self, timestamp: datetime) -> float:
        """Calculate time-of-day factor (fatigue, shift changes)."""
        hour = timestamp.hour
        if 6 <= hour <= 14:  # First shift - highest efficiency
            return 1.0
        elif 14 <= hour <= 22:  # Second shift - slightly lower
            return 1.1
        else:  # Night shift - lowest efficiency
            return 1.3
    
    def _simulate_actual_completion(self, processing_times: List[float]) -> float:
        """Simulate actual completion time with realistic delays."""
        scheduled_makespan = sum(processing_times)  # Simplified
        
        # Add realistic delays
        delays = 0
        
        # Machine breakdowns
        if self.random_state.random() < self.params.failure_rate:
            delays += self.random_state.exponential(30)  # Average 30 min breakdown
        
        # Setup time variations
        setup_delay = self.random_state.normal(0, 5)  # ±5 min setup variation
        
        # Quality issues requiring rework
        if self.random_state.random() < self.params.quality_defect_rate:
            delays += self.random_state.exponential(15)  # Average 15 min rework
        
        return max(scheduled_makespan, scheduled_makespan + delays + setup_delay)
    
    def _generate_quality_score(self, job_type: int, processing_times: List[float]) -> float:
        """Generate quality score based on job complexity and processing time."""
        complexity_penalty = self.job_complexity_factors[job_type] * 0.1
        speed_penalty = max(0, (10 - np.mean(processing_times)) * 0.05)  # Rushed jobs have lower quality
        
        base_quality = 0.95 - complexity_penalty - speed_penalty
        noise = self.random_state.normal(0, 0.05)
        
        return np.clip(base_quality + noise, 0.0, 1.0)
    
    def _calculate_energy_consumption(self, processing_times: List[float], timestamp: datetime) -> float:
        """Calculate energy consumption with time-of-day pricing."""
        base_consumption = sum(processing_times) * 2.5  # kWh per minute
        
        # Peak hours cost more energy
        hour = timestamp.hour
        if 9 <= hour <= 17:  # Peak hours
            cost_multiplier = 1.5
        elif 18 <= hour <= 22:  # Semi-peak
            cost_multiplier = 1.2
        else:  # Off-peak
            cost_multiplier = 0.8
        
        variation = self.random_state.normal(1.0, self.params.energy_cost_variation)
        return base_consumption * cost_multiplier * variation
    
    def _calculate_material_cost(self, job_type: int, timestamp: datetime) -> float:
        """Calculate material cost with market fluctuations."""
        base_cost = self.job_complexity_factors[job_type] * 50  # Base cost in currency units
        
        # Market fluctuations
        market_factor = 1.0 + 0.3 * np.sin(2 * np.pi * timestamp.timetuple().tm_yday / 365)
        
        # Random daily variations
        daily_variation = self.random_state.normal(1.0, 0.1)
        
        return base_cost * market_factor * daily_variation
    
    def _get_machine_availability(self, timestamp: datetime) -> Dict[int, float]:
        """Get machine availability factors (maintenance, breakdowns)."""
        availability = {}
        for machine_id in range(self.params.num_machines):
            # Planned maintenance (lower availability)
            if timestamp.weekday() == 6 and timestamp.hour < 8:  # Sunday morning maintenance
                base_availability = 0.3
            else:
                base_availability = 0.95
            
            # Random variations
            variation = self.random_state.normal(1.0, 0.05)
            availability[machine_id] = np.clip(base_availability * variation, 0.1, 1.0)
        
        return availability
    
    def _get_weather_impact(self, timestamp: datetime) -> float:
        """Get weather impact factor (affects outdoor operations)."""
        # Simulate seasonal weather patterns
        day_of_year = timestamp.timetuple().tm_yday
        
        # Winter/summer have more extreme weather
        seasonal_severity = 0.2 * abs(np.cos(2 * np.pi * day_of_year / 365))
        
        # Random weather events
        weather_event = self.random_state.exponential(1.0) - 1.0
        
        return 1.0 + seasonal_severity * weather_event * 0.1
    
    def generate_demand_forecast_data(self, forecast_days: int = 30) -> pd.DataFrame:
        """Generate demand forecast training data."""
        dates = [datetime.now() + timedelta(days=i) for i in range(forecast_days)]
        
        forecast_data = []
        for date in dates:
            # Base demand with trends and seasonality
            base_demand = 100  # Base daily demand
            
            # Weekly seasonality (lower on weekends)
            weekly_factor = 0.7 if date.weekday() >= 5 else 1.0
            
            # Annual seasonality
            annual_factor = 1.2 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            
            # Growth trend
            days_from_start = (date - datetime.now()).days
            trend_factor = 1.0 + 0.001 * days_from_start  # 0.1% daily growth
            
            # Random noise
            noise = self.random_state.normal(1.0, 0.15)
            
            predicted_demand = max(0, int(base_demand * weekly_factor * annual_factor * trend_factor * noise))
            
            forecast_data.append({
                'date': date,
                'predicted_demand': predicted_demand,
                'day_of_week': date.weekday(),
                'day_of_year': date.timetuple().tm_yday,
                'is_weekend': date.weekday() >= 5,
                'seasonal_factor': annual_factor,
                'trend_factor': trend_factor
            })
        
        return pd.DataFrame(forecast_data)
    
    def generate_feature_matrix(self, jobs_data: pd.DataFrame) -> np.ndarray:
        """Generate feature matrix for ML training."""
        features = []
        
        for _, job in jobs_data.iterrows():
            job_features = [
                job['job_type'],
                job['batch_size'],
                job['day_of_week'],
                job['hour_of_day'],
                job['seasonal_factor'],
                job['weather_impact'],
                np.mean(job['processing_times']),
                np.std(job['processing_times']),
                job['material_cost'],
                job['energy_consumption'],
                int(job['rush_order']),
                job['quality_score']
            ]
            
            # Add machine availability features
            for machine_id in range(self.params.num_machines):
                job_features.append(job['machine_availability'].get(machine_id, 0.95))
            
            features.append(job_features)
        
        return np.array(features)
    
    def create_ml_dataset(self, num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create a complete ML dataset for training scheduling models.
        
        Returns:
            X: Feature matrix
            y: Target values (actual completion times)
            feature_names: List of feature names
        """
        # Generate historical data
        historical_data = self.generate_historical_data(num_samples)
        
        # Create feature matrix
        X = self.generate_feature_matrix(historical_data)
        
        # Target variable (actual completion time)
        y = historical_data['actual_completion_time'].values
        
        # Feature names
        feature_names = [
            'job_type', 'batch_size', 'day_of_week', 'hour_of_day',
            'seasonal_factor', 'weather_impact', 'mean_processing_time',
            'std_processing_time', 'material_cost', 'energy_consumption',
            'rush_order', 'quality_score'
        ]
        
        # Add machine availability features
        for machine_id in range(self.params.num_machines):
            feature_names.append(f'machine_{machine_id}_availability')
        
        return X, y, feature_names


class MultiObjectiveDataGenerator(RealisticDataGenerator):
    """
    Extended data generator for multi-objective optimization scenarios.
    
    Generates data for optimizing multiple conflicting objectives:
    - Makespan minimization
    - Cost minimization  
    - Quality maximization
    - Energy efficiency
    """
    
    def generate_pareto_training_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Generate training data for Pareto frontier learning."""
        data = []
        
        for i in range(num_samples):
            # Generate random job configurations
            num_jobs = self.random_state.randint(3, 15)
            job_types = self.random_state.randint(0, self.params.num_job_types, num_jobs)
            
            # Calculate different objective values for this configuration
            makespan = self._calculate_makespan(job_types)
            cost = self._calculate_total_cost(job_types)
            quality = self._calculate_average_quality(job_types)
            energy = self._calculate_energy_efficiency(job_types)
            
            # Pareto optimality flag (computed later)
            pareto_optimal = False
            
            data.append({
                'configuration_id': i,
                'job_types': job_types,
                'num_jobs': num_jobs,
                'makespan': makespan,
                'total_cost': cost,
                'average_quality': quality,
                'energy_efficiency': energy,
                'pareto_optimal': pareto_optimal
            })
        
        df = pd.DataFrame(data)
        
        # Calculate Pareto optimality
        df['pareto_optimal'] = self._identify_pareto_optimal(df)
        
        return df
    
    def _calculate_makespan(self, job_types: np.ndarray) -> float:
        """Calculate makespan for given job configuration."""
        total_time = 0
        for job_type in job_types:
            processing_times = self._generate_processing_times(job_type, datetime.now())
            total_time += max(processing_times)  # Simplified makespan calculation
        return total_time
    
    def _calculate_total_cost(self, job_types: np.ndarray) -> float:
        """Calculate total cost including material, energy, and labor."""
        total_cost = 0
        for job_type in job_types:
            material_cost = self._calculate_material_cost(job_type, datetime.now())
            energy_cost = self._calculate_energy_consumption(
                self._generate_processing_times(job_type, datetime.now()), 
                datetime.now()
            ) * 0.15  # Cost per kWh
            labor_cost = sum(self._generate_processing_times(job_type, datetime.now())) * 0.5  # Cost per minute
            
            total_cost += material_cost + energy_cost + labor_cost
        
        return total_cost
    
    def _calculate_average_quality(self, job_types: np.ndarray) -> float:
        """Calculate average quality score for job configuration."""
        quality_scores = []
        for job_type in job_types:
            processing_times = self._generate_processing_times(job_type, datetime.now())
            quality = self._generate_quality_score(job_type, processing_times)
            quality_scores.append(quality)
        
        return np.mean(quality_scores)
    
    def _calculate_energy_efficiency(self, job_types: np.ndarray) -> float:
        """Calculate energy efficiency (lower is better)."""
        total_energy = 0
        total_output = 0
        
        for job_type in job_types:
            processing_times = self._generate_processing_times(job_type, datetime.now())
            energy = self._calculate_energy_consumption(processing_times, datetime.now())
            output = self.job_complexity_factors[job_type]  # Proxy for output value
            
            total_energy += energy
            total_output += output
        
        return total_energy / total_output if total_output > 0 else float('inf')
    
    def _identify_pareto_optimal(self, df: pd.DataFrame) -> List[bool]:
        """Identify Pareto optimal solutions."""
        pareto_optimal = []
        
        for i, row in df.iterrows():
            is_optimal = True
            
            for j, other_row in df.iterrows():
                if i == j:
                    continue
                
                # Check if other solution dominates this one
                # (assuming minimization for makespan, cost, energy; maximization for quality)
                dominates = (
                    other_row['makespan'] <= row['makespan'] and
                    other_row['total_cost'] <= row['total_cost'] and
                    other_row['energy_efficiency'] <= row['energy_efficiency'] and
                    other_row['average_quality'] >= row['average_quality'] and
                    (other_row['makespan'] < row['makespan'] or
                     other_row['total_cost'] < row['total_cost'] or
                     other_row['energy_efficiency'] < row['energy_efficiency'] or
                     other_row['average_quality'] > row['average_quality'])
                )
                
                if dominates:
                    is_optimal = False
                    break
            
            pareto_optimal.append(is_optimal)
        
        return pareto_optimal