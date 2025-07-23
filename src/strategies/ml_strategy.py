import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from strategies.base import SchedulerStrategy, OptimizationResult
from models.job import Job
from models.schedule import ScheduleResult


class MLEnhancedScheduler(SchedulerStrategy):
    """
    Machine Learning enhanced scheduling strategy.
    
    This addresses the major limitation of traditional heuristics by using
    ML models to predict processing times, quality, and optimize multiple objectives.
    """
    
    def __init__(self, ml_pipeline=None, base_strategy: str = "predictive_optimization"):
        super().__init__(f"ML Enhanced Scheduler ({base_strategy})")
        self.ml_pipeline = ml_pipeline
        self.base_strategy = base_strategy
        self.confidence_threshold = 0.7
        
        # If no ML pipeline provided, create a dummy one for demonstration
        if self.ml_pipeline is None:
            self._create_dummy_pipeline()
    
    def _create_dummy_pipeline(self):
        """Create dummy ML pipeline for demonstration purposes."""
        class DummyMLPipeline:
            def predict_job_performance(self, job_features):
                # Simulate ML predictions with some realistic variations
                num_jobs = len(job_features)
                return {
                    'processing_time': np.random.exponential(15, num_jobs),
                    'processing_uncertainty': np.random.exponential(2, num_jobs),
                    'quality': np.random.beta(8, 2, num_jobs),  # Skewed towards high quality
                    'makespan': np.random.exponential(20, num_jobs),
                    'cost': np.random.exponential(100, num_jobs),
                    'energy': np.random.exponential(50, num_jobs)
                }
            
            def recommend_optimal_schedule(self, job_features, objective_weights=None):
                predictions = self.predict_job_performance(job_features)
                # Simple ranking based on predicted processing time
                rankings = np.argsort(predictions['processing_time'])
                return {
                    'recommended_sequence': rankings.tolist(),
                    'predicted_performance': predictions,
                    'objective_scores': np.random.random(len(job_features)),
                    'confidence': np.random.uniform(0.6, 0.9)
                }
        
        self.ml_pipeline = DummyMLPipeline()
    
    def find_optimal_schedule(self, jobs: List[Job], num_machines: int,
                            objective_weights: Dict[str, float] = None,
                            use_uncertainty: bool = True,
                            fallback_strategy: str = "balanced",
                            **kwargs) -> ScheduleResult:
        """
        Find optimal schedule using ML predictions and multi-objective optimization.
        
        Args:
            jobs: List of jobs to schedule
            num_machines: Number of machines available
            objective_weights: Weights for different objectives
            use_uncertainty: Whether to consider prediction uncertainty
            fallback_strategy: Strategy to use if ML confidence is low
            **kwargs: Additional parameters
            
        Returns:
            ScheduleResult with ML-optimized schedule
        """
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(jobs, num_machines)
        
        # Create schedule manager
        schedule_manager = self.create_schedule_manager(jobs, num_machines)
        
        print(f"Running ML-Enhanced Scheduling with {len(jobs)} jobs...")
        
        try:
            # Extract job features for ML prediction
            job_features = self._extract_job_features(jobs)
            
            # Get ML recommendations
            ml_recommendations = self.ml_pipeline.recommend_optimal_schedule(
                job_features, objective_weights
            )
            
            # Check confidence level
            confidence = ml_recommendations['confidence']
            print(f"ML Confidence: {confidence:.3f}")
            
            if confidence >= self.confidence_threshold:
                # Use ML recommendations
                sequence = ml_recommendations['recommended_sequence']
                print("Using ML-recommended sequence")
            else:
                # Fallback to traditional heuristic
                print(f"Low ML confidence ({confidence:.3f}), falling back to {fallback_strategy}")
                sequence = self._fallback_scheduling(jobs, fallback_strategy)
            
            # Apply uncertainty-aware adjustments if requested
            if use_uncertainty and confidence >= self.confidence_threshold:
                sequence = self._apply_uncertainty_adjustments(
                    sequence, ml_recommendations['predicted_performance']
                )
            
            # Execute the schedule
            result = schedule_manager.execute_sequence(sequence)
            
            # Enhance result with ML predictions
            result = self._enhance_result_with_ml_insights(result, ml_recommendations)
            
        except Exception as e:
            print(f"ML scheduling failed: {e}, falling back to heuristic")
            # Fallback to safe heuristic
            sequence = self._fallback_scheduling(jobs, fallback_strategy)
            result = schedule_manager.execute_sequence(sequence)
        
        execution_time = time.time() - start_time
        print(f"ML-Enhanced scheduling completed in {execution_time:.3f} seconds")
        
        return result
    
    def _extract_job_features(self, jobs: List[Job]) -> np.ndarray:
        """Extract features from jobs for ML prediction."""
        features = []
        
        for job in jobs:
            # Basic job features
            job_features = [
                job.id,
                len(job.processing_times),  # num_machines
                np.mean(job.processing_times),
                np.std(job.processing_times),
                np.min(job.processing_times),
                np.max(job.processing_times),
                job.get_total_processing_time(),
            ]
            
            # Add processing times for each machine
            job_features.extend(list(job.processing_times))
            
            # Add temporal features (current time)
            from datetime import datetime
            now = datetime.now()
            job_features.extend([
                now.weekday(),
                now.hour,
                now.timetuple().tm_yday,  # day of year
            ])
            
            # Add synthetic features for demonstration
            job_features.extend([
                1.0,  # seasonal_factor
                1.0,  # weather_impact
                50.0,  # material_cost
                25.0,  # energy_consumption
                0,     # rush_order
                0.9,   # quality_score
            ])
            
            # Machine availability (simplified)
            for machine_id in range(5):  # Assume max 5 machines
                if machine_id < len(job.processing_times):
                    job_features.append(0.95)  # 95% availability
                else:
                    job_features.append(0.0)   # Machine not used
            
            features.append(job_features)
        
        return np.array(features)
    
    def _fallback_scheduling(self, jobs: List[Job], strategy: str) -> List[int]:
        """Fallback to traditional heuristic scheduling."""
        job_ids = [job.id for job in jobs]
        
        if strategy == "shortest_processing_time":
            # Sort by total processing time
            job_times = [(job.id, job.get_total_processing_time()) for job in jobs]
            job_times.sort(key=lambda x: x[1])
            return [job_id for job_id, _ in job_times]
        
        elif strategy == "longest_processing_time":
            job_times = [(job.id, job.get_total_processing_time()) for job in jobs]
            job_times.sort(key=lambda x: x[1], reverse=True)
            return [job_id for job_id, _ in job_times]
        
        elif strategy == "balanced":
            # Balance based on variance in processing times
            job_metrics = []
            for job in jobs:
                total_time = job.get_total_processing_time()
                variance = np.var(job.processing_times)
                balance_score = total_time / (1 + variance)
                job_metrics.append((job.id, balance_score))
            
            job_metrics.sort(key=lambda x: x[1], reverse=True)
            return [job_id for job_id, _ in job_metrics]
        
        else:
            # Random fallback
            import random
            sequence = job_ids.copy()
            random.shuffle(sequence)
            return sequence
    
    def _apply_uncertainty_adjustments(self, sequence: List[int], 
                                     predictions: Dict[str, np.ndarray]) -> List[int]:
        """Adjust sequence based on prediction uncertainty."""
        if 'processing_uncertainty' not in predictions:
            return sequence
        
        # Create mapping from job_id to uncertainty
        uncertainty_map = {}
        for i, job_id in enumerate(sequence):
            uncertainty_map[job_id] = predictions['processing_uncertainty'][i]
        
        # Prioritize jobs with lower uncertainty (more reliable predictions)
        adjusted_sequence = sorted(sequence, key=lambda job_id: uncertainty_map.get(job_id, 0))
        
        print(f"Applied uncertainty adjustment: avg uncertainty = {np.mean(list(uncertainty_map.values())):.2f}")
        
        return adjusted_sequence
    
    def _enhance_result_with_ml_insights(self, result: ScheduleResult, 
                                       ml_recommendations: Dict) -> ScheduleResult:
        """Enhance schedule result with ML insights."""
        # Add ML predictions to the result (this would require extending ScheduleResult)
        if hasattr(result, 'ml_insights'):
            result.ml_insights = {
                'confidence': ml_recommendations['confidence'],
                'predicted_quality': np.mean(ml_recommendations['predicted_performance']['quality']),
                'predicted_cost': np.sum(ml_recommendations['predicted_performance']['cost']),
                'objective_scores': ml_recommendations['objective_scores'],
                'ml_strategy': self.base_strategy
            }
        
        return result
    
    def get_strategy_info(self) -> Dict[str, any]:
        """Get information about the ML-enhanced strategy."""
        return {
            'name': self.name,
            'type': 'ml_enhanced',
            'base_strategy': self.base_strategy,
            'time_complexity': 'O(n log n + ML_prediction)',
            'guarantees_optimal': False,
            'ml_enabled': True,
            'confidence_threshold': self.confidence_threshold,
            'characteristics': [
                'Uses machine learning to predict processing times',
                'Multi-objective optimization with learned preferences',
                'Uncertainty-aware scheduling decisions',
                'Adaptive fallback to proven heuristics',
                'Continuous learning from historical performance',
                'Quality, cost, and energy optimization'
            ],
            'objectives_supported': [
                'makespan_minimization',
                'cost_minimization', 
                'quality_maximization',
                'energy_efficiency',
                'uncertainty_reduction'
            ]
        }
    
    def set_objective_weights(self, weights: Dict[str, float]):
        """Set weights for multi-objective optimization."""
        # Validate weights
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Objective weights must sum to 1.0")
        
        self.objective_weights = weights
        print(f"Updated objective weights: {weights}")
    
    def predict_schedule_performance(self, jobs: List[Job], num_machines: int) -> Dict[str, any]:
        """Predict schedule performance without actually executing it."""
        job_features = self._extract_job_features(jobs)
        ml_recommendations = self.ml_pipeline.recommend_optimal_schedule(job_features)
        
        predicted_performance = ml_recommendations['predicted_performance']
        
        return {
            'predicted_makespan': np.max(predicted_performance['makespan']),
            'predicted_total_cost': np.sum(predicted_performance['cost']),
            'predicted_avg_quality': np.mean(predicted_performance['quality']),
            'predicted_energy_consumption': np.sum(predicted_performance['energy']),
            'confidence': ml_recommendations['confidence'],
            'recommended_sequence': ml_recommendations['recommended_sequence']
        }


class ReinforcementLearningScheduler(SchedulerStrategy):
    """
    Reinforcement Learning based scheduler (conceptual implementation).
    
    This would learn optimal scheduling policies through interaction
    with the manufacturing environment.
    """
    
    def __init__(self):
        super().__init__("Reinforcement Learning Scheduler")
        self.policy_network = None
        self.is_trained = False
        self.exploration_rate = 0.1
    
    def find_optimal_schedule(self, jobs: List[Job], num_machines: int, 
                            **kwargs) -> ScheduleResult:
        """
        Find schedule using learned RL policy.
        
        This is a conceptual implementation - would require actual RL training.
        """
        # For demonstration, use a simple learned heuristic
        print("Using Reinforcement Learning policy...")
        
        # Create schedule manager
        schedule_manager = self.create_schedule_manager(jobs, num_machines)
        
        # Simulate RL policy decision
        if not self.is_trained:
            print("RL agent not trained, using exploration...")
            # Random exploration
            import random
            sequence = list(range(len(jobs)))
            random.shuffle(sequence)
        else:
            # Use learned policy (simplified)
            job_priorities = []
            for job in jobs:
                # Simulate learned priority function
                priority = (
                    -job.get_total_processing_time() * 0.6 +  # Prefer shorter jobs
                    -np.var(job.processing_times) * 0.4      # Prefer balanced jobs
                )
                job_priorities.append((job.id, priority))
            
            job_priorities.sort(key=lambda x: x[1], reverse=True)
            sequence = [job_id for job_id, _ in job_priorities]
        
        result = schedule_manager.execute_sequence(sequence)
        
        print(f"RL Scheduler produced sequence: {sequence}")
        return result
    
    def get_strategy_info(self) -> Dict[str, any]:
        """Get information about the RL strategy."""
        return {
            'name': self.name,
            'type': 'reinforcement_learning',
            'time_complexity': 'O(n) - after training',
            'guarantees_optimal': False,
            'requires_training': True,
            'adaptive': True,
            'characteristics': [
                'Learns from environment interaction',
                'Adapts to changing conditions',
                'Balances exploration and exploitation',
                'Can handle complex state spaces',
                'Requires historical data for training'
            ]
        }


class EnsembleMLScheduler(SchedulerStrategy):
    """
    Ensemble of multiple ML strategies for robust scheduling.
    
    Combines predictions from multiple models to improve reliability.
    """
    
    def __init__(self, strategies: List[SchedulerStrategy] = None):
        super().__init__("Ensemble ML Scheduler")
        
        if strategies is None:
            # Create default ensemble
            self.strategies = [
                MLEnhancedScheduler(base_strategy="predictive_optimization"),
                MLEnhancedScheduler(base_strategy="quality_focused"),
                MLEnhancedScheduler(base_strategy="cost_focused"),
            ]
        else:
            self.strategies = strategies
        
        self.voting_method = "weighted_average"
    
    def find_optimal_schedule(self, jobs: List[Job], num_machines: int,
                            **kwargs) -> ScheduleResult:
        """Find optimal schedule using ensemble voting."""
        print(f"Running Ensemble ML Scheduler with {len(self.strategies)} models...")
        
        # Get predictions from all strategies
        all_predictions = []
        all_sequences = []
        
        for i, strategy in enumerate(self.strategies):
            try:
                if hasattr(strategy, 'predict_schedule_performance'):
                    pred = strategy.predict_schedule_performance(jobs, num_machines)
                    all_predictions.append(pred)
                    all_sequences.append(pred['recommended_sequence'])
                else:
                    # Fallback: actually run the strategy
                    result = strategy.find_optimal_schedule(jobs, num_machines, **kwargs)
                    all_sequences.append(result.job_sequence)
                    all_predictions.append({
                        'predicted_makespan': result.makespan,
                        'confidence': 0.8  # Default confidence
                    })
                
                print(f"Strategy {i+1} completed")
                
            except Exception as e:
                print(f"Strategy {i+1} failed: {e}")
                continue
        
        if not all_sequences:
            raise RuntimeError("All ensemble strategies failed")
        
        # Combine predictions using voting
        final_sequence = self._combine_sequences(all_sequences, all_predictions)
        
        # Execute final sequence
        schedule_manager = self.create_schedule_manager(jobs, num_machines)
        result = schedule_manager.execute_sequence(final_sequence)
        
        print(f"Ensemble produced final sequence: {final_sequence}")
        return result
    
    def _combine_sequences(self, sequences: List[List[int]], 
                          predictions: List[Dict]) -> List[int]:
        """Combine multiple sequences using ensemble voting."""
        if not sequences:
            raise ValueError("No sequences to combine")
        
        if len(sequences) == 1:
            return sequences[0]
        
        # Weight by confidence if available
        weights = []
        for pred in predictions:
            weights.append(pred.get('confidence', 1.0))
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Voting: rank-based ensemble
        job_scores = {}
        num_jobs = len(sequences[0])
        
        # Initialize scores
        for job_id in range(num_jobs):
            job_scores[job_id] = 0.0
        
        # Aggregate rankings weighted by confidence
        for seq_idx, sequence in enumerate(sequences):
            weight = weights[seq_idx]
            for position, job_id in enumerate(sequence):
                # Earlier positions get higher scores
                score = (num_jobs - position) * weight
                job_scores[job_id] += score
        
        # Sort by final scores (highest first)
        final_sequence = sorted(job_scores.keys(), key=lambda job_id: job_scores[job_id], reverse=True)
        
        return final_sequence
    
    def get_strategy_info(self) -> Dict[str, any]:
        """Get information about the ensemble strategy."""
        return {
            'name': self.name,
            'type': 'ensemble_ml',
            'num_models': len(self.strategies),
            'voting_method': self.voting_method,
            'time_complexity': 'O(k * n log n) where k = number of models',
            'guarantees_optimal': False,
            'robustness': 'high',
            'characteristics': [
                'Combines multiple ML models for robust predictions',
                'Reduces overfitting through model diversity',
                'Weighted voting based on model confidence',
                'Handles individual model failures gracefully',
                'Better generalization than single models'
            ]
        }


# Example usage and integration
def demonstrate_ml_scheduling():
    """Demonstrate ML-enhanced scheduling capabilities."""
    from models.job import Job
    
    # Create sample jobs
    jobs = [
        Job(id=0, processing_times=[5, 3, 4]),
        Job(id=1, processing_times=[2, 6, 1]),
        Job(id=2, processing_times=[4, 2, 5]),
        Job(id=3, processing_times=[3, 4, 2]),
        Job(id=4, processing_times=[1, 5, 3])
    ]
    
    # Test ML-enhanced scheduler
    ml_scheduler = MLEnhancedScheduler()
    
    # Set multi-objective weights
    ml_scheduler.set_objective_weights({
        'makespan': 0.4,
        'cost': 0.3,
        'quality': 0.2,
        'energy': 0.1
    })
    
    # Get performance prediction
    prediction = ml_scheduler.predict_schedule_performance(jobs, 3)
    print("ML Performance Prediction:")
    for key, value in prediction.items():
        print(f"  {key}: {value}")
    
    # Find optimal schedule
    result = ml_scheduler.find_optimal_schedule(jobs, 3)
    print(f"\nML Optimal Sequence: {result.job_sequence}")
    print(f"Actual Makespan: {result.makespan}")
    
    # Test ensemble scheduler
    ensemble_scheduler = EnsembleMLScheduler()
    ensemble_result = ensemble_scheduler.find_optimal_schedule(jobs, 3)
    print(f"\nEnsemble Optimal Sequence: {ensemble_result.job_sequence}")
    print(f"Ensemble Makespan: {ensemble_result.makespan}")


if __name__ == "__main__":
    demonstrate_ml_scheduling()