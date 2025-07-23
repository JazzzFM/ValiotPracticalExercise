import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import joblib
from datetime import datetime, timedelta

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# Optional ML dependencies with better error handling
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None

def check_ml_dependencies():
    """Check which optional ML dependencies are available."""
    dependencies = {
        'xgboost': HAS_XGBOOST,
        'tensorflow': HAS_TENSORFLOW,
        'sklearn': True,  # Required dependency
        'numpy': True,    # Required dependency
        'pandas': True    # Required dependency
    }
    return dependencies

def warn_missing_dependency(dependency_name: str, alternative: str = None):
    """Warn user about missing optional dependency."""
    message = f"Warning: {dependency_name} not available. "
    if alternative:
        message += f"Using {alternative} as fallback."
    else:
        message += "Some features may be limited."
    print(message)


@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    mse: float
    mae: float
    r2: float
    cross_val_score: float
    training_time: float
    prediction_time: float


class PredictiveModel(ABC):
    """Base class for predictive models in manufacturing scheduling."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = None
        self.performance_metrics = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'PredictiveModel':
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Time prediction
        import time
        start_time = time.time()
        predictions = self.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Cross-validation on training data (if available)
        cv_score = 0.0  # Placeholder
        
        return ModelPerformance(
            mse=mse,
            mae=mae,
            r2=r2,
            cross_val_score=cv_score,
            training_time=0.0,  # Set during training
            prediction_time=prediction_time
        )
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'name': self.name,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.name = model_data.get('name', self.name)
        self.performance_metrics = model_data.get('performance_metrics')
        self.is_trained = True


class ProcessingTimePredictor(PredictiveModel):
    """
    Predicts actual processing times based on job characteristics.
    
    This addresses the limitation of fixed processing times by learning
    from historical data to predict realistic completion times.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        super().__init__(f"ProcessingTimePredictor_{model_type}")
        self.model_type = model_type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying ML model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            if HAS_XGBOOST:
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            else:
                # Fallback to gradient boosting
                warn_missing_dependency("XGBoost", "GradientBoostingRegressor")
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
        elif self.model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.01,
                random_state=42,
                max_iter=500
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None, **kwargs) -> 'ProcessingTimePredictor':
        """Train the processing time prediction model."""
        # Input validation
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. Got X: {X.shape[0]}, y: {y.shape[0]}")
        
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for training")
        
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("X and y cannot contain NaN values")
        
        import time
        start_time = time.time()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Store metadata
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        # Cross-validation performance
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        
        # Store training performance
        y_pred_train = self.model.predict(X_scaled)
        self.performance_metrics = ModelPerformance(
            mse=mean_squared_error(y, y_pred_train),
            mae=mean_absolute_error(y, y_pred_train),
            r2=r2_score(y, y_pred_train),
            cross_val_score=cv_scores.mean(),
            training_time=training_time,
            prediction_time=0.0
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict processing times."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        
        if X.shape[1] != len(self.feature_names) if self.feature_names else True:
            expected_features = len(self.feature_names) if self.feature_names else "unknown"
            raise ValueError(f"X has {X.shape[1]} features, expected {expected_features}")
        
        if np.any(np.isnan(X)):
            raise ValueError("X cannot contain NaN values")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict processing times with uncertainty estimates."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.predict(X)
        
        # Estimate uncertainty using model-specific methods
        if self.model_type == 'random_forest':
            # Use prediction variance across trees
            X_scaled = self.scaler.transform(X)
            tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
            uncertainty = np.std(tree_predictions, axis=0)
        else:
            # Fallback: use training error as uncertainty estimate
            train_mae = self.performance_metrics.mae if self.performance_metrics else 1.0
            uncertainty = np.full(predictions.shape, train_mae)
        
        return predictions, uncertainty
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance_scores = np.abs(self.model.coef_)
        else:
            return {}
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance_scores))]
        return dict(zip(feature_names, importance_scores))


class DemandForecaster(PredictiveModel):
    """
    Forecasts manufacturing demand using time series analysis.
    
    This enables proactive scheduling by predicting future demand patterns.
    """
    
    def __init__(self, model_type: str = 'lstm'):
        super().__init__(f"DemandForecaster_{model_type}")
        self.model_type = model_type
        self.lookback_window = 30
        self.forecast_horizon = 7
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the forecasting model."""
        if self.model_type == 'lstm' and HAS_TENSORFLOW:
            self.model = self._create_lstm_model()
        elif self.model_type == 'xgboost':
            if HAS_XGBOOST:
                self.model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42
                )
            else:
                # Fallback to gradient boosting  
                self.model = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42
                )
        else:
            # Fallback to simple model
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
    
    def _create_lstm_model(self):
        """Create LSTM model for time series forecasting."""
        if not HAS_TENSORFLOW:
            warn_missing_dependency("TensorFlow", "GradientBoostingRegressor")
            raise ImportError("TensorFlow required for LSTM model. Use alternative forecasting method.")
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_window, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def prepare_time_series_data(self, demand_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for training."""
        # Sort by date
        demand_data = demand_data.sort_values('date')
        demand_values = demand_data['predicted_demand'].values
        
        X, y = [], []
        
        for i in range(self.lookback_window, len(demand_values) - self.forecast_horizon + 1):
            X.append(demand_values[i - self.lookback_window:i])
            y.append(demand_values[i:i + self.forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM if needed
        if self.model_type == 'lstm':
            X = X.reshape((X.shape[0], X.shape[1], 1))
        else:
            # Flatten for traditional ML models
            X = X.reshape(X.shape[0], -1)
        
        return X, y
    
    def fit(self, demand_data: pd.DataFrame, **kwargs) -> 'DemandForecaster':
        """Train the demand forecasting model."""
        import time
        start_time = time.time()
        
        # Prepare data
        X, y = self.prepare_time_series_data(demand_data)
        
        # Scale data
        self.scaler = StandardScaler()
        
        if self.model_type == 'lstm':
            # Scale the time series data
            X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
            # Scale targets
            self.target_scaler = StandardScaler()
            y_scaled = self.target_scaler.fit_transform(y)
            
            # Train LSTM
            self.model.fit(
                X_scaled, y_scaled,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
        else:
            # Traditional ML approach
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        # Store performance metrics (simplified)
        self.performance_metrics = ModelPerformance(
            mse=0.0,  # Would calculate on validation set
            mae=0.0,
            r2=0.0,
            cross_val_score=0.0,
            training_time=training_time,
            prediction_time=0.0
        )
        
        return self
    
    def predict(self, recent_demand: np.ndarray) -> np.ndarray:
        """Forecast future demand."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Prepare input
        if self.model_type == 'lstm':
            X = recent_demand.reshape(1, self.lookback_window, 1)
            X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
            prediction_scaled = self.model.predict(X_scaled, verbose=0)
            prediction = self.target_scaler.inverse_transform(prediction_scaled)
        else:
            X = recent_demand.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)
        
        return prediction.flatten()


class QualityPredictor(PredictiveModel):
    """
    Predicts product quality based on manufacturing parameters.
    
    This enables quality-aware scheduling optimization.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        super().__init__(f"QualityPredictor_{model_type}")
        self.model_type = model_type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize quality prediction model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=3,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVR(kernel='rbf', C=100, gamma='scale')
        elif self.model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(80, 40, 20),
                activation='relu',
                solver='adam',
                alpha=0.01,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None, **kwargs) -> 'QualityPredictor':
        """Train the quality prediction model."""
        import time
        start_time = time.time()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Hyperparameter tuning for important models
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 150, 200],
                'max_depth': [8, 12, 16],
                'min_samples_split': [3, 5, 7]
            }
            grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='r2')
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_scaled, y)
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        # Performance evaluation
        y_pred_train = self.model.predict(X_scaled)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        
        self.performance_metrics = ModelPerformance(
            mse=mean_squared_error(y, y_pred_train),
            mae=mean_absolute_error(y, y_pred_train),
            r2=r2_score(y, y_pred_train),
            cross_val_score=cv_scores.mean(),
            training_time=training_time,
            prediction_time=0.0
        )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict quality scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_quality_class(self, X: np.ndarray, 
                            thresholds: List[float] = [0.8, 0.9, 0.95]) -> np.ndarray:
        """Predict quality classes (low, medium, high, excellent)."""
        quality_scores = self.predict(X)
        
        classes = np.zeros(len(quality_scores), dtype=str)
        classes[quality_scores < thresholds[0]] = 'low'
        classes[(quality_scores >= thresholds[0]) & (quality_scores < thresholds[1])] = 'medium'
        classes[(quality_scores >= thresholds[1]) & (quality_scores < thresholds[2])] = 'high'
        classes[quality_scores >= thresholds[2]] = 'excellent'
        
        return classes


class MultiObjectivePredictor:
    """
    Ensemble of models for multi-objective optimization predictions.
    
    Predicts multiple objectives simultaneously for Pareto optimization.
    """
    
    def __init__(self):
        self.models = {
            'makespan': ProcessingTimePredictor('gradient_boosting'),
            'cost': ProcessingTimePredictor('random_forest'),
            'quality': QualityPredictor('neural_network'),
            'energy': ProcessingTimePredictor('gradient_boosting')
        }
        self.is_trained = False
    
    def fit(self, X: np.ndarray, objectives: Dict[str, np.ndarray], 
            feature_names: List[str] = None) -> 'MultiObjectivePredictor':
        """Train all objective predictors."""
        for objective_name, model in self.models.items():
            if objective_name in objectives:
                print(f"Training {objective_name} predictor...")
                model.fit(X, objectives[objective_name], feature_names)
        
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict all objectives."""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = {}
        for objective_name, model in self.models.items():
            if model.is_trained:
                predictions[objective_name] = model.predict(X)
        
        return predictions
    
    def predict_pareto_efficiency(self, X: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Predict likelihood of solutions being Pareto efficient."""
        # This would require training a separate classifier on Pareto optimality
        # For now, return a heuristic based on objective predictions
        predictions = self.predict(X)
        
        # Simple heuristic: solutions with balanced objectives are more likely Pareto efficient
        if len(predictions) >= 2:
            # Normalize predictions and calculate "balance" score
            normalized_preds = {}
            for obj_name, preds in predictions.items():
                normalized_preds[obj_name] = (preds - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-8)
            
            # Calculate variance across objectives (lower variance = more balanced)
            pred_matrix = np.column_stack(list(normalized_preds.values()))
            balance_scores = 1.0 - np.var(pred_matrix, axis=1)
            
            return balance_scores > threshold
        
        return np.ones(len(X), dtype=bool)
    
    def get_model_performance(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all trained models."""
        performance = {}
        for objective_name, model in self.models.items():
            if model.is_trained and model.performance_metrics:
                performance[objective_name] = model.performance_metrics
        return performance


class SchedulingMLPipeline:
    """
    Complete ML pipeline for intelligent manufacturing scheduling.
    
    Integrates all predictive models for comprehensive scheduling optimization.
    """
    
    def __init__(self):
        self.processing_time_model = ProcessingTimePredictor('random_forest')
        self.demand_forecaster = DemandForecaster('gradient_boosting')  # Fallback if no TensorFlow
        self.quality_predictor = QualityPredictor('random_forest')
        self.multi_objective_model = MultiObjectivePredictor()
        
        self.pipeline_trained = False
    
    def train_pipeline(self, historical_data: pd.DataFrame, 
                      demand_data: pd.DataFrame = None):
        """Train the complete ML pipeline."""
        print("Training ML Pipeline for Intelligent Scheduling...")
        
        # Prepare features
        from ..ml.data_generator import RealisticDataGenerator
        generator = RealisticDataGenerator()
        
        # Extract features and targets
        X = generator.generate_feature_matrix(historical_data)
        feature_names = [
            'job_type', 'batch_size', 'day_of_week', 'hour_of_day',
            'seasonal_factor', 'weather_impact', 'mean_processing_time',
            'std_processing_time', 'material_cost', 'energy_consumption',
            'rush_order', 'quality_score'
        ]
        
        # Add machine availability features
        for i in range(5):  # Assuming 5 machines
            feature_names.append(f'machine_{i}_availability')
        
        # Train processing time predictor
        print("1. Training processing time predictor...")
        y_processing = historical_data['actual_completion_time'].values
        self.processing_time_model.fit(X, y_processing, feature_names)
        
        # Train quality predictor
        print("2. Training quality predictor...")
        y_quality = historical_data['quality_score'].values
        self.quality_predictor.fit(X, y_quality, feature_names)
        
        # Train demand forecaster if demand data available
        if demand_data is not None:
            print("3. Training demand forecaster...")
            self.demand_forecaster.fit(demand_data)
        
        # Train multi-objective predictor
        print("4. Training multi-objective predictor...")
        objectives = {
            'makespan': historical_data['actual_completion_time'].values,
            'cost': historical_data['material_cost'].values + historical_data['energy_consumption'].values * 0.15,
            'quality': historical_data['quality_score'].values,
            'energy': historical_data['energy_consumption'].values
        }
        self.multi_objective_model.fit(X, objectives, feature_names)
        
        self.pipeline_trained = True
        print("ML Pipeline training completed!")
    
    def predict_job_performance(self, job_features: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict comprehensive job performance metrics."""
        if not self.pipeline_trained:
            raise ValueError("Pipeline must be trained before prediction")
        
        results = {}
        
        # Processing time prediction with uncertainty
        processing_pred, processing_uncertainty = self.processing_time_model.predict_with_uncertainty(job_features)
        results['processing_time'] = processing_pred
        results['processing_uncertainty'] = processing_uncertainty
        
        # Quality prediction
        quality_pred = self.quality_predictor.predict(job_features)
        results['quality'] = quality_pred
        
        # Multi-objective predictions
        multi_obj_pred = self.multi_objective_model.predict(job_features)
        results.update(multi_obj_pred)
        
        return results
    
    def recommend_optimal_schedule(self, job_features: np.ndarray, 
                                 objective_weights: Dict[str, float] = None) -> Dict[str, any]:
        """Recommend optimal schedule based on ML predictions."""
        if objective_weights is None:
            objective_weights = {'makespan': 0.4, 'cost': 0.3, 'quality': 0.2, 'energy': 0.1}
        
        predictions = self.predict_job_performance(job_features)
        
        # Calculate weighted objective scores
        total_scores = np.zeros(len(job_features))
        
        for objective, weight in objective_weights.items():
            if objective in predictions:
                # Normalize predictions (assuming minimization for makespan, cost, energy; maximization for quality)
                preds = predictions[objective]
                if objective == 'quality':
                    # Higher is better for quality
                    normalized = (preds - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-8)
                else:
                    # Lower is better for makespan, cost, energy
                    normalized = 1.0 - (preds - np.min(preds)) / (np.max(preds) - np.min(preds) + 1e-8)
                
                total_scores += weight * normalized
        
        # Rank jobs by total score
        job_rankings = np.argsort(total_scores)[::-1]  # Best first
        
        return {
            'recommended_sequence': job_rankings.tolist(),
            'predicted_performance': predictions,
            'objective_scores': total_scores,
            'confidence': np.mean(1.0 / (1.0 + predictions.get('processing_uncertainty', np.ones(len(job_features)))))
        }
    
    def save_pipeline(self, base_path: str):
        """Save the complete trained pipeline."""
        if not self.pipeline_trained:
            raise ValueError("Cannot save untrained pipeline")
        
        self.processing_time_model.save_model(f"{base_path}_processing_time.joblib")
        self.quality_predictor.save_model(f"{base_path}_quality.joblib")
        
        # Save demand forecaster if trained
        if self.demand_forecaster.is_trained:
            self.demand_forecaster.save_model(f"{base_path}_demand.joblib")
        
        print(f"ML Pipeline saved to {base_path}_*.joblib")
    
    def load_pipeline(self, base_path: str):
        """Load a pre-trained pipeline."""
        import os
        
        # Load processing time model
        if os.path.exists(f"{base_path}_processing_time.joblib"):
            self.processing_time_model.load_model(f"{base_path}_processing_time.joblib")
        
        # Load quality model
        if os.path.exists(f"{base_path}_quality.joblib"):
            self.quality_predictor.load_model(f"{base_path}_quality.joblib")
        
        # Load demand model if exists
        if os.path.exists(f"{base_path}_demand.joblib"):
            self.demand_forecaster.load_model(f"{base_path}_demand.joblib")
        
        self.pipeline_trained = True
        print(f"ML Pipeline loaded from {base_path}_*.joblib")


# Usage example and model validation
def validate_ml_models():
    """Validate ML models with synthetic data."""
    from .data_generator import RealisticDataGenerator
    
    # Generate training data
    generator = RealisticDataGenerator()
    historical_data = generator.generate_historical_data(1000)
    demand_data = generator.generate_demand_forecast_data(90)
    
    # Create and train pipeline
    pipeline = SchedulingMLPipeline()
    pipeline.train_pipeline(historical_data, demand_data)
    
    # Generate test data
    test_data = generator.generate_historical_data(100)
    X_test = generator.generate_feature_matrix(test_data)
    
    # Make predictions
    predictions = pipeline.predict_job_performance(X_test)
    recommendations = pipeline.recommend_optimal_schedule(X_test)
    
    print("ML Model Validation Results:")
    print(f"Processing Time MAE: {np.mean(np.abs(predictions['processing_time'] - test_data['actual_completion_time'])):.2f}")
    print(f"Quality MAE: {np.mean(np.abs(predictions['quality'] - test_data['quality_score'])):.3f}")
    print(f"Confidence Score: {recommendations['confidence']:.3f}")
    
    return pipeline, predictions, recommendations


if __name__ == "__main__":
    # Run validation
    validate_ml_models()