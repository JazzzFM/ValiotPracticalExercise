import unittest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.predictive_models import (
    ProcessingTimePredictor, check_ml_dependencies, warn_missing_dependency
)


class TestMLDependencies(unittest.TestCase):
    """Test cases for ML dependency handling."""
    
    def test_check_ml_dependencies(self):
        """Test checking ML dependencies."""
        deps = check_ml_dependencies()
        
        # Should always have these required dependencies
        self.assertTrue(deps['sklearn'])
        self.assertTrue(deps['numpy'])  
        self.assertTrue(deps['pandas'])
        
        # Optional dependencies may or may not be available
        self.assertIsInstance(deps['xgboost'], bool)
        self.assertIsInstance(deps['tensorflow'], bool)
    
    @patch('builtins.print')
    def test_warn_missing_dependency(self, mock_print):
        """Test dependency warning function."""
        warn_missing_dependency("TestLib", "Alternative")
        mock_print.assert_called_once()
        
        call_args = mock_print.call_args[0][0]
        self.assertIn("TestLib not available", call_args)
        self.assertIn("Alternative", call_args)


class TestProcessingTimePredictor(unittest.TestCase):
    """Test cases for ProcessingTimePredictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = ProcessingTimePredictor('random_forest')
        
        # Create synthetic training data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.rand(100) * 10 + 5
        self.feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        # Test data
        self.X_test = np.random.rand(20, 5)
        self.y_test = np.random.rand(20) * 10 + 5
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.predictor.model_type, 'random_forest')
        self.assertFalse(self.predictor.is_trained)
        self.assertIsNone(self.predictor.scaler)
    
    def test_model_training(self):
        """Test model training."""
        predictor = self.predictor.fit(self.X_train, self.y_train, self.feature_names)
        
        # Should return self
        self.assertEqual(predictor, self.predictor)
        
        # Should be trained
        self.assertTrue(self.predictor.is_trained)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertEqual(self.predictor.feature_names, self.feature_names)
        self.assertIsNotNone(self.predictor.performance_metrics)
    
    def test_model_training_input_validation(self):
        """Test model training input validation."""
        # Test non-numpy arrays
        with self.assertRaises(TypeError):
            self.predictor.fit([1, 2, 3], [4, 5, 6])
        
        # Test mismatched array sizes
        with self.assertRaises(ValueError):
            self.predictor.fit(self.X_train, self.y_train[:50])
        
        # Test insufficient samples
        with self.assertRaises(ValueError):
            self.predictor.fit(np.array([[1]]), np.array([2]))
        
        # Test NaN values
        X_nan = self.X_train.copy()
        X_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.predictor.fit(X_nan, self.y_train)
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Train model first
        self.predictor.fit(self.X_train, self.y_train, self.feature_names)
        
        # Make predictions
        predictions = self.predictor.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(predictions > 0))  # Should be positive processing times
    
    def test_model_prediction_without_training(self):
        """Test prediction without training raises error."""
        with self.assertRaises(ValueError):
            self.predictor.predict(self.X_test)
    
    def test_model_prediction_input_validation(self):
        """Test model prediction input validation."""
        # Train model first
        self.predictor.fit(self.X_train, self.y_train, self.feature_names)
        
        # Test non-numpy array
        with self.assertRaises(TypeError):
            self.predictor.predict([[1, 2, 3, 4, 5]])
        
        # Test wrong number of features
        with self.assertRaises(ValueError):
            self.predictor.predict(np.array([[1, 2, 3]]))  # Wrong shape
        
        # Test NaN values
        X_nan = self.X_test.copy()
        X_nan[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.predictor.predict(X_nan)
    
    def test_prediction_with_uncertainty(self):
        """Test prediction with uncertainty estimation."""
        # Train model first
        self.predictor.fit(self.X_train, self.y_train, self.feature_names)
        
        # Make predictions with uncertainty
        predictions, uncertainties = self.predictor.predict_with_uncertainty(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(len(uncertainties), len(self.X_test))
        self.assertTrue(np.all(uncertainties >= 0))  # Uncertainties should be non-negative
    
    def test_get_feature_importance(self):
        """Test getting feature importance."""
        # Train model first
        self.predictor.fit(self.X_train, self.y_train, self.feature_names)
        
        # Get feature importance
        importance = self.predictor.get_feature_importance()
        
        self.assertEqual(len(importance), len(self.feature_names))
        for feature_name in self.feature_names:
            self.assertIn(feature_name, importance)
            self.assertIsInstance(importance[feature_name], (float, np.floating))
    
    def test_get_feature_importance_without_training(self):
        """Test getting feature importance without training raises error."""
        with self.assertRaises(ValueError):
            self.predictor.get_feature_importance()
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Train model first
        self.predictor.fit(self.X_train, self.y_train, self.feature_names)
        
        # Evaluate model
        performance = self.predictor.evaluate(self.X_test, self.y_test)
        
        self.assertIsNotNone(performance.mse)
        self.assertIsNotNone(performance.mae)
        self.assertIsNotNone(performance.r2)
        self.assertGreaterEqual(performance.mse, 0)
        self.assertGreaterEqual(performance.mae, 0)
    
    def test_unknown_model_type(self):
        """Test unknown model type raises error."""
        with self.assertRaises(ValueError):
            ProcessingTimePredictor('unknown_model')


if __name__ == '__main__':
    unittest.main()