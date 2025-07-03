import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, model, preprocessor, confidence_level: float = 0.95):
        """
        Initialize the StockPredictor.
        
        Args:
            model: Trained model instance
            preprocessor: Data preprocessor instance
            confidence_level (float): Confidence level for prediction intervals
        """
        self.model = model
        self.preprocessor = preprocessor
        self.confidence_level = confidence_level
        
    def calculate_confidence_interval(self, predictions: np.ndarray, 
                                   std_dev: np.floating) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions (np.ndarray): Model predictions
            std_dev (float): Standard deviation of predictions
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper confidence bounds
        """
        z_score = 1.96  # 95% confidence level
        margin = z_score * float(std_dev)
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        return lower_bound, upper_bound
    
    def generate_predictions(self, X: np.ndarray, 
                           n_steps: int = 30) -> Dict[str, np.ndarray]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            X (np.ndarray): Input sequences
            n_steps (int): Number of future steps to predict
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing predictions and confidence intervals
        """
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        current_sequence = X[-1:]  # Start with the last known sequence
        
        for _ in range(n_steps):
            # Generate prediction for next step
            pred = self.model.predict(current_sequence)
            predictions.append(pred[0, 0])
            
            # Calculate confidence intervals
            # Using historical standard deviation as a simple approach
            std_dev = np.std(current_sequence)
            lower, upper = self.calculate_confidence_interval(pred[0, 0], std_dev)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = pred
            
        # Convert predictions back to original scale
        predictions = self.preprocessor.inverse_transform(
            np.array(predictions).reshape(-1, 1))
        lower_bounds = self.preprocessor.inverse_transform(
            np.array(lower_bounds).reshape(-1, 1))
        upper_bounds = self.preprocessor.inverse_transform(
            np.array(upper_bounds).reshape(-1, 1))
        
        return {
            'predictions': predictions.flatten(),
            'lower_bound': lower_bounds.flatten(),
            'upper_bound': upper_bounds.flatten()
        }
    
    def evaluate_predictions(self, y_true: np.ndarray | pd.Series, 
                           y_pred: np.ndarray | pd.Series) -> Dict[str, float]:
        """
        Calculate various performance metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        # Convert to numpy arrays if not already
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(y_true[1:] - y_true[:-1]) == 
                                 np.sign(y_pred[1:] - y_pred[:-1]))
        directional_accuracy = direction_correct / (len(y_true) - 1)
        
        return {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'MAPE': float(mape),
            'Directional_Accuracy': float(directional_accuracy)
        }
    
    def walk_forward_validation(self, data: pd.DataFrame, 
                              window_size: int = 60) -> Dict[str, float]:
        """
        Perform walk-forward validation.
        
        Args:
            data (pd.DataFrame): Input data
            window_size (int): Size of sliding window
            
        Returns:
            Dict[str, float]: Average performance metrics
        """
        metrics_list = []
        
        for i in range(len(data) - window_size):
            # Prepare training and test data
            train_data = data.iloc[i:i+window_size]
            test_data = data.iloc[i+window_size:i+window_size+1]
            
            # Preprocess data
            X_train, _, _, y_train, _, _ = self.preprocessor.preprocess_data(train_data)
            
            # Generate prediction
            pred = self.model.predict(X_train[-1:])
            pred = self.preprocessor.inverse_transform(pred)
            
            # Calculate metrics
            metrics = self.evaluate_predictions(
                np.array(test_data['Close'].values),
                np.array(pred.flatten())
            )
            metrics_list.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in metrics_list[0].keys():
            avg_metrics[metric] = np.mean([m[metric] for m in metrics_list])
        
        return avg_metrics
