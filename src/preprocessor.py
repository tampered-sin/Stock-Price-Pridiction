import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta

class DataPreprocessor:
    def __init__(self, sequence_length: int = 60):
        """
        Initialize the DataPreprocessor.
        
        Args:
            sequence_length (int): Length of input sequences for LSTM
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame with possible missing values
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        # Forward fill missing values
        df = df.ffill()
        # If any remaining NaN at the beginning, backward fill
        df = df.bfill()
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], n_std: float = 3) -> pd.DataFrame:
        """
        Remove outliers using the z-score method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            n_std (float): Number of standard deviations to use as threshold
            
        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        for column in columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores < n_std]
        return df
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input using sliding window approach.
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (sequences) and y (targets)
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  train_split: float = 0.7, val_split: float = 0.15) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (np.ndarray): Input sequences
            y (np.ndarray): Target values
            train_split (float): Proportion of data for training
            val_split (float): Proportion of data for validation
            
        Returns:
            Tuple[np.ndarray, ...]: Train, validation, and test sets
        """
        n = len(X)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'Close') -> Tuple[np.ndarray, ...]:
        """
        Preprocess the data: handle missing values, scale data, create sequences.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_column (str): Column to predict
            
        Returns:
            Tuple[np.ndarray, ...]: Preprocessed train, validation, and test sets
        """
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df[[target_column]])
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        # Split the data
        return self.split_data(X, y)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.
        
        Args:
            data (np.ndarray): Scaled data
            
        Returns:
            np.ndarray: Data in original scale
        """
        return self.scaler.inverse_transform(data)
    
    def save_scaler(self, path: str):
        """Save the fitted scaler to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path: str):
        """Load the fitted scaler from file."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
