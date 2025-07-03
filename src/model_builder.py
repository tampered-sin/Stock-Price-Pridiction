import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Tuple, List, Optional
import numpy as np

class StockPredictionModel:
    def __init__(self, sequence_length: int, n_features: int, lstm_units: List[int] = [50, 50],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize the LSTM model for stock price prediction.
        
        Args:
            sequence_length (int): Length of input sequences
            n_features (int): Number of features in input data
            lstm_units (List[int]): Number of units in each LSTM layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Initial learning rate
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build and compile the LSTM model.
        
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(self.lstm_units[0], 
                      input_shape=(self.sequence_length, self.n_features),
                      return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(self.lstm_units[1]))
        model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1, activation='linear'))
        
        # Compile model with Adam optimizer and MSE loss
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', 
                     metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        
        return model
    
    def get_callbacks(self, checkpoint_path: str) -> List[tf.keras.callbacks.Callback]:
        """
        Get training callbacks for early stopping and learning rate reduction.
        
        Args:
            checkpoint_path (str): Path to save model checkpoints
            
        Returns:
            List[tf.keras.callbacks.Callback]: List of Keras callbacks
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
        
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        return [early_stopping, checkpoint, lr_reducer]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              checkpoint_path: str = 'models/model_weights.weights.h5') -> tf.keras.callbacks.History:
        """
        Train the model with early stopping and learning rate scheduling.
        
        Args:
            X_train (np.ndarray): Training sequences
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation sequences
            y_val (np.ndarray): Validation targets
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training
            checkpoint_path (str): Path to save model checkpoints
            
        Returns:
            History: Training history
        """
        callbacks = self.get_callbacks(checkpoint_path)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input sequences.
        
        Args:
            X (np.ndarray): Input sequences
            
        Returns:
            np.ndarray: Predicted values
        """
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Evaluate model performance.
        
        Args:
            X (np.ndarray): Input sequences
            y (np.ndarray): True values
            
        Returns:
            Tuple[float, float, float]: MSE, MAE, and RMSE scores
        """
        mse, mae, rmse = self.model.evaluate(X, y, verbose=0)
        return mse, mae, rmse
    
    def save_model(self, path: str):
        """Save the model architecture and weights."""
        self.model.save(path)
    
    def load_model(self, path: str):
        """Load a saved model."""
        self.model = tf.keras.models.load_model(path)
