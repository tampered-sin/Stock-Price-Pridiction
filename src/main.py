import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from data_fetcher import DataFetcher
from technical_indicators import SMA, EMA, RSI, MACD, Bollinger_Bands, Volume_Indicators
from preprocessor import DataPreprocessor
from model_builder import StockPredictionModel
from predictor import StockPredictor
from visualizer import StockVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPricePrediction:
    def __init__(self, tickers: List[str], period: str = '2y',
                 sequence_length: int = 60, cache_dir: str = 'cache',
                 model_dir: str = 'models'):
        """
        Initialize the stock price prediction system.
        
        Args:
            tickers (List[str]): List of stock tickers
            period (str): Data period to fetch
            sequence_length (int): Length of input sequences
            cache_dir (str): Directory for caching data
            model_dir (str): Directory for saving models
        """
        self.tickers = tickers
        self.period = period
        self.sequence_length = sequence_length
        
        # Create required directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.data_fetcher = DataFetcher(cache_dir=cache_dir)
        self.preprocessor = DataPreprocessor(sequence_length=sequence_length)
        self.visualizer = StockVisualizer()
        
        self.model = None
        self.predictor = None
        
    def fetch_and_prepare_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch and prepare data for a single ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: Processed DataFrame with technical indicators
        """
        logger.info(f"Fetching data for {ticker}")
        data = self.data_fetcher.fetch_data([ticker], self.period)[ticker]
        
        # Add technical indicators
        logger.info("Calculating technical indicators")
        data['SMA_20'] = SMA(data['Close'], 20)
        data['SMA_50'] = SMA(data['Close'], 50)
        data['SMA_200'] = SMA(data['Close'], 200)
        
        data['EMA_12'] = EMA(data['Close'], 12)
        data['EMA_26'] = EMA(data['Close'], 26)
        
        data['RSI'] = RSI(data['Close'])
        
        macd_data = MACD(data['Close'])
        data['MACD'] = macd_data['MACD_Line']
        data['Signal_Line'] = macd_data['Signal_Line']
        data['MACD_Hist'] = macd_data['Histogram']
        
        bb_data = Bollinger_Bands(data['Close'])
        data['BB_Upper'] = bb_data['Upper_Band']
        data['BB_Middle'] = bb_data['Middle_Band']
        data['BB_Lower'] = bb_data['Lower_Band']
        
        vol_data = Volume_Indicators(data)
        data['OBV'] = vol_data['OBV']
        data['Volume_MA'] = vol_data['Volume_MA']
        
        # Remove rows with NaN values
        data = data.dropna()
        
        return data
    
    def train_model(self, data: pd.DataFrame) -> None:
        """
        Train the prediction model.
        
        Args:
            data (pd.DataFrame): Processed data with technical indicators
        """
        logger.info("Preprocessing data for training")
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.preprocess_data(data)
        
        n_features = X_train.shape[2]
        
        logger.info("Building and training model")
        self.model = StockPredictionModel(
            sequence_length=self.sequence_length,
            n_features=n_features
        )
        
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,
            batch_size=32,
            checkpoint_path='models/model_weights.h5'
        )
        
        # Initialize predictor with trained model
        self.predictor = StockPredictor(self.model, self.preprocessor)
        
        # Visualize training history
        self.visualizer.plot_training_history(history.history)
        
        # Evaluate model
        logger.info("Evaluating model")
        y_pred = self.model.predict(X_test)
        metrics = self.predictor.evaluate_predictions(y_test, y_pred)
        self.visualizer.plot_performance_metrics(metrics)
        
    def generate_predictions(self, data: pd.DataFrame, n_steps: int = 30) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate future price predictions.
        
        Args:
            data (pd.DataFrame): Current market data
            n_steps (int): Number of steps to predict
            
        Returns:
            Optional[Dict[str, np.ndarray]]: Predictions with confidence intervals, or None if model not trained
        """
        if self.model is None or self.predictor is None:
            logger.error("Model not trained. Please train the model first.")
            return None
            
        logger.info(f"Generating predictions for next {n_steps} days")
        
        # Prepare sequences for prediction
        X, _, _, _, _, _ = self.preprocessor.preprocess_data(data)
        
        # Generate predictions
        predictions = self.predictor.generate_predictions(X, n_steps=n_steps)
        
        # Visualize predictions
        self.visualizer.plot_predictions(
            data['Close'].values[-n_steps:],
            predictions['predictions'],
            confidence_intervals={
                'lower_bound': predictions['lower_bound'],
                'upper_bound': predictions['upper_bound']
            }
        )
        
        return predictions

def main():
    # Example usage
    tickers = ['AAPL']  # Can add more tickers: ['AAPL', 'GOOGL', 'MSFT']
    
    # Initialize the system
    stock_prediction = StockPricePrediction(tickers)
    
    # Process each ticker
    for ticker in tickers:
        # Fetch and prepare data
        data = stock_prediction.fetch_and_prepare_data(ticker)
        
        # Visualize technical indicators
        stock_prediction.visualizer.plot_technical_indicators(
            data,
            {
                'Bollinger_Bands': pd.DataFrame({
                    'Upper_Band': data['BB_Upper'],
                    'Middle_Band': data['BB_Middle'],
                    'Lower_Band': data['BB_Lower']
                }),
                'MACD': pd.DataFrame({
                    'MACD_Line': data['MACD'],
                    'Signal_Line': data['Signal_Line'],
                    'Histogram': data['MACD_Hist']
                }),
                'RSI': pd.DataFrame({'RSI': data['RSI']})
            }
        )
        
        # Train model
        stock_prediction.train_model(data)
        
        # Generate predictions
        predictions = stock_prediction.generate_predictions(data)
        if predictions is not None:
            logger.info(f"Predictions for {ticker}:")
            logger.info(f"Next 30 days: {predictions['predictions']}")
        else:
            logger.error(f"Failed to generate predictions for {ticker}")

if __name__ == "__main__":
    main()
