import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from data_fetcher import DataFetcher
from technical_indicators import SMA, EMA, RSI, MACD, Bollinger_Bands, Volume_Indicators
from preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_fetcher():
    """Test data fetching functionality"""
    logger.info("Testing DataFetcher module...")
    
    fetcher = DataFetcher()
    data = fetcher.fetch_data(['AAPL'], '1y')
    
    print("\nData Fetcher Test Results:")
    print(f"Tickers returned: {list(data.keys())}")
    if 'AAPL' in data:
        print(f"Columns available: {list(data['AAPL'].columns)}")
        print(f"Data points: {len(data['AAPL'])}")
        print(f"Date range: {data['AAPL'].index[0]} to {data['AAPL'].index[-1]}")
    
def test_technical_indicators():
    """Test technical indicators calculations"""
    logger.info("Testing Technical Indicators module...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Close': np.random.normal(100, 10, len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    print("\nTechnical Indicators Test Results:")
    
    # Test SMA
    sma = SMA(sample_data['Close'], window=20)
    print(f"SMA shape: {sma.shape}, NaN values: {sma.isna().sum()}")
    
    # Test RSI
    rsi = RSI(sample_data['Close'])
    print(f"RSI shape: {rsi.shape}, Range: {rsi.min():.2f} to {rsi.max():.2f}")
    
    # Test MACD
    macd = MACD(sample_data['Close'])
    print(f"MACD components: {list(macd.columns)}")
    
    # Test Bollinger Bands
    bb = Bollinger_Bands(sample_data['Close'])
    print(f"Bollinger Bands components: {list(bb.columns)}")

def test_preprocessor():
    """Test data preprocessing functionality"""
    logger.info("Testing DataPreprocessor module...")
    
    # Create sample data with missing values
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Close': np.random.normal(100, 10, len(dates)),
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Add some missing values
    sample_data.loc[sample_data.index[10:15], 'Close'] = np.nan
    
    preprocessor = DataPreprocessor(sequence_length=60)
    
    # Test missing value handling
    cleaned_data = preprocessor.handle_missing_values(sample_data)
    print("\nPreprocessor Test Results:")
    print(f"Original missing values: {sample_data.isna().sum()}")
    print(f"After cleaning missing values: {cleaned_data.isna().sum()}")
    
    # Test sequence creation
    close_values = np.array(sample_data['Close'].values)
    X, y = preprocessor.create_sequences(close_values.reshape(-1, 1))
    print(f"Sequence shape: X={X.shape}, y={y.shape}")

if __name__ == "__main__":
    test_data_fetcher()
    test_technical_indicators()
    test_preprocessor()
