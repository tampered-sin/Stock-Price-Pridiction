"""
Configuration file for Stock Price Prediction System
"""

# Stock symbols to analyze
STOCK_SYMBOLS = [
    'AAPL',   # Apple Inc.
    'GOOGL',  # Alphabet Inc.
    'MSFT',   # Microsoft Corporation
    'TSLA',   # Tesla Inc.
    'AMZN',   # Amazon.com Inc.
    'NVDA',   # NVIDIA Corporation
    'META',   # Meta Platforms Inc.
]

# Data fetching parameters
DATA_CONFIG = {
    'period': '2y',           # Data period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    'cache_dir': 'cache',     # Directory for caching downloaded data
    'refresh_cache': False,   # Set to True to force refresh cached data
}

# Model hyperparameters
MODEL_CONFIG = {
    'sequence_length': 60,    # Number of time steps to look back
    'lstm_units': [50, 50],   # Number of units in each LSTM layer
    'dropout_rate': 0.2,      # Dropout rate for regularization
    'learning_rate': 0.001,   # Initial learning rate
    'batch_size': 32,         # Training batch size
    'epochs': 100,            # Maximum number of training epochs
    'validation_split': 0.15, # Proportion of data for validation
    'test_split': 0.15,       # Proportion of data for testing
}

# Technical indicators parameters
INDICATORS_CONFIG = {
    'sma_periods': [20, 50, 200],     # Simple Moving Average periods
    'ema_periods': [12, 26],          # Exponential Moving Average periods
    'rsi_period': 14,                 # RSI calculation period
    'macd_fast': 12,                  # MACD fast period
    'macd_slow': 26,                  # MACD slow period
    'macd_signal': 9,                 # MACD signal period
    'bb_period': 20,                  # Bollinger Bands period
    'bb_std_dev': 2,                  # Bollinger Bands standard deviations
    'volume_ma_period': 20,           # Volume moving average period
}

# Prediction parameters
PREDICTION_CONFIG = {
    'forecast_days': 30,              # Number of days to predict
    'confidence_level': 0.95,         # Confidence level for prediction intervals
    'walk_forward_window': 60,        # Window size for walk-forward validation
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figure_size': (15, 10),          # Default figure size
    'style': 'seaborn',               # Matplotlib style
    'save_plots': True,               # Whether to save plots to files
    'plot_format': 'png',             # Plot file format
    'dpi': 300,                       # Plot resolution
}

# File paths
PATHS = {
    'models_dir': 'models',
    'outputs_dir': 'outputs',
    'plots_dir': 'outputs/plots',
    'predictions_dir': 'outputs/predictions',
    'logs_dir': 'logs',
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'max_rmse_percent': 5.0,          # Maximum RMSE as percentage of average price
    'min_directional_accuracy': 0.55, # Minimum directional accuracy
    'max_training_time_hours': 2.0,   # Maximum training time in hours
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',                  # Logging level: DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_logging': True,             # Whether to log to file
    'console_logging': True,          # Whether to log to console
}
