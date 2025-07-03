# Stock Price Prediction System

A comprehensive machine learning system for predicting stock prices using LSTM neural networks with technical indicators. This production-ready system includes data fetching, preprocessing, model training, prediction generation, and visualization capabilities.

## Features

- **Real-time Data Fetching**: Automatic stock data retrieval using Yahoo Finance API
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, and volume indicators
- **LSTM Model**: Multi-layer LSTM with dropout regularization and learning rate scheduling
- **Prediction Generation**: Future price predictions with confidence intervals
- **Comprehensive Visualization**: Interactive charts and performance metrics
- **Caching System**: Efficient data caching to avoid repeated API calls
- **Error Handling**: Robust error handling and logging throughout the system

## Project Structure

```
Stock-Price-Pridiction/
├── src/
│   ├── data_fetcher.py          # Data acquisition module
│   ├── technical_indicators.py  # Technical analysis functions
│   ├── preprocessor.py          # Data preprocessing pipeline
│   ├── model_builder.py         # LSTM model architecture
│   ├── predictor.py             # Prediction generation module
│   ├── visualizer.py            # Plotting and visualization functions
│   ├── main.py                  # Main execution script with StockPricePrediction class
│   ├── test_modules.py          # Basic module tests
│   └── test_edge_cases.py       # Comprehensive edge case tests
├── models/                      # Saved model artifacts (model_weights.weights.h5)
├── cache/                       # Cached data files (AAPL_1y.pkl, etc.)
├── config.py                    # Comprehensive configuration parameters
├── requirements.txt             # Python dependencies
├── test_plot.png               # Sample output plot
└── README.md                    # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Stock-Price-Pridiction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create required directories** (if not already present):
   ```bash
   mkdir -p models cache
   ```

### Dependencies

The project includes the following key dependencies:

- **Core**: pandas, numpy, scikit-learn
- **Data Fetching**: yfinance
- **Machine Learning**: tensorflow, keras
- **Visualization**: matplotlib, seaborn, plotly
- **Web Application**: streamlit (optional)
- **Development**: pytest, jupyter, notebook
- **Utilities**: python-dateutil, pytz, requests

## Quick Start

### Basic Usage

```python
from src.main import StockPricePrediction

# Initialize the system with configuration
predictor = StockPricePrediction(
    tickers=['AAPL'], 
    period='2y',
    sequence_length=60,
    cache_dir='cache',
    model_dir='models'
)

# Fetch and prepare data with technical indicators
data = predictor.fetch_and_prepare_data('AAPL')

# Train the LSTM model
predictor.train_model(data)

# Generate future predictions (30 days by default)
predictions = predictor.generate_predictions(data, n_steps=30)

# Access prediction results
if predictions:
    print(f"Predictions: {predictions['predictions']}")
    print(f"Lower bound: {predictions['lower_bound']}")
    print(f"Upper bound: {predictions['upper_bound']}")
```

### Command Line Usage

```bash
# Run the complete pipeline
python src/main.py

# Run tests
python src/test_modules.py
python src/test_edge_cases.py
```

## Configuration

The `config.py` file provides comprehensive configuration options:

### Stock Symbols
```python
STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META']
```

### Data Configuration
- **Period**: '2y' (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- **Cache directory**: 'cache'
- **Cache refresh**: False (set to True to force refresh)

### Model Hyperparameters
- **Sequence length**: 60 time steps
- **LSTM units**: [50, 50] (two layers with 50 units each)
- **Dropout rate**: 0.2
- **Learning rate**: 0.001
- **Batch size**: 32
- **Epochs**: 100
- **Validation split**: 0.15
- **Test split**: 0.15

### Technical Indicators
- **SMA periods**: [20, 50, 200]
- **EMA periods**: [12, 26]
- **RSI period**: 14
- **MACD**: Fast=12, Slow=26, Signal=9
- **Bollinger Bands**: Period=20, Std Dev=2
- **Volume MA period**: 20

### Prediction Settings
- **Forecast days**: 30
- **Confidence level**: 0.95
- **Walk-forward window**: 60

### Visualization Settings
- **Figure size**: (15, 10)
- **Style**: 'seaborn'
- **Save plots**: True
- **Format**: 'png'
- **DPI**: 300

## Model Architecture

The LSTM model includes:

- **Input Layer**: Sequences of 60 time steps with technical indicators
- **LSTM Layers**: Two LSTM layers with 50 units each
- **Dropout**: 20% dropout rate for regularization
- **Dense Layers**: 25-unit dense layer followed by single output
- **Optimization**: Adam optimizer with learning rate scheduling
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction

## Technical Indicators

The system calculates the following technical indicators:

- **Simple Moving Averages (SMA)**: 20, 50, 200-day periods
- **Exponential Moving Averages (EMA)**: 12, 26-day periods
- **Relative Strength Index (RSI)**: 14-day period
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: 20-day period with 2 standard deviations
- **Volume Indicators**: On-Balance Volume (OBV) and Volume MA

## Performance Metrics

The system evaluates model performance using:

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Directional Accuracy**: Percentage of correct price direction predictions

## Data Sources

- **Primary**: Yahoo Finance via yfinance library
- **Supported Tickers**: All major stock exchanges
- **Data Types**: OHLCV (Open, High, Low, Close, Volume)
- **Time Periods**: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

## Testing

The system includes comprehensive testing:

### Basic Tests (`test_modules.py`)
- Data fetching functionality
- Technical indicators calculation
- Data preprocessing pipeline

### Edge Case Tests (`test_edge_cases.py`)
- Invalid ticker handling
- Insufficient data scenarios
- Extreme outlier detection
- Caching performance
- Technical indicator accuracy

### Run Tests
```bash
python src/test_modules.py
python src/test_edge_cases.py
```

## API Reference

### StockPricePrediction Class
```python
from src.main import StockPricePrediction

# Initialize with custom parameters
predictor = StockPricePrediction(
    tickers=['AAPL', 'GOOGL'], 
    period='2y',
    sequence_length=60,
    cache_dir='cache',
    model_dir='models'
)
```

### DataFetcher
```python
from src.data_fetcher import DataFetcher

fetcher = DataFetcher(cache_dir='cache')
data = fetcher.fetch_data(['AAPL'], period='2y')
```

### Technical Indicators
```python
from src.technical_indicators import SMA, EMA, RSI, MACD, Bollinger_Bands, Volume_Indicators

# Calculate individual indicators
sma_20 = SMA(data['Close'], 20)
ema_12 = EMA(data['Close'], 12)
rsi = RSI(data['Close'])
macd_data = MACD(data['Close'])
bb_data = Bollinger_Bands(data['Close'])
vol_data = Volume_Indicators(data)
```

### DataPreprocessor
```python
from src.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(sequence_length=60)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(data)
```

### Model Training
```python
from src.model_builder import StockPredictionModel

model = StockPredictionModel(sequence_length=60, n_features=n_features)
history = model.train(
    X_train, y_train, 
    X_val, y_val,
    epochs=100,
    batch_size=32,
    checkpoint_path='models/model_weights.weights.h5'
)
```

### Prediction Generation
```python
from src.predictor import StockPredictor

predictor = StockPredictor(model, preprocessor)
predictions = predictor.generate_predictions(X, n_steps=30)
```

### Visualization
```python
from src.visualizer import StockVisualizer

visualizer = StockVisualizer()
visualizer.plot_technical_indicators(data, indicators_dict)
visualizer.plot_training_history(history)
visualizer.plot_predictions(actual_prices, predictions, confidence_intervals)
```

## Troubleshooting

### Common Issues

1. **TensorFlow Installation**:
   ```bash
   pip install tensorflow==2.13.0  # For Python 3.8-3.11
   ```

2. **Memory Issues**:
   - Reduce batch size in `config.py`
   - Use smaller sequence lengths
   - Process fewer stocks simultaneously

3. **Data Fetching Errors**:
   - Check internet connection
   - Verify ticker symbols are valid
   - Clear cache directory if data seems corrupted

4. **Model Training Slow**:
   - Enable GPU acceleration if available
   - Reduce number of epochs
   - Use smaller model architecture

### Error Codes

- **404 Error**: Invalid ticker symbol or delisted stock
- **Memory Error**: Insufficient RAM for model training
- **Import Error**: Missing dependencies

## Performance Optimization

### For Better Accuracy
- Increase sequence length (60-120 time steps)
- Add more technical indicators
- Use ensemble methods
- Implement feature engineering

### For Faster Training
- Use GPU acceleration
- Reduce model complexity
- Implement batch processing
- Use data generators for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Stock market predictions are inherently uncertain, and this system should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct your own research before making investment choices.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues in the repository
3. Create a new issue with detailed information
4. Include error messages and system information

## Acknowledgments

- Yahoo Finance for providing free stock data API
- TensorFlow team for the machine learning framework
- Pandas and NumPy communities for data manipulation tools
- Matplotlib and Plotly for visualization capabilities
