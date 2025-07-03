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
stock_prediction_project/
├── src/
│   ├── data_fetcher.py          # Data acquisition module
│   ├── technical_indicators.py  # Technical analysis functions
│   ├── preprocessor.py          # Data preprocessing pipeline
│   ├── model_builder.py         # LSTM model architecture
│   ├── predictor.py             # Prediction generation module
│   ├── visualizer.py            # Plotting and visualization functions
│   ├── main.py                  # Main execution script
│   ├── test_modules.py          # Basic module tests
│   └── test_edge_cases.py       # Comprehensive edge case tests
├── models/                      # Saved model artifacts
├── cache/                       # Cached data files
├── outputs/                     # Generated outputs and plots
├── config.py                    # Configuration parameters
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd stock-price-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create required directories**:
   ```bash
   mkdir -p models cache outputs/plots outputs/predictions logs
   ```

## Quick Start

### Basic Usage

```python
from src.main import StockPricePrediction

# Initialize the system
predictor = StockPricePrediction(['AAPL'])

# Fetch and prepare data
data = predictor.fetch_and_prepare_data('AAPL')

# Train the model
predictor.train_model(data)

# Generate predictions
predictions = predictor.generate_predictions(data, n_steps=30)
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

Edit `config.py` to customize:

- **Stock symbols** to analyze
- **Model hyperparameters** (LSTM units, dropout rate, learning rate)
- **Technical indicator parameters** (periods, thresholds)
- **Prediction settings** (forecast days, confidence levels)
- **Visualization preferences**

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

### DataFetcher
```python
fetcher = DataFetcher(cache_dir='cache')
data = fetcher.fetch_data(['AAPL'], period='1y')
```

### Technical Indicators
```python
from src.technical_indicators import SMA, EMA, RSI, MACD, Bollinger_Bands

sma = SMA(data['Close'], window=20)
rsi = RSI(data['Close'], window=14)
macd = MACD(data['Close'])
bb = Bollinger_Bands(data['Close'])
```

### DataPreprocessor
```python
preprocessor = DataPreprocessor(sequence_length=60)
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(data)
```

### Model Training
```python
from src.model_builder import StockPredictionModel

model = StockPredictionModel(sequence_length=60, n_features=1)
history = model.train(X_train, y_train, X_val, y_val)
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
