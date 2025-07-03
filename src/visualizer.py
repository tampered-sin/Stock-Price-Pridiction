import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StockVisualizer:
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize the visualizer with a plotting style.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        
    def plot_stock_prices(self, data: pd.DataFrame, title: str = "Stock Price History",
                         figsize: Tuple[int, int] = (15, 7)) -> None:
        """
        Plot historical stock prices with volume.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        
        # Plot prices
        ax1.plot(data.index, data['Close'], label='Close Price')
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Plot volume
        ax2.bar(data.index, data['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_technical_indicators(self, data: pd.DataFrame, 
                                indicators: Dict[str, pd.DataFrame],
                                figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot stock prices with technical indicators.
        
        Args:
            data (pd.DataFrame): Original price data
            indicators (Dict[str, pd.DataFrame]): Dictionary of technical indicators
            figsize (Tuple[int, int]): Figure size
        """
        fig = make_subplots(rows=3, cols=1, shared_xaxis=True,
                           vertical_spacing=0.05,
                           row_heights=[0.5, 0.25, 0.25])
        
        # Plot candlestick
        fig.add_trace(go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name='OHLC'),
                     row=1, col=1)
        
        # Plot Bollinger Bands
        if 'Bollinger_Bands' in indicators:
            bb = indicators['Bollinger_Bands']
            fig.add_trace(go.Scatter(x=data.index, y=bb['Upper_Band'],
                                   name='Upper BB', line=dict(dash='dash')),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=bb['Middle_Band'],
                                   name='Middle BB', line=dict(dash='dash')),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=bb['Lower_Band'],
                                   name='Lower BB', line=dict(dash='dash')),
                         row=1, col=1)
        
        # Plot MACD
        if 'MACD' in indicators:
            macd = indicators['MACD']
            fig.add_trace(go.Scatter(x=data.index, y=macd['MACD_Line'],
                                   name='MACD Line'),
                         row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=macd['Signal_Line'],
                                   name='Signal Line'),
                         row=2, col=1)
            fig.add_trace(go.Bar(x=data.index, y=macd['Histogram'],
                               name='MACD Histogram'),
                         row=2, col=1)
        
        # Plot RSI
        if 'RSI' in indicators:
            fig.add_trace(go.Scatter(x=data.index, y=indicators['RSI'],
                                   name='RSI'),
                         row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                         row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         row=3, col=1)
        
        fig.update_layout(
            title='Technical Analysis',
            yaxis_title='Price',
            yaxis2_title='MACD',
            yaxis3_title='RSI',
            xaxis_rangeslider_visible=False
        )
        
        fig.show()
        
    def plot_training_history(self, history: Dict[str, List[float]],
                            figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot training history metrics.
        
        Args:
            history (Dict[str, List[float]]): Training history dictionary
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_predictions(self, actual: np.ndarray, predicted: np.ndarray,
                        confidence_intervals: Optional[Dict[str, np.ndarray]] = None,
                        figsize: Tuple[int, int] = (15, 7)) -> None:
        """
        Plot actual vs predicted values with confidence intervals.
        
        Args:
            actual (np.ndarray): Actual values
            predicted (np.ndarray): Predicted values
            confidence_intervals (Dict[str, np.ndarray]): Confidence interval bounds
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        
        plt.plot(actual, label='Actual', color='blue')
        plt.plot(predicted, label='Predicted', color='red', linestyle='--')
        
        if confidence_intervals is not None:
            plt.fill_between(range(len(predicted)),
                           confidence_intervals['lower_bound'],
                           confidence_intervals['upper_bound'],
                           color='gray', alpha=0.2, label='95% Confidence Interval')
        
        plt.title('Stock Price Predictions')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, feature_names: List[str], 
                              importance_scores: np.ndarray,
                              figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot feature importance scores.
        
        Args:
            feature_names (List[str]): Names of features
            importance_scores (np.ndarray): Importance scores for each feature
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        
        # Sort features by importance
        indices = np.argsort(importance_scores)
        plt.barh(range(len(indices)), importance_scores[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        
        plt.tight_layout()
        plt.show()
        
    def plot_performance_metrics(self, metrics: Dict[str, float],
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot model performance metrics.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())
        
        plt.bar(metrics_names, metrics_values)
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)
        plt.ylabel('Value')
        
        # Add value labels on top of each bar
        for i, v in enumerate(metrics_values):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
