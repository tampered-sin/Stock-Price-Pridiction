import pandas as pd
import numpy as np
from typing import Union

def SMA(data: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return data.rolling(window=window).mean()

def EMA(data: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average"""
    return data.ewm(span=window, adjust=False).mean()

def RSI(data: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0.0)).astype(float)
    loss = (-delta.where(delta < 0, 0.0)).astype(float)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
    ema_fast = EMA(data, fast_period)
    ema_slow = EMA(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'MACD_Line': macd_line,
        'Signal_Line': signal_line,
        'Histogram': histogram
    })

def Bollinger_Bands(data: pd.Series, window: int = 20, num_std_dev: int = 2) -> pd.DataFrame:
    """Bollinger Bands"""
    sma = SMA(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std_dev)
    lower_band = sma - (std * num_std_dev)
    return pd.DataFrame({
        'Upper_Band': upper_band,
        'Middle_Band': sma,
        'Lower_Band': lower_band
    })

def Volume_Indicators(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume-based indicators: On-Balance Volume (OBV) and Volume Moving Average"""
    close = data['Close']
    volume = data['Volume']
    
    obv = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
            
    vol_ma = volume.rolling(window=window).mean()
    return pd.DataFrame({
        'OBV': obv,
        'Volume_MA': vol_ma
    })
