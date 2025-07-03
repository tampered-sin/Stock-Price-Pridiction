import os
import logging
import yfinance as yf
import pandas as pd
from typing import List, Optional
import datetime
import pickle

CACHE_DIR = "cache"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _cache_path(self, ticker: str, period: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker}_{period}.pkl")

    def fetch_data(self, tickers: List[str], period: str = "1y") -> dict:
        """
        Fetch historical stock data for given tickers and period.
        Supports caching to avoid repeated API calls.

        Args:
            tickers (List[str]): List of stock ticker symbols.
            period (str): Data period (e.g., '1y', '2y', '5y', 'max').

        Returns:
            dict: Dictionary of ticker to pandas DataFrame with OHLCV data.
        """
        data = {}
        for ticker in tickers:
            try:
                cache_file = self._cache_path(ticker, period)
                if os.path.exists(cache_file):
                    logger.info(f"Loading cached data for {ticker} ({period})")
                    with open(cache_file, "rb") as f:
                        df = pickle.load(f)
                else:
                    logger.info(f"Fetching data for {ticker} ({period}) from Yahoo Finance")
                    df = yf.Ticker(ticker).history(period=period)
                    if df.empty:
                        logger.warning(f"No data fetched for {ticker} with period {period}")
                    else:
                        with open(cache_file, "wb") as f:
                            pickle.dump(df, f)
                data[ticker] = df
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                data[ticker] = pd.DataFrame()
        return data
