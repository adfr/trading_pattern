"""Historical data management."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf

from ..core.config import Config
from ..core.database import Database
from ..core.logger import get_logger
from ..core.exceptions import InsufficientDataError

logger = get_logger(__name__)


class HistoricalDataManager:
    """
    Manages historical market data from multiple sources.

    Supports:
    - Yahoo Finance for free historical data
    - IBKR for more granular intraday data
    - Local caching with SQLite/CSV
    """

    def __init__(self, config: Config, database: Optional[Database] = None):
        """
        Initialize historical data manager.

        Args:
            config: Configuration object
            database: Database instance (optional)
        """
        self.config = config
        self.database = database

        # Cache settings
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_dir = Path("data/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1min",
        source: str = "auto",
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe (1min, 5min, 15min, 1hour, 1day)
            source: Data source (auto, yahoo, ibkr, cache)

        Returns:
            DataFrame with OHLCV data
        """
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        cache_key = f"{symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"

        # Check memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check file cache
        cache_file = self._cache_dir / f"{cache_key}.parquet"
        if cache_file.exists() and source != "fresh":
            try:
                df = pd.read_parquet(cache_file)
                self._cache[cache_key] = df
                logger.info(f"Loaded {len(df)} bars from cache for {symbol}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Fetch from source
        if source == "auto":
            # Use Yahoo for daily data, IBKR for intraday
            if timeframe in ("1day", "1d"):
                df = self._fetch_from_yahoo(symbol, start_date, end_date, timeframe)
            else:
                # Try Yahoo first for minute data (limited history)
                df = self._fetch_from_yahoo(symbol, start_date, end_date, timeframe)
        elif source == "yahoo":
            df = self._fetch_from_yahoo(symbol, start_date, end_date, timeframe)
        else:
            df = self._fetch_from_yahoo(symbol, start_date, end_date, timeframe)

        if df.empty:
            raise InsufficientDataError(f"No data available for {symbol}")

        # Cache the data
        self._cache[cache_key] = df
        try:
            df.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

        return df

    def _fetch_from_yahoo(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        # Map timeframe to yfinance interval
        interval_map = {
            "1min": "1m",
            "2min": "2m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h",
            "1day": "1d",
            "1d": "1d",
        }

        interval = interval_map.get(timeframe, "1d")

        # Yahoo Finance limitations
        # - 1m data: max 7 days
        # - 2m, 5m, 15m, 30m: max 60 days
        # - 1h: max 730 days
        # - 1d: unlimited

        try:
            ticker = yf.Ticker(symbol)

            # Adjust for Yahoo limitations
            if interval == "1m":
                max_days = 7
                if (end_date - start_date).days > max_days:
                    start_date = end_date - timedelta(days=max_days)
            elif interval in ("2m", "5m", "15m", "30m"):
                max_days = 60
                if (end_date - start_date).days > max_days:
                    start_date = end_date - timedelta(days=max_days)

            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
            )

            if df.empty:
                return df

            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]]

            # Filter to market hours for intraday data
            if interval in ("1m", "2m", "5m", "15m", "30m", "1h"):
                df = self._filter_market_hours(df)

            logger.info(f"Fetched {len(df)} bars from Yahoo for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch from Yahoo: {e}")
            return pd.DataFrame()

    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to regular market hours."""
        if df.empty:
            return df

        # Convert to Eastern time if not already
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df.index = df.index.tz_convert("America/New_York")

        # Filter to market hours (9:30 AM - 4:00 PM ET)
        market_open = pd.Timestamp("09:30").time()
        market_close = pd.Timestamp("16:00").time()

        mask = (df.index.time >= market_open) & (df.index.time <= market_close)
        df = df[mask]

        # Filter to weekdays
        df = df[df.index.dayofweek < 5]

        return df

    def get_multiple_symbols(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1day",
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe

        Returns:
            Dictionary of symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.get_data(symbol, start_date, end_date, timeframe)
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                data[symbol] = pd.DataFrame()

        return data

    def calculate_returns(
        self,
        df: pd.DataFrame,
        method: str = "log"
    ) -> pd.Series:
        """
        Calculate returns from price data.

        Args:
            df: DataFrame with close prices
            method: 'log' for log returns, 'simple' for simple returns

        Returns:
            Series of returns
        """
        if method == "log":
            return np.log(df["close"] / df["close"].shift(1))
        else:
            return df["close"].pct_change()

    def resample(
        self,
        df: pd.DataFrame,
        target_timeframe: str,
    ) -> pd.DataFrame:
        """
        Resample data to a different timeframe.

        Args:
            df: Source DataFrame
            target_timeframe: Target timeframe (5min, 15min, 1hour, 1day)

        Returns:
            Resampled DataFrame
        """
        # Map to pandas resample rule
        rule_map = {
            "5min": "5T",
            "15min": "15T",
            "30min": "30T",
            "1hour": "1H",
            "1day": "1D",
        }

        rule = rule_map.get(target_timeframe)
        if not rule:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")

        resampled = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled

    def get_market_regime(
        self,
        df: pd.DataFrame,
        lookback: int = 20
    ) -> str:
        """
        Determine market regime based on recent price action.

        Args:
            df: Price DataFrame
            lookback: Number of periods to analyze

        Returns:
            Market regime: 'trending_up', 'trending_down', 'ranging', 'volatile'
        """
        if len(df) < lookback:
            return "unknown"

        recent = df.iloc[-lookback:]
        returns = self.calculate_returns(recent, "simple")

        # Calculate metrics
        mean_return = returns.mean()
        std_return = returns.std()
        total_return = (recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1

        # Classify regime
        if abs(total_return) > 0.05:  # Strong directional move
            if total_return > 0:
                return "trending_up"
            else:
                return "trending_down"
        elif std_return > 0.02:  # High volatility
            return "volatile"
        else:
            return "ranging"

    def prepare_for_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1min",
    ) -> pd.DataFrame:
        """
        Prepare data for backtesting.

        Ensures data quality and adds useful columns.

        Args:
            symbol: Ticker symbol
            start_date: Backtest start date
            end_date: Backtest end date
            timeframe: Data timeframe

        Returns:
            DataFrame ready for backtesting
        """
        # Get data with some buffer for indicator calculations
        buffer_days = 30
        data_start = start_date - timedelta(days=buffer_days)

        df = self.get_data(symbol, data_start, end_date, timeframe)

        if df.empty:
            raise InsufficientDataError(f"No data for {symbol} backtest")

        # Add useful columns
        df["returns"] = self.calculate_returns(df, "log")
        df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(252)

        # Add ATR
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # Filter to actual backtest period
        df = df[df.index >= start_date]

        logger.info(f"Prepared {len(df)} bars for {symbol} backtest")
        return df

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cached data."""
        if symbol:
            # Clear specific symbol
            keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]

            # Remove cache files
            for cache_file in self._cache_dir.glob(f"{symbol}_*.parquet"):
                cache_file.unlink()
        else:
            # Clear all
            self._cache.clear()
            for cache_file in self._cache_dir.glob("*.parquet"):
                cache_file.unlink()

        logger.info(f"Cache cleared for {symbol or 'all symbols'}")
