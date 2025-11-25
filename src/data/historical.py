"""Historical data management - IBKR as primary source."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pytz

import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.database import Database
from ..core.logger import get_logger
from ..core.exceptions import InsufficientDataError, IBKRConnectionError

logger = get_logger(__name__)


class HistoricalDataManager:
    """
    Manages historical market data with IBKR as the primary source.

    IMPORTANT: Uses IBKR for all data to ensure consistency between
    backtesting and live trading. This prevents discrepancies that
    can cause strategies to fail in production.
    """

    def __init__(
        self,
        config: Config,
        ibkr_client=None,
        database: Optional[Database] = None,
    ):
        """
        Initialize historical data manager.

        Args:
            config: Configuration object
            ibkr_client: IBKR client instance (required for data)
            database: Database instance (optional)
        """
        self.config = config
        self.ibkr_client = ibkr_client
        self.database = database

        # Cache settings
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_dir = Path("data/cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Track data source for transparency
        self._data_sources: dict[str, str] = {}

    def set_ibkr_client(self, ibkr_client) -> None:
        """Set or update the IBKR client."""
        self.ibkr_client = ibkr_client
        logger.info("IBKR client set for historical data")

    def get_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol from IBKR.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe (1min, 5min, 15min, 1hour, 1day)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data

        Raises:
            IBKRConnectionError: If IBKR is not connected
            InsufficientDataError: If no data available
        """
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        cache_key = f"{symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"

        # Check memory cache
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached data for {symbol}")
            return self._cache[cache_key]

        # Check file cache
        cache_file = self._cache_dir / f"{cache_key}.parquet"
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                # Verify cache was from IBKR
                meta_file = self._cache_dir / f"{cache_key}.meta"
                if meta_file.exists():
                    with open(meta_file) as f:
                        source = f.read().strip()
                    if source == "ibkr":
                        self._cache[cache_key] = df
                        self._data_sources[cache_key] = "ibkr_cache"
                        logger.info(f"Loaded {len(df)} bars from IBKR cache for {symbol}")
                        return df
                    else:
                        logger.warning(f"Cache for {symbol} is from {source}, fetching fresh from IBKR")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Fetch from IBKR
        df = self._fetch_from_ibkr(symbol, start_date, end_date, timeframe)

        if df.empty:
            raise InsufficientDataError(
                f"No data available for {symbol}. "
                "Ensure IBKR is connected and symbol is valid."
            )

        # Cache the data
        self._cache[cache_key] = df
        self._data_sources[cache_key] = "ibkr"

        try:
            df.to_parquet(cache_file)
            # Write metadata
            meta_file = self._cache_dir / f"{cache_key}.meta"
            with open(meta_file, "w") as f:
                f.write("ibkr")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

        return df

    def _fetch_from_ibkr(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch data from IBKR."""
        if not self.ibkr_client:
            raise IBKRConnectionError(
                "IBKR client not configured. "
                "The system requires IBKR for data to ensure consistency "
                "between backtesting and live trading."
            )

        if not self.ibkr_client.connected:
            raise IBKRConnectionError(
                "IBKR not connected. Please connect to TWS/Gateway first. "
                "Run: ibkr_client.connect()"
            )

        # Map timeframe to IBKR bar size
        bar_size_map = {
            "1min": "1 min",
            "2min": "2 mins",
            "5min": "5 mins",
            "15min": "15 mins",
            "30min": "30 mins",
            "1hour": "1 hour",
            "1day": "1 day",
        }

        bar_size = bar_size_map.get(timeframe, "1 min")

        # Calculate duration
        days = (end_date - start_date).days
        if days <= 1:
            duration = "1 D"
        elif days <= 7:
            duration = f"{days} D"
        elif days <= 30:
            duration = f"{days} D"
        elif days <= 365:
            months = days // 30
            duration = f"{months} M"
        else:
            years = days // 365
            duration = f"{years} Y"

        try:
            logger.info(f"Fetching {symbol} from IBKR: {duration} of {bar_size} bars")

            df = self.ibkr_client.get_historical_data(
                symbol=symbol,
                duration=duration,
                bar_size=bar_size,
                what_to_show="TRADES",
                use_rth=True,
            )

            if df.empty:
                logger.warning(f"IBKR returned no data for {symbol}")
                return df

            # Filter to requested date range
            # Handle timezone-aware comparison if needed
            if start_date:
                # If df.index is timezone-aware, make start_date timezone-aware too
                if df.index.tz is not None:
                    if start_date.tzinfo is None:
                        start_date = pytz.timezone('US/Eastern').localize(start_date)
                    else:
                        start_date = start_date.astimezone(pytz.timezone('US/Eastern'))
                df = df[df.index >= start_date]
            if end_date:
                # If df.index is timezone-aware, make end_date timezone-aware too
                if df.index.tz is not None:
                    if end_date.tzinfo is None:
                        end_date = pytz.timezone('US/Eastern').localize(end_date)
                    else:
                        end_date = end_date.astimezone(pytz.timezone('US/Eastern'))
                df = df[df.index <= end_date]

            # Ensure standard column names
            df.columns = [c.lower() for c in df.columns]

            logger.info(f"Fetched {len(df)} bars from IBKR for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch from IBKR: {e}")
            raise IBKRConnectionError(f"IBKR data fetch failed: {e}")

    def get_realtime_snapshot(self, symbol: str) -> dict:
        """
        Get real-time price snapshot from IBKR.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with current price data
        """
        if not self.ibkr_client or not self.ibkr_client.connected:
            raise IBKRConnectionError("IBKR not connected")

        price = self.ibkr_client.get_current_price(symbol)

        return {
            "symbol": symbol,
            "price": price,
            "timestamp": datetime.now(),
            "source": "ibkr",
        }

    def get_multiple_symbols(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1day",
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols from IBKR.

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
        Prepare data for backtesting using IBKR data.

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
            raise InsufficientDataError(f"No IBKR data for {symbol} backtest")

        # Verify data source
        logger.info(f"Backtest data source: IBKR (ensuring consistency with live trading)")

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
        # Handle timezone-aware comparison
        filter_start = start_date
        if df.index.tz is not None and start_date.tzinfo is None:
            filter_start = pytz.timezone('US/Eastern').localize(start_date)
        df = df[df.index >= filter_start]

        logger.info(f"Prepared {len(df)} IBKR bars for {symbol} backtest")
        return df

    def verify_data_consistency(self, symbol: str) -> dict:
        """
        Verify that cached data matches current IBKR data.

        Useful for ensuring no stale cache issues.

        Args:
            symbol: Ticker symbol

        Returns:
            Verification results
        """
        results = {
            "symbol": symbol,
            "verified": False,
            "issues": [],
        }

        try:
            # Get fresh data
            fresh_df = self.get_data(symbol, use_cache=False)

            # Get cached data
            cache_keys = [k for k in self._cache if k.startswith(symbol)]

            for key in cache_keys:
                cached_df = self._cache[key]

                # Compare overlapping period
                overlap_start = max(cached_df.index[0], fresh_df.index[0])
                overlap_end = min(cached_df.index[-1], fresh_df.index[-1])

                cached_overlap = cached_df[overlap_start:overlap_end]
                fresh_overlap = fresh_df[overlap_start:overlap_end]

                if len(cached_overlap) != len(fresh_overlap):
                    results["issues"].append(f"Row count mismatch in {key}")
                else:
                    # Check prices match
                    price_diff = abs(cached_overlap["close"] - fresh_overlap["close"]).max()
                    if price_diff > 0.01:
                        results["issues"].append(f"Price discrepancy in {key}: {price_diff}")

            results["verified"] = len(results["issues"]) == 0

        except Exception as e:
            results["issues"].append(f"Verification failed: {e}")

        return results

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cached data."""
        if symbol:
            # Clear specific symbol
            keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]

            # Remove cache files
            for cache_file in self._cache_dir.glob(f"{symbol}_*"):
                cache_file.unlink()
        else:
            # Clear all
            self._cache.clear()
            for cache_file in self._cache_dir.glob("*"):
                cache_file.unlink()

        logger.info(f"Cache cleared for {symbol or 'all symbols'}")

    def get_data_source(self, symbol: str) -> str:
        """Get the data source used for a symbol."""
        for key, source in self._data_sources.items():
            if key.startswith(symbol):
                return source
        return "unknown"
