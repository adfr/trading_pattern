"""Market data management."""

from datetime import datetime, timedelta
from typing import Any, Callable, Optional
import threading
from collections import defaultdict

import pandas as pd
import numpy as np

from ..core.config import Config
from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.logger import get_logger
from .ibkr_client import IBKRClient

logger = get_logger(__name__)


class MarketDataManager:
    """
    Manages real-time and historical market data.

    Provides a unified interface for accessing market data
    with caching and real-time updates.
    """

    def __init__(
        self,
        config: Config,
        ibkr_client: Optional[IBKRClient] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize market data manager.

        Args:
            config: Configuration object
            ibkr_client: IBKR client (optional, will create if not provided)
            event_bus: Event bus (optional)
        """
        self.config = config
        self.ibkr_client = ibkr_client
        self.event_bus = event_bus or get_event_bus()

        # Data storage
        self._bars: dict[str, pd.DataFrame] = {}
        self._ticks: dict[str, list[dict]] = defaultdict(list)
        self._subscriptions: dict[str, int] = {}

        # Callbacks
        self._bar_callbacks: dict[str, list[Callable]] = defaultdict(list)

        # Threading
        self._lock = threading.RLock()
        self._running = False

        # Configuration
        self._max_bars = 10000  # Maximum bars to keep per symbol

    def start(self) -> None:
        """Start the market data manager."""
        if self._running:
            return

        self._running = True
        logger.info("Market data manager started")

    def stop(self) -> None:
        """Stop the market data manager."""
        self._running = False

        # Unsubscribe from all symbols
        for symbol in list(self._subscriptions.keys()):
            self.unsubscribe(symbol)

        logger.info("Market data manager stopped")

    def subscribe(
        self,
        symbol: str,
        callback: Optional[Callable[[dict], None]] = None
    ) -> None:
        """
        Subscribe to real-time data for a symbol.

        Args:
            symbol: Ticker symbol
            callback: Function to call with each new bar
        """
        if symbol in self._subscriptions:
            if callback:
                self._bar_callbacks[symbol].append(callback)
            return

        if not self.ibkr_client or not self.ibkr_client.connected:
            logger.warning(f"IBKR not connected, cannot subscribe to {symbol}")
            return

        def on_bar(bar_data: dict) -> None:
            self._on_new_bar(bar_data)

        sub_id = self.ibkr_client.subscribe_realtime_bars(symbol, on_bar)
        self._subscriptions[symbol] = sub_id

        if callback:
            self._bar_callbacks[symbol].append(callback)

        logger.info(f"Subscribed to real-time data for {symbol}")

    def unsubscribe(self, symbol: str) -> None:
        """Unsubscribe from real-time data."""
        if symbol in self._subscriptions:
            del self._subscriptions[symbol]
            self._bar_callbacks[symbol].clear()
            logger.info(f"Unsubscribed from {symbol}")

    def _on_new_bar(self, bar_data: dict) -> None:
        """Handle new bar data."""
        symbol = bar_data["symbol"]

        with self._lock:
            # Add to bars DataFrame
            if symbol not in self._bars:
                self._bars[symbol] = pd.DataFrame()

            new_row = pd.DataFrame([{
                "open": bar_data["open"],
                "high": bar_data["high"],
                "low": bar_data["low"],
                "close": bar_data["close"],
                "volume": bar_data["volume"],
            }], index=[pd.Timestamp(bar_data["timestamp"])])

            self._bars[symbol] = pd.concat([self._bars[symbol], new_row])

            # Trim to max bars
            if len(self._bars[symbol]) > self._max_bars:
                self._bars[symbol] = self._bars[symbol].iloc[-self._max_bars:]

        # Publish event
        self.event_bus.publish(Event(
            type=EventType.BAR,
            data=bar_data,
            source="MarketDataManager"
        ))

        # Call callbacks
        for callback in self._bar_callbacks.get(symbol, []):
            try:
                callback(bar_data)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}")

    def get_bars(
        self,
        symbol: str,
        count: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get bar data for a symbol.

        Args:
            symbol: Ticker symbol
            count: Number of bars to return (from end)
            start_time: Start time filter
            end_time: End time filter

        Returns:
            DataFrame with OHLCV data
        """
        with self._lock:
            if symbol not in self._bars:
                return pd.DataFrame()

            df = self._bars[symbol].copy()

        # Apply filters
        if start_time:
            df = df[df.index >= start_time]
        if end_time:
            df = df[df.index <= end_time]
        if count:
            df = df.iloc[-count:]

        return df

    def load_historical(
        self,
        symbol: str,
        duration: str = "5 D",
        bar_size: str = "1 min",
    ) -> pd.DataFrame:
        """
        Load historical data from IBKR.

        Args:
            symbol: Ticker symbol
            duration: Duration string (e.g., "5 D", "1 M")
            bar_size: Bar size (e.g., "1 min", "5 mins")

        Returns:
            DataFrame with OHLCV data
        """
        if not self.ibkr_client or not self.ibkr_client.connected:
            logger.warning(f"IBKR not connected, cannot load historical data for {symbol}")
            return pd.DataFrame()

        df = self.ibkr_client.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
        )

        with self._lock:
            if symbol in self._bars and not self._bars[symbol].empty:
                # Merge with existing data
                self._bars[symbol] = pd.concat([df, self._bars[symbol]])
                self._bars[symbol] = self._bars[symbol][~self._bars[symbol].index.duplicated(keep="last")]
                self._bars[symbol].sort_index(inplace=True)
            else:
                self._bars[symbol] = df

            # Trim to max bars
            if len(self._bars[symbol]) > self._max_bars:
                self._bars[symbol] = self._bars[symbol].iloc[-self._max_bars:]

        logger.info(f"Loaded {len(df)} historical bars for {symbol}")
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol."""
        with self._lock:
            if symbol in self._bars and not self._bars[symbol].empty:
                return self._bars[symbol]["close"].iloc[-1]

        # Try to get from IBKR
        if self.ibkr_client and self.ibkr_client.connected:
            return self.ibkr_client.get_current_price(symbol)

        return None

    def get_ohlcv(
        self,
        symbol: str,
        periods: int = 100
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get OHLCV arrays for technical analysis.

        Args:
            symbol: Ticker symbol
            periods: Number of periods

        Returns:
            Tuple of (open, high, low, close, volume) arrays
        """
        df = self.get_bars(symbol, count=periods)

        if df.empty:
            empty = np.array([])
            return empty, empty, empty, empty, empty

        return (
            df["open"].values,
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["volume"].values,
        )

    def calculate_indicators(
        self,
        symbol: str,
        indicators: list[str],
        periods: int = 100
    ) -> dict[str, np.ndarray]:
        """
        Calculate technical indicators.

        Args:
            symbol: Ticker symbol
            indicators: List of indicator names
            periods: Number of periods

        Returns:
            Dictionary of indicator name to values
        """
        df = self.get_bars(symbol, count=periods + 50)  # Extra for calculations

        if df.empty or len(df) < 20:
            return {}

        results = {}

        try:
            import pandas_ta as ta

            for indicator in indicators:
                if indicator == "sma_20":
                    results["sma_20"] = ta.sma(df["close"], length=20).values[-periods:]
                elif indicator == "sma_50":
                    results["sma_50"] = ta.sma(df["close"], length=50).values[-periods:]
                elif indicator == "ema_9":
                    results["ema_9"] = ta.ema(df["close"], length=9).values[-periods:]
                elif indicator == "ema_21":
                    results["ema_21"] = ta.ema(df["close"], length=21).values[-periods:]
                elif indicator == "rsi":
                    results["rsi"] = ta.rsi(df["close"], length=14).values[-periods:]
                elif indicator == "macd":
                    macd = ta.macd(df["close"])
                    if macd is not None:
                        results["macd"] = macd.iloc[:, 0].values[-periods:]
                        results["macd_signal"] = macd.iloc[:, 1].values[-periods:]
                        results["macd_hist"] = macd.iloc[:, 2].values[-periods:]
                elif indicator == "bbands":
                    bb = ta.bbands(df["close"], length=20)
                    if bb is not None:
                        results["bb_upper"] = bb.iloc[:, 0].values[-periods:]
                        results["bb_middle"] = bb.iloc[:, 1].values[-periods:]
                        results["bb_lower"] = bb.iloc[:, 2].values[-periods:]
                elif indicator == "atr":
                    results["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14).values[-periods:]
                elif indicator == "vwap":
                    # Simple VWAP calculation
                    typical_price = (df["high"] + df["low"] + df["close"]) / 3
                    results["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
                    results["vwap"] = results["vwap"].values[-periods:]

        except ImportError:
            logger.warning("pandas_ta not installed, using basic calculations")
            # Fallback to basic calculations
            close = df["close"].values

            if "sma_20" in indicators:
                results["sma_20"] = self._sma(close, 20)[-periods:]
            if "sma_50" in indicators:
                results["sma_50"] = self._sma(close, 50)[-periods:]

        return results

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        weights = np.ones(period) / period
        return np.convolve(data, weights, mode="valid")

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        from pytz import timezone
        from datetime import time

        tz = timezone(self.config.market.timezone)
        now = datetime.now(tz)

        # Check day of week
        if now.strftime("%A") not in self.config.market.trading_days:
            return False

        # Check time
        open_time = datetime.strptime(self.config.market.open_time, "%H:%M").time()
        close_time = datetime.strptime(self.config.market.close_time, "%H:%M").time()

        return open_time <= now.time() <= close_time
