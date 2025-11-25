"""Pattern matching and detection."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PatternDefinition:
    """Definition of a trading pattern."""
    id: str
    name: str
    pattern_type: str  # bullish, bearish, neutral
    conditions: list[dict[str, Any]]
    min_conditions_met: int
    entry: dict[str, Any]
    exit: dict[str, Any]
    filters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatternDefinition":
        """Create PatternDefinition from dictionary."""
        detection = data.get("detection", {})
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            pattern_type=data.get("pattern_type", "neutral"),
            conditions=detection.get("conditions", []),
            min_conditions_met=detection.get("min_conditions_met", 1),
            entry=data.get("entry", {}),
            exit=data.get("exit", {}),
            filters=data.get("filters", {}),
            metadata={
                "description": data.get("description", ""),
                "rationale": data.get("rationale", ""),
                "symbol": data.get("symbol", ""),
                "timeframe": data.get("timeframe", ""),
                "created_at": data.get("created_at", ""),
            }
        )


@dataclass
class PatternMatch:
    """Result of pattern matching."""
    pattern_id: str
    pattern_name: str
    pattern_type: str
    timestamp: datetime
    confidence: float
    conditions_met: int
    total_conditions: int
    bar_index: int
    price: float
    indicators: dict[str, float] = field(default_factory=dict)


class PatternMatcher:
    """
    Matches patterns against market data.

    Evaluates pattern conditions and generates matches with confidence scores.
    """

    def __init__(self):
        """Initialize pattern matcher."""
        self._patterns: dict[str, PatternDefinition] = {}
        self._indicator_cache: dict[str, np.ndarray] = {}

        # Indicator calculators
        self._indicator_funcs: dict[str, Callable] = {
            "sma": self._calc_sma,
            "ema": self._calc_ema,
            "rsi": self._calc_rsi,
            "macd": self._calc_macd,
            "macd_signal": self._calc_macd_signal,
            "macd_hist": self._calc_macd_hist,
            "bbands_upper": self._calc_bbands_upper,
            "bbands_lower": self._calc_bbands_lower,
            "bbands_middle": self._calc_bbands_middle,
            "atr": self._calc_atr,
            "volume_sma": self._calc_volume_sma,
            "high": lambda df, p: df["high"].values,
            "low": lambda df, p: df["low"].values,
            "close": lambda df, p: df["close"].values,
            "open": lambda df, p: df["open"].values,
            "volume": lambda df, p: df["volume"].values,
        }

    def add_pattern(self, pattern: PatternDefinition) -> None:
        """Add a pattern to the matcher."""
        self._patterns[pattern.id] = pattern
        logger.info(f"Added pattern: {pattern.name} ({pattern.id})")

    def remove_pattern(self, pattern_id: str) -> None:
        """Remove a pattern from the matcher."""
        if pattern_id in self._patterns:
            del self._patterns[pattern_id]
            logger.info(f"Removed pattern: {pattern_id}")

    def load_pattern_from_dict(self, data: dict[str, Any]) -> PatternDefinition:
        """Load pattern from dictionary and add to matcher."""
        pattern = PatternDefinition.from_dict(data)
        self.add_pattern(pattern)
        return pattern

    def find_matches(
        self,
        df: pd.DataFrame,
        pattern_ids: Optional[list[str]] = None,
        min_confidence: float = 0.5,
    ) -> list[PatternMatch]:
        """
        Find pattern matches in data.

        Args:
            df: DataFrame with OHLCV data
            pattern_ids: Specific patterns to check (None = all)
            min_confidence: Minimum confidence threshold

        Returns:
            List of pattern matches
        """
        if df.empty:
            return []

        matches = []
        patterns_to_check = pattern_ids or list(self._patterns.keys())

        # Clear indicator cache for new data
        self._indicator_cache.clear()

        for pattern_id in patterns_to_check:
            if pattern_id not in self._patterns:
                continue

            pattern = self._patterns[pattern_id]

            # Check each bar
            for i in range(max(50, len(df) // 10), len(df)):
                match = self._check_pattern_at_bar(df, pattern, i)
                if match and match.confidence >= min_confidence:
                    matches.append(match)

        # Sort by timestamp
        matches.sort(key=lambda m: m.timestamp)

        return matches

    def check_current(
        self,
        df: pd.DataFrame,
        pattern_ids: Optional[list[str]] = None,
        min_confidence: float = 0.5,
    ) -> list[PatternMatch]:
        """
        Check for pattern matches at the current bar only.

        Args:
            df: DataFrame with OHLCV data
            pattern_ids: Specific patterns to check (None = all)
            min_confidence: Minimum confidence threshold

        Returns:
            List of matches at current bar
        """
        if df.empty:
            return []

        matches = []
        patterns_to_check = pattern_ids or list(self._patterns.keys())

        # Clear cache
        self._indicator_cache.clear()

        # Check only the last bar
        bar_idx = len(df) - 1

        for pattern_id in patterns_to_check:
            if pattern_id not in self._patterns:
                continue

            pattern = self._patterns[pattern_id]
            match = self._check_pattern_at_bar(df, pattern, bar_idx)

            if match and match.confidence >= min_confidence:
                matches.append(match)

        return matches

    def _check_pattern_at_bar(
        self,
        df: pd.DataFrame,
        pattern: PatternDefinition,
        bar_idx: int,
    ) -> Optional[PatternMatch]:
        """Check if pattern matches at specific bar."""
        conditions_met = 0
        indicators_used = {}

        for condition in pattern.conditions:
            try:
                result = self._evaluate_condition(df, condition, bar_idx)
                if result:
                    conditions_met += 1
            except Exception as e:
                logger.debug(f"Condition evaluation error: {e}")
                continue

        # Check if enough conditions are met
        if conditions_met < pattern.min_conditions_met:
            return None

        # Apply filters
        if not self._check_filters(df, pattern.filters, bar_idx):
            return None

        # Calculate confidence
        total_conditions = len(pattern.conditions)
        confidence = conditions_met / total_conditions if total_conditions > 0 else 0

        return PatternMatch(
            pattern_id=pattern.id,
            pattern_name=pattern.name,
            pattern_type=pattern.pattern_type,
            timestamp=df.index[bar_idx] if isinstance(df.index[bar_idx], datetime) else datetime.now(),
            confidence=confidence,
            conditions_met=conditions_met,
            total_conditions=total_conditions,
            bar_index=bar_idx,
            price=df["close"].iloc[bar_idx],
            indicators=indicators_used,
        )

    def _evaluate_condition(
        self,
        df: pd.DataFrame,
        condition: dict[str, Any],
        bar_idx: int,
    ) -> bool:
        """Evaluate a single condition."""
        indicator = condition.get("indicator", "")
        operator = condition.get("operator", "")
        value = condition.get("value")
        lookback = condition.get("lookback", 0)

        # Get indicator value
        left_value = self._get_indicator_value(df, indicator, bar_idx - lookback)

        if left_value is None or np.isnan(left_value):
            return False

        # Get comparison value
        if isinstance(value, str):
            # Value is another indicator
            right_value = self._get_indicator_value(df, value, bar_idx - lookback)
            if right_value is None or np.isnan(right_value):
                return False
        else:
            right_value = float(value)

        # Evaluate operator
        if operator == ">":
            return left_value > right_value
        elif operator == "<":
            return left_value < right_value
        elif operator == ">=":
            return left_value >= right_value
        elif operator == "<=":
            return left_value <= right_value
        elif operator == "==":
            return abs(left_value - right_value) < 0.0001
        elif operator == "crosses_above":
            prev_left = self._get_indicator_value(df, indicator, bar_idx - lookback - 1)
            prev_right = self._get_indicator_value(df, value, bar_idx - lookback - 1) if isinstance(value, str) else right_value
            if prev_left is None or prev_right is None:
                return False
            return prev_left <= prev_right and left_value > right_value
        elif operator == "crosses_below":
            prev_left = self._get_indicator_value(df, indicator, bar_idx - lookback - 1)
            prev_right = self._get_indicator_value(df, value, bar_idx - lookback - 1) if isinstance(value, str) else right_value
            if prev_left is None or prev_right is None:
                return False
            return prev_left >= prev_right and left_value < right_value

        return False

    def _get_indicator_value(
        self,
        df: pd.DataFrame,
        indicator: str,
        bar_idx: int,
    ) -> Optional[float]:
        """Get indicator value at bar index."""
        if bar_idx < 0 or bar_idx >= len(df):
            return None

        # Parse indicator name and period
        parts = indicator.split("_")
        base_name = parts[0].lower()
        period = int(parts[-1]) if len(parts) > 1 and parts[-1].isdigit() else 14

        # Check cache
        cache_key = f"{indicator}_{len(df)}"
        if cache_key in self._indicator_cache:
            values = self._indicator_cache[cache_key]
            if bar_idx < len(values):
                return values[bar_idx]

        # Calculate indicator
        if base_name in self._indicator_funcs:
            values = self._indicator_funcs[base_name](df, period)
            self._indicator_cache[cache_key] = values
            if bar_idx < len(values):
                return values[bar_idx]

        # Try direct column access
        if indicator in df.columns:
            return df[indicator].iloc[bar_idx]

        return None

    def _check_filters(
        self,
        df: pd.DataFrame,
        filters: dict[str, Any],
        bar_idx: int,
    ) -> bool:
        """Check if filters pass."""
        if not filters:
            return True

        # Volume filter
        if "volume_min" in filters:
            vol = df["volume"].iloc[bar_idx]
            if vol < filters["volume_min"]:
                return False

        # Volatility filter
        if "volatility_range" in filters:
            atr = self._calc_atr(df, 14)
            if bar_idx < len(atr):
                current_atr = atr[bar_idx]
                vol_range = filters["volatility_range"]
                if current_atr < vol_range.get("min", 0):
                    return False
                if current_atr > vol_range.get("max", float("inf")):
                    return False

        # Time filter
        if "time_of_day" in filters:
            time_filter = filters["time_of_day"]
            if hasattr(df.index[bar_idx], "time"):
                bar_time = df.index[bar_idx].time()
                start_time = datetime.strptime(time_filter.get("start", "00:00"), "%H:%M").time()
                end_time = datetime.strptime(time_filter.get("end", "23:59"), "%H:%M").time()
                if not (start_time <= bar_time <= end_time):
                    return False

        return True

    # Indicator calculations
    def _calc_sma(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        return df["close"].rolling(period).mean().values

    def _calc_ema(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        return df["close"].ewm(span=period, adjust=False).mean().values

    def _calc_rsi(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values

    def _calc_macd(self, df: pd.DataFrame, period: int = 12) -> np.ndarray:
        """Calculate MACD line."""
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        return (ema12 - ema26).values

    def _calc_macd_signal(self, df: pd.DataFrame, period: int = 9) -> np.ndarray:
        """Calculate MACD signal line."""
        macd = self._calc_macd(df, 12)
        return pd.Series(macd).ewm(span=9, adjust=False).mean().values

    def _calc_macd_hist(self, df: pd.DataFrame, period: int = 9) -> np.ndarray:
        """Calculate MACD histogram."""
        macd = self._calc_macd(df, 12)
        signal = self._calc_macd_signal(df, 9)
        return macd - signal

    def _calc_bbands_upper(self, df: pd.DataFrame, period: int = 20) -> np.ndarray:
        """Calculate Bollinger Bands upper."""
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        return (sma + 2 * std).values

    def _calc_bbands_lower(self, df: pd.DataFrame, period: int = 20) -> np.ndarray:
        """Calculate Bollinger Bands lower."""
        sma = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        return (sma - 2 * std).values

    def _calc_bbands_middle(self, df: pd.DataFrame, period: int = 20) -> np.ndarray:
        """Calculate Bollinger Bands middle."""
        return df["close"].rolling(period).mean().values

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean().values

    def _calc_volume_sma(self, df: pd.DataFrame, period: int = 20) -> np.ndarray:
        """Calculate Volume SMA."""
        return df["volume"].rolling(period).mean().values
