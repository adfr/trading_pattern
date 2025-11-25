"""Signal generation from pattern matches."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.logger import get_logger
from .pattern import PatternMatch, PatternMatcher

logger = get_logger(__name__)


class SignalType(Enum):
    """Type of trading signal."""
    LONG_ENTRY = auto()
    LONG_EXIT = auto()
    SHORT_ENTRY = auto()
    SHORT_EXIT = auto()
    HOLD = auto()


@dataclass
class Signal:
    """Trading signal."""
    signal_type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    pattern_match: Optional[PatternMatch] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_entry(self) -> bool:
        """Check if signal is an entry."""
        return self.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY)

    @property
    def is_exit(self) -> bool:
        """Check if signal is an exit."""
        return self.signal_type in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT)

    @property
    def is_long(self) -> bool:
        """Check if signal is for long position."""
        return self.signal_type in (SignalType.LONG_ENTRY, SignalType.LONG_EXIT)


class SignalGenerator:
    """
    Generates trading signals from pattern matches.

    Combines pattern detection with entry/exit logic and risk parameters.
    """

    def __init__(
        self,
        pattern_matcher: PatternMatcher,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize signal generator.

        Args:
            pattern_matcher: Pattern matcher instance
            event_bus: Event bus (optional)
        """
        self.pattern_matcher = pattern_matcher
        self.event_bus = event_bus or get_event_bus()

        # Signal generation settings
        self.min_confidence = 0.6
        self.cooldown_bars = 5  # Minimum bars between signals

        # Track recent signals
        self._last_signal_bar: dict[str, int] = {}

    def generate_signals(
        self,
        symbol: str,
        df: pd.DataFrame,
        current_position: str = "flat",  # flat, long, short
    ) -> list[Signal]:
        """
        Generate signals from market data.

        Args:
            symbol: Ticker symbol
            df: DataFrame with OHLCV data
            current_position: Current position state

        Returns:
            List of signals
        """
        if df.empty:
            return []

        signals = []

        # Check for pattern matches at current bar
        matches = self.pattern_matcher.check_current(
            df,
            min_confidence=self.min_confidence,
        )

        current_bar = len(df) - 1
        current_price = df["close"].iloc[-1]
        current_time = df.index[-1] if hasattr(df.index[-1], "isoformat") else datetime.now()

        # Check cooldown
        last_bar = self._last_signal_bar.get(symbol, -999)
        if current_bar - last_bar < self.cooldown_bars:
            return []

        for match in matches:
            signal = self._create_signal_from_match(
                match=match,
                symbol=symbol,
                df=df,
                current_position=current_position,
                current_price=current_price,
                timestamp=current_time,
            )

            if signal:
                signals.append(signal)
                self._last_signal_bar[symbol] = current_bar

                # Publish event
                self.event_bus.publish(Event(
                    type=EventType.SIGNAL_GENERATED,
                    data={
                        "signal_type": signal.signal_type.name,
                        "symbol": symbol,
                        "price": signal.price,
                        "pattern": match.pattern_name,
                        "confidence": signal.confidence,
                    },
                    source="SignalGenerator"
                ))

        return signals

    def _create_signal_from_match(
        self,
        match: PatternMatch,
        symbol: str,
        df: pd.DataFrame,
        current_position: str,
        current_price: float,
        timestamp: datetime,
    ) -> Optional[Signal]:
        """Create a signal from a pattern match."""
        pattern = self.pattern_matcher._patterns.get(match.pattern_id)
        if not pattern:
            return None

        # Determine signal type based on pattern type and position
        signal_type = self._determine_signal_type(
            pattern_type=match.pattern_type,
            current_position=current_position,
        )

        if signal_type == SignalType.HOLD:
            return None

        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_exit_levels(
            pattern=pattern,
            df=df,
            signal_type=signal_type,
            current_price=current_price,
        )

        return Signal(
            signal_type=signal_type,
            symbol=symbol,
            timestamp=timestamp,
            price=current_price,
            pattern_match=match,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=match.confidence,
            metadata={
                "pattern_id": match.pattern_id,
                "pattern_name": match.pattern_name,
                "conditions_met": match.conditions_met,
            }
        )

    def _determine_signal_type(
        self,
        pattern_type: str,
        current_position: str,
    ) -> SignalType:
        """Determine signal type based on pattern and position."""
        if pattern_type == "bullish":
            if current_position == "flat":
                return SignalType.LONG_ENTRY
            elif current_position == "short":
                return SignalType.SHORT_EXIT  # Exit short before going long
            else:
                return SignalType.HOLD  # Already long

        elif pattern_type == "bearish":
            if current_position == "flat":
                return SignalType.SHORT_ENTRY
            elif current_position == "long":
                return SignalType.LONG_EXIT  # Exit long before going short
            else:
                return SignalType.HOLD  # Already short

        return SignalType.HOLD

    def _calculate_exit_levels(
        self,
        pattern: Any,
        df: pd.DataFrame,
        signal_type: SignalType,
        current_price: float,
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        exit_rules = pattern.exit

        # Calculate ATR for dynamic stops
        atr = self._calculate_atr(df)

        stop_loss = None
        take_profit = None

        # Stop loss
        sl_config = exit_rules.get("stop_loss", {})
        sl_type = sl_config.get("type", "atr")
        sl_value = float(sl_config.get("value", 2.0))

        if sl_type == "atr":
            sl_distance = atr * sl_value
        elif sl_type == "percent":
            sl_distance = current_price * (sl_value / 100)
        else:  # fixed
            sl_distance = sl_value

        # Take profit
        tp_config = exit_rules.get("take_profit", {})
        tp_type = tp_config.get("type", "atr")
        tp_value = float(tp_config.get("value", 3.0))

        if tp_type == "atr":
            tp_distance = atr * tp_value
        elif tp_type == "percent":
            tp_distance = current_price * (tp_value / 100)
        else:  # fixed
            tp_distance = tp_value

        # Apply direction
        if signal_type in (SignalType.LONG_ENTRY,):
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance
        elif signal_type in (SignalType.SHORT_ENTRY,):
            stop_loss = current_price + sl_distance
            take_profit = current_price - tp_distance

        return stop_loss, take_profit

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR."""
        if len(df) < period:
            return df["close"].iloc[-1] * 0.02  # Default 2%

        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else df["close"].iloc[-1] * 0.02

    def check_exit_conditions(
        self,
        symbol: str,
        df: pd.DataFrame,
        position: dict[str, Any],
    ) -> Optional[Signal]:
        """
        Check if exit conditions are met for a position.

        Args:
            symbol: Ticker symbol
            df: Current market data
            position: Position dictionary

        Returns:
            Exit signal if conditions met, None otherwise
        """
        if df.empty:
            return None

        current_price = df["close"].iloc[-1]
        current_time = df.index[-1] if hasattr(df.index[-1], "isoformat") else datetime.now()

        direction = position.get("direction", "long")
        entry_price = position.get("entry_price", current_price)
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        entry_bar = position.get("entry_bar", 0)

        # Check stop loss
        if stop_loss:
            if direction == "long" and current_price <= stop_loss:
                return Signal(
                    signal_type=SignalType.LONG_EXIT,
                    symbol=symbol,
                    timestamp=current_time,
                    price=current_price,
                    metadata={"exit_reason": "stop_loss"}
                )
            elif direction == "short" and current_price >= stop_loss:
                return Signal(
                    signal_type=SignalType.SHORT_EXIT,
                    symbol=symbol,
                    timestamp=current_time,
                    price=current_price,
                    metadata={"exit_reason": "stop_loss"}
                )

        # Check take profit
        if take_profit:
            if direction == "long" and current_price >= take_profit:
                return Signal(
                    signal_type=SignalType.LONG_EXIT,
                    symbol=symbol,
                    timestamp=current_time,
                    price=current_price,
                    metadata={"exit_reason": "take_profit"}
                )
            elif direction == "short" and current_price <= take_profit:
                return Signal(
                    signal_type=SignalType.SHORT_EXIT,
                    symbol=symbol,
                    timestamp=current_time,
                    price=current_price,
                    metadata={"exit_reason": "take_profit"}
                )

        # Check time-based exit
        pattern_id = position.get("pattern_id")
        if pattern_id and pattern_id in self.pattern_matcher._patterns:
            pattern = self.pattern_matcher._patterns[pattern_id]
            time_exit = pattern.exit.get("time_exit", {})

            if time_exit.get("enabled", False):
                max_bars = time_exit.get("bars", 60)
                current_bar = len(df) - 1

                if current_bar - entry_bar >= max_bars:
                    signal_type = SignalType.LONG_EXIT if direction == "long" else SignalType.SHORT_EXIT
                    return Signal(
                        signal_type=signal_type,
                        symbol=symbol,
                        timestamp=current_time,
                        price=current_price,
                        metadata={"exit_reason": "time_exit"}
                    )

        return None
