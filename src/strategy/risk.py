"""Risk management and position sizing."""

from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Optional

import numpy as np

from ..core.config import Config
from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.exceptions import RiskLimitExceeded
from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    total_exposure: float
    position_count: int
    daily_pnl: float
    daily_trades: int
    max_drawdown: float
    current_drawdown: float


class PositionSizer:
    """
    Calculates position sizes based on risk parameters.

    Implements Kelly criterion and fixed fractional sizing.
    """

    def __init__(self, config: Config):
        """
        Initialize position sizer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.kelly_fraction = config.risk.kelly_fraction
        self.max_position_pct = config.trading.max_position_size

    def calculate_size(
        self,
        account_value: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
    ) -> int:
        """
        Calculate position size.

        Args:
            account_value: Total account value
            entry_price: Entry price
            stop_loss: Stop loss price
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win / average loss

        Returns:
            Number of shares to trade
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0

        # Calculate Kelly fraction
        kelly = self._kelly_criterion(win_rate, avg_win_loss_ratio)

        # Apply fractional Kelly
        adj_kelly = kelly * self.kelly_fraction

        # Cap at maximum position size
        position_pct = min(adj_kelly, self.max_position_pct)

        # Calculate position value
        position_value = account_value * position_pct

        # Calculate number of shares based on risk
        # Risk amount = position_pct * account_value (but we risk the stop loss distance)
        risk_amount = account_value * self.config.risk.kelly_fraction * 0.01  # 1% risk per trade
        shares_from_risk = int(risk_amount / risk_per_share)

        # Also limit by position value
        shares_from_value = int(position_value / entry_price)

        # Take the smaller of the two
        shares = min(shares_from_risk, shares_from_value)

        # Ensure at least 1 share if we're trading
        return max(1, shares) if shares > 0 else 0

    def _kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly criterion.

        Kelly % = W - (1-W)/R
        Where:
            W = Win rate
            R = Win/Loss ratio

        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / average loss

        Returns:
            Optimal fraction of capital to risk
        """
        if win_loss_ratio == 0:
            return 0

        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Kelly can be negative (don't trade) or very high (unrealistic)
        return max(0, min(kelly, 0.5))

    def calculate_fixed_fractional(
        self,
        account_value: float,
        entry_price: float,
        risk_pct: float = 0.01,
        stop_loss: Optional[float] = None,
    ) -> int:
        """
        Calculate position size using fixed fractional method.

        Args:
            account_value: Total account value
            entry_price: Entry price
            risk_pct: Percentage of account to risk (default 1%)
            stop_loss: Stop loss price (optional)

        Returns:
            Number of shares
        """
        risk_amount = account_value * risk_pct

        if stop_loss:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                return int(risk_amount / risk_per_share)

        # Default: use entry price as basis
        max_position_value = account_value * self.max_position_pct
        return int(max_position_value / entry_price)


class RiskManager:
    """
    Manages overall portfolio risk.

    Enforces risk limits and monitors exposure.
    """

    def __init__(
        self,
        config: Config,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize risk manager.

        Args:
            config: Configuration object
            event_bus: Event bus (optional)
        """
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        self.position_sizer = PositionSizer(config)

        # Track daily metrics
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._current_date: date = date.today()

        # Track positions
        self._positions: dict[str, dict[str, Any]] = {}

        # Track equity curve for drawdown
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0

    def check_trade_allowed(
        self,
        symbol: str,
        direction: str,
        size: int,
        entry_price: float,
        account_value: float,
    ) -> tuple[bool, str]:
        """
        Check if a trade is allowed under risk rules.

        Args:
            symbol: Ticker symbol
            direction: Trade direction (long/short)
            size: Position size
            entry_price: Entry price
            account_value: Current account value

        Returns:
            Tuple of (allowed, reason)
        """
        self._reset_daily_if_needed()

        # Check daily loss limit
        if self._daily_pnl < 0:
            daily_loss_pct = abs(self._daily_pnl) / account_value
            if daily_loss_pct >= self.config.trading.daily_loss_limit:
                self._publish_risk_event("daily_loss_limit")
                return False, "Daily loss limit reached"

        # Check max trades per day
        if self._daily_trades >= self.config.risk.max_trades_per_day:
            return False, "Maximum daily trades reached"

        # Check total exposure
        position_value = size * entry_price
        current_exposure = self._calculate_total_exposure()
        new_exposure = (current_exposure + position_value) / account_value

        if new_exposure > self.config.trading.max_total_exposure:
            self._publish_risk_event("exposure_limit")
            return False, f"Total exposure would exceed {self.config.trading.max_total_exposure:.0%}"

        # Check single position size
        position_pct = position_value / account_value
        if position_pct > self.config.trading.max_position_size:
            return False, f"Position size exceeds {self.config.trading.max_position_size:.0%}"

        # Check max drawdown
        if self._current_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            if drawdown >= self.config.backtest.max_drawdown:
                self._publish_risk_event("max_drawdown")
                return False, f"Maximum drawdown of {self.config.backtest.max_drawdown:.0%} reached"

        return True, "Trade allowed"

    def register_trade(
        self,
        symbol: str,
        direction: str,
        size: int,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Register a new trade."""
        self._daily_trades += 1

        self._positions[symbol] = {
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_time": datetime.now(),
        }

        logger.info(f"Registered trade: {direction} {size} {symbol} @ {entry_price}")

    def close_position(
        self,
        symbol: str,
        exit_price: float,
    ) -> float:
        """
        Close a position and return P&L.

        Args:
            symbol: Ticker symbol
            exit_price: Exit price

        Returns:
            Position P&L
        """
        if symbol not in self._positions:
            return 0.0

        position = self._positions[symbol]
        direction = position["direction"]
        size = position["size"]
        entry_price = position["entry_price"]

        if direction == "long":
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size

        self._daily_pnl += pnl
        del self._positions[symbol]

        logger.info(f"Closed position: {symbol} P&L: ${pnl:.2f}")
        return pnl

    def update_equity(self, equity: float) -> None:
        """Update current equity for drawdown tracking."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def get_metrics(self, account_value: float) -> RiskMetrics:
        """Get current risk metrics."""
        exposure = self._calculate_total_exposure()

        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity

        return RiskMetrics(
            total_exposure=exposure / account_value if account_value > 0 else 0,
            position_count=len(self._positions),
            daily_pnl=self._daily_pnl,
            daily_trades=self._daily_trades,
            max_drawdown=self.config.backtest.max_drawdown,
            current_drawdown=max(0, drawdown),
        )

    def get_position(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, dict[str, Any]]:
        """Get all positions."""
        return self._positions.copy()

    def _calculate_total_exposure(self) -> float:
        """Calculate total position exposure."""
        total = 0.0
        for pos in self._positions.values():
            total += pos["size"] * pos["entry_price"]
        return total

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counters if new day."""
        today = date.today()
        if today != self._current_date:
            self._current_date = today
            self._daily_pnl = 0.0
            self._daily_trades = 0
            logger.info("Daily risk counters reset")

    def _publish_risk_event(self, reason: str) -> None:
        """Publish a risk warning event."""
        self.event_bus.publish(Event(
            type=EventType.RISK_LIMIT_WARNING,
            data={"reason": reason},
            source="RiskManager"
        ))

    def calculate_position_size(
        self,
        account_value: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float = 0.5,
        win_loss_ratio: float = 1.5,
    ) -> int:
        """
        Calculate recommended position size.

        Args:
            account_value: Account value
            entry_price: Entry price
            stop_loss: Stop loss price
            win_rate: Historical win rate
            win_loss_ratio: Average win/loss ratio

        Returns:
            Recommended position size
        """
        return self.position_sizer.calculate_size(
            account_value=account_value,
            entry_price=entry_price,
            stop_loss=stop_loss,
            win_rate=win_rate,
            avg_win_loss_ratio=win_loss_ratio,
        )
