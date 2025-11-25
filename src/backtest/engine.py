"""Backtesting engine."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.database import Database
from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.logger import get_logger
from ..strategy.pattern import PatternDefinition, PatternMatcher
from ..strategy.pattern_manager import get_pattern_manager
from ..strategy.signal import Signal, SignalGenerator, SignalType
from .metrics import MetricsCalculator, PerformanceMetrics

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    pattern_id: str
    pattern_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # Trades
    trades: list[dict[str, Any]] = field(default_factory=list)

    # Metrics
    metrics: Optional[PerformanceMetrics] = None

    # Validation
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    # Equity curve
    equity_curve: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_trades": len(self.trades),
            "winning_trades": self.metrics.winning_trades if self.metrics else 0,
            "losing_trades": self.metrics.losing_trades if self.metrics else 0,
            "win_rate": self.metrics.win_rate if self.metrics else 0,
            "profit_factor": self.metrics.profit_factor if self.metrics else 0,
            "sharpe_ratio": self.metrics.sharpe_ratio if self.metrics else 0,
            "sortino_ratio": self.metrics.sortino_ratio if self.metrics else 0,
            "max_drawdown": self.metrics.max_drawdown if self.metrics else 0,
            "total_return": self.metrics.total_return if self.metrics else 0,
            "annualized_return": self.metrics.annualized_return if self.metrics else 0,
            "t_statistic": self.metrics.t_statistic if self.metrics else 0,
            "p_value": self.metrics.p_value if self.metrics else 1,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
            "detailed_results": {
                "trades": self.trades,
                "metrics": self.metrics.to_dict() if self.metrics else {},
            }
        }


class BacktestEngine:
    """
    Rigorous backtesting engine for pattern evaluation.

    Simulates trading with realistic constraints and calculates
    comprehensive performance metrics.
    """

    def __init__(
        self,
        config: Config,
        database: Optional[Database] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Configuration object
            database: Database for storing results (optional)
            event_bus: Event bus (optional)
        """
        self.config = config
        self.database = database
        self.event_bus = event_bus or get_event_bus()
        self.metrics_calculator = MetricsCalculator()

        # Pattern manager for persistent patterns
        self.pattern_manager = get_pattern_manager(database)

        # Simulation parameters
        self.initial_capital = 100000
        self.commission = 0.001  # 0.1% per trade
        self.slippage = 0.0005  # 0.05% slippage

    def run(
        self,
        pattern: dict[str, Any],
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> BacktestResult:
        """
        Run backtest for a pattern.

        Args:
            pattern: Pattern dictionary
            data: OHLCV DataFrame
            start_date: Backtest start date (optional)
            end_date: Backtest end date (optional)

        Returns:
            BacktestResult object
        """
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        if data.empty:
            return self._create_empty_result(pattern)

        # Add pattern to manager (keeps it in memory after backtest)
        pattern_def = self.pattern_manager.add_pattern(pattern)
        signal_generator = SignalGenerator(self.pattern_manager.pattern_matcher)

        # Publish start event
        self.event_bus.publish(Event(
            type=EventType.BACKTEST_START,
            data={"pattern_id": pattern.get("id"), "bars": len(data)},
            source="BacktestEngine"
        ))

        # Run simulation
        result = self._simulate(
            pattern=pattern_def,
            data=data,
            signal_generator=signal_generator,
        )

        result.pattern_id = pattern.get("id", "")
        result.pattern_name = pattern.get("name", "")
        result.start_date = data.index[0] if hasattr(data.index[0], "isoformat") else datetime.now()
        result.end_date = data.index[-1] if hasattr(data.index[-1], "isoformat") else datetime.now()

        # Calculate metrics
        result.metrics = self.metrics_calculator.calculate(
            trades=result.trades,
            equity_curve=result.equity_curve,
            initial_capital=self.initial_capital,
        )

        # Validate against criteria
        result.passed, result.failure_reasons = self.metrics_calculator.validate_against_criteria(
            result.metrics,
            {
                "min_trades": self.config.backtest.min_trades,
                "min_win_rate": self.config.backtest.min_win_rate,
                "min_profit_factor": self.config.backtest.min_profit_factor,
                "min_sharpe_ratio": self.config.backtest.min_sharpe_ratio,
                "max_drawdown": self.config.backtest.max_drawdown,
            }
        )

        # Store result in database
        if self.database:
            self.database.save_backtest_result(result.to_dict())

        # Publish complete event
        self.event_bus.publish(Event(
            type=EventType.BACKTEST_COMPLETE,
            data={
                "pattern_id": result.pattern_id,
                "passed": result.passed,
                "total_trades": len(result.trades),
            },
            source="BacktestEngine"
        ))

        logger.info(
            f"Backtest complete: {result.pattern_name} - "
            f"{'PASSED' if result.passed else 'FAILED'} "
            f"({len(result.trades)} trades, {result.metrics.win_rate:.1%} win rate)"
        )

        return result

    def _simulate(
        self,
        pattern: PatternDefinition,
        data: pd.DataFrame,
        signal_generator: SignalGenerator,
    ) -> BacktestResult:
        """Run the trading simulation."""
        result = BacktestResult(
            pattern_id="",
            pattern_name="",
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
        )

        # State tracking
        capital = self.initial_capital
        position: Optional[dict] = None
        equity_values = [capital]

        logger.info(f"Starting backtest simulation for '{pattern.name}': {len(data)} bars, starting at bar 50")

        # Iterate through data
        signals_checked = 0
        for i in range(50, len(data)):  # Start after warmup period
            signals_checked += 1
            current_data = data.iloc[:i+1]
            current_bar = data.iloc[i]
            current_price = current_bar["close"]
            current_time = data.index[i]

            # Check exit conditions if in position
            if position:
                exit_signal = self._check_exit(
                    position=position,
                    current_price=current_price,
                    current_bar_idx=i,
                    pattern=pattern,
                )

                if exit_signal:
                    # Execute exit
                    pnl = self._execute_exit(
                        position=position,
                        exit_price=current_price,
                        exit_time=current_time,
                        exit_reason=exit_signal,
                    )
                    capital += pnl
                    result.trades.append(position)
                    position = None

            # Check for new entry signals if flat
            if position is None:
                signals = signal_generator.generate_signals(
                    symbol=pattern.metadata.get("symbol", ""),
                    df=current_data,
                    current_position="flat",
                )

                for signal in signals:
                    if signal.is_entry and signal.confidence >= self.config.patterns.min_confidence:
                        # Execute entry
                        position = self._execute_entry(
                            signal=signal,
                            entry_price=current_price,
                            entry_time=current_time,
                            entry_bar_idx=i,
                            capital=capital,
                        )
                        break  # Only one position at a time

            # Update equity curve
            if position:
                # Mark to market
                unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)
                equity_values.append(capital + unrealized_pnl)
            else:
                equity_values.append(capital)

        # Close any open position at end
        if position:
            final_price = data["close"].iloc[-1]
            pnl = self._execute_exit(
                position=position,
                exit_price=final_price,
                exit_time=data.index[-1],
                exit_reason="end_of_backtest",
            )
            capital += pnl
            result.trades.append(position)

        result.final_capital = capital
        result.equity_curve = np.array(equity_values)

        logger.info(f"Backtest simulation complete: checked {signals_checked} bars, generated {len(result.trades)} trades")

        return result

    def _execute_entry(
        self,
        signal: Signal,
        entry_price: float,
        entry_time: Any,
        entry_bar_idx: int,
        capital: float,
    ) -> dict[str, Any]:
        """Execute an entry trade."""
        # Apply slippage
        if signal.signal_type == SignalType.LONG_ENTRY:
            adjusted_price = entry_price * (1 + self.slippage)
            direction = "long"
        else:
            adjusted_price = entry_price * (1 - self.slippage)
            direction = "short"

        # Calculate position size (simplified for backtest)
        risk_per_trade = capital * 0.01  # 1% risk
        if signal.stop_loss:
            risk_per_share = abs(adjusted_price - signal.stop_loss)
            if risk_per_share > 0:
                size = int(risk_per_trade / risk_per_share)
            else:
                size = int((capital * self.config.trading.max_position_size) / adjusted_price)
        else:
            size = int((capital * self.config.trading.max_position_size) / adjusted_price)

        size = max(1, size)

        # Apply commission
        commission = adjusted_price * size * self.commission

        return {
            "id": str(uuid.uuid4()),
            "direction": direction,
            "entry_price": adjusted_price,
            "entry_time": entry_time,
            "entry_bar_idx": entry_bar_idx,
            "size": size,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "commission_entry": commission,
            "pattern_id": signal.pattern_match.pattern_id if signal.pattern_match else None,
            "confidence": signal.confidence,
        }

    def _execute_exit(
        self,
        position: dict[str, Any],
        exit_price: float,
        exit_time: Any,
        exit_reason: str,
    ) -> float:
        """Execute an exit trade and return P&L."""
        direction = position["direction"]
        entry_price = position["entry_price"]
        size = position["size"]

        # Apply slippage
        if direction == "long":
            adjusted_exit = exit_price * (1 - self.slippage)
            pnl = (adjusted_exit - entry_price) * size
        else:
            adjusted_exit = exit_price * (1 + self.slippage)
            pnl = (entry_price - adjusted_exit) * size

        # Apply commission
        commission = adjusted_exit * size * self.commission
        pnl -= commission + position.get("commission_entry", 0)

        # Update position
        position["exit_price"] = adjusted_exit
        position["exit_time"] = exit_time
        position["exit_reason"] = exit_reason
        position["pnl"] = pnl
        position["pnl_percent"] = pnl / (entry_price * size)
        position["commission_total"] = commission + position.get("commission_entry", 0)

        # Calculate duration
        if hasattr(position.get("entry_time"), "timestamp") and hasattr(exit_time, "timestamp"):
            duration = (exit_time - position["entry_time"]).total_seconds() / 60
            position["duration_minutes"] = duration

        return pnl

    def _check_exit(
        self,
        position: dict[str, Any],
        current_price: float,
        current_bar_idx: int,
        pattern: PatternDefinition,
    ) -> Optional[str]:
        """Check if exit conditions are met."""
        direction = position["direction"]
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        entry_bar_idx = position.get("entry_bar_idx", 0)

        # Check stop loss
        if stop_loss:
            if direction == "long" and current_price <= stop_loss:
                return "stop_loss"
            elif direction == "short" and current_price >= stop_loss:
                return "stop_loss"

        # Check take profit
        if take_profit:
            if direction == "long" and current_price >= take_profit:
                return "take_profit"
            elif direction == "short" and current_price <= take_profit:
                return "take_profit"

        # Check time-based exit
        time_exit = pattern.exit.get("time_exit", {})
        if time_exit.get("enabled", False):
            max_bars = time_exit.get("bars", 60)
            if current_bar_idx - entry_bar_idx >= max_bars:
                return "time_exit"

        return None

    def _calculate_unrealized_pnl(
        self,
        position: dict[str, Any],
        current_price: float,
    ) -> float:
        """Calculate unrealized P&L for open position."""
        direction = position["direction"]
        entry_price = position["entry_price"]
        size = position["size"]

        if direction == "long":
            return (current_price - entry_price) * size
        else:
            return (entry_price - current_price) * size

    def _create_empty_result(self, pattern: dict) -> BacktestResult:
        """Create an empty result for when there's no data."""
        return BacktestResult(
            pattern_id=pattern.get("id", ""),
            pattern_name=pattern.get("name", ""),
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            passed=False,
            failure_reasons=["No data available for backtest"],
        )

    def run_multiple(
        self,
        patterns: list[dict[str, Any]],
        data: pd.DataFrame,
    ) -> list[BacktestResult]:
        """Run backtests for multiple patterns."""
        results = []
        for pattern in patterns:
            try:
                result = self.run(pattern, data)
                results.append(result)
            except Exception as e:
                logger.error(f"Backtest failed for {pattern.get('name', 'unknown')}: {e}")

        return results
