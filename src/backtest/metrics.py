"""Performance metrics calculation."""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy import stats

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0

    # Statistical tests
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False

    # Additional metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration: float = 0.0
    expectancy: float = 0.0

    # Trade details
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "avg_trade_return": self.avg_trade_return,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_trade_duration": self.avg_trade_duration,
            "expectancy": self.expectancy,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
        }


class MetricsCalculator:
    """
    Calculates comprehensive performance metrics from trade data.

    Includes statistical tests for significance.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        trading_days_per_year: int = 252,
        significance_level: float = 0.05,
    ):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            trading_days_per_year: Number of trading days per year
            significance_level: P-value threshold for significance
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        self.significance_level = significance_level

    def calculate(
        self,
        trades: list[dict[str, Any]],
        equity_curve: Optional[np.ndarray] = None,
        initial_capital: float = 100000,
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            trades: List of trade dictionaries
            equity_curve: Optional equity curve array
            initial_capital: Starting capital

        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()

        if not trades:
            return metrics

        # Extract P&L from trades
        pnls = np.array([t.get("pnl", 0) for t in trades])
        returns = np.array([t.get("pnl_percent", 0) for t in trades])

        # Basic counts
        metrics.total_trades = len(trades)
        metrics.winning_trades = int(np.sum(pnls > 0))
        metrics.losing_trades = int(np.sum(pnls < 0))
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0

        # P&L metrics
        winning_pnls = pnls[pnls > 0]
        losing_pnls = pnls[pnls < 0]

        metrics.gross_profit = float(np.sum(winning_pnls)) if len(winning_pnls) > 0 else 0
        metrics.gross_loss = float(np.sum(losing_pnls)) if len(losing_pnls) > 0 else 0
        metrics.avg_win = float(np.mean(winning_pnls)) if len(winning_pnls) > 0 else 0
        metrics.avg_loss = float(np.mean(losing_pnls)) if len(losing_pnls) > 0 else 0
        metrics.largest_win = float(np.max(pnls)) if len(pnls) > 0 else 0
        metrics.largest_loss = float(np.min(pnls)) if len(pnls) > 0 else 0

        # Return metrics
        metrics.total_return = float(np.sum(pnls)) / initial_capital if initial_capital > 0 else 0
        metrics.avg_trade_return = float(np.mean(returns)) if len(returns) > 0 else 0

        # Profit factor
        if metrics.gross_loss != 0:
            metrics.profit_factor = abs(metrics.gross_profit / metrics.gross_loss)
        else:
            metrics.profit_factor = float('inf') if metrics.gross_profit > 0 else 0

        # Expectancy
        metrics.expectancy = (
            metrics.win_rate * metrics.avg_win +
            (1 - metrics.win_rate) * metrics.avg_loss
        )

        # Calculate from equity curve or trades
        if equity_curve is not None and len(equity_curve) > 1:
            daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        else:
            daily_returns = returns

        # Sharpe Ratio
        metrics.sharpe_ratio = self._calculate_sharpe(daily_returns)

        # Sortino Ratio
        metrics.sortino_ratio = self._calculate_sortino(daily_returns)

        # Drawdown metrics
        if equity_curve is not None:
            metrics.max_drawdown, metrics.avg_drawdown = self._calculate_drawdowns(equity_curve)
        else:
            cumulative = np.cumsum(pnls) + initial_capital
            metrics.max_drawdown, metrics.avg_drawdown = self._calculate_drawdowns(cumulative)

        # Calmar Ratio
        if metrics.max_drawdown > 0:
            # Annualize return
            if len(trades) > 0:
                days = self._estimate_trading_days(trades)
                annual_mult = self.trading_days / max(days, 1)
                metrics.annualized_return = metrics.total_return * annual_mult
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

        # Consecutive wins/losses
        metrics.max_consecutive_wins, metrics.max_consecutive_losses = self._calculate_streaks(pnls)

        # Average trade duration
        durations = [t.get("duration_minutes", 0) for t in trades if "duration_minutes" in t]
        metrics.avg_trade_duration = float(np.mean(durations)) if durations else 0

        # Statistical significance test
        metrics.t_statistic, metrics.p_value = self._ttest_returns(returns)
        metrics.is_significant = metrics.p_value < self.significance_level

        return metrics

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.trading_days)
        std = np.std(excess_returns, ddof=1)

        if std == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / std * np.sqrt(self.trading_days)
        return float(sharpe) if not np.isnan(sharpe) else 0.0

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino Ratio (penalizes only downside volatility)."""
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / self.trading_days)
        downside_returns = returns[returns < 0]

        if len(downside_returns) < 2:
            return 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        sortino = np.mean(excess_returns) / downside_std * np.sqrt(self.trading_days)
        return float(sortino) if not np.isnan(sortino) else 0.0

    def _calculate_drawdowns(self, equity_curve: np.ndarray) -> tuple[float, float]:
        """Calculate maximum and average drawdown."""
        if len(equity_curve) < 2:
            return 0.0, 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / peak

        max_dd = float(np.max(drawdowns))
        avg_dd = float(np.mean(drawdowns[drawdowns > 0])) if np.any(drawdowns > 0) else 0.0

        return max_dd, avg_dd

    def _calculate_streaks(self, pnls: np.ndarray) -> tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if len(pnls) == 0:
            return 0, 0

        wins = pnls > 0
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for win in wins:
            if win:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _estimate_trading_days(self, trades: list[dict]) -> int:
        """Estimate number of trading days from trades."""
        if not trades:
            return 1

        # Try to get from entry/exit times
        timestamps = []
        for t in trades:
            if "entry_time" in t:
                timestamps.append(t["entry_time"])
            if "exit_time" in t:
                timestamps.append(t["exit_time"])

        if len(timestamps) >= 2:
            # Get unique dates
            dates = set()
            for ts in timestamps:
                if hasattr(ts, "date"):
                    dates.add(ts.date())
            return max(len(dates), 1)

        return len(trades)  # Fallback

    def _ttest_returns(self, returns: np.ndarray) -> tuple[float, float]:
        """
        Perform one-sample t-test on returns.

        Tests if mean return is significantly different from zero.

        Returns:
            Tuple of (t-statistic, p-value)
        """
        if len(returns) < 3:
            return 0.0, 1.0

        # Remove NaN and inf
        clean_returns = returns[np.isfinite(returns)]
        if len(clean_returns) < 3:
            return 0.0, 1.0

        try:
            t_stat, p_value = stats.ttest_1samp(clean_returns, 0)
            return float(t_stat), float(p_value)
        except Exception:
            return 0.0, 1.0

    def validate_against_criteria(
        self,
        metrics: PerformanceMetrics,
        criteria: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Validate metrics against backtest criteria.

        Args:
            metrics: Performance metrics
            criteria: Criteria dictionary

        Returns:
            Tuple of (passed, list of failure reasons)
        """
        failures = []

        # Minimum trades
        min_trades = criteria.get("min_trades", 30)
        if metrics.total_trades < min_trades:
            failures.append(f"Insufficient trades: {metrics.total_trades} < {min_trades}")

        # Win rate
        min_win_rate = criteria.get("min_win_rate", 0.4)
        if metrics.win_rate < min_win_rate:
            failures.append(f"Win rate too low: {metrics.win_rate:.2%} < {min_win_rate:.2%}")

        # Profit factor
        min_pf = criteria.get("min_profit_factor", 1.5)
        if metrics.profit_factor < min_pf:
            failures.append(f"Profit factor too low: {metrics.profit_factor:.2f} < {min_pf}")

        # Sharpe ratio
        min_sharpe = criteria.get("min_sharpe_ratio", 1.0)
        if metrics.sharpe_ratio < min_sharpe:
            failures.append(f"Sharpe ratio too low: {metrics.sharpe_ratio:.2f} < {min_sharpe}")

        # Max drawdown
        max_dd = criteria.get("max_drawdown", 0.15)
        if metrics.max_drawdown > max_dd:
            failures.append(f"Max drawdown too high: {metrics.max_drawdown:.2%} > {max_dd:.2%}")

        # Statistical significance
        if not metrics.is_significant:
            failures.append(f"Results not statistically significant (p={metrics.p_value:.4f})")

        passed = len(failures) == 0
        return passed, failures
