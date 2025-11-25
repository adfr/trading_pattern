"""Walk-forward optimization for pattern validation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.logger import get_logger
from .engine import BacktestEngine, BacktestResult
from .metrics import MetricsCalculator, PerformanceMetrics

logger = get_logger(__name__)


@dataclass
class WalkForwardWindow:
    """A single walk-forward optimization window."""
    window_index: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    in_sample_result: Optional[BacktestResult] = None
    out_of_sample_result: Optional[BacktestResult] = None


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""
    pattern_id: str
    pattern_name: str
    windows: list[WalkForwardWindow]

    # Aggregated metrics
    in_sample_metrics: Optional[PerformanceMetrics] = None
    out_of_sample_metrics: Optional[PerformanceMetrics] = None

    # Validation
    passed: bool = False
    failure_reasons: list[str] = None

    # Robustness metrics
    oos_is_ratio: float = 0.0  # Out-of-sample / In-sample performance ratio
    consistency_score: float = 0.0  # How consistent across windows

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "num_windows": len(self.windows),
            "in_sample_sharpe": self.in_sample_metrics.sharpe_ratio if self.in_sample_metrics else 0,
            "out_of_sample_sharpe": self.out_of_sample_metrics.sharpe_ratio if self.out_of_sample_metrics else 0,
            "oos_is_ratio": self.oos_is_ratio,
            "consistency_score": self.consistency_score,
            "passed": self.passed,
            "failure_reasons": self.failure_reasons or [],
        }


class WalkForwardOptimizer:
    """
    Walk-forward optimization for robust pattern validation.

    Tests pattern performance on out-of-sample data to detect overfitting.
    """

    def __init__(
        self,
        config: Config,
        backtest_engine: Optional[BacktestEngine] = None,
    ):
        """
        Initialize optimizer.

        Args:
            config: Configuration object
            backtest_engine: Backtest engine (optional, will create if not provided)
        """
        self.config = config
        self.backtest_engine = backtest_engine or BacktestEngine(config)
        self.metrics_calculator = MetricsCalculator()

        # Walk-forward parameters
        self.num_windows = config.backtest.walk_forward_windows
        self.oos_ratio = config.backtest.out_of_sample_ratio

    def run(
        self,
        pattern: dict[str, Any],
        data: pd.DataFrame,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            pattern: Pattern dictionary
            data: Full OHLCV DataFrame

        Returns:
            WalkForwardResult object
        """
        if data.empty:
            return self._create_empty_result(pattern, ["No data available"])

        # Create windows
        windows = self._create_windows(data)

        if len(windows) < 2:
            return self._create_empty_result(pattern, ["Insufficient data for walk-forward"])

        # Run backtests for each window
        all_is_trades = []
        all_oos_trades = []

        for window in windows:
            # In-sample backtest
            is_data = data[
                (data.index >= window.in_sample_start) &
                (data.index <= window.in_sample_end)
            ]

            if len(is_data) > 50:
                window.in_sample_result = self.backtest_engine.run(
                    pattern=pattern,
                    data=is_data,
                )
                all_is_trades.extend(window.in_sample_result.trades)

            # Out-of-sample backtest
            oos_data = data[
                (data.index >= window.out_of_sample_start) &
                (data.index <= window.out_of_sample_end)
            ]

            if len(oos_data) > 50:
                window.out_of_sample_result = self.backtest_engine.run(
                    pattern=pattern,
                    data=oos_data,
                )
                all_oos_trades.extend(window.out_of_sample_result.trades)

        # Calculate aggregated metrics
        result = WalkForwardResult(
            pattern_id=pattern.get("id", ""),
            pattern_name=pattern.get("name", ""),
            windows=windows,
        )

        if all_is_trades:
            result.in_sample_metrics = self.metrics_calculator.calculate(all_is_trades)

        if all_oos_trades:
            result.out_of_sample_metrics = self.metrics_calculator.calculate(all_oos_trades)

        # Calculate robustness metrics
        result.oos_is_ratio = self._calculate_oos_ratio(result)
        result.consistency_score = self._calculate_consistency(windows)

        # Validate
        result.passed, result.failure_reasons = self._validate(result)

        logger.info(
            f"Walk-forward complete: {result.pattern_name} - "
            f"{'PASSED' if result.passed else 'FAILED'} "
            f"(OOS/IS ratio: {result.oos_is_ratio:.2f}, consistency: {result.consistency_score:.2f})"
        )

        return result

    def _create_windows(self, data: pd.DataFrame) -> list[WalkForwardWindow]:
        """Create walk-forward windows."""
        if data.empty:
            return []

        total_bars = len(data)
        window_size = total_bars // self.num_windows

        if window_size < 100:  # Minimum window size
            # Reduce number of windows
            actual_windows = max(2, total_bars // 100)
            window_size = total_bars // actual_windows
        else:
            actual_windows = self.num_windows

        windows = []
        oos_size = int(window_size * self.oos_ratio)
        is_size = window_size - oos_size

        for i in range(actual_windows - 1):  # Last window is only OOS
            start_idx = i * window_size
            is_end_idx = start_idx + is_size
            oos_end_idx = start_idx + window_size

            # Ensure we don't exceed data bounds
            if oos_end_idx >= total_bars:
                break

            window = WalkForwardWindow(
                window_index=i,
                in_sample_start=data.index[start_idx],
                in_sample_end=data.index[is_end_idx],
                out_of_sample_start=data.index[is_end_idx + 1],
                out_of_sample_end=data.index[oos_end_idx],
            )
            windows.append(window)

        return windows

    def _calculate_oos_ratio(self, result: WalkForwardResult) -> float:
        """Calculate out-of-sample to in-sample performance ratio."""
        if not result.in_sample_metrics or not result.out_of_sample_metrics:
            return 0.0

        is_sharpe = result.in_sample_metrics.sharpe_ratio
        oos_sharpe = result.out_of_sample_metrics.sharpe_ratio

        if is_sharpe <= 0:
            return 0.0 if oos_sharpe <= 0 else 1.0

        return oos_sharpe / is_sharpe

    def _calculate_consistency(self, windows: list[WalkForwardWindow]) -> float:
        """
        Calculate consistency score across windows.

        Returns value between 0-1 where 1 is perfect consistency.
        """
        oos_returns = []

        for window in windows:
            if window.out_of_sample_result and window.out_of_sample_result.metrics:
                oos_returns.append(window.out_of_sample_result.metrics.total_return)

        if len(oos_returns) < 2:
            return 0.0

        # Calculate percentage of profitable windows
        profitable_windows = sum(1 for r in oos_returns if r > 0)
        profitability_score = profitable_windows / len(oos_returns)

        # Calculate coefficient of variation (lower is more consistent)
        mean_return = np.mean(oos_returns)
        std_return = np.std(oos_returns)

        if mean_return <= 0:
            cv_score = 0.0
        else:
            cv = std_return / mean_return
            cv_score = max(0, 1 - cv)  # Convert to 0-1 score

        # Combined score
        return (profitability_score * 0.6 + cv_score * 0.4)

    def _validate(self, result: WalkForwardResult) -> tuple[bool, list[str]]:
        """Validate walk-forward results."""
        failures = []

        # Check OOS performance
        if result.out_of_sample_metrics:
            oos = result.out_of_sample_metrics

            # Must be profitable out-of-sample
            if oos.total_return <= 0:
                failures.append("Out-of-sample returns are negative")

            # OOS Sharpe should be positive
            if oos.sharpe_ratio <= 0:
                failures.append("Out-of-sample Sharpe ratio is non-positive")

            # Check minimum trades
            if oos.total_trades < 10:
                failures.append(f"Insufficient out-of-sample trades: {oos.total_trades}")
        else:
            failures.append("No out-of-sample results available")

        # Check OOS/IS ratio - should be reasonable
        if result.oos_is_ratio < 0.5:
            failures.append(f"Out-of-sample degradation too high (OOS/IS ratio: {result.oos_is_ratio:.2f})")

        # Check consistency
        if result.consistency_score < 0.4:
            failures.append(f"Results inconsistent across windows (score: {result.consistency_score:.2f})")

        passed = len(failures) == 0
        return passed, failures

    def _create_empty_result(
        self,
        pattern: dict[str, Any],
        reasons: list[str],
    ) -> WalkForwardResult:
        """Create an empty result for failure cases."""
        return WalkForwardResult(
            pattern_id=pattern.get("id", ""),
            pattern_name=pattern.get("name", ""),
            windows=[],
            passed=False,
            failure_reasons=reasons,
        )


class MonteCarloSimulator:
    """
    Monte Carlo simulation for robustness testing.

    Shuffles trades to test if results could be due to luck.
    """

    def __init__(self, num_simulations: int = 1000):
        """
        Initialize simulator.

        Args:
            num_simulations: Number of Monte Carlo simulations
        """
        self.num_simulations = num_simulations
        self.metrics_calculator = MetricsCalculator()

    def run(
        self,
        trades: list[dict[str, Any]],
        initial_capital: float = 100000,
    ) -> dict[str, Any]:
        """
        Run Monte Carlo simulation.

        Args:
            trades: List of trade dictionaries
            initial_capital: Starting capital

        Returns:
            Simulation results dictionary
        """
        if len(trades) < 10:
            return {
                "passed": False,
                "reason": "Insufficient trades for Monte Carlo simulation",
            }

        # Extract P&Ls
        pnls = np.array([t.get("pnl", 0) for t in trades])
        actual_return = np.sum(pnls) / initial_capital

        # Run simulations
        simulated_returns = []

        for _ in range(self.num_simulations):
            # Shuffle trades
            shuffled_pnls = np.random.permutation(pnls)

            # Calculate cumulative return
            sim_return = np.sum(shuffled_pnls) / initial_capital
            simulated_returns.append(sim_return)

        simulated_returns = np.array(simulated_returns)

        # Calculate percentile of actual return
        percentile = np.sum(simulated_returns <= actual_return) / self.num_simulations

        # Calculate confidence interval
        ci_low = np.percentile(simulated_returns, 5)
        ci_high = np.percentile(simulated_returns, 95)

        # Determine if results are significant
        is_significant = percentile >= 0.95

        return {
            "actual_return": actual_return,
            "mean_simulated_return": np.mean(simulated_returns),
            "std_simulated_return": np.std(simulated_returns),
            "percentile": percentile,
            "confidence_interval": (ci_low, ci_high),
            "is_significant": is_significant,
            "passed": is_significant,
        }
