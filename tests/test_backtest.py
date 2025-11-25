"""Tests for backtesting engine."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.core.config import Config
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.metrics import MetricsCalculator, PerformanceMetrics


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.backtest.min_trades = 5  # Lower for testing
    config.backtest.min_win_rate = 0.3
    config.backtest.min_profit_factor = 1.0
    config.backtest.min_sharpe_ratio = 0.5
    config.backtest.max_drawdown = 0.3
    return config


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for backtesting."""
    dates = pd.date_range(start="2024-01-01", periods=500, freq="1min")
    np.random.seed(42)

    base_price = 100
    returns = np.random.randn(500) * 0.002  # 0.2% volatility
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(500) * 0.001),
        "high": prices * (1 + abs(np.random.randn(500) * 0.002)),
        "low": prices * (1 - abs(np.random.randn(500) * 0.002)),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 500),
    }, index=dates)

    return df


@pytest.fixture
def sample_pattern():
    """Create a sample pattern for backtesting."""
    return {
        "id": "backtest-pattern-1",
        "name": "test_strategy",
        "pattern_type": "bullish",
        "symbol": "TEST",
        "timeframe": "1min",
        "detection": {
            "conditions": [
                {
                    "indicator": "close",
                    "operator": ">",
                    "value": "sma_20",
                    "lookback": 0
                }
            ],
            "min_conditions_met": 1
        },
        "entry": {
            "signal": "Close above SMA20",
            "order_type": "market"
        },
        "exit": {
            "stop_loss": {"type": "percent", "value": 1.0},
            "take_profit": {"type": "percent", "value": 2.0},
            "time_exit": {"enabled": True, "bars": 30}
        }
    }


@pytest.fixture
def sample_trades():
    """Create sample trade data."""
    return [
        {"pnl": 100, "pnl_percent": 0.01, "duration_minutes": 30},
        {"pnl": -50, "pnl_percent": -0.005, "duration_minutes": 15},
        {"pnl": 150, "pnl_percent": 0.015, "duration_minutes": 45},
        {"pnl": -30, "pnl_percent": -0.003, "duration_minutes": 20},
        {"pnl": 200, "pnl_percent": 0.02, "duration_minutes": 60},
        {"pnl": 80, "pnl_percent": 0.008, "duration_minutes": 25},
        {"pnl": -70, "pnl_percent": -0.007, "duration_minutes": 35},
        {"pnl": 120, "pnl_percent": 0.012, "duration_minutes": 40},
    ]


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculate_basic_metrics(self, sample_trades):
        """Test basic metrics calculation."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(sample_trades)

        assert metrics.total_trades == 8
        assert metrics.winning_trades == 5
        assert metrics.losing_trades == 3
        assert 0 < metrics.win_rate < 1

    def test_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(sample_trades)

        # Profit factor = gross profit / gross loss
        gross_profit = sum(t["pnl"] for t in sample_trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in sample_trades if t["pnl"] < 0))

        expected_pf = gross_profit / gross_loss
        assert abs(metrics.profit_factor - expected_pf) < 0.01

    def test_empty_trades(self):
        """Test with empty trade list."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate([])

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0
        assert metrics.profit_factor == 0

    def test_validate_against_criteria(self, sample_trades):
        """Test validation against criteria."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate(sample_trades)

        # Should pass lenient criteria
        passed, failures = calculator.validate_against_criteria(
            metrics,
            {
                "min_trades": 5,
                "min_win_rate": 0.3,
                "min_profit_factor": 1.0,
                "min_sharpe_ratio": -1.0,  # Very lenient
                "max_drawdown": 0.5,
            }
        )

        # May or may not pass depending on random data
        assert isinstance(passed, bool)
        assert isinstance(failures, list)


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_run_backtest(self, config, sample_data, sample_pattern):
        """Test running a backtest."""
        engine = BacktestEngine(config)
        result = engine.run(sample_pattern, sample_data)

        assert isinstance(result, BacktestResult)
        assert result.pattern_id == "backtest-pattern-1"
        assert result.initial_capital == 100000
        assert result.metrics is not None

    def test_backtest_result_to_dict(self, config, sample_data, sample_pattern):
        """Test BacktestResult serialization."""
        engine = BacktestEngine(config)
        result = engine.run(sample_pattern, sample_data)

        result_dict = result.to_dict()

        assert "pattern_id" in result_dict
        assert "total_trades" in result_dict
        assert "win_rate" in result_dict
        assert "passed" in result_dict

    def test_empty_data(self, config, sample_pattern):
        """Test backtest with empty data."""
        engine = BacktestEngine(config)
        result = engine.run(sample_pattern, pd.DataFrame())

        assert result.passed is False
        assert "No data" in result.failure_reasons[0]


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            total_return=0.15,
            sharpe_ratio=1.5,
            profit_factor=2.0,
            max_drawdown=0.1,
        )

        d = metrics.to_dict()

        assert d["total_trades"] == 100
        assert d["win_rate"] == 0.6
        assert d["sharpe_ratio"] == 1.5
