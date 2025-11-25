"""Tests for pattern matching and detection."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.strategy.pattern import PatternMatcher, PatternDefinition


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1min")
    np.random.seed(42)

    # Create trending price data
    base_price = 100
    trend = np.cumsum(np.random.randn(200) * 0.1)
    prices = base_price + trend

    df = pd.DataFrame({
        "open": prices + np.random.randn(200) * 0.05,
        "high": prices + abs(np.random.randn(200) * 0.1),
        "low": prices - abs(np.random.randn(200) * 0.1),
        "close": prices,
        "volume": np.random.randint(1000, 10000, 200),
    }, index=dates)

    return df


@pytest.fixture
def sample_pattern():
    """Create a sample pattern definition."""
    return {
        "id": "test-pattern-1",
        "name": "test_momentum",
        "pattern_type": "bullish",
        "detection": {
            "conditions": [
                {
                    "indicator": "close",
                    "operator": ">",
                    "value": "sma_20",
                    "lookback": 0
                },
                {
                    "indicator": "rsi_14",
                    "operator": ">",
                    "value": 50,
                    "lookback": 0
                }
            ],
            "min_conditions_met": 2
        },
        "entry": {
            "signal": "All conditions met",
            "order_type": "market"
        },
        "exit": {
            "stop_loss": {"type": "atr", "value": 2.0},
            "take_profit": {"type": "atr", "value": 3.0},
            "time_exit": {"enabled": True, "bars": 60}
        }
    }


class TestPatternMatcher:
    """Tests for PatternMatcher class."""

    def test_add_pattern(self, sample_pattern):
        """Test adding a pattern."""
        matcher = PatternMatcher()
        pattern_def = matcher.load_pattern_from_dict(sample_pattern)

        assert pattern_def.id == "test-pattern-1"
        assert pattern_def.name == "test_momentum"
        assert len(pattern_def.conditions) == 2

    def test_find_matches(self, sample_data, sample_pattern):
        """Test finding pattern matches."""
        matcher = PatternMatcher()
        matcher.load_pattern_from_dict(sample_pattern)

        matches = matcher.find_matches(sample_data, min_confidence=0.5)

        # Should find some matches
        assert isinstance(matches, list)
        for match in matches:
            assert match.confidence >= 0.5
            assert match.pattern_id == "test-pattern-1"

    def test_check_current(self, sample_data, sample_pattern):
        """Test checking current bar for matches."""
        matcher = PatternMatcher()
        matcher.load_pattern_from_dict(sample_pattern)

        matches = matcher.check_current(sample_data, min_confidence=0.5)

        assert isinstance(matches, list)

    def test_indicator_calculation(self, sample_data):
        """Test indicator calculations."""
        matcher = PatternMatcher()

        # Test SMA
        sma = matcher._calc_sma(sample_data, 20)
        assert len(sma) == len(sample_data)
        assert not np.isnan(sma[-1])

        # Test RSI
        rsi = matcher._calc_rsi(sample_data, 14)
        assert len(rsi) == len(sample_data)
        # RSI should be between 0 and 100
        valid_rsi = rsi[~np.isnan(rsi)]
        assert all(0 <= r <= 100 for r in valid_rsi)

        # Test ATR
        atr = matcher._calc_atr(sample_data, 14)
        assert len(atr) == len(sample_data)
        assert all(a >= 0 for a in atr[~np.isnan(atr)])


class TestPatternDefinition:
    """Tests for PatternDefinition class."""

    def test_from_dict(self, sample_pattern):
        """Test creating PatternDefinition from dict."""
        pattern = PatternDefinition.from_dict(sample_pattern)

        assert pattern.id == "test-pattern-1"
        assert pattern.name == "test_momentum"
        assert pattern.pattern_type == "bullish"
        assert pattern.min_conditions_met == 2
        assert len(pattern.conditions) == 2

    def test_from_dict_minimal(self):
        """Test creating PatternDefinition with minimal data."""
        minimal = {
            "name": "minimal_pattern",
            "detection": {
                "conditions": [],
                "min_conditions_met": 0
            },
            "entry": {"signal": "test"},
            "exit": {}
        }

        pattern = PatternDefinition.from_dict(minimal)

        assert pattern.name == "minimal_pattern"
        assert pattern.pattern_type == "neutral"  # default
