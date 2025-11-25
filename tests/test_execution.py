"""Tests for execution components."""

import pytest
from datetime import datetime

from src.core.config import Config
from src.execution.order_manager import (
    OrderManager,
    Order,
    OrderStatus,
    OrderType,
    OrderSide,
)
from src.execution.position_tracker import PositionTracker, Position


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.trading.mode = "paper"
    return config


class TestOrderManager:
    """Tests for OrderManager."""

    def test_create_market_order(self, config):
        """Test creating a market order."""
        manager = OrderManager(config)
        order = manager.create_market_order(
            symbol="QQQ",
            side="buy",
            quantity=100,
        )

        assert order.symbol == "QQQ"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING

    def test_create_limit_order(self, config):
        """Test creating a limit order."""
        manager = OrderManager(config)
        order = manager.create_limit_order(
            symbol="SPY",
            side="sell",
            quantity=50,
            limit_price=450.00,
        )

        assert order.symbol == "SPY"
        assert order.side == OrderSide.SELL
        assert order.quantity == 50
        assert order.limit_price == 450.00
        assert order.order_type == OrderType.LIMIT

    def test_create_stop_order(self, config):
        """Test creating a stop order."""
        manager = OrderManager(config)
        order = manager.create_stop_order(
            symbol="AAPL",
            side="sell",
            quantity=25,
            stop_price=175.00,
        )

        assert order.symbol == "AAPL"
        assert order.stop_price == 175.00
        assert order.order_type == OrderType.STOP

    def test_create_bracket_orders(self, config):
        """Test creating bracket orders."""
        manager = OrderManager(config)
        entry, stop, tp = manager.create_bracket_orders(
            symbol="QQQ",
            side="buy",
            quantity=100,
            entry_price=400.00,
            stop_loss=395.00,
            take_profit=410.00,
        )

        assert entry.side == OrderSide.BUY
        assert stop.side == OrderSide.SELL
        assert tp.side == OrderSide.SELL
        assert stop.parent_order_id == entry.id
        assert tp.parent_order_id == entry.id

    def test_submit_order_paper(self, config):
        """Test submitting order in paper mode."""
        manager = OrderManager(config)
        order = manager.create_market_order("QQQ", "buy", 100)

        result = manager.submit_order(order)

        assert result is True
        assert order.status == OrderStatus.SUBMITTED

    def test_cancel_order(self, config):
        """Test canceling an order."""
        manager = OrderManager(config)
        order = manager.create_limit_order("QQQ", "buy", 100, 400.00)
        manager.submit_order(order)

        result = manager.cancel_order(order.id)

        assert result is True
        assert order.status == OrderStatus.CANCELLED

    def test_get_pending_orders(self, config):
        """Test getting pending orders."""
        manager = OrderManager(config)

        # Create and submit multiple orders
        order1 = manager.create_limit_order("QQQ", "buy", 100, 400.00)
        order2 = manager.create_limit_order("SPY", "buy", 50, 450.00)
        manager.submit_order(order1)
        manager.submit_order(order2)

        pending = manager.get_pending_orders()

        assert len(pending) == 2


class TestPositionTracker:
    """Tests for PositionTracker."""

    def test_open_position(self, config):
        """Test opening a position."""
        tracker = PositionTracker(config)
        position = tracker.open_position(
            symbol="QQQ",
            direction="long",
            quantity=100,
            price=400.00,
            stop_loss=395.00,
            take_profit=410.00,
        )

        assert position.symbol == "QQQ"
        assert position.direction == "long"
        assert position.quantity == 100
        assert position.entry_price == 400.00

    def test_close_position(self, config):
        """Test closing a position."""
        tracker = PositionTracker(config)
        tracker.open_position(
            symbol="QQQ",
            direction="long",
            quantity=100,
            price=400.00,
        )

        closed = tracker.close_position(
            symbol="QQQ",
            price=405.00,
            reason="take_profit",
        )

        assert closed is not None
        assert closed.pnl == 500.00  # (405 - 400) * 100
        assert tracker.get_position("QQQ") is None

    def test_update_prices(self, config):
        """Test updating position prices."""
        tracker = PositionTracker(config)
        tracker.open_position(
            symbol="QQQ",
            direction="long",
            quantity=100,
            price=400.00,
        )

        tracker.update_prices({"QQQ": 405.00})

        position = tracker.get_position("QQQ")
        assert position.current_price == 405.00
        assert position.unrealized_pnl == 500.00

    def test_has_position(self, config):
        """Test checking for position."""
        tracker = PositionTracker(config)

        assert tracker.has_position("QQQ") is False

        tracker.open_position("QQQ", "long", 100, 400.00)

        assert tracker.has_position("QQQ") is True

    def test_total_unrealized_pnl(self, config):
        """Test calculating total unrealized P&L."""
        tracker = PositionTracker(config)

        tracker.open_position("QQQ", "long", 100, 400.00)
        tracker.open_position("SPY", "long", 50, 450.00)

        tracker.update_prices({"QQQ": 405.00, "SPY": 455.00})

        total_pnl = tracker.get_total_unrealized_pnl()

        expected = (405 - 400) * 100 + (455 - 450) * 50
        assert total_pnl == expected


class TestPosition:
    """Tests for Position class."""

    def test_update_price_long(self):
        """Test price update for long position."""
        position = Position(
            symbol="QQQ",
            direction="long",
            quantity=100,
            entry_price=400.00,
            entry_time=datetime.now(),
        )

        position.update_price(410.00)

        assert position.unrealized_pnl == 1000.00
        assert position.unrealized_pnl_pct == 0.025  # 2.5%

    def test_update_price_short(self):
        """Test price update for short position."""
        position = Position(
            symbol="QQQ",
            direction="short",
            quantity=100,
            entry_price=400.00,
            entry_time=datetime.now(),
        )

        position.update_price(390.00)

        assert position.unrealized_pnl == 1000.00  # Profit on short

    def test_to_dict(self):
        """Test converting position to dictionary."""
        position = Position(
            symbol="QQQ",
            direction="long",
            quantity=100,
            entry_price=400.00,
            entry_time=datetime.now(),
            stop_loss=395.00,
            take_profit=410.00,
        )

        d = position.to_dict()

        assert d["symbol"] == "QQQ"
        assert d["direction"] == "long"
        assert d["quantity"] == 100
        assert d["stop_loss"] == 395.00
