"""Position tracking for portfolio management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..core.config import Config
from ..core.database import Database
from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.logger import get_logger
from .order_manager import Order, OrderSide

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str  # "long" or "short"
    quantity: int
    entry_price: float
    entry_time: datetime

    # Risk levels
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Metadata
    pattern_id: Optional[str] = None
    entry_order_id: Optional[str] = None

    def update_price(self, price: float) -> None:
        """Update current price and P&L."""
        self.current_price = price

        if self.direction == "long":
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - price) * self.quantity

        cost_basis = self.entry_price * self.quantity
        self.unrealized_pnl_pct = self.unrealized_pnl / cost_basis if cost_basis > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "pattern_id": self.pattern_id,
        }


@dataclass
class ClosedTrade:
    """Represents a closed trade."""
    symbol: str
    direction: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    exit_reason: str
    pattern_id: Optional[str] = None


class PositionTracker:
    """
    Tracks open positions and closed trades.

    Manages position lifecycle from entry to exit.
    """

    def __init__(
        self,
        config: Config,
        database: Optional[Database] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize position tracker.

        Args:
            config: Configuration object
            database: Database for trade storage
            event_bus: Event bus
        """
        self.config = config
        self.database = database
        self.event_bus = event_bus or get_event_bus()

        # Position tracking
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[ClosedTrade] = []

        # Subscribe to order events
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)

    def _on_order_filled(self, event: Event) -> None:
        """Handle order filled events."""
        order_data = event.data

        symbol = order_data.get("symbol")
        side = order_data.get("side")
        quantity = order_data.get("filled_quantity", order_data.get("quantity", 0))
        fill_price = order_data.get("fill_price", 0)
        pattern_id = order_data.get("pattern_id")
        order_id = order_data.get("id")

        if not symbol or not side or quantity <= 0:
            return

        # Determine if this opens or closes a position
        current_pos = self._positions.get(symbol)

        if current_pos is None:
            # Opening new position
            direction = "long" if side == "BUY" else "short"
            self._open_position(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                price=fill_price,
                pattern_id=pattern_id,
                order_id=order_id,
            )
        else:
            # Check if closing or adding
            is_closing = (
                (current_pos.direction == "long" and side == "SELL") or
                (current_pos.direction == "short" and side == "BUY")
            )

            if is_closing:
                self._close_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=fill_price,
                    reason="signal",
                )
            else:
                # Adding to position
                self._add_to_position(
                    symbol=symbol,
                    quantity=quantity,
                    price=fill_price,
                )

    def _open_position(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        price: float,
        pattern_id: Optional[str] = None,
        order_id: Optional[str] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Position:
        """Open a new position."""
        position = Position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_id=pattern_id,
            entry_order_id=order_id,
            current_price=price,
        )

        self._positions[symbol] = position

        self.event_bus.publish(Event(
            type=EventType.POSITION_OPENED,
            data=position.to_dict(),
            source="PositionTracker"
        ))

        logger.info(f"Opened position: {direction} {quantity} {symbol} @ {price}")
        return position

    def _add_to_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
    ) -> None:
        """Add to existing position."""
        position = self._positions.get(symbol)
        if not position:
            return

        # Calculate new average price
        total_cost = (position.entry_price * position.quantity) + (price * quantity)
        total_quantity = position.quantity + quantity
        position.entry_price = total_cost / total_quantity
        position.quantity = total_quantity

        logger.info(f"Added to position: {quantity} {symbol} @ {price}")

    def _close_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        reason: str,
    ) -> Optional[ClosedTrade]:
        """Close a position (fully or partially)."""
        position = self._positions.get(symbol)
        if not position:
            return None

        # Calculate P&L
        if position.direction == "long":
            pnl = (price - position.entry_price) * quantity
        else:
            pnl = (position.entry_price - price) * quantity

        pnl_pct = pnl / (position.entry_price * quantity)

        # Create closed trade record
        closed = ClosedTrade(
            symbol=symbol,
            direction=position.direction,
            quantity=quantity,
            entry_price=position.entry_price,
            exit_price=price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            pattern_id=position.pattern_id,
        )

        self._closed_trades.append(closed)

        # Update or remove position
        if quantity >= position.quantity:
            # Fully closed
            del self._positions[symbol]
            logger.info(f"Closed position: {symbol} @ {price}, P&L: ${pnl:.2f}")
        else:
            # Partially closed
            position.quantity -= quantity
            logger.info(f"Partially closed {quantity} {symbol} @ {price}, P&L: ${pnl:.2f}")

        # Store in database
        if self.database:
            self.database.save_trade({
                "symbol": closed.symbol,
                "direction": closed.direction,
                "entry_price": closed.entry_price,
                "exit_price": closed.exit_price,
                "quantity": closed.quantity,
                "entry_time": closed.entry_time,
                "exit_time": closed.exit_time,
                "pnl": closed.pnl,
                "pnl_percent": closed.pnl_pct,
                "status": "closed",
                "exit_reason": closed.exit_reason,
                "pattern_id": closed.pattern_id,
                "mode": self.config.trading.mode,
            })

        self.event_bus.publish(Event(
            type=EventType.POSITION_CLOSED,
            data={
                "symbol": symbol,
                "pnl": pnl,
                "reason": reason,
            },
            source="PositionTracker"
        ))

        return closed

    def open_position(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        pattern_id: Optional[str] = None,
    ) -> Position:
        """Manually open a position."""
        return self._open_position(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_id=pattern_id,
        )

    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "manual",
        quantity: Optional[int] = None,
    ) -> Optional[ClosedTrade]:
        """Manually close a position."""
        position = self._positions.get(symbol)
        if not position:
            return None

        qty = quantity or position.quantity
        return self._close_position(symbol, qty, price, reason)

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update position prices.

        Args:
            prices: Dictionary of symbol to current price
        """
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].update_price(price)

        self.event_bus.publish(Event(
            type=EventType.POSITION_UPDATED,
            data={"positions": len(self._positions)},
            source="PositionTracker"
        ))

    def set_stop_loss(self, symbol: str, stop_loss: float) -> None:
        """Set stop loss for a position."""
        if symbol in self._positions:
            self._positions[symbol].stop_loss = stop_loss
            logger.info(f"Set stop loss for {symbol}: {stop_loss}")

    def set_take_profit(self, symbol: str, take_profit: float) -> None:
        """Set take profit for a position."""
        if symbol in self._positions:
            self._positions[symbol].take_profit = take_profit
            logger.info(f"Set take profit for {symbol}: {take_profit}")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all open positions."""
        return self._positions.copy()

    def get_position_value(self, symbol: str) -> float:
        """Get current value of a position."""
        pos = self._positions.get(symbol)
        if pos:
            return pos.current_price * pos.quantity
        return 0.0

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure."""
        return sum(
            pos.current_price * pos.quantity
            for pos in self._positions.values()
        )

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def get_closed_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[ClosedTrade]:
        """Get closed trades."""
        trades = self._closed_trades

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        return trades[-limit:]

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return symbol in self._positions

    def get_direction(self, symbol: str) -> Optional[str]:
        """Get direction of position for a symbol."""
        pos = self._positions.get(symbol)
        return pos.direction if pos else None
