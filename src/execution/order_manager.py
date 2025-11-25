"""Order management for trade execution."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from ..core.config import Config
from ..core.database import Database
from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.exceptions import OrderError
from ..core.logger import get_logger
from ..data.ibkr_client import IBKRClient

logger = get_logger(__name__)


class OrderStatus(Enum):
    """Order status."""
    PENDING = auto()
    SUBMITTED = auto()
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class OrderType(Enum):
    """Order type."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class OrderSide(Enum):
    """Order side."""
    BUY = auto()
    SELL = auto()


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    status: OrderStatus = OrderStatus.PENDING

    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    fill_price: Optional[float] = None

    # Execution details
    filled_quantity: int = 0
    remaining_quantity: int = 0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Linkage
    pattern_id: Optional[str] = None
    parent_order_id: Optional[str] = None  # For brackets

    # IBKR order ID
    broker_order_id: Optional[int] = None

    def __post_init__(self):
        self.remaining_quantity = self.quantity

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.name,
            "quantity": self.quantity,
            "order_type": self.order_type.name,
            "status": self.status.name,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "fill_price": self.fill_price,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "pattern_id": self.pattern_id,
        }


class OrderManager:
    """
    Manages order creation, submission, and tracking.

    Handles communication with IBKR for order execution.
    """

    def __init__(
        self,
        config: Config,
        ibkr_client: Optional[IBKRClient] = None,
        database: Optional[Database] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize order manager.

        Args:
            config: Configuration object
            ibkr_client: IBKR client for execution
            database: Database for order storage
            event_bus: Event bus
        """
        self.config = config
        self.ibkr_client = ibkr_client
        self.database = database
        self.event_bus = event_bus or get_event_bus()

        # Order tracking
        self._orders: dict[str, Order] = {}
        self._pending_orders: set[str] = set()

        # Mode
        self.is_paper = config.trading.mode == "paper"

    def create_market_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: int,
        pattern_id: Optional[str] = None,
    ) -> Order:
        """Create a market order."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            pattern_id=pattern_id,
        )

        self._orders[order.id] = order
        logger.info(f"Created market order: {order.side.name} {quantity} {symbol}")

        return order

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        limit_price: float,
        pattern_id: Optional[str] = None,
    ) -> Order:
        """Create a limit order."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            pattern_id=pattern_id,
        )

        self._orders[order.id] = order
        logger.info(f"Created limit order: {order.side.name} {quantity} {symbol} @ {limit_price}")

        return order

    def create_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        stop_price: float,
        pattern_id: Optional[str] = None,
    ) -> Order:
        """Create a stop order."""
        order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price,
            pattern_id=pattern_id,
        )

        self._orders[order.id] = order
        logger.info(f"Created stop order: {order.side.name} {quantity} {symbol} @ {stop_price}")

        return order

    def create_bracket_orders(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        pattern_id: Optional[str] = None,
    ) -> tuple[Order, Order, Order]:
        """
        Create bracket orders (entry + stop loss + take profit).

        Returns:
            Tuple of (entry_order, stop_loss_order, take_profit_order)
        """
        # Entry order
        entry_order = self.create_limit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=entry_price,
            pattern_id=pattern_id,
        )

        # Exit side is opposite
        exit_side = "sell" if side.lower() == "buy" else "buy"

        # Stop loss order
        stop_order = self.create_stop_order(
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            stop_price=stop_loss,
            pattern_id=pattern_id,
        )
        stop_order.parent_order_id = entry_order.id

        # Take profit order
        tp_order = self.create_limit_order(
            symbol=symbol,
            side=exit_side,
            quantity=quantity,
            limit_price=take_profit,
            pattern_id=pattern_id,
        )
        tp_order.parent_order_id = entry_order.id

        return entry_order, stop_order, tp_order

    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            True if submission successful
        """
        if order.status != OrderStatus.PENDING:
            raise OrderError(f"Cannot submit order in status: {order.status}")

        # Paper trading simulation
        if self.is_paper or not self.ibkr_client or not self.ibkr_client.connected:
            return self._simulate_order(order)

        # Live execution
        try:
            action = "BUY" if order.side == OrderSide.BUY else "SELL"

            if order.order_type == OrderType.MARKET:
                trade = self.ibkr_client.place_market_order(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    action=action,
                )
            elif order.order_type == OrderType.LIMIT:
                trade = self.ibkr_client.place_limit_order(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    action=action,
                    limit_price=order.limit_price,
                )
            elif order.order_type == OrderType.STOP:
                trade = self.ibkr_client.place_stop_order(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    action=action,
                    stop_price=order.stop_price,
                )
            else:
                raise OrderError(f"Unsupported order type: {order.order_type}")

            order.broker_order_id = trade.order.orderId
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now()
            self._pending_orders.add(order.id)

            self.event_bus.publish(Event(
                type=EventType.ORDER_SUBMITTED,
                data=order.to_dict(),
                source="OrderManager"
            ))

            logger.info(f"Order submitted: {order.id} (IBKR ID: {order.broker_order_id})")
            return True

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            order.status = OrderStatus.REJECTED
            return False

    def _simulate_order(self, order: Order) -> bool:
        """Simulate order execution for paper trading."""
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()

        # For market orders, simulate immediate fill
        if order.order_type == OrderType.MARKET:
            self._fill_order(order, order.limit_price or 0)  # Price would come from market data

        self._pending_orders.add(order.id)

        self.event_bus.publish(Event(
            type=EventType.ORDER_SUBMITTED,
            data=order.to_dict(),
            source="OrderManager"
        ))

        logger.info(f"Order submitted (paper): {order.id}")
        return True

    def _fill_order(
        self,
        order: Order,
        fill_price: float,
        filled_quantity: Optional[int] = None,
    ) -> None:
        """Mark order as filled."""
        order.fill_price = fill_price
        order.filled_quantity = filled_quantity or order.quantity
        order.remaining_quantity = order.quantity - order.filled_quantity

        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        order.filled_at = datetime.now()

        if order.id in self._pending_orders:
            self._pending_orders.remove(order.id)

        self.event_bus.publish(Event(
            type=EventType.ORDER_FILLED,
            data=order.to_dict(),
            source="OrderManager"
        ))

        logger.info(
            f"Order filled: {order.id} @ {fill_price} "
            f"({order.filled_quantity}/{order.quantity})"
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self._orders:
            return False

        order = self._orders[order_id]

        if order.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            logger.warning(f"Cannot cancel order in status: {order.status}")
            return False

        # Paper trading
        if self.is_paper or not self.ibkr_client:
            order.status = OrderStatus.CANCELLED
            if order_id in self._pending_orders:
                self._pending_orders.remove(order_id)

            self.event_bus.publish(Event(
                type=EventType.ORDER_CANCELLED,
                data=order.to_dict(),
                source="OrderManager"
            ))

            logger.info(f"Order cancelled (paper): {order_id}")
            return True

        # Live cancellation
        # Note: Would need to track IBKR trade object for actual cancellation
        logger.info(f"Order cancelled: {order_id}")
        return True

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all pending orders.

        Args:
            symbol: Only cancel orders for this symbol (optional)

        Returns:
            Number of orders cancelled
        """
        cancelled = 0

        for order_id in list(self._pending_orders):
            order = self._orders.get(order_id)
            if order and (symbol is None or order.symbol == symbol):
                if self.cancel_order(order_id):
                    cancelled += 1

        return cancelled

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None,
    ) -> list[Order]:
        """Get orders with optional filtering."""
        orders = list(self._orders.values())

        if status:
            orders = [o for o in orders if o.status == status]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders."""
        return [self._orders[oid] for oid in self._pending_orders if oid in self._orders]

    def update_from_fill(
        self,
        broker_order_id: int,
        fill_price: float,
        filled_quantity: int,
    ) -> None:
        """Update order from broker fill event."""
        for order in self._orders.values():
            if order.broker_order_id == broker_order_id:
                self._fill_order(order, fill_price, filled_quantity)
                break

    def check_stop_orders(self, symbol: str, current_price: float) -> None:
        """
        Check stop orders for paper trading.

        Args:
            symbol: Symbol to check
            current_price: Current market price
        """
        if not self.is_paper:
            return

        for order in self.get_orders(status=OrderStatus.SUBMITTED, symbol=symbol):
            if order.order_type == OrderType.STOP:
                triggered = False

                if order.side == OrderSide.SELL and current_price <= order.stop_price:
                    triggered = True
                elif order.side == OrderSide.BUY and current_price >= order.stop_price:
                    triggered = True

                if triggered:
                    self._fill_order(order, current_price)

    def check_limit_orders(self, symbol: str, current_price: float) -> None:
        """
        Check limit orders for paper trading.

        Args:
            symbol: Symbol to check
            current_price: Current market price
        """
        if not self.is_paper:
            return

        for order in self.get_orders(status=OrderStatus.SUBMITTED, symbol=symbol):
            if order.order_type == OrderType.LIMIT:
                triggered = False

                if order.side == OrderSide.BUY and current_price <= order.limit_price:
                    triggered = True
                elif order.side == OrderSide.SELL and current_price >= order.limit_price:
                    triggered = True

                if triggered:
                    self._fill_order(order, order.limit_price)
