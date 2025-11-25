"""Interactive Brokers API client."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import pandas as pd
from ib_insync import IB, Contract, Stock, Order, MarketOrder, LimitOrder, StopOrder, Trade

from ..core.config import Config
from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.exceptions import IBKRConnectionError, OrderError
from ..core.logger import get_logger

logger = get_logger(__name__)


class IBKRClient:
    """
    Client for Interactive Brokers TWS/Gateway API.

    Uses ib_insync for async communication with IBKR.
    """

    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        Initialize IBKR client.

        Args:
            config: Configuration object
            event_bus: Event bus for publishing events (optional)
        """
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        self.ib = IB()
        self._connected = False
        self._callbacks: dict[str, list[Callable]] = {
            "tick": [],
            "bar": [],
            "order": [],
            "error": [],
        }

    @property
    def connected(self) -> bool:
        """Check if connected to IBKR."""
        return self._connected and self.ib.isConnected()

    def connect(self) -> None:
        """Connect to IBKR TWS/Gateway."""
        if self.connected:
            logger.info("Already connected to IBKR")
            return

        try:
            self.ib.connect(
                host=self.config.ibkr.host,
                port=self.config.ibkr.port,
                clientId=self.config.ibkr.client_id,
                timeout=self.config.ibkr.timeout,
                readonly=self.config.ibkr.readonly,
            )
            self._connected = True

            # Set up event handlers
            self.ib.errorEvent += self._on_error
            self.ib.orderStatusEvent += self._on_order_status
            self.ib.newOrderEvent += self._on_new_order

            logger.info(
                f"Connected to IBKR at {self.config.ibkr.host}:{self.config.ibkr.port}"
            )

            self.event_bus.publish(Event(
                type=EventType.SYSTEM_START,
                data={"component": "ibkr_client"},
                source="IBKRClient"
            ))

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            raise IBKRConnectionError(f"Failed to connect to IBKR: {e}")

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self.connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Any) -> None:
        """Handle IBKR errors."""
        # Filter out informational messages
        if errorCode in (2104, 2106, 2158):  # Market data farm messages
            return

        logger.error(f"IBKR Error {errorCode}: {errorString}")

        self.event_bus.publish(Event(
            type=EventType.SYSTEM_ERROR,
            data={
                "req_id": reqId,
                "error_code": errorCode,
                "error_string": errorString,
            },
            source="IBKRClient"
        ))

        for callback in self._callbacks["error"]:
            callback(errorCode, errorString)

    def _on_order_status(self, trade: Trade) -> None:
        """Handle order status updates."""
        status = trade.orderStatus.status
        logger.info(f"Order {trade.order.orderId}: {status}")

        event_type = {
            "Filled": EventType.ORDER_FILLED,
            "Cancelled": EventType.ORDER_CANCELLED,
            "Submitted": EventType.ORDER_SUBMITTED,
        }.get(status)

        if event_type:
            self.event_bus.publish(Event(
                type=event_type,
                data={
                    "order_id": trade.order.orderId,
                    "symbol": trade.contract.symbol,
                    "status": status,
                    "filled": trade.orderStatus.filled,
                    "remaining": trade.orderStatus.remaining,
                    "avg_fill_price": trade.orderStatus.avgFillPrice,
                },
                source="IBKRClient"
            ))

        for callback in self._callbacks["order"]:
            callback(trade)

    def _on_new_order(self, trade: Trade) -> None:
        """Handle new order events."""
        logger.info(f"New order: {trade.order.orderId}")

    def create_contract(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD") -> Contract:
        """
        Create a contract object.

        Args:
            symbol: Ticker symbol
            sec_type: Security type (STK, OPT, FUT, etc.)
            exchange: Exchange
            currency: Currency

        Returns:
            Contract object
        """
        if sec_type == "STK":
            contract = Stock(symbol, exchange, currency)
        else:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = sec_type
            contract.exchange = exchange
            contract.currency = currency

        # Qualify the contract to get full details
        if self.connected:
            self.ib.qualifyContracts(contract)

        return contract

    def get_historical_data(
        self,
        symbol: str,
        duration: str = "5 D",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical bar data.

        Args:
            symbol: Ticker symbol
            duration: How far back to get data (e.g., "5 D", "1 M", "1 Y")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day")
            what_to_show: Type of data (TRADES, MIDPOINT, BID, ASK)
            use_rth: Use regular trading hours only

        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        contract = self.create_contract(symbol)

        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )

        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "timestamp": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        } for bar in bars])

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        return df

    def subscribe_realtime_bars(
        self,
        symbol: str,
        callback: Callable[[dict], None],
        bar_size: int = 5,
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> int:
        """
        Subscribe to real-time bars.

        Args:
            symbol: Ticker symbol
            callback: Function to call with each bar
            bar_size: Bar size in seconds (5 seconds is minimum)
            what_to_show: Type of data
            use_rth: Use regular trading hours only

        Returns:
            Request ID for the subscription
        """
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        contract = self.create_contract(symbol)

        def on_bar(bars, hasNewBar):
            if hasNewBar and bars:
                bar = bars[-1]
                bar_data = {
                    "symbol": symbol,
                    "timestamp": bar.time,
                    "open": bar.open_,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                callback(bar_data)

                self.event_bus.publish(Event(
                    type=EventType.BAR,
                    data=bar_data,
                    source="IBKRClient"
                ))

        bars = self.ib.reqRealTimeBars(
            contract,
            barSize=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
        )

        bars.updateEvent += on_bar

        return id(bars)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price.

        Args:
            symbol: Ticker symbol

        Returns:
            Current price or None if unavailable
        """
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        contract = self.create_contract(symbol)
        ticker = self.ib.reqMktData(contract, "", False, False)

        # Wait briefly for data
        self.ib.sleep(1)

        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)

        return price if price and price > 0 else None

    def place_market_order(
        self,
        symbol: str,
        quantity: int,
        action: str,  # "BUY" or "SELL"
    ) -> Trade:
        """
        Place a market order.

        Args:
            symbol: Ticker symbol
            quantity: Number of shares
            action: "BUY" or "SELL"

        Returns:
            Trade object
        """
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        if self.config.ibkr.readonly:
            raise OrderError("Client is in readonly mode")

        contract = self.create_contract(symbol)
        order = MarketOrder(action, quantity)

        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Placed market order: {action} {quantity} {symbol}")

        return trade

    def place_limit_order(
        self,
        symbol: str,
        quantity: int,
        action: str,
        limit_price: float,
    ) -> Trade:
        """
        Place a limit order.

        Args:
            symbol: Ticker symbol
            quantity: Number of shares
            action: "BUY" or "SELL"
            limit_price: Limit price

        Returns:
            Trade object
        """
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        if self.config.ibkr.readonly:
            raise OrderError("Client is in readonly mode")

        contract = self.create_contract(symbol)
        order = LimitOrder(action, quantity, limit_price)

        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Placed limit order: {action} {quantity} {symbol} @ {limit_price}")

        return trade

    def place_stop_order(
        self,
        symbol: str,
        quantity: int,
        action: str,
        stop_price: float,
    ) -> Trade:
        """
        Place a stop order.

        Args:
            symbol: Ticker symbol
            quantity: Number of shares
            action: "BUY" or "SELL"
            stop_price: Stop price

        Returns:
            Trade object
        """
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        if self.config.ibkr.readonly:
            raise OrderError("Client is in readonly mode")

        contract = self.create_contract(symbol)
        order = StopOrder(action, quantity, stop_price)

        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Placed stop order: {action} {quantity} {symbol} @ {stop_price}")

        return trade

    def place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        action: str,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
    ) -> list[Trade]:
        """
        Place a bracket order (entry + take profit + stop loss).

        Args:
            symbol: Ticker symbol
            quantity: Number of shares
            action: "BUY" or "SELL"
            entry_price: Entry limit price
            take_profit: Take profit price
            stop_loss: Stop loss price

        Returns:
            List of Trade objects
        """
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        if self.config.ibkr.readonly:
            raise OrderError("Client is in readonly mode")

        contract = self.create_contract(symbol)

        bracket = self.ib.bracketOrder(
            action,
            quantity,
            entry_price,
            take_profit,
            stop_loss,
        )

        trades = [self.ib.placeOrder(contract, o) for o in bracket]
        logger.info(
            f"Placed bracket order: {action} {quantity} {symbol} "
            f"entry={entry_price} tp={take_profit} sl={stop_loss}"
        )

        return trades

    def cancel_order(self, trade: Trade) -> None:
        """Cancel an order."""
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        self.ib.cancelOrder(trade.order)
        logger.info(f"Cancelled order: {trade.order.orderId}")

    def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions."""
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        positions = self.ib.positions()
        return [{
            "symbol": pos.contract.symbol,
            "quantity": pos.position,
            "avg_cost": pos.avgCost,
            "market_value": pos.position * pos.avgCost,
        } for pos in positions]

    def get_account_summary(self) -> dict[str, Any]:
        """Get account summary."""
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        account_values = self.ib.accountSummary()
        summary = {}

        for av in account_values:
            if av.tag in ("NetLiquidation", "TotalCashValue", "GrossPositionValue",
                         "MaintMarginReq", "AvailableFunds", "BuyingPower"):
                summary[av.tag] = float(av.value)

        return summary

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Get open orders."""
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        trades = self.ib.openTrades()
        return [{
            "order_id": trade.order.orderId,
            "symbol": trade.contract.symbol,
            "action": trade.order.action,
            "quantity": trade.order.totalQuantity,
            "order_type": trade.order.orderType,
            "status": trade.orderStatus.status,
            "filled": trade.orderStatus.filled,
        } for trade in trades]

    def run_loop(self) -> None:
        """Run the IB event loop (blocking)."""
        if not self.connected:
            raise IBKRConnectionError("Not connected to IBKR")

        self.ib.run()

    def sleep(self, seconds: float) -> None:
        """Sleep while keeping the connection alive."""
        if self.connected:
            self.ib.sleep(seconds)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
