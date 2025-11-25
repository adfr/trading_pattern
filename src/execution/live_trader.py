"""Live trading execution engine."""

import threading
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from ..core.config import Config
from ..core.database import Database
from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.logger import get_logger
from ..data.ibkr_client import IBKRClient
from ..data.market_data import MarketDataManager
from ..strategy.pattern import PatternMatcher
from ..strategy.signal import SignalGenerator, SignalType
from ..strategy.risk import RiskManager
from .order_manager import OrderManager
from .position_tracker import PositionTracker

logger = get_logger(__name__)


class LiveTrader:
    """
    Live trading execution engine.

    Coordinates all components for real-time trading.
    """

    def __init__(
        self,
        config: Config,
        database: Optional[Database] = None,
        ibkr_client: Optional[IBKRClient] = None,
    ):
        """
        Initialize live trader.

        Args:
            config: Configuration object
            database: Database instance
            ibkr_client: IBKR client (optional, will create if not provided)
        """
        self.config = config
        self.database = database or Database(config.database_path)
        self.event_bus = get_event_bus()

        # Initialize IBKR client
        self.ibkr_client = ibkr_client

        # Initialize components
        self.market_data = MarketDataManager(config, self.ibkr_client, self.event_bus)
        self.order_manager = OrderManager(config, self.ibkr_client, self.database, self.event_bus)
        self.position_tracker = PositionTracker(config, self.database, self.event_bus)
        self.risk_manager = RiskManager(config, self.event_bus)

        # Strategy components
        self.pattern_matcher = PatternMatcher()
        self.signal_generator = SignalGenerator(self.pattern_matcher, self.event_bus)

        # Trading state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._symbols: list[str] = config.trading.symbols

        # Account info
        self._account_value: float = 100000  # Default, will be updated

        # Deployed patterns
        self._active_patterns: list[dict[str, Any]] = []

    def load_deployed_patterns(self) -> None:
        """Load patterns that have been deployed for trading."""
        patterns = self.database.get_patterns(status="deployed")

        for pattern_data in patterns:
            self.pattern_matcher.load_pattern_from_dict(pattern_data)
            self._active_patterns.append(pattern_data)

        logger.info(f"Loaded {len(patterns)} deployed patterns")

    def start(self) -> None:
        """Start live trading."""
        if self._running:
            logger.warning("Trader already running")
            return

        logger.info("Starting live trader...")

        # Connect to IBKR if available and not connected
        if self.ibkr_client and not self.ibkr_client.connected:
            try:
                self.ibkr_client.connect()
            except Exception as e:
                logger.error(f"Failed to connect to IBKR: {e}")
                if self.config.trading.mode == "live":
                    raise

        # Load deployed patterns
        self.load_deployed_patterns()

        if not self._active_patterns:
            logger.warning("No deployed patterns found. Add patterns before trading.")

        # Update account value
        self._update_account_value()

        # Start market data
        self.market_data.start()

        # Subscribe to symbols
        for symbol in self._symbols:
            self._subscribe_symbol(symbol)

        # Start trading loop
        self._running = True
        self._thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._thread.start()

        self.event_bus.publish(Event(
            type=EventType.SYSTEM_START,
            data={"mode": self.config.trading.mode},
            source="LiveTrader"
        ))

        logger.info(
            f"Live trader started in {self.config.trading.mode} mode "
            f"with {len(self._symbols)} symbols"
        )

    def stop(self) -> None:
        """Stop live trading."""
        if not self._running:
            return

        logger.info("Stopping live trader...")

        self._running = False

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        # Cancel all pending orders
        cancelled = self.order_manager.cancel_all_orders()
        logger.info(f"Cancelled {cancelled} pending orders")

        # Stop market data
        self.market_data.stop()

        # Disconnect from IBKR
        if self.ibkr_client and self.ibkr_client.connected:
            self.ibkr_client.disconnect()

        self.event_bus.publish(Event(
            type=EventType.SYSTEM_STOP,
            data={},
            source="LiveTrader"
        ))

        logger.info("Live trader stopped")

    def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to market data for a symbol."""
        def on_bar(bar_data: dict) -> None:
            self._on_bar(bar_data)

        # Load historical data first
        self.market_data.load_historical(symbol, duration="2 D", bar_size="1 min")

        # Subscribe to real-time
        self.market_data.subscribe(symbol, on_bar)

    def _on_bar(self, bar_data: dict) -> None:
        """Handle new bar data."""
        symbol = bar_data["symbol"]
        price = bar_data["close"]

        # Update position prices
        self.position_tracker.update_prices({symbol: price})

        # Check paper trading orders
        if self.config.trading.mode == "paper":
            self.order_manager.check_stop_orders(symbol, price)
            self.order_manager.check_limit_orders(symbol, price)

    def _trading_loop(self) -> None:
        """Main trading loop."""
        last_check = {}

        while self._running:
            try:
                # Check if market is open
                if not self.market_data.is_market_open():
                    time.sleep(60)
                    continue

                for symbol in self._symbols:
                    # Rate limit checks
                    now = time.time()
                    if symbol in last_check:
                        if now - last_check[symbol] < self.config.risk.min_trade_interval_minutes * 60:
                            continue
                    last_check[symbol] = now

                    # Get current data
                    df = self.market_data.get_bars(symbol, count=200)
                    if df.empty:
                        continue

                    # Update position price
                    current_price = df["close"].iloc[-1]
                    self.position_tracker.update_prices({symbol: current_price})

                    # Check exit conditions for open positions
                    position = self.position_tracker.get_position(symbol)
                    if position:
                        exit_signal = self.signal_generator.check_exit_conditions(
                            symbol=symbol,
                            df=df,
                            position=position.to_dict(),
                        )

                        if exit_signal:
                            self._execute_exit(symbol, exit_signal, current_price)
                            continue

                    # Check for new entry signals
                    current_position = "flat"
                    if position:
                        current_position = position.direction

                    signals = self.signal_generator.generate_signals(
                        symbol=symbol,
                        df=df,
                        current_position=current_position,
                    )

                    for signal in signals:
                        if signal.is_entry:
                            self._execute_entry(symbol, signal, current_price)

                # Small sleep to prevent CPU spinning
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(5)

    def _execute_entry(self, symbol: str, signal: Any, price: float) -> None:
        """Execute an entry signal."""
        # Check risk limits
        direction = "long" if signal.signal_type == SignalType.LONG_ENTRY else "short"

        # Calculate position size
        size = self.risk_manager.calculate_position_size(
            account_value=self._account_value,
            entry_price=price,
            stop_loss=signal.stop_loss or price * 0.98,
        )

        if size <= 0:
            logger.warning(f"Position size is 0 for {symbol}")
            return

        # Check if trade is allowed
        allowed, reason = self.risk_manager.check_trade_allowed(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=price,
            account_value=self._account_value,
        )

        if not allowed:
            logger.warning(f"Trade not allowed: {reason}")
            return

        # Create and submit order
        side = "buy" if direction == "long" else "sell"
        order = self.order_manager.create_market_order(
            symbol=symbol,
            side=side,
            quantity=size,
            pattern_id=signal.metadata.get("pattern_id"),
        )

        if self.order_manager.submit_order(order):
            # Open position tracking
            self.position_tracker.open_position(
                symbol=symbol,
                direction=direction,
                quantity=size,
                price=price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                pattern_id=signal.metadata.get("pattern_id"),
            )

            # Register with risk manager
            self.risk_manager.register_trade(
                symbol=symbol,
                direction=direction,
                size=size,
                entry_price=price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            logger.info(
                f"Entered {direction} position: {size} {symbol} @ {price} "
                f"(SL: {signal.stop_loss}, TP: {signal.take_profit})"
            )

    def _execute_exit(self, symbol: str, signal: Any, price: float) -> None:
        """Execute an exit signal."""
        position = self.position_tracker.get_position(symbol)
        if not position:
            return

        # Create and submit exit order
        side = "sell" if position.direction == "long" else "buy"
        order = self.order_manager.create_market_order(
            symbol=symbol,
            side=side,
            quantity=position.quantity,
        )

        if self.order_manager.submit_order(order):
            # Close position
            closed = self.position_tracker.close_position(
                symbol=symbol,
                price=price,
                reason=signal.metadata.get("exit_reason", "signal"),
            )

            if closed:
                # Update risk manager
                self.risk_manager.close_position(symbol, price)

                logger.info(
                    f"Exited position: {symbol} @ {price}, "
                    f"P&L: ${closed.pnl:.2f} ({closed.pnl_pct:.2%})"
                )

    def _update_account_value(self) -> None:
        """Update account value from IBKR."""
        if self.ibkr_client and self.ibkr_client.connected:
            try:
                summary = self.ibkr_client.get_account_summary()
                self._account_value = summary.get("NetLiquidation", self._account_value)
                self.risk_manager.update_equity(self._account_value)
            except Exception as e:
                logger.error(f"Failed to get account value: {e}")

    def add_pattern(self, pattern: dict[str, Any]) -> None:
        """Add a pattern for live trading."""
        self.pattern_matcher.load_pattern_from_dict(pattern)
        self._active_patterns.append(pattern)
        logger.info(f"Added pattern for trading: {pattern.get('name')}")

    def remove_pattern(self, pattern_id: str) -> None:
        """Remove a pattern from live trading."""
        self.pattern_matcher.remove_pattern(pattern_id)
        self._active_patterns = [p for p in self._active_patterns if p.get("id") != pattern_id]
        logger.info(f"Removed pattern: {pattern_id}")

    def get_status(self) -> dict[str, Any]:
        """Get current trading status."""
        positions = self.position_tracker.get_all_positions()
        risk_metrics = self.risk_manager.get_metrics(self._account_value)

        return {
            "running": self._running,
            "mode": self.config.trading.mode,
            "connected": self.ibkr_client.connected if self.ibkr_client else False,
            "symbols": self._symbols,
            "active_patterns": len(self._active_patterns),
            "open_positions": len(positions),
            "positions": {s: p.to_dict() for s, p in positions.items()},
            "total_unrealized_pnl": self.position_tracker.get_total_unrealized_pnl(),
            "account_value": self._account_value,
            "daily_pnl": risk_metrics.daily_pnl,
            "daily_trades": risk_metrics.daily_trades,
            "total_exposure": risk_metrics.total_exposure,
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        closed_trades = self.position_tracker.get_closed_trades()

        if not closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0,
            }

        pnls = [t.pnl for t in closed_trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(closed_trades) if closed_trades else 0,
            "total_pnl": sum(pnls),
            "avg_win": sum(winning) / len(winning) if winning else 0,
            "avg_loss": sum(losing) / len(losing) if losing else 0,
            "largest_win": max(pnls) if pnls else 0,
            "largest_loss": min(pnls) if pnls else 0,
        }
