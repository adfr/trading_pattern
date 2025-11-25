"""Event system for the trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional
from collections import defaultdict
import threading


class EventType(Enum):
    """Types of events in the trading system."""
    # Market data events
    TICK = auto()
    BAR = auto()
    QUOTE = auto()

    # Pattern events
    PATTERN_DETECTED = auto()
    PATTERN_GENERATED = auto()
    PATTERN_VALIDATED = auto()

    # Signal events
    SIGNAL_GENERATED = auto()
    SIGNAL_CONFIRMED = auto()

    # Order events
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()

    # Position events
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()

    # Risk events
    RISK_LIMIT_WARNING = auto()
    RISK_LIMIT_BREACH = auto()
    DAILY_LOSS_LIMIT = auto()

    # System events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    SYSTEM_ERROR = auto()

    # Backtest events
    BACKTEST_START = auto()
    BACKTEST_COMPLETE = auto()
    BACKTEST_TRADE = auto()


@dataclass
class Event:
    """Represents an event in the system."""
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""

    def __str__(self) -> str:
        return f"Event({self.type.name}, source={self.source}, data={self.data})"


class EventBus:
    """
    Thread-safe event bus for publishing and subscribing to events.

    Implements the observer pattern for loose coupling between components.
    """

    def __init__(self):
        self._subscribers: dict[EventType, list[Callable[[Event], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._event_history: list[Event] = []
        self._max_history = 1000

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is published
        """
        with self._lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None]
    ) -> None:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            # Get subscribers (copy to avoid modification during iteration)
            subscribers = list(self._subscribers[event.type])

        # Call subscribers outside lock to prevent deadlocks
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                # Log but don't stop other subscribers
                print(f"Error in event subscriber: {e}")

    def publish_async(self, event: Event) -> threading.Thread:
        """
        Publish an event asynchronously.

        Args:
            event: Event to publish

        Returns:
            Thread handling the event
        """
        thread = threading.Thread(target=self.publish, args=(event,))
        thread.start()
        return thread

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> list[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        with self._lock:
            if event_type:
                events = [e for e in self._event_history if e.type == event_type]
            else:
                events = list(self._event_history)
            return events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._event_history.clear()


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
