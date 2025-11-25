"""Custom exceptions for the trading system."""


class TradingSystemError(Exception):
    """Base exception for trading system errors."""
    pass


class ConfigurationError(TradingSystemError):
    """Configuration-related errors."""
    pass


class IBKRConnectionError(TradingSystemError):
    """IBKR connection errors."""
    pass


class InsufficientDataError(TradingSystemError):
    """Not enough data for analysis."""
    pass


class RiskLimitExceeded(TradingSystemError):
    """Risk limit has been exceeded."""
    pass


class BacktestError(TradingSystemError):
    """Backtesting errors."""
    pass


class PatternError(TradingSystemError):
    """Pattern-related errors."""
    pass


class OrderError(TradingSystemError):
    """Order execution errors."""
    pass


class ValidationError(TradingSystemError):
    """Data validation errors."""
    pass
