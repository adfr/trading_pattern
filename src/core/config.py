"""Configuration management for the trading system."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigurationError


@dataclass
class IBKRConfig:
    """IBKR connection configuration."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 60
    readonly: bool = False


@dataclass
class AIConfig:
    """Claude AI configuration."""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key: Optional[str] = None


@dataclass
class TradingConfig:
    """Trading parameters configuration."""
    mode: str = "paper"
    symbols: list[str] = field(default_factory=lambda: ["QQQ", "SPY"])
    timeframe: str = "1min"
    max_position_size: float = 0.05
    max_total_exposure: float = 0.30
    daily_loss_limit: float = 0.02


@dataclass
class RiskConfig:
    """Risk management configuration."""
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    max_trades_per_day: int = 10
    min_trade_interval_minutes: int = 5
    kelly_fraction: float = 0.25


@dataclass
class BacktestConfig:
    """Backtesting criteria configuration."""
    min_trades: int = 30
    min_win_rate: float = 0.40
    min_profit_factor: float = 1.5
    min_sharpe_ratio: float = 1.0
    max_drawdown: float = 0.15
    walk_forward_windows: int = 5
    out_of_sample_ratio: float = 0.3


@dataclass
class PatternConfig:
    """Pattern generation configuration."""
    window_size: int = 60
    similarity_threshold: float = 0.85
    min_confidence: float = 0.7
    max_patterns_per_generation: int = 5


@dataclass
class MarketConfig:
    """Market hours configuration."""
    open_time: str = "09:30"
    close_time: str = "16:00"
    timezone: str = "America/New_York"
    trading_days: list[str] = field(
        default_factory=lambda: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "logs/trading.log"
    max_size_mb: int = 10
    backup_count: int = 5


class Config:
    """Main configuration class that loads and provides access to all settings."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file and environment.

        Args:
            config_path: Path to YAML config file (optional)
        """
        load_dotenv()

        self._config_data: dict[str, Any] = {}

        if config_path:
            self._load_from_file(config_path)
        else:
            default_path = Path("config/config.yaml")
            if default_path.exists():
                self._load_from_file(str(default_path))

        self._initialize_configs()

    def _load_from_file(self, path: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(path, "r") as f:
                self._config_data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")

    def _initialize_configs(self) -> None:
        """Initialize all configuration dataclasses."""
        self.ibkr = IBKRConfig(**self._config_data.get("ibkr", {}))

        ai_data = self._config_data.get("ai", {})
        ai_data["api_key"] = os.getenv("ANTHROPIC_API_KEY", ai_data.get("api_key"))
        self.ai = AIConfig(**ai_data)

        self.trading = TradingConfig(**self._config_data.get("trading", {}))
        self.risk = RiskConfig(**self._config_data.get("risk", {}))
        self.backtest = BacktestConfig(**self._config_data.get("backtest", {}))
        self.patterns = PatternConfig(**self._config_data.get("patterns", {}))
        self.market = MarketConfig(**self._config_data.get("market", {}))
        self.logging = LoggingConfig(**self._config_data.get("logging", {}))

        self.database_path = self._config_data.get("database", {}).get(
            "path", "data/trading.db"
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.ai.api_key:
            raise ConfigurationError(
                "ANTHROPIC_API_KEY not set. Set it via environment variable or config file."
            )

        if self.trading.mode not in ("paper", "live"):
            raise ConfigurationError(
                f"Invalid trading mode: {self.trading.mode}. Must be 'paper' or 'live'."
            )

        if not 0 < self.trading.max_position_size <= 1:
            raise ConfigurationError(
                "max_position_size must be between 0 and 1"
            )

        if not 0 < self.trading.max_total_exposure <= 1:
            raise ConfigurationError(
                "max_total_exposure must be between 0 and 1"
            )

        if self.backtest.min_trades < 1:
            raise ConfigurationError(
                "min_trades must be at least 1"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "ibkr": self.ibkr.__dict__,
            "ai": {k: v for k, v in self.ai.__dict__.items() if k != "api_key"},
            "trading": self.trading.__dict__,
            "risk": self.risk.__dict__,
            "backtest": self.backtest.__dict__,
            "patterns": self.patterns.__dict__,
            "market": self.market.__dict__,
            "logging": self.logging.__dict__,
            "database_path": self.database_path,
        }
