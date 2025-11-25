"""AI-powered pattern generation."""

import json
import uuid
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.config import Config
from ..core.database import Database
from ..core.events import Event, EventBus, EventType, get_event_bus
from ..core.logger import get_logger
from .claude_client import ClaudeClient
from .prompts import PromptTemplates

logger = get_logger(__name__)


class PatternGenerator:
    """
    Generates trading patterns using Claude AI.

    Analyzes market data and generates backtestable trading patterns
    with precise entry/exit rules.
    """

    def __init__(
        self,
        config: Config,
        claude_client: Optional[ClaudeClient] = None,
        database: Optional[Database] = None,
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize pattern generator.

        Args:
            config: Configuration object
            claude_client: Claude client (optional, will create if not provided)
            database: Database for storing patterns (optional)
            event_bus: Event bus (optional)
        """
        self.config = config
        self.claude = claude_client or ClaudeClient(config)
        self.database = database
        self.event_bus = event_bus or get_event_bus()

    def generate_pattern(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        timeframe: str = "1min",
        additional_context: str = "",
    ) -> dict[str, Any]:
        """
        Generate a new trading pattern based on market data.

        Args:
            symbol: Ticker symbol
            market_data: DataFrame with OHLCV data
            timeframe: Data timeframe
            additional_context: Additional context for pattern generation

        Returns:
            Generated pattern dictionary
        """
        # Prepare market summary
        price_summary = self._create_price_summary(market_data)
        atr = self._calculate_atr(market_data)
        volume_profile = self._create_volume_profile(market_data)
        market_regime = self._determine_market_regime(market_data)

        # Generate pattern using Claude
        prompt = PromptTemplates.GENERATE_PATTERN.format(
            symbol=symbol,
            timeframe=timeframe,
            price_summary=price_summary,
            atr=f"{atr:.4f}",
            volume_profile=volume_profile,
            market_regime=market_regime,
            additional_context=additional_context or "None provided",
        )

        try:
            pattern = self.claude.generate_json(
                prompt=prompt,
                system_prompt=PromptTemplates.SYSTEM_PROMPT,
                temperature=0.7,
            )

            # Add metadata
            pattern["id"] = str(uuid.uuid4())
            pattern["symbol"] = symbol
            pattern["timeframe"] = timeframe
            pattern["created_at"] = datetime.now().isoformat()
            pattern["created_by"] = "ai"
            pattern["status"] = "pending"

            # Validate pattern structure
            self._validate_pattern(pattern)

            # Save to database if available
            if self.database:
                self.database.save_pattern({
                    "id": pattern["id"],
                    "name": pattern["name"],
                    "description": pattern.get("description", ""),
                    "pattern_type": pattern.get("pattern_type", "neutral"),
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "definition": pattern,
                    "entry_rules": pattern.get("entry", {}),
                    "exit_rules": pattern.get("exit", {}),
                    "created_by": "ai",
                    "status": "pending",
                })

            # Publish event
            self.event_bus.publish(Event(
                type=EventType.PATTERN_GENERATED,
                data={"pattern_id": pattern["id"], "name": pattern["name"]},
                source="PatternGenerator"
            ))

            logger.info(f"Generated pattern: {pattern['name']} ({pattern['id']})")
            return pattern

        except Exception as e:
            logger.error(f"Failed to generate pattern: {e}")
            raise

    def generate_multiple_patterns(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        num_patterns: int = 3,
        timeframe: str = "1min",
    ) -> list[dict[str, Any]]:
        """
        Generate multiple patterns for a symbol.

        Args:
            symbol: Ticker symbol
            market_data: DataFrame with OHLCV data
            num_patterns: Number of patterns to generate
            timeframe: Data timeframe

        Returns:
            List of generated patterns
        """
        patterns = []
        contexts = [
            "Focus on momentum-based entries",
            "Focus on mean-reversion opportunities",
            "Focus on breakout patterns",
            "Focus on range-bound trading",
            "Focus on volatility expansion",
        ]

        for i in range(min(num_patterns, len(contexts))):
            try:
                pattern = self.generate_pattern(
                    symbol=symbol,
                    market_data=market_data,
                    timeframe=timeframe,
                    additional_context=contexts[i],
                )
                patterns.append(pattern)
            except Exception as e:
                logger.error(f"Failed to generate pattern {i+1}: {e}")

        return patterns

    def optimize_pattern(
        self,
        pattern: dict[str, Any],
        backtest_results: dict[str, Any],
        failed_criteria: list[str],
    ) -> dict[str, Any]:
        """
        Optimize a pattern based on backtest results.

        Args:
            pattern: Original pattern
            backtest_results: Backtest results dictionary
            failed_criteria: List of criteria that failed

        Returns:
            Optimized pattern
        """
        prompt = PromptTemplates.OPTIMIZE_PATTERN.format(
            pattern_json=json.dumps(pattern, indent=2),
            backtest_results=json.dumps(backtest_results, indent=2),
            failed_criteria=", ".join(failed_criteria),
        )

        optimized = self.claude.generate_json(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_PROMPT,
            temperature=0.5,
        )

        # Preserve metadata
        optimized["id"] = str(uuid.uuid4())
        optimized["parent_id"] = pattern.get("id")
        optimized["symbol"] = pattern.get("symbol")
        optimized["timeframe"] = pattern.get("timeframe")
        optimized["created_at"] = datetime.now().isoformat()
        optimized["created_by"] = "ai_optimization"
        optimized["status"] = "pending"
        optimized["optimization_round"] = pattern.get("optimization_round", 0) + 1

        self._validate_pattern(optimized)

        logger.info(f"Optimized pattern: {optimized['name']} (round {optimized['optimization_round']})")
        return optimized

    def analyze_backtest_results(
        self,
        pattern: dict[str, Any],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Get AI analysis of backtest results.

        Args:
            pattern: Pattern dictionary
            results: Backtest results

        Returns:
            Analysis dictionary
        """
        # Format trade distribution
        trade_dist = ""
        if "trades" in results:
            trades = results["trades"]
            if trades:
                pnls = [t.get("pnl", 0) for t in trades]
                trade_dist = f"""
- Average P&L: ${np.mean(pnls):.2f}
- Median P&L: ${np.median(pnls):.2f}
- Best trade: ${max(pnls):.2f}
- Worst trade: ${min(pnls):.2f}
- Std Dev: ${np.std(pnls):.2f}
"""

        prompt = PromptTemplates.ANALYZE_BACKTEST.format(
            pattern_name=pattern.get("name", "Unknown"),
            start_date=results.get("start_date", "N/A"),
            end_date=results.get("end_date", "N/A"),
            total_trades=results.get("total_trades", 0),
            win_rate=results.get("win_rate", 0),
            profit_factor=results.get("profit_factor", 0),
            sharpe_ratio=results.get("sharpe_ratio", 0),
            max_drawdown=results.get("max_drawdown", 0),
            total_return=results.get("total_return", 0),
            trade_distribution=trade_dist,
        )

        return self.claude.generate_json(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_PROMPT,
            temperature=0.3,
        )

    def generate_variations(
        self,
        base_pattern: dict[str, Any],
        num_variations: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Generate variations of a base pattern.

        Args:
            base_pattern: Base pattern to vary
            num_variations: Number of variations to generate

        Returns:
            List of pattern variations
        """
        prompt = PromptTemplates.GENERATE_VARIATIONS.format(
            num_variations=num_variations,
            pattern_json=json.dumps(base_pattern, indent=2),
        )

        response = self.claude.generate_json(
            prompt=prompt,
            system_prompt=PromptTemplates.SYSTEM_PROMPT,
            temperature=0.8,
        )

        # Handle both array and object responses
        if isinstance(response, list):
            variations = response
        elif isinstance(response, dict) and "variations" in response:
            variations = response["variations"]
        else:
            variations = [response]

        # Add metadata to each variation
        for i, var in enumerate(variations):
            var["id"] = str(uuid.uuid4())
            var["parent_id"] = base_pattern.get("id")
            var["symbol"] = base_pattern.get("symbol")
            var["timeframe"] = base_pattern.get("timeframe")
            var["created_at"] = datetime.now().isoformat()
            var["created_by"] = "ai_variation"
            var["status"] = "pending"
            var["variation_index"] = i + 1

        return variations

    def _create_price_summary(self, df: pd.DataFrame) -> str:
        """Create a price summary from DataFrame."""
        if df.empty:
            return "No data available"

        recent = df.iloc[-100:] if len(df) > 100 else df

        return f"""
- Current price: {recent['close'].iloc[-1]:.2f}
- Period high: {recent['high'].max():.2f}
- Period low: {recent['low'].min():.2f}
- Price change: {((recent['close'].iloc[-1] / recent['close'].iloc[0]) - 1) * 100:.2f}%
- Average bar range: {(recent['high'] - recent['low']).mean():.4f}
"""

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period:
            return 0.0

        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else 0.0

    def _create_volume_profile(self, df: pd.DataFrame) -> str:
        """Create a volume profile description."""
        if df.empty or "volume" not in df.columns:
            return "Volume data not available"

        recent = df.iloc[-100:] if len(df) > 100 else df
        avg_vol = recent["volume"].mean()
        current_vol = recent["volume"].iloc[-1]

        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        return f"""
- Average volume: {avg_vol:,.0f}
- Current volume: {current_vol:,.0f}
- Volume ratio (current/avg): {vol_ratio:.2f}x
"""

    def _determine_market_regime(self, df: pd.DataFrame) -> str:
        """Determine current market regime."""
        if len(df) < 50:
            return "insufficient data"

        recent = df.iloc[-50:]
        returns = recent["close"].pct_change().dropna()

        mean_return = returns.mean()
        volatility = returns.std()
        total_return = (recent["close"].iloc[-1] / recent["close"].iloc[0]) - 1

        if abs(total_return) > 0.05:
            if total_return > 0:
                return "trending_up"
            else:
                return "trending_down"
        elif volatility > 0.02:
            return "volatile"
        else:
            return "ranging"

    def _validate_pattern(self, pattern: dict[str, Any]) -> None:
        """Validate pattern structure."""
        required_fields = ["name", "detection", "entry", "exit"]

        for field in required_fields:
            if field not in pattern:
                raise ValueError(f"Pattern missing required field: {field}")

        # Validate detection
        detection = pattern["detection"]
        if "conditions" not in detection:
            raise ValueError("Pattern detection missing conditions")

        # Validate entry
        entry = pattern["entry"]
        if "signal" not in entry:
            raise ValueError("Pattern entry missing signal")

        # Validate exit
        exit_rules = pattern["exit"]
        if "stop_loss" not in exit_rules:
            raise ValueError("Pattern exit missing stop_loss")

        logger.debug(f"Pattern {pattern['name']} validated successfully")
