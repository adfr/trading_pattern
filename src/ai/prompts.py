"""Prompt templates for Claude AI pattern generation."""


class PromptTemplates:
    """Collection of prompt templates for pattern generation."""

    SYSTEM_PROMPT = """You are an expert quantitative trading strategist specializing in technical analysis and pattern recognition. Your task is to generate rigorous, backtestable trading patterns.

When generating patterns, you must:
1. Define precise mathematical conditions for pattern recognition
2. Specify exact entry and exit rules with no ambiguity
3. Include risk management parameters
4. Consider market microstructure and realistic execution
5. Avoid overfitting by using robust, generalizable conditions

Your patterns must be implementable in code with these components:
- Detection conditions (using price, volume, and standard indicators)
- Entry signals with specific trigger conditions
- Exit rules including stop loss and take profit levels
- Position sizing recommendations

Always respond with valid JSON that can be parsed programmatically."""

    GENERATE_PATTERN = """Analyze the following market data and generate a trading pattern.

Market Context:
- Symbol: {symbol}
- Timeframe: {timeframe}
- Recent price action: {price_summary}
- Volatility (ATR): {atr}
- Volume profile: {volume_profile}
- Market regime: {market_regime}

Additional context:
{additional_context}

Generate a trading pattern in the following JSON format:
{{
    "name": "pattern_name_snake_case",
    "description": "Brief description of the pattern",
    "pattern_type": "bullish|bearish|neutral",
    "rationale": "Explanation of why this pattern should work",
    "detection": {{
        "conditions": [
            {{
                "indicator": "indicator_name",
                "operator": ">|<|>=|<=|==|crosses_above|crosses_below",
                "value": "number or indicator_name",
                "lookback": "number of periods"
            }}
        ],
        "min_conditions_met": "number of conditions that must be true"
    }},
    "entry": {{
        "signal": "description of entry trigger",
        "order_type": "market|limit",
        "limit_offset": "offset from current price if limit order"
    }},
    "exit": {{
        "stop_loss": {{
            "type": "atr|percent|fixed",
            "value": "multiplier or percentage"
        }},
        "take_profit": {{
            "type": "atr|percent|fixed|trailing",
            "value": "multiplier or percentage"
        }},
        "time_exit": {{
            "enabled": true|false,
            "bars": "number of bars before forced exit"
        }}
    }},
    "filters": {{
        "volume_min": "minimum volume threshold",
        "volatility_range": {{"min": "value", "max": "value"}},
        "trend_filter": "description of trend condition",
        "time_of_day": {{"start": "HH:MM", "end": "HH:MM"}}
    }},
    "risk": {{
        "max_position_pct": "max position as % of account",
        "risk_per_trade_pct": "max risk per trade as % of account"
    }}
}}

Ensure the pattern is:
1. Specific and unambiguous
2. Based on sound technical analysis principles
3. Suitable for the given timeframe
4. Not overly complex (max 5 detection conditions)"""

    ANALYZE_BACKTEST = """Analyze the following backtest results and provide recommendations.

Pattern: {pattern_name}
Backtest Period: {start_date} to {end_date}

Results:
- Total Trades: {total_trades}
- Win Rate: {win_rate:.2%}
- Profit Factor: {profit_factor:.2f}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Max Drawdown: {max_drawdown:.2%}
- Total Return: {total_return:.2%}

Trade Distribution:
{trade_distribution}

Provide analysis in JSON format:
{{
    "assessment": "pass|fail|needs_optimization",
    "strengths": ["list of pattern strengths"],
    "weaknesses": ["list of pattern weaknesses"],
    "recommendations": [
        {{
            "type": "entry|exit|filter|risk",
            "current": "current setting",
            "suggested": "suggested change",
            "rationale": "why this change might help"
        }}
    ],
    "confidence_score": 0.0-1.0,
    "market_conditions": "description of conditions where pattern works best"
}}"""

    OPTIMIZE_PATTERN = """Given the following pattern and its backtest results, suggest optimizations.

Current Pattern:
{pattern_json}

Backtest Results:
{backtest_results}

Failed Criteria:
{failed_criteria}

Generate an optimized version of the pattern that addresses the failed criteria while maintaining the core logic. Return the complete optimized pattern in the same JSON format as the original.

Focus on:
1. Adjusting thresholds that may be too tight or loose
2. Adding or modifying filters to improve win rate
3. Optimizing stop loss and take profit levels
4. Improving entry timing

Do not make changes that would lead to overfitting. Changes should be based on sound reasoning."""

    EXPLAIN_PATTERN = """Explain the following trading pattern in detail.

Pattern:
{pattern_json}

Provide a comprehensive explanation including:
1. What market conditions this pattern identifies
2. The logic behind each detection condition
3. Why the entry and exit rules are structured this way
4. What risks are associated with this pattern
5. When this pattern is most likely to succeed or fail

Make the explanation suitable for a trader who wants to understand the pattern deeply."""

    GENERATE_VARIATIONS = """Generate {num_variations} variations of the following base pattern.

Base Pattern:
{pattern_json}

Each variation should:
1. Maintain the core concept but adjust parameters
2. Target different market conditions (trending, ranging, volatile)
3. Have meaningfully different entry/exit rules
4. Be designed to capture similar opportunities with different risk/reward profiles

Return an array of pattern JSONs, each following the standard pattern format."""

    MARKET_ANALYSIS = """Analyze the current market conditions based on the following data.

Symbol: {symbol}
Timeframe: {timeframe}

Recent Price Data (last {num_bars} bars):
{price_data}

Technical Indicators:
{indicators}

Provide analysis in JSON format:
{{
    "regime": "trending_up|trending_down|ranging|volatile",
    "trend_strength": 0.0-1.0,
    "volatility_level": "low|medium|high",
    "key_levels": {{
        "support": [list of support levels],
        "resistance": [list of resistance levels]
    }},
    "pattern_suggestions": [
        "list of pattern types that might work well in current conditions"
    ],
    "risks": ["list of current market risks"],
    "opportunities": ["list of potential opportunities"]
}}"""
