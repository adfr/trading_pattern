# IBKR AI Trading System

A rigorous, AI-powered algorithmic trading system for Interactive Brokers that uses Claude to generate, backtest, and deploy trading patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IBKR AI Trading System                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Claude    │───▶│  Strategy   │───▶│  Backtest   │───▶│ Production  │  │
│  │  AI Engine  │    │  Generator  │    │   Engine    │    │   Deploy    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        Core Infrastructure                              ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     ││
│  │  │  Config  │ │  Events  │ │ Database │ │  Logger  │ │   Risk   │     ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                     │             │
│         ▼                                                     ▼             │
│  ┌─────────────┐                                      ┌─────────────┐      │
│  │  IBKR API   │◀────────────────────────────────────▶│  Execution  │      │
│  │   Client    │                                      │   Engine    │      │
│  └─────────────┘                                      └─────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **AI-Powered Pattern Generation**: Claude generates trading patterns based on market analysis
- **Rigorous Backtesting**: Statistical validation with walk-forward optimization
- **IBKR Integration**: Native TWS/Gateway API integration for live trading
- **Risk Management**: Position sizing, stop losses, exposure limits
- **Event-Driven Architecture**: Scalable, real-time processing
- **Production-Ready**: Comprehensive logging, monitoring, and alerting

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure settings
cp config/config.example.yaml config/config.yaml
# Edit config/config.yaml with your settings
```

## Configuration

1. **IBKR Setup**: Configure TWS or IB Gateway
   - Enable API connections in TWS: Configure → API → Settings
   - Set port (default: 7497 for paper, 7496 for live)

2. **Claude API**: Set your Anthropic API key
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

3. **Configuration File**: Edit `config/config.yaml`

## Usage

### 1. Generate Trading Patterns with AI

```bash
# Generate new patterns using Claude
python -m src.cli generate --symbol QQQ --timeframe 1min

# Generate with specific market conditions
python -m src.cli generate --symbol SPY --conditions "high volatility"
```

### 2. Backtest Patterns

```bash
# Backtest all pending patterns
python -m src.cli backtest --all

# Backtest specific pattern
python -m src.cli backtest --pattern-id abc123

# Backtest with custom date range
python -m src.cli backtest --pattern-id abc123 --start 2024-01-01 --end 2024-06-01
```

### 3. Deploy to Production

```bash
# Deploy patterns that passed backtesting
python -m src.cli deploy --pattern-id abc123

# Start live trading
python -m src.cli trade --mode live

# Paper trading mode
python -m src.cli trade --mode paper
```

### 4. Monitor System

```bash
# View system status
python -m src.cli status

# View active positions
python -m src.cli positions

# View pattern performance
python -m src.cli performance
```

## Project Structure

```
trading_pattern/
├── config/
│   ├── config.yaml           # Main configuration
│   └── config.example.yaml   # Example configuration
├── src/
│   ├── __init__.py
│   ├── cli.py                # Command-line interface
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   ├── events.py         # Event system
│   │   ├── database.py       # SQLite database
│   │   ├── logger.py         # Logging setup
│   │   └── exceptions.py     # Custom exceptions
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ibkr_client.py    # IBKR API client
│   │   ├── market_data.py    # Market data handling
│   │   └── historical.py     # Historical data management
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── claude_client.py  # Claude API integration
│   │   ├── pattern_generator.py  # AI pattern generation
│   │   └── prompts.py        # Prompt templates
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── pattern.py        # Pattern definitions
│   │   ├── signal.py         # Signal generation
│   │   └── risk.py           # Risk management
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py         # Backtesting engine
│   │   ├── metrics.py        # Performance metrics
│   │   └── optimizer.py      # Walk-forward optimization
│   └── execution/
│       ├── __init__.py
│       ├── order_manager.py  # Order management
│       ├── position_tracker.py  # Position tracking
│       └── live_trader.py    # Live trading loop
├── tests/
│   ├── __init__.py
│   ├── test_backtest.py
│   ├── test_patterns.py
│   └── test_execution.py
├── data/                     # Data storage (gitignored)
├── logs/                     # Log files (gitignored)
├── requirements.txt
└── README.md
```

## Backtesting Criteria

Patterns must pass rigorous statistical tests before production deployment:

1. **Minimum Trades**: At least 30 trades for statistical significance
2. **Win Rate**: > 50% for mean-reversion, > 40% for trend-following
3. **Profit Factor**: > 1.5 (gross profit / gross loss)
4. **Sharpe Ratio**: > 1.0 annualized
5. **Max Drawdown**: < 15% of account equity
6. **Out-of-Sample Performance**: Must perform in walk-forward test
7. **Statistical Significance**: t-test p-value < 0.05

## Risk Management

- **Position Sizing**: Kelly criterion with fractional Kelly (0.25)
- **Stop Loss**: ATR-based dynamic stops
- **Max Position**: 5% of account per trade
- **Max Exposure**: 30% total account exposure
- **Daily Loss Limit**: 2% maximum daily loss

## License

MIT License - See LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss.
Past performance does not guarantee future results. Use at your own risk.
