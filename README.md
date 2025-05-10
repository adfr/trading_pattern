# TimeRAG for Financial Pattern Recognition

## Overview

TimeRAG (Time Series Retrieval Augmented Generation) is a framework for detecting patterns in financial time series data. This implementation focuses on minute-level data analysis for stock market patterns, specifically configured for QQQ ETF.

## Key Components

The implementation consists of four main parts:

### 1. Core TimeRAG Implementation (`timerag_minute_implementation.py`)

- Adapts the TimeRAG framework for intraday pattern recognition
- Uses pattern matching techniques including Dynamic Time Warping (DTW) and Euclidean distance
- Includes pre-defined intraday patterns (ascending triangles, bull flags, etc.)
- Handles minute-level data specific challenges (filtering for market hours, etc.)

### 2. Test Script for QQQ Data (`test_minute_timerag_qqq.py`)

- Simplified implementation focused on pattern detection in QQQ minute data
- Uses Euclidean distance for pattern comparison
- Includes visualization tools for detected patterns
- Implements CSV export for further analysis
- Supports custom pattern templates

### 3. Demonstration Script (`timerag_minute_demo.py`)

- Provides comprehensive analysis of detected patterns
- Includes simulation of live trading based on patterns
- Visualizes pattern distribution and trading results
- Compares pattern detection across multiple symbols

### 4. Live Trading System (`timerag_live_trading.py`)

- Implements a real-time trading framework
- Handles continuous data updates
- Executes trades based on pattern detection
- Includes position management and performance tracking

## Key Features for Financial Traders

### Intraday Pattern Recognition

- Detects common intraday patterns:
  - Ascending triangles
  - Bull flags
  - Double bottoms
  - Head and shoulders
  - Cup and handle
  - Custom patterns (ascending/descending wedges, consolidations)
- Calculates confidence scores for each match
- Works with various timeframes (1-minute, 5-minute, etc.)

### Trading Strategy Integration

- Automatically generates buy/sell signals based on pattern types
- Includes configurable confidence thresholds for trade entry
- Implements time-based exits for risk management

### Performance Analysis

- Tracks pattern detection with detailed statistics
- Visualizes pattern distribution and confidence levels
- Exports detection results to CSV for further analysis

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd trading_pattern
```

2. Create a virtual environment and activate it:
```bash
python -m venv timerag_env
source timerag_env/bin/activate  # On Windows: timerag_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The system can work with data from Yahoo Finance or pre-downloaded CSV files:

### Using Yahoo Finance Data
```bash
python test_minute_timerag_qqq.py --symbol QQQ --period 5d --interval 1m
```

### Using CSV Data
```bash
python test_minute_timerag_qqq.py --csv QQQ_minute_data.csv
```

## Usage Examples

### Basic Pattern Detection

```bash
python test_minute_timerag_qqq.py
```

### Customized Pattern Detection

```bash
python test_minute_timerag_qqq.py --threshold 3.0 --window 60 --step 5 --max-patterns 5
```

### Adding Custom Patterns

```bash
python test_minute_timerag_qqq.py --custom-pattern ascending_wedge
```

### Saving Results

```bash
python test_minute_timerag_qqq.py --save-csv --save-fig
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--csv` | CSV file with minute data | `QQQ_minute_data.csv` |
| `--symbol` | Ticker symbol to analyze | `QQQ` |
| `--period` | Period to download from Yahoo Finance | `5d` |
| `--interval` | Data interval | `1m` |
| `--window` | Pattern window size in minutes | `60` |
| `--step` | Step size for sliding window | `5` |
| `--threshold` | Pattern detection threshold (lower is stricter) | `5.0` |
| `--max-patterns` | Maximum patterns to visualize | `3` |
| `--save-csv` | Save detected patterns to CSV file | `False` |
| `--save-fig` | Save visualization to PNG file | `False` |
| `--custom-pattern` | Add a custom pattern type | `None` |

## For Developers

### Pattern Detection API

```python
from test_minute_timerag_qqq import load_data, create_patterns, detect_patterns

# Load data
data = load_data(csv_file='QQQ_minute_data.csv')

# Create pattern templates
patterns = create_patterns()

# Detect patterns
detected = detect_patterns(data, patterns, window_size=60, threshold=5.0)
```

### Creating Custom Patterns

```python
from test_minute_timerag_qqq import create_custom_pattern

# Create a custom pattern template
wedge_pattern = create_custom_pattern("ascending_wedge", length=60)
```

## Technical Details

- Pattern matching is performed using Euclidean distance between normalized price sequences
- Time series data is normalized to range [0-1] for comparison
- Pattern templates are randomly generated with slight variations to improve robustness
- Confidence scores are calculated as 1 - (distance / threshold)

This implementation demonstrates how the TimeRAG framework can be adapted for financial time series pattern recognition at the minute level, combining the power of time series pattern matching with real-time data processing for trading applications.