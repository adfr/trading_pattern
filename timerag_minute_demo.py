"""
TimeRAG Minute-Level Pattern Detection Demo

This script demonstrates how to use TimeRAG for intraday pattern detection
with minute-level data, including simulating real-time pattern detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import TimeRAG implementation
from timerag_minute_implementation import (
    TimeRAG, create_intraday_patterns, detect_intraday_patterns,
    visualize_intraday_patterns, fetch_minute_data, filter_market_hours
)


def analyze_pattern_statistics(detected_patterns):
    """
    Analyze the statistics of detected patterns
    
    Args:
        detected_patterns: List of detected patterns
        
    Returns:
        Dictionary with pattern statistics
    """
    if not detected_patterns:
        return {"error": "No patterns detected"}
    
    # Count occurrences of each pattern
    pattern_counts = {}
    for pattern in detected_patterns:
        pattern_type = pattern['pattern']
        if pattern_type in pattern_counts:
            pattern_counts[pattern_type] += 1
        else:
            pattern_counts[pattern_type] = 1
    
    # Calculate average confidence by pattern type
    pattern_confidence = {}
    for pattern in detected_patterns:
        pattern_type = pattern['pattern']
        if pattern_type in pattern_confidence:
            pattern_confidence[pattern_type].append(pattern['confidence'])
        else:
            pattern_confidence[pattern_type] = [pattern['confidence']]
    
    avg_confidence = {k: sum(v)/len(v) for k, v in pattern_confidence.items()}
    
    # Get time distribution of patterns
    pattern_times = [pattern['time'].hour * 60 + pattern['time'].minute for pattern in detected_patterns]
    
    # Calculate pattern statistics
    stats = {
        'total_patterns': len(detected_patterns),
        'pattern_counts': pattern_counts,
        'avg_confidence': avg_confidence,
        'most_common_pattern': max(pattern_counts.items(), key=lambda x: x[1])[0],
        'highest_confidence_pattern': max(avg_confidence.items(), key=lambda x: x[1])[0],
        'earliest_pattern_time': min(pattern['time'] for pattern in detected_patterns),
        'latest_pattern_time': max(pattern['time'] for pattern in detected_patterns),
        'avg_pattern_per_hour': len(detected_patterns) / ((max(pattern_times) - min(pattern_times)) / 60) if len(pattern_times) > 1 else 0
    }
    
    return stats


def visualize_pattern_distribution(detected_patterns):
    """
    Visualize the distribution of detected patterns
    
    Args:
        detected_patterns: List of detected patterns
    """
    if not detected_patterns:
        print("No patterns to visualize")
        return
    
    # Extract pattern types and confidence
    pattern_types = [pattern['pattern'] for pattern in detected_patterns]
    confidences = [pattern['confidence'] for pattern in detected_patterns]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Pattern count bar chart
    pattern_counts = pd.Series(pattern_types).value_counts()
    ax1.bar(pattern_counts.index, pattern_counts.values)
    ax1.set_title('Pattern Type Distribution')
    ax1.set_xlabel('Pattern Type')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Confidence box plot by pattern type
    data = []
    labels = []
    for pattern_type in pattern_counts.index:
        pattern_confidences = [p['confidence'] for p in detected_patterns if p['pattern'] == pattern_type]
        data.append(pattern_confidences)
        labels.append(pattern_type)
    
    ax2.boxplot(data, labels=labels)
    ax2.set_title('Pattern Confidence Distribution')
    ax2.set_xlabel('Pattern Type')
    ax2.set_ylabel('Confidence')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def simulate_live_trading(timerag, data, window_size=60, threshold=1.5, delay=0.1):
    """
    Simulate live trading with TimeRAG pattern detection
    
    Args:
        timerag: Initialized TimeRAG instance
        data: Historical minute data to simulate with
        window_size: Pattern window size
        threshold: Detection threshold
        delay: Simulation delay in seconds
    """
    print("\nSimulating live trading with TimeRAG pattern detection...")
    print("=" * 80)
    
    # Get minute data
    close_data = data['Close']
    
    # Initialize trading state
    position = None
    entry_price = 0
    pnl = 0
    trades = []
    detected_patterns_live = []
    
    # Create trading rules based on patterns
    bullish_patterns = ['momentum_burst', 'intraday_bull_flag', 'consolidation_breakout', 'v_shape_reversal', 'double_bottom', 'vwap_bounce', 'closing_drive']
    bearish_patterns = ['intraday_bear_flag', 'consolidation_breakdown', 'vwap_rejection']
    
    # Number of minutes to simulate looking ahead after pattern detection
    look_ahead = 20  # Look ahead 20 minutes after pattern detection
    
    # Start simulation
    print(f"Starting simulation with {len(data) - window_size} data points")
    
    # Use tqdm for progress bar
    for i in tqdm(range(len(close_data) - window_size)):
        # Get current window
        current_time = data.index[i + window_size - 1]
        current_window = close_data.iloc[i:i+window_size].values
        current_price = close_data.iloc[i + window_size - 1]
        
        # Detect patterns in current window
        similar_patterns = timerag.retrieve_similar_sequences(current_window)
        
        # Check if any pattern is detected
        if similar_patterns and similar_patterns[0]['distance'] < threshold:
            pattern_type = similar_patterns[0]['label'].split('_var')[0].split('_stretch')[0]
            confidence = 1.0 - (similar_patterns[0]['distance'] / threshold)
            
            # Record pattern detection
            pattern_info = {
                'time': current_time,
                'pattern': pattern_type,
                'confidence': confidence,
                'price': current_price,
                'distance': similar_patterns[0]['distance']
            }
            detected_patterns_live.append(pattern_info)
            
            # Print pattern detection
            print(f"\n[{current_time}] Detected {pattern_type} pattern with {confidence:.2f} confidence at price {current_price:.2f}")
            
            # Trading logic
            if pattern_type in bullish_patterns and confidence > 0.7:
                if position is None:
                    # Enter long position
                    entry_price = current_price
                    position = 'LONG'
                    print(f"[{current_time}] LONG entry at {entry_price:.2f}")
                elif position == 'SHORT':
                    # Exit short position
                    exit_price = current_price
                    trade_pnl = entry_price - exit_price
                    pnl += trade_pnl
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': trade_pnl,
                        'position': position,
                        'pattern': last_pattern
                    })
                    print(f"[{current_time}] SHORT exit at {exit_price:.2f}, Trade PnL: {trade_pnl:.2f}")
                    
                    # Enter long position
                    entry_price = current_price
                    position = 'LONG'
                    print(f"[{current_time}] LONG entry at {entry_price:.2f}")
            
            elif pattern_type in bearish_patterns and confidence > 0.7:
                if position is None:
                    # Enter short position
                    entry_price = current_price
                    position = 'SHORT'
                    print(f"[{current_time}] SHORT entry at {entry_price:.2f}")
                elif position == 'LONG':
                    # Exit long position
                    exit_price = current_price
                    trade_pnl = exit_price - entry_price
                    pnl += trade_pnl
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': trade_pnl,
                        'position': position,
                        'pattern': last_pattern
                    })
                    print(f"[{current_time}] LONG exit at {exit_price:.2f}, Trade PnL: {trade_pnl:.2f}")
                    
                    # Enter short position
                    entry_price = current_price
                    position = 'SHORT'
                    print(f"[{current_time}] SHORT entry at {entry_price:.2f}")
            
            # Save entry time and pattern
            entry_time = current_time
            last_pattern = pattern_type
        
        # Check for time-based exit (after look_ahead minutes)
        if position and i >= window_size + look_ahead:
            look_back_time = data.index[i - look_ahead]
            
            # Find if we have any open trade from look_back_time
            open_trade_found = False
            for trade in trades:
                if trade['entry_time'] == look_back_time:
                    open_trade_found = True
                    break
            
            if not open_trade_found and entry_time == look_back_time:
                exit_price = current_price
                if position == 'LONG':
                    trade_pnl = exit_price - entry_price
                else:  # SHORT
                    trade_pnl = entry_price - exit_price
                
                pnl += trade_pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': trade_pnl,
                    'position': position,
                    'pattern': last_pattern,
                    'exit_reason': 'time_based'
                })
                print(f"[{current_time}] {position} exit at {exit_price:.2f} after {look_ahead} minutes, Trade PnL: {trade_pnl:.2f}")
                position = None
        
        # Add a small delay to simulate real-time
        time.sleep(delay)
    
    # Close any open position at the end
    if position:
        exit_price = close_data.iloc[-1]
        if position == 'LONG':
            trade_pnl = exit_price - entry_price
        else:  # SHORT
            trade_pnl = entry_price - exit_price
        
        pnl += trade_pnl
        trades.append({
            'entry_time': entry_time,
            'exit_time': data.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': trade_pnl,
            'position': position,
            'pattern': last_pattern,
            'exit_reason': 'end_of_data'
        })
        print(f"[{data.index[-1]}] {position} exit at {exit_price:.2f} (end of data), Trade PnL: {trade_pnl:.2f}")
    
    # Print trading summary
    print("\nTrading Summary:")
    print(f"Total PnL: {pnl:.2f}")
    print(f"Number of trades: {len(trades)}")
    
    if trades:
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Average win: {avg_win:.2f}")
        print(f"Average loss: {avg_loss:.2f}")
        print(f"Profit factor: {abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades)) if sum(t['pnl'] for t in losing_trades) else float('inf'):.2f}")
    
    return detected_patterns_live, trades, pnl


def visualize_trading_results(data, trades, detected_patterns):
    """
    Visualize trading results
    
    Args:
        data: Price data
        trades: List of trade dictionaries
        detected_patterns: List of detected patterns
    """
    if not trades:
        print("No trades to visualize")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price data
    ax.plot(data.index, data['Close'], label='Close Price', color='black', alpha=0.7)
    
    # Highlight patterns
    for pattern in detected_patterns:
        pattern_time = pattern['time']
        pattern_type = pattern['pattern']
        pattern_price = pattern['price']
        is_bullish = pattern_type in ['momentum_burst', 'intraday_bull_flag', 'consolidation_breakout', 
                                    'v_shape_reversal', 'double_bottom', 'vwap_bounce', 'closing_drive']
        
        color = 'green' if is_bullish else 'red'
        marker = '^' if is_bullish else 'v'
        
        ax.scatter(pattern_time, pattern_price, color=color, marker=marker, s=100, alpha=0.7)
        ax.annotate(pattern_type, (pattern_time, pattern_price), 
                    xytext=(0, 10 if is_bullish else -10), 
                    textcoords='offset points',
                    fontsize=8, rotation=45, ha='center')
    
    # Plot trades
    for trade in trades:
        entry_time = trade['entry_time']
        exit_time = trade['exit_time']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        position = trade['position']
        pnl = trade['pnl']
        
        # Plot entry point
        entry_color = 'green' if position == 'LONG' else 'red'
        ax.scatter(entry_time, entry_price, color=entry_color, marker='o', s=100)
        
        # Plot exit point
        exit_color = 'blue'
        ax.scatter(exit_time, exit_price, color=exit_color, marker='x', s=100)
        
        # Connect entry and exit with a line
        ax.plot([entry_time, exit_time], [entry_price, exit_price], 
                color='green' if pnl > 0 else 'red', 
                linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Customize plot
    ax.set_title('TimeRAG Trading Simulation Results')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(['Close Price', 'Pattern', 'Entry', 'Exit'])
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show date and time
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Create trade performance chart
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot PnL per trade
    pnls = [trade['pnl'] for trade in trades]
    axs[0].bar(range(len(pnls)), pnls, color=['green' if pnl > 0 else 'red' for pnl in pnls])
    axs[0].set_title('PnL per Trade')
    axs[0].set_xlabel('Trade Number')
    axs[0].set_ylabel('PnL')
    axs[0].grid(True, alpha=0.3)
    
    # Plot cumulative PnL
    cumulative_pnl = np.cumsum(pnls)
    axs[1].plot(cumulative_pnl, marker='o')
    axs[1].set_title('Cumulative PnL')
    axs[1].set_xlabel('Trade Number')
    axs[1].set_ylabel('Cumulative PnL')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_multiple_symbols(symbols=['QQQ', 'SPY', 'AAPL'], interval='1m', period='5d', window_size=60):
    """
    Compare pattern detection across multiple symbols
    
    Args:
        symbols: List of symbols to analyze
        interval: Data interval
        period: Time period to fetch
        window_size: Pattern window size
    """
    results = {}
    
    # Initialize TimeRAG
    timerag = TimeRAG(window_size=window_size, step_size=5, n_clusters=10, top_k=3)
    
    # Create intraday patterns
    patterns = create_intraday_patterns(minutes=window_size)
    
    # Build knowledge base
    timerag.build_knowledge_base(patterns)
    
    # Loop through symbols
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        
        try:
            # Fetch minute data
            minute_data = fetch_minute_data(symbol=symbol, period=period, interval=interval)
            
            # Filter to market hours
            market_data = filter_market_hours(minute_data)
            
            # Detect patterns
            detected_patterns = detect_intraday_patterns(
                timerag, market_data, window_size=window_size, threshold=1.5
            )
            
            # Calculate statistics
            stats = analyze_pattern_statistics(detected_patterns)
            
            # Store results
            results[symbol] = {
                'data': market_data,
                'patterns': detected_patterns,
                'stats': stats
            }
            
            print(f"Detected {len(detected_patterns)} patterns for {symbol}")
            if detected_patterns:
                print(f"Most common pattern: {stats['most_common_pattern']}")
                print(f"Highest confidence pattern: {stats['highest_confidence_pattern']}")
        
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Compare results
    if results:
        # Create comparison table
        comparison = {
            'Symbol': [],
            'Total Patterns': [],
            'Most Common Pattern': [],
            'Highest Confidence Pattern': [],
            'Avg. Confidence': []
        }
        
        for symbol, result in results.items():
            if 'stats' in result and 'error' not in result['stats']:
                stats = result['stats']
                comparison['Symbol'].append(symbol)
                comparison['Total Patterns'].append(stats['total_patterns'])
                comparison['Most Common Pattern'].append(stats['most_common_pattern'])
                comparison['Highest Confidence Pattern'].append(stats['highest_confidence_pattern'])
                comparison['Avg. Confidence'].append(np.mean(list(stats['avg_confidence'].values())))
        
        # Print comparison table
        if comparison['Symbol']:
            comparison_df = pd.DataFrame(comparison)
            print("\nPattern Detection Comparison:")
            print(comparison_df)
        
        # Visualize pattern distribution for each symbol
        for symbol, result in results.items():
            if 'patterns' in result and result['patterns']:
                print(f"\nPattern distribution for {symbol}:")
                visualize_pattern_distribution(result['patterns'])
    
    return results


def run_intraday_demo():
    """Run the complete intraday pattern detection demo"""
    print("\n" + "="*80)
    print("TimeRAG Intraday Pattern Detection Demo")
    print("="*80 + "\n")
    
    # 1. Initialize TimeRAG
    print("Initializing TimeRAG for minute-level pattern detection...")
    window_size = 60  # 60-minute window
    timerag = TimeRAG(window_size=window_size, step_size=5, n_clusters=10, top_k=3)
    
    # 2. Create intraday patterns
    print("Creating intraday patterns for knowledge base...")
    patterns = create_intraday_patterns(minutes=window_size)
    
    # Display pattern names
    print("Intraday patterns in knowledge base:")
    for pattern_name in patterns.keys():
        print(f"- {pattern_name}")
    
    # 3. Build knowledge base
    timerag.build_knowledge_base(patterns)
    
    # 4. Fetch minute data
    symbol = 'QQQ'
    print(f"\nFetching minute data for {symbol}...")
    
    try:
        minute_data = fetch_minute_data(symbol=symbol, period='5d', interval='1m')
        
        # 5. Filter to market hours
        market_data = filter_market_hours(minute_data)
        
        # 6. Detect patterns
        print("\nDetecting patterns...")
        detected_patterns = detect_intraday_patterns(
            timerag, market_data, window_size=window_size, threshold=1.5
        )
        
        # 7. Print results
        print(f"\nDetected {len(detected_patterns)} patterns")
        
        if detected_patterns:
            # Sort by time
            detected_patterns.sort(key=lambda x: x['time'])
            
            # Print first 5 patterns
            print("\nFirst 5 detected patterns:")
            for i, pattern in enumerate(detected_patterns[:5]):
                print(f"{i+1}. Time: {pattern['time']}")
                print(f"   Pattern: {pattern['pattern']}")
                print(f"   Confidence: {pattern['confidence']:.2f}")
                print(f"   DTW Distance: {pattern['matches'][0]['distance']:.2f}")
                print()
            
            # 8. Calculate statistics
            stats = analyze_pattern_statistics(detected_patterns)
            
            print("\nPattern Statistics:")
            print(f"Total patterns: {stats['total_patterns']}")
            print(f"Pattern counts: {stats['pattern_counts']}")
            print(f"Most common pattern: {stats['most_common_pattern']}")
            print(f"Highest confidence pattern: {stats['highest_confidence_pattern']}")
            print(f"Average confidence: {stats['avg_confidence']}")
            print(f"Earliest pattern time: {stats['earliest_pattern_time']}")
            print(f"Latest pattern time: {stats['latest_pattern_time']}")
            print(f"Average patterns per hour: {stats['avg_pattern_per_hour']:.2f}")
            
            # 9. Visualize patterns
            print("\nVisualizing detected patterns...")
            visualize_intraday_patterns(market_data, detected_patterns[:5], window_size=window_size)
            
            # 10. Visualize pattern distribution
            print("\nVisualizing pattern distribution...")
            visualize_pattern_distribution(detected_patterns)
            
            # 11. Simulate live trading
            print("\nSimulating live trading with detected patterns...")
            detected_patterns_live, trades, pnl = simulate_live_trading(
                timerag, market_data, window_size=window_size, threshold=1.5, delay=0.001
            )
            
            # 12. Visualize trading results
            if trades:
                print("\nVisualizing trading results...")
                visualize_trading_results(market_data, trades, detected_patterns_live)
        
        # 13. Compare multiple symbols
        print("\nComparing multiple symbols...")
        symbol_results = compare_multiple_symbols(
            symbols=['QQQ', 'SPY', 'AAPL', 'MSFT'], 
            interval='1m', 
            period='1d',  # Use 1d to make it faster
            window_size=window_size
        )
        
    except Exception as e:
        print(f"Error in intraday demo: {e}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_intraday_demo()