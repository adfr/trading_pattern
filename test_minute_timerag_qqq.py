#!/usr/bin/env python
"""
Minute-Level TimeRAG Test for QQQ
A simplified implementation to detect financial patterns in minute data
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse
import os

def normalize(sequence):
    """Normalize a sequence to range [0, 1]"""
    sequence = np.array(sequence, dtype=float).flatten()  # Ensure 1D array of floats
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    if max_val == min_val:
        return np.zeros_like(sequence)
    return (sequence - min_val) / (max_val - min_val)

def create_patterns():
    """Create common intraday chart patterns"""
    patterns = {}
    
    # Ascending Triangle - corrected pattern
    triangle = np.zeros(60)
    
    # Flat resistance line at 0.7 that gets tested multiple times
    resistance_level = 0.7
    
    # Initial approach to resistance
    triangle[:10] = np.linspace(0.3, resistance_level - 0.02, 10) + np.random.normal(0, 0.01, 10)
    
    # First test of resistance and pullback
    triangle[10:15] = resistance_level - 0.026 + np.random.normal(0, 0.013, 5)
    triangle[15:20] = np.linspace(resistance_level - 0.026, 0.435, 5) + np.random.normal(0, 0.013, 5)
    
    # Second approach with higher low
    triangle[20:25] = np.linspace(0.435, resistance_level - 0.013, 5) + np.random.normal(0, 0.013, 5)
    triangle[25:30] = np.linspace(resistance_level - 0.013, 0.515, 5) + np.random.normal(0, 0.013, 5)
    
    # Third approach with higher low
    triangle[30:35] = np.linspace(0.515, resistance_level, 5) + np.random.normal(0, 0.013, 5) 
    triangle[35:40] = np.linspace(resistance_level, 0.53, 5) + np.random.normal(0, 0.013, 5)
    
    # Final approach 
    triangle[40:52] = np.linspace(0.53, resistance_level, 12) + np.random.normal(0, 0.013, 12)
    
    # Breakout (using just 8 time stamps instead of 10)
    triangle[52:60] = np.linspace(resistance_level, 0.97, 8) + np.random.normal(0, 0.026, 8)  # Breakout
    
    patterns["ascending_triangle"] = triangle
    
    # Bull Flag - updated with fewer, more pronounced oscillations
    flag = np.zeros(60)
    
    # Strong upward move (flagpole)
    flag[:20] = np.linspace(0.2, 0.9, 20) + np.random.normal(0, 0.02, 20)
    
    # Flag consolidation with 4 oscillations (32 periods / 4 oscillations = 8 periods per cycle)
    oscillation_periods = 4
    periods_per_oscillation = 8
    
    for osc in range(oscillation_periods):
        start_idx = 20 + osc * periods_per_oscillation
        mid_idx = start_idx + periods_per_oscillation // 2
        end_idx = start_idx + periods_per_oscillation
        
        # Upper part of oscillation (first half)
        flag[start_idx:mid_idx] = (0.9 - 0.02 * osc  # Slight downward drift for the flag
                                   + np.linspace(0, -0.05, periods_per_oscillation // 2)  # Downslope within oscillation
                                   + np.random.normal(0, 0.01, periods_per_oscillation // 2))  # Add noise
        
        # Lower part of oscillation (second half)
        flag[mid_idx:end_idx] = (0.85 - 0.02 * osc  # Continues the downward drift
                                 + np.linspace(-0.05, 0, periods_per_oscillation // 2)  # Upslope within oscillation
                                 + np.random.normal(0, 0.01, periods_per_oscillation // 2))  # Add noise
    
    # Add a breakout at the end (8 time stamps)
    flag[52:60] = np.linspace(0.83, 1.0, 8) + np.random.normal(0, 0.02, 8)  # Breakout after flag
    
    patterns["bull_flag"] = flag
    
    # Double Bottom
    double = np.zeros(60)
    double[:20] = 0.5 - 0.4 * np.sin(np.linspace(0, np.pi, 20)) + np.random.normal(0, 0.01, 20)
    double[20:30] = np.linspace(0.5, 0.4, 10) + np.random.normal(0, 0.01, 10)
    double[30:52] = 0.4 - 0.3 * np.sin(np.linspace(0, np.pi, 22)) + np.random.normal(0, 0.01, 22)
    
    # Breakout (8 time stamps)
    double[52:60] = np.linspace(0.4, 0.7, 8) + np.random.normal(0, 0.01, 8)
    
    patterns["double_bottom"] = double
    
    # Head and Shoulders
    head_shoulders = np.zeros(60)
    head_shoulders[:15] = 0.3 + 0.2 * np.sin(np.linspace(0, np.pi, 15)) + np.random.normal(0, 0.01, 15)  # Left shoulder
    head_shoulders[15:30] = 0.3 + 0.4 * np.sin(np.linspace(0, np.pi, 15)) + np.random.normal(0, 0.01, 15)  # Head
    head_shoulders[30:45] = 0.3 + 0.2 * np.sin(np.linspace(0, np.pi, 15)) + np.random.normal(0, 0.01, 15)  # Right shoulder
    
    # Breakdown (8 time stamps)
    head_shoulders[52:60] = 0.3 - 0.2 * np.linspace(0, 1, 8) + np.random.normal(0, 0.01, 8)  # Breakdown
    
    # Add neckline test
    head_shoulders[45:52] = 0.3 + np.random.normal(0, 0.01, 7)  # Testing neckline
    
    patterns["head_and_shoulders"] = head_shoulders
    
    # Cup and Handle
    cup = np.zeros(60)
    cup[:30] = 0.5 - 0.4 * np.sin(np.linspace(0, np.pi, 30)) + np.random.normal(0, 0.01, 30)  # Cup
    cup[30:52] = 0.5 - 0.1 * np.sin(np.linspace(0, np.pi, 22)) + np.random.normal(0, 0.01, 22)  # Handle
    
    # Breakout (8 time stamps)
    cup[52:60] = np.linspace(0.5, 0.8, 8) + np.random.normal(0, 0.01, 8)  # Breakout
    
    patterns["cup_and_handle"] = cup
    
    return patterns

def create_custom_pattern(pattern_type, length=60):
    """Create a custom pattern based on type"""
    if pattern_type == "ascending_wedge":
        pattern = np.zeros(length)
        # Resistance line with decreasing slope
        resistance = np.linspace(0.8, 0.6, length)
        # Support line with increasing slope
        support = np.linspace(0.2, 0.5, length)
        # Price oscillates between support and resistance, converging
        for i in range(length):
            if i % 4 < 2:
                pattern[i] = resistance[i] - np.random.uniform(0, 0.05)
            else:
                pattern[i] = support[i] + np.random.uniform(0, 0.05)
        return pattern
        
    elif pattern_type == "descending_wedge":
        pattern = np.zeros(length)
        # Resistance line with decreasing slope
        resistance = np.linspace(0.8, 0.5, length)
        # Support line with decreasing slope (steeper)
        support = np.linspace(0.6, 0.2, length)
        # Price oscillates between support and resistance, converging
        for i in range(length):
            if i % 4 < 2:
                pattern[i] = resistance[i] - np.random.uniform(0, 0.05)
            else:
                pattern[i] = support[i] + np.random.uniform(0, 0.05)
        return pattern
        
    elif pattern_type == "sideways_consolidation":
        pattern = 0.5 + np.random.normal(0, 0.05, length)
        return pattern
        
    else:
        print(f"Pattern type '{pattern_type}' not recognized, returning random noise")
        return np.random.normal(0.5, 0.1, length)

def load_data(csv_file=None, symbol='QQQ', period='5d', interval='1m'):
    """Load data from CSV or fetch from Yahoo Finance"""
    if csv_file and os.path.exists(csv_file):
        try:
            # Read CSV file, skipping the second row (which contains 'QQQ' as headers)
            data = pd.read_csv(csv_file, skiprows=[1])
            
            # Convert datetime column to datetime format
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
                data.set_index('Datetime', inplace=True)
            
            # Ensure all numeric columns are properly converted to float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            print(f"Loaded {len(data)} rows from {csv_file}")
            return data
        except Exception as e:
            print(f"Error loading CSV: {e}")
    
    try:
        print(f"Downloading {symbol} data ({period}, {interval})...")
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save downloaded data to the data directory
        csv_path = f"data/{symbol}_{interval}_data.csv"
        data.to_csv(csv_path)
        print(f"Downloaded {len(data)} rows of {symbol} data and saved to {csv_path}")
        
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

def filter_market_hours(data):
    """Filter data to market hours (9:30 AM to 4:00 PM)"""
    if not isinstance(data.index, pd.DatetimeIndex):
        print("Data doesn't have a datetime index, skipping market hours filtering")
        return data
        
    market_open = pd.to_datetime('09:30').time()
    market_close = pd.to_datetime('16:00').time()
    
    try:
        filtered_data = data.between_time(market_open, market_close)
        print(f"Filtered from {len(data)} to {len(filtered_data)} data points (market hours only)")
        return filtered_data
    except Exception as e:
        print(f"Error filtering market hours: {e}, using full dataset")
        return data

def detect_patterns(data, patterns, window_size=60, threshold=5.0, step_size=5):
    """Detect patterns in time series data"""
    detected = []
    
    # Get the prices as a 1D array
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        # Convert to numeric, replacing non-numeric values with NaN
        prices = pd.to_numeric(data['Close'], errors='coerce').ffill().values.flatten()
    else:
        prices = np.array(data, dtype=float).flatten()
    
    print(f"Analyzing {len(prices) - window_size + 1} possible windows...")
    
    # Process each window
    for i in range(0, len(prices) - window_size + 1, step_size):  # Use configurable step size
        window = prices[i:i+window_size]
        
        # Skip windows with any NaN values
        if np.isnan(window).any():
            continue
            
        normalized_window = normalize(window)
        
        # Compare with each pattern
        best_match = None
        best_score = float('inf')
        
        for name, pattern in patterns.items():
            # Ensure both sequences are 1D arrays of floats
            pattern_flat = np.array(pattern, dtype=float).flatten()
            normalized_window_flat = normalized_window.flatten()
            
            # Ensure equal length
            min_len = min(len(normalized_window_flat), len(pattern_flat))
            window_use = normalized_window_flat[:min_len]
            pattern_use = pattern_flat[:min_len]
            
            try:
                # Use simple Euclidean distance instead of DTW
                distance = euclidean(window_use, pattern_use)
                
                if distance < best_score:
                    best_score = distance
                    best_match = name
            except Exception as e:
                print(f"Error calculating distance: {e}")
                continue
        
        # Record if match is good enough
        if best_score < threshold:
            if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex):
                timestamp = data.index[i]
            else:
                timestamp = i
                
            detected.append({
                'position': i,
                'time': timestamp,
                'pattern': best_match,
                'confidence': 1.0 - (best_score / threshold),
                'distance': best_score
            })
    
    return detected

def save_patterns_to_csv(detected_patterns, filename='data/detected_patterns.csv'):
    """Save detected patterns to CSV file"""
    if not detected_patterns:
        print("No patterns to save")
        return
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Create a DataFrame from the detected patterns
    df = pd.DataFrame(detected_patterns)
    
    # Convert position to int
    if 'position' in df.columns:
        df['position'] = df['position'].astype(int)
    
    # Format confidence and distance
    if 'confidence' in df.columns:
        df['confidence'] = df['confidence'].round(4)
    if 'distance' in df.columns:
        df['distance'] = df['distance'].round(4)
        
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} patterns to {filename}")

def save_individual_pattern_plots(data, patterns, detected, window_size=60, future_periods=60, confidence_threshold=0.6):
    """Save individual plot for each high-confidence pattern with future price movement"""
    # Check if we have any patterns to process
    if not detected:
        print("No patterns detected to plot")
        return
    
    # Create images directory if it doesn't exist
    image_dir = "images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        print(f"Created directory: {image_dir}")
    
    # Filter for high confidence patterns
    high_confidence_patterns = [p for p in detected if p['confidence'] >= confidence_threshold]
    print(f"Saving plots for {len(high_confidence_patterns)} patterns with confidence >= {confidence_threshold}")
    
    # Keep track of patterns processed
    patterns_processed = 0
    patterns_plotted = 0
    
    for i, pattern in enumerate(high_confidence_patterns):
        # Extract pattern information
        pos = pattern['position']
        confidence = pattern['confidence']
        pattern_name = pattern['pattern']
        pattern_time = pattern['time']
        
        # Define the pattern window and future window
        pattern_end_pos = pos + window_size
        future_end_pos = min(pattern_end_pos + future_periods, len(data))
        
        # Skip if we don't have enough future data
        if future_end_pos <= pattern_end_pos:
            print(f"  Skipping pattern at {pattern_time} - not enough future data")
            continue
        
        patterns_processed += 1
        
        # Extract the data for the pattern window and future window
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            pattern_data = data.iloc[pos:pattern_end_pos]
            full_data = data.iloc[pos:future_end_pos]
            
            # Check if the entire future window is within trading hours
            if isinstance(full_data.index, pd.DatetimeIndex):
                # Get the last timestamp of the pattern and the full window
                pattern_end_time = pattern_data.index[-1]
                full_window_end_time = full_data.index[-1]
                
                # Check if we're crossing days
                if pattern_end_time.date() != full_window_end_time.date():
                    print(f"  Skipping pattern at {pattern_time} - future window crosses days")
                    continue
                
                # Check if we're potentially crossing beyond market hours
                market_close = pd.to_datetime('16:00').time()
                if full_window_end_time.time() > market_close:
                    print(f"  Skipping pattern at {pattern_time} - future window extends beyond market hours")
                    continue
                
                # Check if there are any gaps in the data (might indicate non-trading periods)
                time_diffs = np.diff(full_data.index.astype(np.int64)) / 1e9 / 60  # Convert to minutes
                if np.any(time_diffs > 5):  # If any gap is more than 5 minutes
                    print(f"  Skipping pattern at {pattern_time} - gaps in data suggest non-continuous trading")
                    continue
            
            # Create the figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
            
            # Plot the pattern template in the first subplot
            pattern_template = patterns[pattern_name]
            ax1.plot(np.arange(len(pattern_template)), pattern_template, 'b-', label='Pattern Template')
            ax1.set_title(f"Pattern Template: {pattern_name}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot the pattern window and future data
            ax2.plot(full_data.index, full_data['Close'], 'b-', label='Price Data')
            
            # Highlight the pattern window
            ax2.axvspan(pattern_data.index[0], pattern_data.index[-1], 
                       color='green', alpha=0.2, label='Pattern Window')
            
            # Highlight the future window
            ax2.axvspan(pattern_data.index[-1], full_data.index[-1], 
                       color='orange', alpha=0.2, label='Future Window')
            
            # Format x-axis for dates
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax2.tick_params(axis='x', rotation=45)
            
            # Overlay the normalized pattern
            min_price = pattern_data['Close'].min()
            max_price = pattern_data['Close'].max()
            
            # Ensure pattern_template is the right length
            template_to_use = pattern_template
            if len(pattern_template) > len(pattern_data):
                template_to_use = pattern_template[:len(pattern_data)]
            elif len(pattern_template) < len(pattern_data):
                template_to_use = np.pad(pattern_template, 
                                       (0, len(pattern_data) - len(pattern_template)), 
                                       'constant', 
                                       constant_values=pattern_template[-1])
                
            scaled_pattern = min_price + normalize(template_to_use) * (max_price - min_price)
            ax2.plot(pattern_data.index, scaled_pattern, 'r--', label='Matched Pattern')
            
            # Add annotations
            ax2.set_title(f"Pattern: {pattern_name} at {pattern_time}\nConfidence: {confidence:.2f}")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add future price change annotation
            future_price_change = (full_data['Close'].iloc[-1] / pattern_data['Close'].iloc[-1] - 1) * 100
            future_max_change = (full_data['Close'].max() / pattern_data['Close'].iloc[-1] - 1) * 100
            
            # Add text box with future performance
            textstr = f"Future change: {future_price_change:.2f}%\nMax future change: {future_max_change:.2f}%"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Save the figure
            timestamp_str = pattern_time.strftime('%Y%m%d_%H%M')
            filename = f"{image_dir}/{pattern_name}_{timestamp_str}_conf{confidence:.2f}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=150)
            plt.close(fig)
            
            patterns_plotted += 1
            print(f"  Saved plot {patterns_plotted}/{len(high_confidence_patterns)}: {filename}")
        else:
            print(f"  Skipping pattern at position {pos} - data format not supported")
    
    print(f"Processed {patterns_processed} patterns, saved {patterns_plotted} plots to {image_dir}/")
    if patterns_processed > patterns_plotted:
        print(f"Skipped {patterns_processed - patterns_plotted} patterns due to trading hours constraints")

def visualize_patterns(data, patterns, detected, window_size=60, max_patterns=3, save_fig=False):
    """Visualize detected patterns"""
    if not detected:
        print("No patterns detected to visualize")
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'])
        plt.title('QQQ Price (No Patterns Detected)')
        if save_fig:
            plt.savefig('no_patterns.png')
        plt.show()
        return
    
    # Sort by confidence
    top_patterns = sorted(detected, key=lambda x: x['confidence'], reverse=True)
    if len(top_patterns) > max_patterns:
        top_patterns = top_patterns[:max_patterns]
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(top_patterns) + 1, figsize=(12, 4 * (len(top_patterns) + 1)))
    
    # If only one subplot, make it an array for consistent indexing
    if len(top_patterns) == 0:
        axes = [axes]
    elif len(top_patterns) == 1:
        axes = [axes[0], axes[1]]
    
    # Plot full price chart
    axes[0].plot(data.index, data['Close'])
    axes[0].set_title('QQQ Price with Detected Patterns')
    
    # Format x-axis for dates
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    axes[0].tick_params(axis='x', rotation=45)
    
    # Highlight detected patterns
    colors = ['r', 'g', 'b', 'c', 'm']
    for i, pattern in enumerate(top_patterns):
        pos = pattern['position']
        pattern_end = min(pos + window_size, len(data))
        color = colors[i % len(colors)]
        
        # Highlight in main chart
        axes[0].axvspan(data.index[pos], data.index[pattern_end-1], alpha=0.2, color=color)
        
        # Add text label
        axes[0].text(data.index[pos], data['Close'].iloc[pos] * 1.02, 
                    f"{pattern['pattern']}\n{pattern['confidence']:.2f}", 
                    fontsize=8, color=color)
        
        # Plot individual pattern
        if len(top_patterns) > 0:
            # Extract pattern window
            window_data = data['Close'].iloc[pos:pattern_end]
            axes[i+1].plot(window_data.index, window_data, label='Price Data')
            
            # Format x-axis for dates
            axes[i+1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[i+1].tick_params(axis='x', rotation=45)
            
            # Get the pattern template
            pattern_template = patterns[pattern['pattern']]
            
            # Scale pattern to match price range for visualization
            min_price = window_data.min()
            max_price = window_data.max()
            
            # Ensure pattern_template is the right length
            template_to_use = pattern_template
            if len(pattern_template) > len(window_data):
                template_to_use = pattern_template[:len(window_data)]
            elif len(pattern_template) < len(window_data):
                # Pad with the last value
                template_to_use = np.pad(pattern_template, 
                                        (0, len(window_data) - len(pattern_template)), 
                                        'constant', 
                                        constant_values=pattern_template[-1])
                
            scaled_pattern = min_price + normalize(template_to_use) * (max_price - min_price)
            
            # Plot pattern template
            axes[i+1].plot(window_data.index, scaled_pattern, '--', label='Pattern Template')
            axes[i+1].set_title(f"Pattern: {pattern['pattern']} (Confidence: {pattern['confidence']:.2f})")
            axes[i+1].legend()
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('detected_patterns.png')
        print("Saved visualization to detected_patterns.png")
        
    plt.show()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='QQQ Minute-Level Pattern Detection')
    parser.add_argument('--csv', type=str, default='data/QQQ_minute_data.csv',
                        help='CSV file with minute data (default: data/QQQ_minute_data.csv)')
    parser.add_argument('--symbol', type=str, default='QQQ',
                        help='Ticker symbol to analyze if CSV not available (default: QQQ)')
    parser.add_argument('--period', type=str, default='5d',
                        help='Period to download from Yahoo Finance (default: 5d)')
    parser.add_argument('--interval', type=str, default='1m',
                        help='Data interval (default: 1m)')
    parser.add_argument('--window', type=int, default=60,
                        help='Pattern window size in minutes (default: 60)')
    parser.add_argument('--step', type=int, default=5,
                        help='Step size for sliding window (default: 5)')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Pattern detection threshold (default: 5.0)')
    parser.add_argument('--max-patterns', type=int, default=3,
                        help='Maximum patterns to visualize (default: 3)')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save detected patterns to CSV file')
    parser.add_argument('--save-fig', action='store_true',
                        help='Save visualization to PNG file')
    parser.add_argument('--save-patterns', action='store_true',
                        help='Save individual pattern plots to images folder')
    parser.add_argument('--confidence', type=float, default=0.6,
                        help='Confidence threshold for saving individual patterns (default: 0.6)')
    parser.add_argument('--future-periods', type=int, default=60,
                        help='Number of future periods to include in individual pattern plots (default: 60)')
    parser.add_argument('--custom-pattern', type=str,
                        help='Add a custom pattern type (options: ascending_wedge, descending_wedge, sideways_consolidation)')
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    print("\n=== QQQ Minute-Level Pattern Detection ===\n")
    
    # 1. Create pattern templates
    patterns = create_patterns()
    print(f"Created {len(patterns)} pattern templates")
    
    # Add custom pattern if specified
    if args.custom_pattern:
        custom_pattern = create_custom_pattern(args.custom_pattern, args.window)
        patterns[args.custom_pattern] = custom_pattern
        print(f"Added custom pattern: {args.custom_pattern}")
    
    # 2. Load data
    data = load_data(csv_file=args.csv, symbol=args.symbol, period=args.period, interval=args.interval)
    if data is None:
        print("Failed to load data")
        return
    
    # 3. Filter to market hours
    market_data = filter_market_hours(data)
    
    # 4. Set detection parameters
    window_size = args.window
    threshold = args.threshold
    
    # 5. Detect patterns
    detected = detect_patterns(market_data, patterns, window_size, threshold, args.step)
    
    # 6. Print results
    print(f"\nDetected {len(detected)} patterns:")
    
    # Sort by confidence
    detected.sort(key=lambda x: x['confidence'], reverse=True)
    
    for i, pattern in enumerate(detected[:10]):  # Show top 10
        print(f"{i+1}. Time: {pattern['time']}")
        print(f"   Pattern: {pattern['pattern']}")
        print(f"   Confidence: {pattern['confidence']:.2f}")
        print(f"   Distance: {pattern['distance']:.4f}")
    
    # 7. Save patterns to CSV if requested
    if args.save_csv:
        save_patterns_to_csv(detected)
    
    # 8. Generate individual pattern plots if requested
    if args.save_patterns:
        save_individual_pattern_plots(
            market_data, 
            patterns, 
            detected, 
            window_size=window_size, 
            future_periods=args.future_periods,
            confidence_threshold=args.confidence
        )
    
    # 9. Visualize patterns
    try:
        visualize_patterns(market_data, patterns, detected, window_size, args.max_patterns, args.save_fig)
    except Exception as e:
        print(f"Error visualizing patterns: {e}")
    
    print("\nAnalysis complete!")
    return market_data, patterns, detected

if __name__ == "__main__":
    main() 