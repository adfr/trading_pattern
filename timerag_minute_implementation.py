"""
TimeRAG for Minute-Level Financial Pattern Recognition

This version of TimeRAG is adapted to work with intraday, minute-level 
financial data for identifying short-term patterns.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import pickle
import os


class TimeRAG:
    """
    TimeRAG: Boosting LLM Time Series Forecasting via Retrieval-Augmented Generation
    Adapted for minute-level financial pattern recognition
    """
    
    def __init__(self, 
                window_size: int = 60,  # Default 60 minutes (1 hour)
                step_size: int = 5,     # Check every 5 minutes
                n_clusters: int = 10,
                top_k: int = 5,
                normalize: bool = True):
        """
        Initialize TimeRAG for minute-level data
        
        Args:
            window_size: Length of sliding window for sequence slicing (in minutes)
            step_size: Step size for sliding window
            n_clusters: Number of clusters for K-means
            top_k: Number of top similar sequences to retrieve
            normalize: Whether to normalize sequences
        """
        self.window_size = window_size
        self.step_size = step_size
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.normalize = normalize
        self.knowledge_base = []  # Changed from dict to list for consistent iteration
        self.fitted = False
    
    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize sequence to range [0, 1]"""
        if not self.normalize:
            return sequence
            
        min_val = np.min(sequence)
        max_val = np.max(sequence)
        if max_val == min_val:
            return np.zeros_like(sequence)
        return (sequence - min_val) / (max_val - min_val)
    
    def denormalize(self, sequence: np.ndarray, original_min=0, original_max=1) -> np.ndarray:
        """Convert normalized sequence back to original scale"""
        return sequence * (original_max - original_min) + original_min
    
    def sliding_window(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Create sliding windows from data
        
        Args:
            data: Input time series data
            
        Returns:
            List of normalized sequences
        """
        sequences = []
        for i in range(0, len(data) - self.window_size + 1, self.step_size):
            sequence = data[i:i+self.window_size]
            # Normalize each sequence
            sequence = self.normalize_sequence(sequence)
            sequences.append(sequence)
        return sequences
    
    def build_knowledge_base(self, patterns: Dict[str, np.ndarray]):
        """
        Build knowledge base with known intraday patterns and their variations
        
        Args:
            patterns: Dictionary with pattern names as keys and pattern arrays as values
        """
        all_patterns = []
        pattern_labels = []
        
        # Process each pattern to create variations
        for pattern_name, pattern_data in patterns.items():
            # Normalize the pattern
            normalized_pattern = self.normalize_sequence(pattern_data)
            
            # Add original pattern
            all_patterns.append(normalized_pattern)
            pattern_labels.append(pattern_name)
            
            # Add variations with noise
            for i in range(5):  # Create 5 variations with different noise levels
                noise = np.random.normal(0, 0.03 * (i + 1), normalized_pattern.shape)
                noisy_pattern = normalized_pattern + noise
                noisy_pattern = self.normalize_sequence(noisy_pattern)  # Re-normalize
                all_patterns.append(noisy_pattern)
                pattern_labels.append(f"{pattern_name}_var{i+1}")
            
            # Add time-stretched variations (make pattern slightly longer/shorter)
            for stretch in [0.8, 0.9, 1.1, 1.2]:
                indices = np.linspace(0, len(normalized_pattern)-1, int(len(normalized_pattern)*stretch))
                stretched_pattern = np.interp(indices, np.arange(len(normalized_pattern)), normalized_pattern)
                
                # Ensure all patterns are the same length as window_size
                if len(stretched_pattern) > self.window_size:
                    stretched_pattern = stretched_pattern[:self.window_size]
                elif len(stretched_pattern) < self.window_size:
                    padding = np.zeros(self.window_size - len(stretched_pattern))
                    stretched_pattern = np.concatenate([stretched_pattern, padding])
                
                all_patterns.append(stretched_pattern)
                pattern_labels.append(f"{pattern_name}_stretch{stretch}")
        
        # Store patterns directly in knowledge base
        for i, pattern in enumerate(all_patterns):
            self.knowledge_base.append({
                'sequence': pattern,
                'label': pattern_labels[i],
                'normalize': True  # Flag to track normalization status
            })
        
        self.fitted = True
        print(f"Knowledge base built with {len(self.knowledge_base)} pattern variations")
    
    def build_knowledge_base_from_data(self, data: np.ndarray, labels: Optional[List[str]] = None):
        """
        Build knowledge base from minute-level time series data using K-means clustering
        
        Args:
            data: Input time series data
            labels: Optional labels for the sequences
        """
        # Slice data into sequences
        sequences = self.sliding_window(data)
        
        if len(sequences) < self.n_clusters:
            self.n_clusters = len(sequences)
            print(f"Reducing number of clusters to {self.n_clusters} due to limited data")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        sequences_array = np.array(sequences)
        cluster_assignments = kmeans.fit_predict(sequences_array)
        
        # Clear existing knowledge base
        self.knowledge_base = []
        
        # Build knowledge base by selecting representative sequence from each cluster
        for cluster_idx in range(self.n_clusters):
            # Get indices of sequences in this cluster
            cluster_member_indices = np.where(cluster_assignments == cluster_idx)[0]
            
            # Find the sequence closest to the centroid
            centroid = kmeans.cluster_centers_[cluster_idx]
            min_dist = float('inf')
            representative_idx = -1
            
            for member_idx in cluster_member_indices:
                dist = np.linalg.norm(sequences_array[member_idx] - centroid)
                if dist < min_dist:
                    min_dist = dist
                    representative_idx = member_idx
            
            # Store representative sequence in knowledge base
            if representative_idx >= 0:
                label = f"cluster_{cluster_idx}"
                if labels is not None and representative_idx < len(labels):
                    label = labels[representative_idx]
                
                self.knowledge_base.append({
                    'sequence': sequences_array[representative_idx],
                    'label': label,
                    'normalize': True  # Flag to track normalization status
                })
        
        self.fitted = True
        print(f"Knowledge base built with {len(self.knowledge_base)} clusters")
    
    def retrieve_similar_sequences(self, query_sequence):
        """
        Retrieve similar sequences from the knowledge base using DTW
        
        Args:
            query_sequence: The sequence to compare against the knowledge base
            
        Returns:
            List of dictionaries containing similar sequences sorted by distance
        """
        similarities = []
        
        # Convert query to numpy array if it isn't already
        if not isinstance(query_sequence, np.ndarray):
            if isinstance(query_sequence, pd.DataFrame):
                query_sequence = query_sequence.values
            elif isinstance(query_sequence, pd.Series):
                query_sequence = query_sequence.values
            else:
                try:
                    query_sequence = np.array(query_sequence)
                except:
                    raise ValueError("Query sequence must be convertible to a numpy array")
        
        # Ensure query sequence is 1-dimensional 
        if query_sequence.ndim > 1:
            query_sequence = query_sequence.flatten()
            
        # Normalize the query sequence if needed
        if self.normalize:
            query_sequence = self.normalize_sequence(query_sequence)
        
        # Check if knowledge base is empty
        if not self.knowledge_base:
            print("Warning: Knowledge base is empty.")
            return []
            
        # Process each entry in the knowledge base
        for idx, entry in enumerate(self.knowledge_base):
            # Get the pattern sequence from the entry
            kb_sequence = entry['sequence']
            
            # Convert kb_sequence to numpy array if it isn't already
            if not isinstance(kb_sequence, np.ndarray):
                try:
                    kb_sequence = np.array(kb_sequence)
                except:
                    print(f"Warning: Knowledge base entry {idx} could not be converted to numpy array. Skipping.")
                    continue
            
            # Ensure knowledge base sequence is 1-dimensional
            if kb_sequence.ndim > 1:
                kb_sequence = kb_sequence.flatten()
            
            # Ensure both sequences have the same length
            min_len = min(len(query_sequence), len(kb_sequence))
            query_seq_use = query_sequence[:min_len]
            kb_seq_use = kb_sequence[:min_len]
            
            try:
                # Calculate DTW distance
                distance, _ = fastdtw(query_seq_use, kb_seq_use, dist=euclidean)
                
                similarities.append({
                    'id': idx,
                    'sequence': kb_sequence,
                    'label': entry['label'],
                    'distance': distance
                })
            except Exception as e:
                print(f"Error calculating DTW distance for entry {idx}: {e}")
                continue
        
        # Sort by distance (ascending)
        similarities.sort(key=lambda x: x['distance'])
        
        # Return only top_k results if we have more than top_k
        if len(similarities) > self.top_k:
            return similarities[:self.top_k]
            
        return similarities
    
    def save_knowledge_base(self, filepath: str):
        """Save knowledge base to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        print(f"Knowledge base saved to {filepath}")
    
    def load_knowledge_base(self, filepath: str):
        """Load knowledge base from file"""
        with open(filepath, 'rb') as f:
            self.knowledge_base = pickle.load(f)
        self.fitted = True
        print(f"Knowledge base loaded from {filepath} with {len(self.knowledge_base)} entries")


def create_intraday_patterns(minutes: int = 60) -> Dict[str, np.ndarray]:
    """
    Create common intraday chart patterns
    
    Args:
        minutes: Pattern length in minutes
        
    Returns:
        Dictionary of pattern name -> pattern array
    """
    patterns = {}
    
    # Minute-based x-axis
    x = np.linspace(0, 1, minutes)
    
    # 1. V-shaped reversal (quick drop and recovery)
    v_shape = 1 - 0.6 * np.sin(x * np.pi) + np.random.normal(0, 0.02, minutes)
    patterns['v_shape_reversal'] = v_shape
    
    # 2. Intraday double bottom (W pattern)
    w_shape = 0.7 - 0.5 * np.sin(2 * x * np.pi) + np.random.normal(0, 0.02, minutes)
    patterns['double_bottom'] = w_shape
    
    # 3. Morning momentum burst
    momentum_burst = np.zeros(minutes)
    # First 15 minutes strong move
    momentum_burst[:15] = np.linspace(0.2, 0.8, 15) + np.random.normal(0, 0.02, 15)
    # Rest of day consolidation
    momentum_burst[15:] = 0.8 + np.random.normal(0, 0.05, minutes-15)
    patterns['momentum_burst'] = momentum_burst
    
    # 4. Bull flag (intraday)
    bull_flag = np.zeros(minutes)
    # First part: sharp uptrend
    bull_flag[:20] = np.linspace(0.2, 0.9, 20) + np.random.normal(0, 0.02, 20)
    # Second part: flag (slight downtrend in a channel)
    flag_top = np.linspace(0.9, 0.8, minutes-20)
    flag_bottom = np.linspace(0.7, 0.6, minutes-20)
    for i in range(20, minutes):
        idx = i - 20
        if i % 2 == 0:
            bull_flag[i] = flag_top[idx] - np.random.uniform(0, 0.03)
        else:
            bull_flag[i] = flag_bottom[idx] + np.random.uniform(0, 0.03)
    patterns['intraday_bull_flag'] = bull_flag
    
    # 5. Bear flag (intraday)
    bear_flag = np.zeros(minutes)
    # First part: sharp downtrend
    bear_flag[:20] = np.linspace(0.9, 0.2, 20) + np.random.normal(0, 0.02, 20)
    # Second part: flag (slight uptrend in a channel)
    flag_top = np.linspace(0.4, 0.5, minutes-20)
    flag_bottom = np.linspace(0.2, 0.3, minutes-20)
    for i in range(20, minutes):
        idx = i - 20
        if i % 2 == 0:
            bear_flag[i] = flag_top[idx] - np.random.uniform(0, 0.03)
        else:
            bear_flag[i] = flag_bottom[idx] + np.random.uniform(0, 0.03)
    patterns['intraday_bear_flag'] = bear_flag
    
    # 6. Consolidation breakout
    breakout = np.zeros(minutes)
    # First part: consolidation
    breakout[:40] = 0.5 + np.random.normal(0, 0.05, 40)
    # Second part: breakout
    breakout[40:] = np.linspace(0.5, 0.9, minutes-40) + np.random.normal(0, 0.02, minutes-40)
    patterns['consolidation_breakout'] = breakout
    
    # 7. Consolidation breakdown
    breakdown = np.zeros(minutes)
    # First part: consolidation
    breakdown[:40] = 0.5 + np.random.normal(0, 0.05, 40)
    # Second part: breakdown
    breakdown[40:] = np.linspace(0.5, 0.1, minutes-40) + np.random.normal(0, 0.02, minutes-40)
    patterns['consolidation_breakdown'] = breakdown
    
    # 8. VWAP bounce
    vwap_bounce = np.zeros(minutes)
    # First part: decline to VWAP
    vwap_bounce[:30] = np.linspace(0.8, 0.3, 30) + np.random.normal(0, 0.02, 30)
    # Second part: bounce from VWAP
    vwap_bounce[30:] = np.linspace(0.3, 0.7, minutes-30) + np.random.normal(0, 0.02, minutes-30)
    patterns['vwap_bounce'] = vwap_bounce
    
    # 9. VWAP rejection
    vwap_rejection = np.zeros(minutes)
    # First part: rise to VWAP
    vwap_rejection[:30] = np.linspace(0.2, 0.7, 30) + np.random.normal(0, 0.02, 30)
    # Second part: rejection from VWAP
    vwap_rejection[30:] = np.linspace(0.7, 0.3, minutes-30) + np.random.normal(0, 0.02, minutes-30)
    patterns['vwap_rejection'] = vwap_rejection
    
    # 10. Closing drive (end of day momentum)
    closing_drive = np.zeros(minutes)
    # First part: consolidation
    closing_drive[:40] = 0.5 + np.random.normal(0, 0.05, 40)
    # Second part: end of day momentum
    closing_drive[40:] = np.linspace(0.5, 0.9, minutes-40) + np.random.normal(0, 0.02, minutes-40)
    patterns['closing_drive'] = closing_drive
    
    return patterns


def fetch_minute_data(symbol='QQQ', period='5d', interval='1m'):
    """
    Fetch minute-level data from yfinance
    
    Args:
        symbol: Ticker symbol
        period: Time period to fetch
        interval: Data interval (1m, 2m, 5m, etc.)
        
    Returns:
        DataFrame with minute-level price data
    """
    print(f"Fetching {interval} data for {symbol} over the last {period}...")
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    
    if hist.empty:
        raise ValueError(f"No data returned for {symbol} with {interval} interval. Try a different symbol or interval.")
    
    print(f"Retrieved {len(hist)} data points from {hist.index[0]} to {hist.index[-1]}")
    
    return hist


def filter_market_hours(data, market_open='09:30', market_close='16:00'):
    """
    Filter data to only include market hours
    
    Args:
        data: DataFrame with datetime index
        market_open: Market open time (HH:MM)
        market_close: Market close time (HH:MM)
        
    Returns:
        DataFrame filtered to market hours
    """
    # Convert times to pandas.Timestamp.time
    open_time = pd.to_datetime(market_open).time()
    close_time = pd.to_datetime(market_close).time()
    
    # Filter data to only include market hours
    filtered_data = data.between_time(open_time, close_time)
    
    print(f"Filtered from {len(data)} to {len(filtered_data)} data points (market hours only)")
    
    return filtered_data


def detect_intraday_patterns(timerag, data, window_size=60, threshold=1.5):
    """
    Detect intraday patterns in minute-level financial time series data
    
    Args:
        timerag: Initialized TimeRAG instance
        data: Financial time series data (close prices)
        window_size: Size of sliding window in minutes
        threshold: Distance threshold for pattern matching
        
    Returns:
        List of detected patterns with their positions and confidence
    """
    detected_patterns = []
    
    # Ensure we have a 1D array of close prices for proper processing
    if isinstance(data, pd.DataFrame):
        if 'Close' in data.columns:
            close_prices = data['Close'].values
        else:
            # Use the first numeric column if 'Close' isn't available
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                close_prices = data[numeric_cols[0]].values
            else:
                raise ValueError("No numeric columns found in the data")
    elif isinstance(data, pd.Series):
        close_prices = data.values
    elif isinstance(data, np.ndarray):
        # If it's already a numpy array, ensure it's 1D
        close_prices = data.flatten() if data.ndim > 1 else data
    else:
        raise ValueError("Data must be a pandas DataFrame, Series, or numpy array")
        
    # Process each window in the data
    for i in range(0, len(close_prices) - window_size + 1):
        window = close_prices[i:i+window_size]
        
        # Make sure window is 1D 
        if hasattr(window, 'ndim') and window.ndim > 1:
            window = window.flatten()
        
        window_time = data.index[i] if hasattr(data, 'index') else i
        
        # Retrieve similar patterns
        similar_patterns = timerag.retrieve_similar_sequences(window)
        
        # Check if any pattern is similar enough
        if similar_patterns and similar_patterns[0]['distance'] < threshold:
            pattern_info = {
                'position': i,
                'time': window_time,
                'pattern': similar_patterns[0]['label'].split('_var')[0].split('_stretch')[0],  # Get base pattern name
                'confidence': 1.0 - (similar_patterns[0]['distance'] / threshold),
                'matches': similar_patterns
            }
            detected_patterns.append(pattern_info)
    
    return detected_patterns


def visualize_intraday_patterns(data, detected_patterns, window_size=60, max_patterns=5):
    """
    Visualize detected intraday patterns
    
    Args:
        data: Financial time series data with datetime index
        detected_patterns: List of detected patterns
        window_size: Size of pattern window in minutes
        max_patterns: Maximum number of patterns to visualize
    """
    # Limit the number of patterns to visualize
    if len(detected_patterns) > max_patterns:
        # Sort by confidence and take top patterns
        detected_patterns = sorted(detected_patterns, key=lambda x: x['confidence'], reverse=True)[:max_patterns]
    
    num_patterns = len(detected_patterns)
    if num_patterns == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'])
        plt.title("No patterns detected")
        plt.show()
        return
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(num_patterns + 1, 3, figure=fig)
    
    # Plot the full time series in the top subplot
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.plot(data.index, data['Close'])
    ax_full.set_title("Minute-Level Price Chart with Detected Patterns")
    ax_full.tick_params(axis='x', rotation=45)
    
    # Format x-axis to show date and time
    ax_full.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d %H:%M'))
    
    # Highlight the detected patterns in the full series
    for i, pattern in enumerate(detected_patterns):
        pos = pattern['position']
        pattern_end = min(pos + window_size, len(data))
        pattern_range = range(pos, pattern_end)
        
        # Highlight the pattern region
        ax_full.axvspan(data.index[pos], data.index[pattern_end-1], 
                       alpha=0.2, color=f'C{i+1}')
        
        # Add text label
        ax_full.text(data.index[pos], data.iloc[pos]['Close'] * 1.02, 
                    f"{pattern['pattern']}\n{pattern['confidence']:.2f}", 
                    fontsize=8, color=f'C{i+1}')
    
    # Plot each detected pattern in a separate subplot
    for i, pattern in enumerate(detected_patterns):
        pos = pattern['position']
        pattern_end = min(pos + window_size, len(data))
        
        # Pattern data subplot
        ax_pattern = fig.add_subplot(gs[i+1, 0])
        pattern_data = data.iloc[pos:pattern_end]
        ax_pattern.plot(pattern_data.index, pattern_data['Close'])
        ax_pattern.set_title(f"Pattern: {pattern['pattern']}, Confidence: {pattern['confidence']:.2f}")
        ax_pattern.tick_params(axis='x', rotation=45)
        ax_pattern.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        # Original template subplot
        ax_template = fig.add_subplot(gs[i+1, 1])
        template_seq = pattern['matches'][0]['sequence']
        ax_template.plot(range(len(template_seq)), template_seq)
        ax_template.set_title(f"Template: {pattern['matches'][0]['label']}")
        ax_template.set_xticks([])
        
        # Add a subplot for the normalized comparison
        ax_compare = fig.add_subplot(gs[i+1, 2])
        
        # Normalize the data window for comparison
        window_data = data.iloc[pos:pattern_end]['Close'].values
        norm_window = (window_data - window_data.min()) / (window_data.max() - window_data.min())
        
        # Plot both normalized sequences
        ax_compare.plot(norm_window, label='Data Window')
        ax_compare.plot(template_seq[:len(norm_window)], label='Template', linestyle='--')
        ax_compare.set_title(f"Comparison (DTW Distance: {pattern['matches'][0]['distance']:.2f})")
        ax_compare.legend()
        ax_compare.set_xticks([])
    
    plt.tight_layout()
    plt.show()


def run_intraday_pattern_detection(symbol='QQQ', interval='1m', period='5d', window_size=60, threshold=1.5):
    """
    Run the complete intraday pattern detection pipeline
    
    Args:
        symbol: Ticker symbol
        interval: Data interval (1m, 2m, 5m, etc.)
        period: Time period to fetch
        window_size: Pattern window size in minutes
        threshold: Detection threshold
    """
    # 1. Initialize TimeRAG
    timerag = TimeRAG(window_size=window_size, step_size=5, n_clusters=10, top_k=3)
    
    # 2. Create intraday patterns
    patterns = create_intraday_patterns(minutes=window_size)
    
    # 3. Build knowledge base
    timerag.build_knowledge_base(patterns)
    
    # 4. Fetch minute data
    try:
        minute_data = fetch_minute_data(symbol=symbol, period=period, interval=interval)
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Falling back to SPY data")
        minute_data = fetch_minute_data(symbol='SPY', period=period, interval=interval)
    
    # 5. Filter to market hours
    market_data = filter_market_hours(minute_data)
    
    # 6. Detect patterns
    detected_patterns = detect_intraday_patterns(
        timerag, market_data, window_size=window_size, threshold=threshold
    )
    
    # 7. Print results
    print(f"\nDetected {len(detected_patterns)} patterns")
    
    if detected_patterns:
        # Sort by time
        detected_patterns.sort(key=lambda x: x['time'])
        
        for i, pattern in enumerate(detected_patterns):
            print(f"{i+1}. Time: {pattern['time']}")
            print(f"   Pattern: {pattern['pattern']}")
            print(f"   Confidence: {pattern['confidence']:.2f}")
            print(f"   DTW Distance: {pattern['matches'][0]['distance']:.2f}")
            print()
    
    # 8. Visualize patterns
    visualize_intraday_patterns(market_data, detected_patterns, window_size=window_size)
    
    return timerag, market_data, detected_patterns


if __name__ == "__main__":
    run_intraday_pattern_detection(symbol='QQQ', interval='1m', period='5d')