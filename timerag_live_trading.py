"""
TimeRAG Live Trading Script

This script demonstrates how to use TimeRAG for real-time pattern detection
and trading with minute-level financial data.

Note: This script simulates live trading using recent historical data.
In a real production environment, you would need to connect to a broker API
for actual trade execution.
"""

import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yfinance as yf
import threading
import queue
import logging
from IPython.display import clear_output

# Import TimeRAG implementation
from timerag_minute_implementation import (
    TimeRAG, create_intraday_patterns, detect_intraday_patterns,
    visualize_intraday_patterns, fetch_minute_data, filter_market_hours
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("timerag_live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TimeRAG-Live")


class TimeRAGLiveTrader:
    """TimeRAG Live Trading System using minute-level data"""
    
    def __init__(self, symbols=['QQQ'], window_size=60, threshold=1.5, update_interval=60):
        """
        Initialize the live trading system
        
        Args:
            symbols: List of symbols to trade
            window_size: Pattern window size in minutes
            threshold: Pattern detection threshold
            update_interval: Data update interval in seconds
        """
        self.symbols = symbols
        self.window_size = window_size
        self.threshold = threshold
        self.update_interval = update_interval
        
        # Initialize TimeRAG
        self.timerag = TimeRAG(window_size=window_size, step_size=1, n_clusters=10, top_k=3)
        
        # Create pattern knowledge base
        patterns = create_intraday_patterns(minutes=window_size)
        self.timerag.build_knowledge_base(patterns)
        
        # Data storage
        self.data = {}
        self.latest_prices = {}
        self.detected_patterns = {}
        
        # Trading state
        self.positions = {symbol: None for symbol in symbols}
        self.entry_prices = {symbol: 0.0 for symbol in symbols}
        self.entry_times = {symbol: None for symbol in symbols}
        self.trade_history = []
        self.pnl = 0.0
        
        # Pattern definitions
        self.bullish_patterns = [
            'momentum_burst', 'intraday_bull_flag', 'consolidation_breakout',
            'v_shape_reversal', 'double_bottom', 'vwap_bounce', 'closing_drive'
        ]
        self.bearish_patterns = [
            'intraday_bear_flag', 'consolidation_breakdown', 'vwap_rejection'
        ]
        
        # System state
        self.running = False
        self.data_thread = None
        self.trading_thread = None
        self.message_queue = queue.Queue()
        
        # Visualization
        self.fig = None
        self.axs = None
        self.animation = None
    
    def fetch_initial_data(self):
        """Fetch initial historical data for all symbols"""
        logger.info("Fetching initial historical data...")
        
        for symbol in self.symbols:
            try:
                # Fetch enough data to fill the window
                hist_data = fetch_minute_data(
                    symbol=symbol, 
                    period='1d',  # Just 1 day for initial data
                    interval='1m'
                )
                
                # Filter to market hours
                market_data = filter_market_hours(hist_data)
                
                if len(market_data) < self.window_size:
                    logger.warning(f"Not enough data for {symbol}. Need at least {self.window_size} minutes.")
                    continue
                
                self.data[symbol] = market_data
                self.latest_prices[symbol] = market_data['Close'].iloc[-1]
                self.detected_patterns[symbol] = []
                
                logger.info(f"Loaded {len(market_data)} data points for {symbol}")
            
            except Exception as e:
                logger.error(f"Error fetching initial data for {symbol}: {e}")
    
    def update_data(self):
        """Update data for all symbols"""
        while self.running:
            try:
                for symbol in self.symbols:
                    # In a real system, you would fetch only the latest data
                    # Here we're simulating by fetching the last 5 minutes
                    latest_data = fetch_minute_data(
                        symbol=symbol,
                        period='5m',
                        interval='1m'
                    )
                    
                    if latest_data.empty:
                        continue
                    
                    # Filter to market hours
                    latest_data = filter_market_hours(latest_data)
                    
                    if latest_data.empty:
                        continue
                    
                    # Update data
                    if symbol in self.data:
                        # Append new data, avoid duplicates by checking timestamps
                        existing_times = self.data[symbol].index
                        new_data = latest_data[~latest_data.index.isin(existing_times)]
                        
                        if not new_data.empty:
                            self.data[symbol] = pd.concat([self.data[symbol], new_data])
                            self.data[symbol] = self.data[symbol].iloc[-5000:]  # Keep last 5000 points to save memory
                            
                            # Update latest price
                            self.latest_prices[symbol] = self.data[symbol]['Close'].iloc[-1]
                            
                            # Log new data
                            logger.info(f"Updated {len(new_data)} new data points for {symbol}")
                            
                            # Put message in queue for the trading thread
                            self.message_queue.put(f"NEW_DATA:{symbol}")
                    else:
                        self.data[symbol] = latest_data
                        self.latest_prices[symbol] = latest_data['Close'].iloc[-1]
                
                # Sleep until next update
                time.sleep(self.update_interval)
            
            except Exception as e:
                logger.error(f"Error updating data: {e}")
                time.sleep(5)  # Wait a bit before trying again
    
    def detect_patterns_live(self, symbol):
        """
        Detect patterns in the latest data
        
        Args:
            symbol: Symbol to detect patterns for
        """
        try:
            if symbol not in self.data or len(self.data[symbol]) < self.window_size:
                return []
            
            # Get the latest window of data
            latest_window = self.data[symbol]['Close'].values[-self.window_size:]
            current_time = self.data[symbol].index[-1]
            current_price = self.latest_prices[symbol]
            
            # Retrieve similar patterns
            similar_patterns = self.timerag.retrieve_similar_sequences(latest_window)
            
            # Check if any pattern is similar enough
            if similar_patterns and similar_patterns[0]['distance'] < self.threshold:
                pattern_type = similar_patterns[0]['label'].split('_var')[0].split('_stretch')[0]
                confidence = 1.0 - (similar_patterns[0]['distance'] / self.threshold)
                
                # Record pattern detection
                pattern_info = {
                    'symbol': symbol,
                    'time': current_time,
                    'pattern': pattern_type,
                    'confidence': confidence,
                    'price': current_price,
                    'distance': similar_patterns[0]['distance']
                }
                
                self.detected_patterns[symbol].append(pattern_info)
                self.detected_patterns[symbol] = self.detected_patterns[symbol][-100:]  # Keep last 100 patterns
                
                logger.info(f"Detected {pattern_type} pattern for {symbol} with {confidence:.2f} confidence at {current_time}")
                
                return pattern_info
            
            return None
        
        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {e}")
            return None
    
    def execute_trade(self, symbol, action, reason):
        """
        Execute a trade
        
        Args:
            symbol: Symbol to trade
            action: 'BUY', 'SELL', or 'EXIT'
            reason: Reason for the trade
        """
        current_time = datetime.datetime.now()
        current_price = self.latest_prices[symbol]
        
        if action == 'BUY' and self.positions[symbol] != 'LONG':
            # Exit any existing position first
            if self.positions[symbol] == 'SHORT':
                self.execute_trade(symbol, 'EXIT', 'Reversing position')
            
            # Enter long position
            self.positions[symbol] = 'LONG'
            self.entry_prices[symbol] = current_price
            self.entry_times[symbol] = current_time
            
            logger.info(f"BUY {symbol} at {current_price:.2f} ({reason})")
        
        elif action == 'SELL' and self.positions[symbol] != 'SHORT':
            # Exit any existing position first
            if self.positions[symbol] == 'LONG':
                self.execute_trade(symbol, 'EXIT', 'Reversing position')
            
            # Enter short position
            self.positions[symbol] = 'SHORT'
            self.entry_prices[symbol] = current_price
            self.entry_times[symbol] = current_time
            
            logger.info(f"SELL {symbol} at {current_price:.2f} ({reason})")
        
        elif action == 'EXIT' and self.positions[symbol] is not None:
            # Exit position
            position_type = self.positions[symbol]
            entry_price = self.entry_prices[symbol]
            entry_time = self.entry_times[symbol]
            
            # Calculate PnL
            if position_type == 'LONG':
                trade_pnl = current_price - entry_price
            else:  # SHORT
                trade_pnl = entry_price - current_price
            
            self.pnl += trade_pnl
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'position': position_type,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': current_time,
                'exit_price': current_price,
                'pnl': trade_pnl,
                'reason': reason
            }
            
            self.trade_history.append(trade_record)
            
            logger.info(f"EXIT {position_type} {symbol} at {current_price:.2f}, PnL: {trade_pnl:.2f} ({reason})")
            
            # Reset position
            self.positions[symbol] = None
            self.entry_prices[symbol] = 0.0
            self.entry_times[symbol] = None
    
    def trading_logic(self):
        """Main trading logic loop"""
        while self.running:
            try:
                # Wait for a message from the data thread
                try:
                    message = self.message_queue.get(timeout=1)
                    
                    # Process message
                    if message.startswith("NEW_DATA:"):
                        symbol = message.split(":")[1]
                        
                        # Detect patterns
                        pattern = self.detect_patterns_live(symbol)
                        
                        # Execute trading logic based on detected pattern
                        if pattern:
                            pattern_type = pattern['pattern']
                            confidence = pattern['confidence']
                            
                            if pattern_type in self.bullish_patterns and confidence > 0.7:
                                self.execute_trade(symbol, 'BUY', f"{pattern_type} pattern with {confidence:.2f} confidence")
                            
                            elif pattern_type in self.bearish_patterns and confidence > 0.7:
                                self.execute_trade(symbol, 'SELL', f"{pattern_type} pattern with {confidence:.2f} confidence")
                    
                    self.message_queue.task_done()
                
                except queue.Empty:
                    pass
                
                # Check for time-based exits (positions held for too long)
                for symbol in self.symbols:
                    if self.positions[symbol] and self.entry_times[symbol]:
                        # Check if position has been held for more than 60 minutes
                        time_held = datetime.datetime.now() - self.entry_times[symbol]
                        if time_held.total_seconds() > 3600:  # 60 minutes
                            self.execute_trade(symbol, 'EXIT', 'Time-based exit (60 min)')
            
            except Exception as e:
                logger.error(f"Error in trading logic: {e}")
    
    def start(self):
        """Start the live trading system"""
        if self.running:
            logger.warning("System is already running")
            return
        
        logger.info("Starting TimeRAG Live Trading System")
        
        # Fetch initial data
        self.fetch_initial_data()
        
        # Start system
        self.running = True
        
        # Start data thread
        self.data_thread = threading.Thread(target=self.update_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self.trading_logic)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        logger.info("TimeRAG Live Trading System started")
    
    def stop(self):
        """Stop the live trading system"""
        if not self.running:
            logger.warning("System is not running")
            return
        
        logger.info("Stopping TimeRAG Live Trading System")
        
        # Stop system
        self.running = False
        
        # Wait for threads to finish
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=5)
        
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        
        # Exit any open positions
        for symbol in self.symbols:
            if self.positions[symbol]:
                self.execute_trade(symbol, 'EXIT', 'System shutdown')
        
        logger.info("TimeRAG Live Trading System stopped")
        
        # Print final summary
        self.print_summary()
    
    def print_summary(self):
        """Print trading summary"""
        logger.info("\n" + "="*50)
        logger.info("Trading Summary")
        logger.info("="*50)
        
        logger.info(f"Total PnL: {self.pnl:.2f}")
        logger.info(f"Number of trades: {len(self.trade_history)}")
        
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            losing_trades = [t for t in self.trade_history if t['pnl'] <= 0]
            win_rate = len(winning_trades) / len(self.trade_history) * 100
            
            logger.info(f"Win rate: {win_rate:.2f}%")
            
            if winning_trades:
                avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
                logger.info(f"Average win: {avg_win:.2f}")
            
            if losing_trades:
                avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
                logger.info(f"Average loss: {avg_loss:.2f}")
            
            if winning_trades and losing_trades and sum(t['pnl'] for t in losing_trades) != 0:
                profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades))
                logger.info(f"Profit factor: {profit_factor:.2f}")
        
        logger.info("="*50)
    
    def start_visualization(self):
        """Start real-time visualization"""
        if not self.running:
            logger.warning("System must be running to start visualization")
            return
        
        if len(self.symbols) == 0:
            logger.warning("No symbols to visualize")
            return
        
        # Create figure
        self.fig, self.axs = plt.subplots(len(self.symbols), 1, figsize=(15, 5 * len(self.symbols)))
        
        # If only one symbol, axs is not a list
        if len(self.symbols) == 1:
            self.axs = [self.axs]
        
        # Function to update the plot
        def update(frame):
            for i, symbol in enumerate(self.symbols):
                ax = self.axs[i]
                
                # Clear previous plot
                ax.clear()
                
                if symbol not in self.data or self.data[symbol].empty:
                    ax.text(0.5, 0.5, f"No data for {symbol}", 
                            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    continue
                
                # Get the last 120 data points to plot
                plot_data = self.data[symbol].iloc[-120:]
                
                # Plot price data
                ax.plot(plot_data.index, plot_data['Close'], color='black')
                
                # Plot detected patterns
                if symbol in self.detected_patterns:
                    for pattern in self.detected_patterns[symbol]:
                        pattern_time = pattern['time']
                        
                        # Only plot if in the visible range
                        if pattern_time in plot_data.index:
                            pattern_price = pattern['price']
                            pattern_type = pattern['pattern']
                            
                            is_bullish = pattern_type in self.bullish_patterns
                            color = 'green' if is_bullish else 'red'
                            marker = '^' if is_bullish else 'v'
                            
                            ax.scatter(pattern_time, pattern_price, color=color, marker=marker, s=100)
                            ax.annotate(pattern_type, (pattern_time, pattern_price),
                                        xytext=(0, 10 if is_bullish else -10),
                                        textcoords='offset points',
                                        fontsize=8, rotation=45, ha='center')
                
                # Highlight current position
                if self.positions[symbol]:
                    position_color = 'green' if self.positions[symbol] == 'LONG' else 'red'
                    ax.axhline(y=self.entry_prices[symbol], color=position_color, linestyle='--', alpha=0.7)
                    
                    # Add position label
                    ax.text(plot_data.index[0], self.entry_prices[symbol], 
                            f"{self.positions[symbol]} @ {self.entry_prices[symbol]:.2f}",
                            color=position_color, fontsize=9)
                
                # Set title and labels
                ax.set_title(f"{symbol} - Price: {self.latest_prices[symbol]:.2f}")
                ax.grid(True, alpha=0.3)
                
                # Format x-axis to show time
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
        
        # Create animation
        self.animation = FuncAnimation(self.fig, update, interval=1000, cache_frame_data=False)
        plt.show()


def run_live_trading_demo(symbols=['QQQ'], duration_seconds=300):
    """
    Run a live trading demo for a specified duration
    
    Args:
        symbols: List of symbols to trade
        duration_seconds: Duration of the demo in seconds
    """
    try:
        # Create trader
        trader = TimeRAGLiveTrader(
            symbols=symbols,
            window_size=60,  # 60-minute patterns
            threshold=1.5,
            update_interval=60  # Update every 60 seconds
        )
        
        # Start trading system
        trader.start()
        
        # Let it run for a while
        logger.info(f"Running live trading demo for {duration_seconds} seconds...")
        
        # Uncomment to enable visualization
        # trader.start_visualization()
        
        # Instead of visualization, print updates
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            # Print current state
            clear_output(wait=True)
            
            print(f"TimeRAG Live Trading - Running for {int(time.time() - start_time)} seconds")
            print("="*50)
            print(f"Symbols: {trader.symbols}")
            print(f"Current positions: {trader.positions}")
            print(f"Current PnL: {trader.pnl:.2f}")
            print(f"Trades executed: {len(trader.trade_history)}")
            
            # Print latest prices
            print("\nLatest Prices:")
            for symbol, price in trader.latest_prices.items():
                print(f"{symbol}: {price:.2f}")
            
            # Print recent patterns
            print("\nRecent Patterns:")
            for symbol in trader.symbols:
                if symbol in trader.detected_patterns and trader.detected_patterns[symbol]:
                    recent_patterns = trader.detected_patterns[symbol][-3:]  # Last 3 patterns
                    for pattern in recent_patterns:
                        print(f"{pattern['time']} - {symbol} - {pattern['pattern']} ({pattern['confidence']:.2f})")
            
            # Print recent trades
            print("\nRecent Trades:")
            for trade in trader.trade_history[-5:]:  # Last 5 trades
                print(f"{trade['exit_time']} - {trade['symbol']} - {trade['position']} - PnL: {trade['pnl']:.2f} - {trade['reason']}")
            
            # Sleep
            time.sleep(5)
        
        # Stop trading system
        trader.stop()
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        if 'trader' in locals():
            trader.stop()
    
    except Exception as e:
        logger.error(f"Error in live trading demo: {e}")
        if 'trader' in locals():
            trader.stop()


if __name__ == "__main__":
    symbols = ['QQQ', 'SPY', 'AAPL']
    run_live_trading_demo(symbols=symbols, duration_seconds=600)  # Run for 10 minutes