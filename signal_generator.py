import json
import threading
import time

import pandas as pd
import numpy as np
import websocket
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

class RealTimeChart:
    def __init__(self, signal_generator):
        self.legend_created = False
        self.signal_generator = signal_generator
        self.fig  = plt.figure(figsize=(12, 6))
        self.ax = plt.gca()
        self.line, = self.ax.plot([], [])
        self.scatter = None  # Store scatter plot
        self.signal_scatter = None
        self.entry_signals = []
        self.ani = None
        self.last_axis_redraw = time.time()
        self.axis_redraw_interval = 30  # Redraw axis every 30 seconds

    def init_chart(self):
        """Initialize the chart."""
        self.ax.set_title(f"Real-Time Chart: {self.signal_generator.symbol}")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        return self.line,

    def update_chart(self, frame):
        """Update the chart with new data."""
        ohlcs = self.signal_generator.ohlcs
        tickers = self.signal_generator.tickers

        # Update price line
        print(f"Updating price line: tickers.index size: {len(tickers.index)}, tickers['last'] size: {len(tickers['last'])}")
        self.line.set_data(tickers.index, tickers['last'])

        # Plot last spots
        if self.scatter is None:
            self.scatter = self.ax.scatter(
                tickers.index,
                tickers['last'],
                marker='D',  # Diamond shape
                color='#1E88E5',  # Material Design blue
                edgecolors='#FFFFFF',  # White border
                linewidth=1.0,
                s=100,
                zorder=4,
                alpha=0.85,
                label="Last Price"
            )
        else:
            print(f"Updating last scatter offsets with {len(ohlcs)} points")
            timestamp = mdates.date2num(tickers.index.to_numpy())
            self.scatter.set_offsets(np.column_stack([timestamp , tickers['last']]))

        # Create a persistent entry signals scatter plot
        if self.signal_scatter is None:
            signal_times = [signal['time'] for signal in self.entry_signals]
            signal_prices = [signal['price'] for signal in self.entry_signals]

            self.signal_scatter = self.ax.scatter(
                signal_times,
                signal_prices,
                marker='*',  # Star shape - stands out well
                color='#43A047',  # Material Design green
                edgecolors='#FFFFFF',  # White edge
                linewidth=1.5,
                s=160,  # Notably larger
                zorder=5,
                alpha=1.0,
                label="Entry Signal"
            )
        elif self.entry_signals:
            # Update existing signal scatter
            signal_times = [signal['time'] for signal in self.entry_signals]
            signal_prices = [signal['price'] for signal in self.entry_signals]
            print(f"Updating signal scatter offsets with {len(signal_times)} signals")
            self.signal_scatter.set_offsets(np.column_stack([signal_times, signal_prices]))

        if ohlcs.empty or (len(tickers) < 5 and len(ohlcs) < 5):
            return self.line, self.scatter, self.signal_scatter

        # Force redraw axis after first data is received
        current_time = time.time()
        if len(tickers) >= 5:
            print(f"First {len(tickers)} data points received - redrawing chart axes")
            self.redraw_axis(tickers)
            self.last_axis_redraw = current_time

        # Periodically redraw axis to keep up with data changes
        elif current_time - self.last_axis_redraw > self.axis_redraw_interval:
            print("Periodic axis redraw")
            self.redraw_axis(tickers)
            self.last_axis_redraw = current_time
        else:
            # Force redraw
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        # For proper legend handling
        if not self.legend_created:
            self.ax.legend()
            self.legend_created = True

        return self.line, self.scatter, self.signal_scatter

    def redraw_axis(self, data):
        """Properly recalculate and set axis limits based on data"""
        if len(data) == 0:
            return

        # Calculate appropriate y-axis range with buffer
        price_min = data['low'].min()
        price_max = data['high'].max()

        # Add more buffer for more volatile assets
        buffer_percentage = 0.05  # 5% buffer
        y_range = price_max - price_min

        y_min = price_min - (y_range * buffer_percentage)
        y_max = price_max + (y_range * buffer_percentage)

        # Set y-axis limits with buffer
        self.ax.set_ylim(y_min, y_max)

        # Set x-axis limits to show all data points
        self.ax.set_xlim(data.index.min(), data.index.max())

        # Format date labels properly
        self.fig.autofmt_xdate()

        # Force redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        print(f"Chart axes redrawn. Price range: {y_min:.2f} to {y_max:.2f}")

    def add_entry_signal(self, time, price):
        """Add an entry signal to the chart."""
        self.entry_signals.append({'time': time, 'price': price})

    def start(self):
        """Start the real-time chart."""
        self.ani = FuncAnimation(self.fig, self.update_chart, init_func=self.init_chart, interval=1000, blit=True,
                            cache_frame_data=False)
        plt.show()

# Global configuration
ORDER_QTY = 1.0  # Fixed order quantity for all trades

class TradingSignalGenerator:
    def __init__(self, symbol, target_return=3.0, hold_time=72):
        self.symbol = symbol
        self.target_return = target_return
        self.hold_time = hold_time

        # Separate DataFrames for ohlcs and tickers
        self.ohlcs = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap'])
        self.ohlcs.set_index('timestamp', inplace=True)
        self.tickers = pd.DataFrame(columns=['timestamp', 'bid', 'bid_qty', 'ask', 'ask_qty', 'last', 'volume', 'low'])
        self.tickers.set_index('timestamp', inplace=True)

        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        self.entry_time = None
        self.entry_price = None

        print(f"Signal generator initialized for {symbol}")
        print(f"Strategy parameters: Target Return = {target_return}%, Hold Time = {hold_time}h")

        self.chart = RealTimeChart(self)

    def add_ohlc(self, ohlc_data):
        print(f"Adding new ohlc at {ohlc_data['timestamp']} for {ohlc_data.get('symbol', self.symbol)}")

        """Add a new ohlc to the dataframe and update indicators"""
        try:
            timestamp = pd.to_datetime(int(ohlc_data['timestamp']), unit='s')

            # Add the new data
            new_data = pd.DataFrame([{
                'open': float(ohlc_data['open']),
                'high': float(ohlc_data['high']),
                'low': float(ohlc_data['low']),
                'close': float(ohlc_data['close']),
                'volume': float(ohlc_data['volume']),
                'vwap': float(ohlc_data['vwap']) if 'vwap' in ohlc_data else None
            }], index=[timestamp])

            self.ohlcs = pd.concat([self.ohlcs, new_data])

            # Keep only recent data to conserve memory (e.g., last 1000 ohlcs)
            if len(self.ohlcs) > 1000:
                self.ohlcs = self.ohlcs.iloc[-1000:]

            # Set this to 50 for testing purposes.
            # Update indicators if we have enough data
            if len(self.ohlcs) >= 30:
                self.calculate_indicators()
                self.check_signals(timestamp)
            else:
                print(f"Building history: {len(self.ohlcs)}/200 ohlcs")

            return True
        except Exception as e:
            print(f"Error processing ohlc: {e}")
            return False

    def add_ticker(self, formatted_ticker):
        print(f"Adding ticker ticker data at {formatted_ticker['timestamp']} for {formatted_ticker.get('symbol', self.symbol)}")

        """Add the ticker ticker data as a new row and update the chart."""
        try:
            timestamp = pd.to_datetime(int(formatted_ticker['timestamp']), unit='s')

            # Create a DataFrame for the ticker
            ticker = pd.DataFrame([{
                'bid': float(formatted_ticker['bid']),
                'bid_qty': float(formatted_ticker['bid_qty']),
                'ask': float(formatted_ticker['ask']),
                'ask_qty': float(formatted_ticker['ask_qty']),
                'last': float(formatted_ticker['last']),
                'volume': float(formatted_ticker['volume']),
                'low': float(formatted_ticker['low']),
                'high': float(formatted_ticker['high'])
            }], index=[timestamp])

            # Append to self.tickers, keeping only the last 1000 rows
            self.tickers = pd.concat([self.tickers, ticker])
            if len(self.tickers) > 1000:
                self.tickers = self.tickers.iloc[-1000:]

        except Exception as e:
            print(f"Error processing ticker ticker: {e}")

    def calculate_indicators(self):
        """Add your own implementation: calculate technical indicators for the latest data"""

    def check_signals(self, current_time):
        """Check for entry and exit signals"""
        # Skip if we don't have enough data
        if len(self.ohlcs) < 2:
            return

        df = self.ohlcs
        current_price = df['close'].iloc[-1]

        # Replace with your actual indicator calculations
        entry_signal = False

        if entry_signal:
            self.enter_trade(current_price, current_time)

    def enter_trade(self, price, time):
        """Enter a new trade"""
        self.in_position = True
        self.entry_price = price
        self.entry_time = time

        print(f"\n[{time}] ENTRY SIGNAL: {self.symbol}")
        print(f"Price: {price:.2f}")
        print(f"Target: {price * (1 + self.target_return / 100):.2f} (+{self.target_return}%)")
        print(f"Max hold time: {self.hold_time} hours")

        # Create and submit trade signal message
        trade_signal = {
            "type": "signal",
            "action": "entry",
            "symbol": self.symbol,
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "price": price,
            "target_price": price * (1 + self.target_return / 100),
            "target_return": self.target_return,
            "hold_time_hours": self.hold_time,
            "order_qty": ORDER_QTY  # Use the global order quantity
        }

        # Pass the message to the parent WebSocketClient to send
        if hasattr(self, 'ws_client') and self.ws_client and self.ws_client.ws:
            self.ws_client.ws.send(json.dumps(trade_signal))
            print(f"Trade entry signal sent to WebSocket")

        self.chart.add_entry_signal(time, price)  # Add signal to the chart

class WebSocketClient:
    def __init__(self, url, symbol):
        self.url = url
        self.symbol = symbol  # Now just a single symbol instead of a list
        self.ws = None
        self.signal_generator = None  # Single signal generator
        self.is_running = True
        self.reconnect_interval = 5  # seconds between reconnection attempts
        self.max_reconnect_attempts = 10  # maximum number of reconnection attempts
        self.reconnect_count = 0
        self.connection_established = False
        self.message_count = 0  # Track message count for debugging

        # Initialize the signal generator for the single symbol
        self.signal_generator = TradingSignalGenerator(
            symbol=symbol,
            target_return=3.0,  # 3% target return
            hold_time=72  # 72 hour hold limit
        )
        self.signal_generator.ws_client = self  # Add reference to the WebSocket client

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)

            # Log first few messages to help with debugging
            if self.message_count < 5:
                print(f"WebSocket message received: {message[:100]}...")
                self.message_count += 1

            # Handle message based on structure
            if 'channel' in data and data['channel'] == 'ohlc' and 'data' in data:
                # Process each ohlc in the data array
                for ohlc_data in data['data']:
                    if 'symbol' not in ohlc_data:
                        continue

                    symbol = ohlc_data['symbol']

                    # Only process if it matches our symbol
                    if symbol == self.symbol:
                        # Format ohlc data to match expected structure
                        formatted_ohlc = {
                            'symbol': symbol,
                            'timestamp': int(time.time()),
                            'open': ohlc_data['open'],
                            'high': ohlc_data['high'],
                            'low': ohlc_data['low'],
                            'close': ohlc_data['close'],
                            'volume': ohlc_data['volume'],
                            'vwap': ohlc_data.get('vwap', None)
                        }

                        self.signal_generator.add_ohlc(formatted_ohlc)
            # Handle ticker messages
            elif 'channel' in data and data['channel'] == 'ticker' and 'data' in data:
                for ticker_data in data['data']:
                    if 'symbol' not in ticker_data:
                        continue

                    symbol = ticker_data['symbol']

                    # Only process if it matches our symbol
                    if symbol == self.symbol:
                        # Format ohlc data to match expected structure
                        formatted_ticker = {
                            'symbol': symbol,
                            'timestamp': int(pd.to_datetime(data['timestamp']).timestamp()),
                            'bid': ticker_data['bid'],
                            'bid_qty': ticker_data['bid_qty'],
                            'ask': ticker_data['ask'],
                            'ask_qty': ticker_data['ask_qty'],
                            'last': ticker_data['last'],
                            'volume': ticker_data['volume'],
                            'low': ticker_data['low'],
                            'high': ticker_data['high']
                        }

                        self.signal_generator.add_ticker(formatted_ticker)

        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        # More detailed error information
        import traceback
        traceback.print_exc()

        # Try to get more details about the connection
        if isinstance(error, ConnectionRefusedError):
            print(f"Connection refused to {self.url}. Is the server running and accessible?")
        elif isinstance(error, TimeoutError):
            print(f"Connection timed out to {self.url}. Check network connectivity.")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

        # Log additional info about whether connection was ever established
        if not self.connection_established:
            print("WARNING: Connection closed without ever being successfully established")

        if self.is_running:
            self.reconnect()

    def on_open(self, ws):
        print(f"WebSocket connection established to {self.url}")
        self.connection_established = True
        # Reset reconnect count on successful connection
        self.reconnect_count = 0

        # Subscribe to market data for our single symbol
        subscription_msg = {
            "type": "subscribe",
            "symbol": self.symbol
        }
        try:
            ws.send(json.dumps(subscription_msg))
            print(f"Subscribed to {self.symbol}")
        except Exception as e:
            print(f"Error subscribing to {self.symbol}: {e}")

    def connect(self):
        """Establish WebSocket connection"""
        self.is_running = True

        # Enable trace for first connection attempt for debugging
        websocket.enableTrace(True)

        print(f"Attempting to connect to WebSocket at {self.url}")

        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        # Start WebSocket connection in a separate thread
        wst = threading.Thread(target=self._run_websocket)
        wst.daemon = True
        wst.start()

        time.sleep(30)

        self.signal_generator.chart.start()  # Start the real-time chart

        print(f"WebSocket client started for symbol: {self.symbol}")

        # Keep the main thread alive
        try:
            start_time = time.time()
            while self.is_running:
                time.sleep(1)
                # If connection hasn't established after 30 seconds, show diagnostics
                if not self.connection_established and time.time() - start_time > 30:
                    print(f"Connection not established after 30 seconds. Troubleshooting tips:")
                    print(f"1. Check if server at {self.url} is running")
                    print(f"2. Verify network connectivity and firewall settings")
                    print(f"3. Try connecting to the WebSocket from a browser or tool like wscat")
                    print(f"4. Check server logs for connection attempts")
                    print(f"5. Try different WebSocket libraries or tools to test")
                    start_time = time.time()  # Reset timer
        except KeyboardInterrupt:
            self.stop()

    def _run_websocket(self):
        """Run the WebSocket connection with retry parameters"""
        # Using ping_interval and ping_timeout helps detect disconnections
        try:
            print(f"Starting WebSocket connection to {self.url}")
            self.ws.run_forever(ping_interval=30, ping_timeout=10)
            print("WebSocket run_forever loop exited")
        except Exception as e:
            print(f"Exception in WebSocket connection: {e}")
            import traceback
            traceback.print_exc()

    def reconnect(self):
        """Attempt to reconnect the WebSocket"""
        if not self.is_running:
            return

        self.reconnect_count += 1
        if self.reconnect_count > self.max_reconnect_attempts:
            print(f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached. Stopping.")
            self.stop()
            return

        # Disable trace for reconnections to reduce noise
        websocket.enableTrace(False)

        backoff_time = min(self.reconnect_interval * (1.5 ** (self.reconnect_count - 1)), 60)
        print(f"Attempting to reconnect (attempt {self.reconnect_count}/{self.max_reconnect_attempts}) in {backoff_time:.1f} seconds...")
        time.sleep(backoff_time)  # Use exponential backoff

        # Create a new thread for the reconnection
        wst = threading.Thread(target=self._run_websocket)
        wst.daemon = True
        wst.start()

    def stop(self):
        """Stop the WebSocket client"""
        self.is_running = False
        if self.ws:
            try:
                self.ws.close()
                print("WebSocket connection closed successfully")
            except Exception as e:
                print(f"Error closing WebSocket: {e}")
        print("WebSocket client stopped")

    def test_connection(self):
        """Test connection to the server without WebSocket"""
        global sock
        import socket
        import urllib.parse

        # Parse the URL to get host and port
        parsed_url = urllib.parse.urlparse(self.url)

        # Extract host and port
        host = parsed_url.netloc.split(':')[0]
        port = parsed_url.port or (443 if parsed_url.scheme == 'wss' else 80)

        print(f"Testing TCP connection to {host}:{port}...")

        # Try to establish a TCP connection
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            if result == 0:
                print(f"TCP connection to {host}:{port} successful!")
            else:
                print(f"TCP connection to {host}:{port} failed with error code {result}")
                print("Possible issues:")
                print("- Server is not running")
                print("- Firewall is blocking the connection")
                print("- Incorrect host or port")
        except Exception as e:
            print(f"Error testing connection: {e}")
        finally:
            sock.close()

if __name__ == "__main__":
    # Example usage
    ws_url = "ws://localhost:8080"  # Replace with actual WebSocket endpoint
    symbol = "BTC/USD"  # Replace with symbols you want to monitor

    client = WebSocketClient(ws_url, symbol)

    # Test TCP connection first before attempting WebSocket
    client.test_connection()

    # Then try the full WebSocket connection
    client.connect()