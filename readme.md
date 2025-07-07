# Kraken Signal Generator

A sample real-time cryptocurrency trading signal generator that connects to 
[tingh-dev-kraken-trading server](https://github.com/tinghau/tingh-dev-kraken-trading), calculates technical indicators (which requires implementation), and 
generates trading signals based on configurable strategies.

## Features

- Real-time WebSocket connection to market data
- Live chart visualisation 
- Automatic reconnection with exponential backoff

## Components

### TradingSignalGenerator

Processes market data and generates trading signals based on technical indicators:
- Configurable target returns and holding periods
- Customisable technical indicators
- Entry and exit signal generation, two functions support this but require implementation:
  - **calculate_indicators()**: update the indicators based on the latest data
  - **generate_signals()**: consider the indicators and generate buy/sell signals

### RealTimeChart

Provides live visualisation of price data and trading signals:
- Real-time price charts
- Highlighted entry signals
- Automatic axis scaling
- Responsive updates

### WebSocketClient

Manages the WebSocket connection to the tingh-dev-kraken-trading app:
- Handles message processing
- Automatic reconnection with backoff
- Connection diagnostics
- Data subscription management

## Usage

```python
# Example usage
ws_url = "ws://localhost:8080"  # WebSocket server endpoint
symbol = "BTC/USD"              # Trading pair to monitor

client = WebSocketClient(ws_url, symbol)

# Test connection before attempting WebSocket
client.test_connection()

# Connect and start processing data
client.connect()
```

## Configuration

- `ORDER_QTY`: Fixed order quantity for all trades
- `target_return`: Target profit percentage (default: 3.0%)
- `hold_time`: Maximum position holding time in hours (default: 72)
- Technical indicator parameters can be customised in the `calculate_indicators()` method
- Entry and exit signal generation logic can be customised in the `generate_signals()` method

## Requirements

- Python 3.x
- pandas
- numpy
- websocket-client
- matplotlib
- ta (Technical Analysis library)

### Further
Blog post: https://tingh.dev/2025/07/04/kraken-signal.html