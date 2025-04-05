# Crypto Oracle - Advanced Trading Analysis System

A comprehensive cryptocurrency trading analysis system integrating technical analysis, machine learning, sentiment analysis, on-chain metrics, and AI-driven strategy generation.

## Overview

Crypto Oracle is a powerful trading analysis system designed to help traders make more informed decisions by combining multiple data sources and analytical approaches:

- **Technical Analysis**: Comprehensive indicators including RSI, MACD, Bollinger Bands, and more
- **Machine Learning**: LSTM neural networks for price forecasting and GARCH models for volatility prediction
- **Multi-Timeframe Analysis**: Examining patterns across different timeframes (1h, 4h, 1d)
- **Sentiment Analysis**: Social media, news, and market sentiment integration
- **On-Chain Metrics**: Blockchain data analysis for supported cryptocurrencies
- **GPT-Powered Strategy Generation**: AI-generated trading strategies based on comprehensive market data
- **Backtesting**: Historical performance validation of trading strategies
- **Portfolio Management**: Position tracking and risk management

## Features

### Market Analysis

- Fetch and analyze crypto data from major exchanges
- Compute 20+ technical indicators
- Multi-timeframe analysis to identify trends across different time periods
- Identify support/resistance levels and market phases
- Market scanning to discover trading opportunities

### Machine Learning Models

- LSTM price prediction with confidence scoring
- GARCH volatility modeling
- Value at Risk (VaR) calculations
- Automated model caching for improved performance

### Advanced Data Integration

- Social media sentiment from Twitter, Reddit, and more
- News sentiment analysis and aggregation
- Fear & Greed Index integration
- On-chain metrics for major cryptocurrencies
- Exchange inflow/outflow monitoring

### Trading Strategy Generation

- AI-driven strategy creation using GPT-4
- Entry, exit, stop-loss, and take-profit recommendations
- Position sizing suggestions based on risk parameters
- Strategy confidence scoring

### Backtesting and Validation

- Historical performance testing
- Strategy quality scoring
- Risk/reward evaluation
- Win rate, profit factor, and drawdown analysis

### Portfolio Management

- Position tracking and performance monitoring
- Risk management
- Portfolio performance statistics

## Installation

### Prerequisites

- Python 3.8+
- Required packages (install via pip):
  ```
  pip install -r requirements.txt
  ```

### Configuration

1. Create a `.env` file with the following variables:
   ```
   # Exchange API keys
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_API_SECRET=your_binance_api_secret
   
   # OpenAI API key for GPT-4 integration
   OPENAI_API_KEY=your_openai_api_key
   
   # Optional: For sentiment and on-chain metrics
   ENABLE_SENTIMENT=true
   ENABLE_ONCHAIN=true
   TWITTER_BEARER_TOKEN=your_twitter_token
   GLASSNODE_API_KEY=your_glassnode_key
   CRYPTOQUANT_API_KEY=your_cryptoquant_key
   ```

2. Customize settings in the configuration section of the main script if needed.

## Usage

### API Server

Run the API server to access all functionality via REST endpoints:

```bash
python crypto_oracle.py api --port 8000
```

### Command Line Interface

Analyze a specific cryptocurrency:

```bash
python crypto_oracle.py analyze BTC/USDT --timeframes 1h 4h 1d
```

Scan the market for opportunities:

```bash
python crypto_oracle.py scan --max-symbols 30 --min-confidence 0.7
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/tickers` | GET | Available trading pairs |
| `/timeframes` | GET | Available timeframes |
| `/analyze` | POST | Full analysis for a symbol |
| `/scan` | POST | Market scan for opportunities |
| `/sentiment/{symbol}` | GET | Sentiment data for a symbol |
| `/onchain/{symbol}` | GET | On-chain metrics for a symbol |
| `/backtest` | POST | Backtest a strategy |
| `/status` | GET | System status |

## Example

Analyzing Bitcoin:

```python
import requests
import json

response = requests.post('http://localhost:8000/analyze', json={
    'symbol': 'BTC/USDT',
    'timeframes': ['1h', '4h', '1d'],
    'lookback': 1000,
    'include_sentiment': True,
    'include_onchain': True
})

result = response.json()
print(json.dumps(result, indent=2))

# Access the strategy recommendation
strategy = result['data']['strategy']
print(f"Trade direction: {strategy['trade_direction']}")
print(f"Entry price: {strategy['optimal_entry']}")
print(f"Confidence score: {strategy['confidence_score']}")
```

## Dependencies

- CCXT: Exchange connectivity
- TensorFlow: Machine learning models
- FastAPI: API server
- Pandas/NumPy: Data processing
- TA-Lib (optional): Technical indicators
- OpenAI: GPT strategy generation

## Disclaimer

This software is for educational and research purposes only. It is not financial advice, and you should not make trading or investment decisions based solely on its output. Always do your own research and consult with a qualified financial advisor before trading.

## License

MIT License
