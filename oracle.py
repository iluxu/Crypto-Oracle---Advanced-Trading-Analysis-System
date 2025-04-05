if API_AVAILABLE:
    # Create FastAPI application
    app = FastAPI(
        title="Crypto Oracle API",
        description="Advanced cryptocurrency analysis and trading strategy generation",
        version="2.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this to your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models for request/response
    class TimeframeList(BaseModel):
        timeframes: List[str] = DEFAULT_TIMEFRAMES
    
    class AnalysisRequest(BaseModel):
        symbol: str
        timeframes: List[str] = DEFAULT_TIMEFRAMES
        lookback: int = 1000
        account_balance: float = DEFAULT_ACCOUNT_BALANCE
        max_leverage: float = 10.0
        include_sentiment: bool = True
        include_onchain: bool = True
        run_backtest: bool = True
    
    class ScanRequest(BaseModel):
        max_symbols: int = 20
        min_confidence: float = 0.6
        min_backtest_score: float = 0.5
        prefer_direction: Optional[str] = None  # 'long', 'short', or None for both
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_TASKS
        timeframes: List[str] = DEFAULT_TIMEFRAMES[:2]
    
    class PortfolioRequest(BaseModel):
        account_balance: float = DEFAULT_ACCOUNT_BALANCE
        max_positions: int = MAX_POSITIONS
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE
        max_leverage: float = 10.0
    
    @app.get("/", tags=["General"])
    async def root():
        """API root - provides basic system information"""
        return {
            "name": "Crypto Oracle API",
            "version": "2.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "technical_analysis": True,
                "machine_learning": ML_AVAILABLE,
                "sentiment_analysis": ENABLE_SENTIMENT,
                "onchain_metrics": ENABLE_ONCHAIN,
                "gpt_strategy": openai_client is not None
            }
        }
    
    @app.get("/tickers", tags=["Data"])
    async def get_tickers():
        """Get available trading pairs"""
        if not exchange_client:
            raise HTTPException(status_code=503, detail="Exchange client not available")
        
        try:
            # Ensure markets are loaded
            if not exchange_client.markets:
                loaded = await load_exchange_markets(exchange_client)
                if not loaded:
                    raise HTTPException(status_code=503, detail="Failed to load markets")
            
            # Filter for active USDT futures
            tickers = []
            for symbol, market in exchange_client.markets.items():
                if (market.get('active', False) and 
                    market.get('quote') == 'USDT' and 
                    market.get('swap', False) and 
                    market.get('future', False)):
                    tickers.append({
                        'symbol': symbol,
                        'base': market.get('base', ''),
                        'quote': market.get('quote', ''),
                        'type': 'perpetual'
                    })
            
            return {
                "success": True,
                "count": len(tickers),
                "tickers": tickers
            }
        
        except Exception as e:
            log.error(f"Error fetching tickers: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to fetch tickers: {str(e)}")
    
    @app.get("/timeframes", tags=["Data"])
    async def get_timeframes():
        """Get available timeframes"""
        return {
            "success": True,
            "timeframes": DEFAULT_TIMEFRAMES,
            "primary_timeframe": PRIMARY_TIMEFRAME
        }
    
    @app.post("/analyze", tags=["Analysis"])
    async def analyze_symbol_endpoint(request: AnalysisRequest):
        """
        Perform comprehensive analysis on a cryptocurrency
        """
        try:
            # Run the analysis
            result = await analyze_symbol(
                request.symbol,
                timeframes=request.timeframes,
                lookback=request.lookback,
                account_balance=request.account_balance,
                max_leverage=request.max_leverage,
                include_sentiment=request.include_sentiment,
                include_onchain=request.include_onchain,
                run_backtest=request.run_backtest
            )
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "symbol": request.symbol,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "data": result
            }
        
        except Exception as e:
            log.error(f"Error in analyze endpoint: {e}", exc_info=True)
            raise HTTPException(status# -*- coding: utf-8 -*-
"""
Crypto Oracle - Enhanced Trading Analysis System
================================================
A comprehensive crypto trading analysis system integrating technical analysis,
machine learning, sentiment analysis, on-chain metrics, and AI-driven strategy generation.

Key Features:
- Multi-timeframe analysis
- LSTM price forecasting
- GARCH volatility modeling
- GPT-driven strategy generation
- On-chain metrics integration
- Social media sentiment analysis
- Advanced backtesting
- Portfolio management

Author: CryptoOracle
Version: 2.0.0
"""

import os
import pandas as pd
import numpy as np
import time
import asyncio
import logging
import warnings
import json
import re
import ccxt
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import traceback
import requests
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="ConvergenceWarning", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("crypto_oracle.log")
    ]
)
log = logging.getLogger("crypto_oracle")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    log.info("Environment variables loaded from .env file")
except ImportError:
    log.warning("python-dotenv not installed. Using environment variables directly.")

#################################################
# TECHNICAL ANALYSIS MODULES
#################################################

# Silence TensorFlow warnings BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import ML/Stats libraries
try:
    from arch import arch_model
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, GRU, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    
    # Configure TensorFlow for better performance
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log.info(f"TensorFlow: Detected {len(gpus)} GPU(s). Memory growth enabled.")
            
            # Set memory limit if needed
            # for gpu in gpus:
            #     tf.config.set_logical_device_configuration(
            #         gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
            
        except RuntimeError as e:
            log.error(f"TensorFlow: Error setting GPU memory growth - {e}")
    else:
        log.info("TensorFlow: No GPU detected. Using CPU.")
        
    # Set thread count for CPU-based operations
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
    
    log.info("ML/Stats libraries imported successfully")
    ML_AVAILABLE = True
except ImportError as e:
    log.warning(f"Some ML/Stats libraries unavailable: {e}. ML features will be disabled.")
    ML_AVAILABLE = False

# Import OpenAI
try:
    import openai
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        log.info("OpenAI client initialized")
    else:
        openai_client = None
        log.warning("OPENAI_API_KEY not found. GPT features will be disabled.")
except ImportError:
    openai_client = None
    log.warning("OpenAI library not installed. GPT features will be disabled.")

# FastAPI for web service
try:
    from fastapi import FastAPI, HTTPException, Body, Query, Path, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, ConfigDict, validator
    import uvicorn
    from starlette.responses import FileResponse
    API_AVAILABLE = True
    log.info("API libraries imported successfully")
except ImportError as e:
    API_AVAILABLE = False
    log.warning(f"API libraries unavailable: {e}. API features will be disabled.")

# TA-Lib for technical indicators if available
try:
    import talib
    TALIB_AVAILABLE = True
    log.info("TA-Lib imported successfully")
except ImportError:
    TALIB_AVAILABLE = False
    log.warning("TA-Lib not installed. Using custom indicator implementations.")

#################################################
# CONFIGURATION
#################################################

# Default LSTM model parameters
LSTM_TIME_STEPS = 60
LSTM_EPOCHS = 15
LSTM_BATCH_SIZE = 64

# Default GARCH model parameters
GARCH_P = 1
GARCH_Q = 1

# Backtesting parameters
MAX_TRADES_PER_BACKTEST = 100
MAX_TRADE_DURATION_BARS = 96  # Max holding period (e.g., 4 days on 1h timeframe)

# Concurrency limits
DEFAULT_MAX_CONCURRENT_TASKS = 5  # For scanning

# Portfolio management defaults
MAX_POSITIONS = 5
DEFAULT_RISK_PER_TRADE = 0.02  # 2% risk per trade
DEFAULT_ACCOUNT_BALANCE = 10000  # Default account balance in USD

# Timeframes to analyze (multi-timeframe analysis)
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
PRIMARY_TIMEFRAME = "1h"  # Primary timeframe for signals

# Sentiment & on-chain metrics
ENABLE_SENTIMENT = os.getenv("ENABLE_SENTIMENT", "false").lower() in ('true', '1', 'yes')
ENABLE_ONCHAIN = os.getenv("ENABLE_ONCHAIN", "false").lower() in ('true', '1', 'yes')

# API keys for external services
CRYPTOWATCH_API_KEY = os.getenv("CRYPTOWATCH_API_KEY", "")
CRYPTOQUANT_API_KEY = os.getenv("CRYPTOQUANT_API_KEY", "")
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")

#################################################
# SENTIMENT ANALYSIS & SOCIAL DATA
#################################################

async def fetch_sentiment_data(symbol: str) -> Dict[str, Any]:
    """
    Fetch sentiment data from various sources
    """
    if not ENABLE_SENTIMENT:
        return {"enabled": False}
    
    # Extract base asset (e.g., BTC from BTC/USDT)
    base_asset = symbol.split('/')[0] if '/' in symbol else symbol.split(':')[0]
    
    result = {
        "enabled": True,
        "symbol": symbol,
        "base_asset": base_asset,
        "timestamp": datetime.now().isoformat(),
        "sources": {},
        "aggregated": {
            "sentiment_score": None,
            "social_volume": None,
            "trending_score": None
        }
    }
    
    try:
        # Process all sentiment sources concurrently
        tasks = [
            fetch_twitter_sentiment(base_asset),
            fetch_news_sentiment(base_asset),
            fetch_reddit_sentiment(base_asset),
            fetch_fear_greed_index()
        ]
        
        sources_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        twitter_data, news_data, reddit_data, fear_greed = sources_data
        
        # Store data from each source
        if isinstance(twitter_data, dict):
            result["sources"]["twitter"] = twitter_data
        else:
            log.warning(f"Failed to fetch Twitter sentiment: {twitter_data}")
            result["sources"]["twitter"] = {"error": str(twitter_data) if twitter_data else "Unknown error"}
            
        if isinstance(news_data, dict):
            result["sources"]["news"] = news_data
        else:
            log.warning(f"Failed to fetch news sentiment: {news_data}")
            result["sources"]["news"] = {"error": str(news_data) if news_data else "Unknown error"}
            
        if isinstance(reddit_data, dict):
            result["sources"]["reddit"] = reddit_data
        else:
            log.warning(f"Failed to fetch Reddit sentiment: {reddit_data}")
            result["sources"]["reddit"] = {"error": str(reddit_data) if reddit_data else "Unknown error"}
            
        if isinstance(fear_greed, dict):
            result["sources"]["fear_greed"] = fear_greed
        else:
            log.warning(f"Failed to fetch Fear & Greed index: {fear_greed}")
            result["sources"]["fear_greed"] = {"error": str(fear_greed) if fear_greed else "Unknown error"}
        
        # Aggregate sentiment scores (weighted average)
        scores = []
        weights = []
        
        # Twitter (weight: 0.3)
        if "twitter" in result["sources"] and "sentiment_score" in result["sources"]["twitter"]:
            scores.append(result["sources"]["twitter"]["sentiment_score"])
            weights.append(0.3)
            
        # News (weight: 0.3)
        if "news" in result["sources"] and "sentiment_score" in result["sources"]["news"]:
            scores.append(result["sources"]["news"]["sentiment_score"])
            weights.append(0.3)
            
        # Reddit (weight: 0.2)
        if "reddit" in result["sources"] and "sentiment_score" in result["sources"]["reddit"]:
            scores.append(result["sources"]["reddit"]["sentiment_score"])
            weights.append(0.2)
            
        # Fear & Greed (weight: 0.2)
        if "fear_greed" in result["sources"] and "value" in result["sources"]["fear_greed"]:
            # Normalize to -1 to 1 range: (value - 50) / 50
            normalized_score = (result["sources"]["fear_greed"]["value"] - 50) / 50
            scores.append(normalized_score)
            weights.append(0.2)
        
        # Calculate weighted average if we have scores
        if scores and weights:
            result["aggregated"]["sentiment_score"] = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Aggregate social volume (simple sum)
        social_volumes = []
        if "twitter" in result["sources"] and "volume" in result["sources"]["twitter"]:
            social_volumes.append(result["sources"]["twitter"]["volume"])
        if "reddit" in result["sources"] and "volume" in result["sources"]["reddit"]:
            social_volumes.append(result["sources"]["reddit"]["volume"])
            
        if social_volumes:
            result["aggregated"]["social_volume"] = sum(social_volumes)
        
        # Calculate trending score (combination of sentiment and volume)
        if result["aggregated"]["sentiment_score"] is not None and result["aggregated"]["social_volume"] is not None:
            sentiment = result["aggregated"]["sentiment_score"]  # -1 to 1
            volume = min(1.0, result["aggregated"]["social_volume"] / 1000)  # Normalize volume to 0-1
            # Trending score formula: volume * (1 + sentiment) for positive sentiment, volume * (1 - |sentiment|) for negative
            if sentiment >= 0:
                result["aggregated"]["trending_score"] = volume * (1 + sentiment)
            else:
                result["aggregated"]["trending_score"] = volume * (1 - abs(sentiment))
        
        return result
    
    except Exception as e:
        log.error(f"Error aggregating sentiment for {symbol}: {e}", exc_info=True)
        result["error"] = str(e)
        return result

async def fetch_twitter_sentiment(asset: str) -> Dict[str, Any]:
    """
    Fetch Twitter sentiment for a cryptocurrency using Twitter API v2
    """
    if not TWITTER_BEARER_TOKEN:
        return {"error": "Twitter API key not configured"}
    
    try:
        # Use asset name and common crypto tags
        query = f"#{asset} OR {asset} crypto OR {asset} cryptocurrency -is:retweet"
        
        # In real implementation, this would call Twitter API
        # Here we simulate a response
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Simulated response
        simulated_data = {
            "sentiment_score": random.uniform(-0.5, 0.8),  # -1 to 1 scale
            "volume": random.randint(100, 5000),
            "positive_count": random.randint(50, 250),
            "negative_count": random.randint(20, 150),
            "neutral_count": random.randint(30, 200),
            "trending_hashtags": [f"#{asset}", "#crypto", "#trading"]
        }
        
        return simulated_data
    
    except Exception as e:
        log.error(f"Error fetching Twitter sentiment for {asset}: {e}")
        return {"error": str(e)}

async def fetch_news_sentiment(asset: str) -> Dict[str, Any]:
    """
    Fetch news sentiment for a cryptocurrency using news API
    """
    try:
        # In real implementation, this would call a news API
        # Here we simulate a response
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Simulated response
        simulated_data = {
            "sentiment_score": random.uniform(-0.3, 0.7),  # -1 to 1 scale
            "article_count": random.randint(5, 50),
            "recent_headlines": [
                f"New developments in {asset} blockchain",
                f"{asset} price analysis: Technical indicators suggest bullish momentum",
                f"Major exchange adds new {asset} trading pairs"
            ]
        }
        
        return simulated_data
    
    except Exception as e:
        log.error(f"Error fetching news sentiment for {asset}: {e}")
        return {"error": str(e)}

async def fetch_reddit_sentiment(asset: str) -> Dict[str, Any]:
    """
    Fetch Reddit sentiment for a cryptocurrency
    """
    try:
        # In real implementation, this would call Reddit API
        # Here we simulate a response
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Simulated response
        simulated_data = {
            "sentiment_score": random.uniform(-0.6, 0.6),  # -1 to 1 scale
            "volume": random.randint(50, 2000),
            "popular_posts": random.randint(3, 20),
            "active_users": random.randint(100, 5000),
            "top_subreddits": [f"r/{asset}", "r/CryptoCurrency", "r/CryptoMarkets"]
        }
        
        return simulated_data
    
    except Exception as e:
        log.error(f"Error fetching Reddit sentiment for {asset}: {e}")
        return {"error": str(e)}

async def fetch_fear_greed_index() -> Dict[str, Any]:
    """
    Fetch Crypto Fear & Greed Index
    """
    try:
        # In real implementation, this would call the Fear & Greed API
        # Here we simulate a response
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Generate a value between 0-100
        value = random.randint(20, 80)
        
        # Determine classification
        if value >= 0 and value <= 24:
            classification = "Extreme Fear"
        elif value <= 49:
            classification = "Fear"
        elif value <= 74:
            classification = "Greed"
        else:
            classification = "Extreme Greed"
        
        # Simulated response
        simulated_data = {
            "value": value,
            "classification": classification,
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "previous_day": value + random.randint(-5, 5),
            "previous_week": value + random.randint(-10, 10)
        }
        
        return simulated_data
    
    except Exception as e:
        log.error(f"Error fetching Fear & Greed index: {e}")
        return {"error": str(e)}


#################################################
# ON-CHAIN METRICS
#################################################

async def fetch_onchain_metrics(symbol: str) -> Dict[str, Any]:
    """
    Fetch on-chain metrics for a cryptocurrency
    """
    if not ENABLE_ONCHAIN:
        return {"enabled": False}
    
    # Extract base asset (e.g., BTC from BTC/USDT)
    base_asset = symbol.split('/')[0] if '/' in symbol else symbol.split(':')[0]
    
    # Only proceed for major coins that typically have on-chain data
    supported_assets = {'BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'ADA', 'DOT', 'LINK', 'XLM', 'SOL'}
    
    if base_asset not in supported_assets:
        return {
            "enabled": True,
            "symbol": symbol,
            "base_asset": base_asset,
            "supported": False,
            "message": f"On-chain metrics not supported for {base_asset}"
        }
    
    result = {
        "enabled": True,
        "symbol": symbol,
        "base_asset": base_asset,
        "supported": True,
        "timestamp": datetime.now().isoformat(),
        "sources": {},
        "metrics": {}
    }
    
    try:
        # Process all on-chain data sources concurrently
        tasks = [
            fetch_glassnode_metrics(base_asset),
            fetch_cryptoquant_metrics(base_asset),
            fetch_chain_txn_metrics(base_asset)
        ]
        
        sources_data = await asyncio.gather(*tasks, return_exceptions=True)
        glassnode_data, cryptoquant_data, chain_txn_data = sources_data
        
        # Store data from each source
        if isinstance(glassnode_data, dict):
            result["sources"]["glassnode"] = glassnode_data
        else:
            log.warning(f"Failed to fetch Glassnode metrics: {glassnode_data}")
            result["sources"]["glassnode"] = {"error": str(glassnode_data) if glassnode_data else "Unknown error"}
            
        if isinstance(cryptoquant_data, dict):
            result["sources"]["cryptoquant"] = cryptoquant_data
        else:
            log.warning(f"Failed to fetch CryptoQuant metrics: {cryptoquant_data}")
            result["sources"]["cryptoquant"] = {"error": str(cryptoquant_data) if cryptoquant_data else "Unknown error"}
            
        if isinstance(chain_txn_data, dict):
            result["sources"]["chain_txn"] = chain_txn_data
        else:
            log.warning(f"Failed to fetch blockchain transaction metrics: {chain_txn_data}")
            result["sources"]["chain_txn"] = {"error": str(chain_txn_data) if chain_txn_data else "Unknown error"}
        
        # Compile the most important metrics from all sources
        metrics = result["metrics"]
        
        # Transaction metrics
        if "chain_txn" in result["sources"] and "transaction_count" in result["sources"]["chain_txn"]:
            metrics["daily_transactions"] = result["sources"]["chain_txn"]["transaction_count"]
            metrics["avg_transaction_value"] = result["sources"]["chain_txn"].get("avg_transaction_value")
            metrics["active_addresses"] = result["sources"]["chain_txn"].get("active_addresses")
        
        # Network metrics
        if "glassnode" in result["sources"]:
            gn = result["sources"]["glassnode"]
            metrics["hash_rate"] = gn.get("hash_rate")
            metrics["staking_rate"] = gn.get("staking_rate")
            metrics["nvt_ratio"] = gn.get("nvt_ratio")
            metrics["sopr"] = gn.get("sopr")  # Spent Output Profit Ratio
            metrics["supply_last_active"] = gn.get("supply_last_active_1y_percent")
        
        # Exchange and holder metrics
        if "cryptoquant" in result["sources"]:
            cq = result["sources"]["cryptoquant"]
            metrics["exchange_inflow"] = cq.get("exchange_inflow")
            metrics["exchange_outflow"] = cq.get("exchange_outflow")
            metrics["miners_to_exchange"] = cq.get("miners_to_exchange")
            metrics["whales_buying"] = cq.get("whales_buying")
        
        # Add some derived metrics if possible
        if "exchange_inflow" in metrics and "exchange_outflow" in metrics:
            if metrics["exchange_inflow"] is not None and metrics["exchange_outflow"] is not None:
                metrics["net_exchange_flow"] = metrics["exchange_outflow"] - metrics["exchange_inflow"]
        
        # High-level metrics for immediate use
        if base_asset == "BTC":
            # Bitcoin-specific metrics
            metrics["percent_supply_in_profit"] = result["sources"].get("glassnode", {}).get("percent_supply_in_profit")
            metrics["puell_multiple"] = result["sources"].get("glassnode", {}).get("puell_multiple")
        elif base_asset == "ETH":
            # Ethereum-specific metrics
            metrics["gas_price"] = result["sources"].get("chain_txn", {}).get("gas_price")
            metrics["defi_locked_value"] = result["sources"].get("glassnode", {}).get("defi_locked_value")
        
        # Calculate on-chain activity score (simplified)
        # This is a custom metric that combines several indicators into one score
        activity_components = []
        
        if "daily_transactions" in metrics and metrics["daily_transactions"] is not None:
            # Normalize to a 0-1 scale (asset-dependent calibration)
            max_txns = 1000000 if base_asset == "BTC" else 2000000  # Example thresholds
            activity_components.append(min(1.0, metrics["daily_transactions"] / max_txns))
        
        if "active_addresses" in metrics and metrics["active_addresses"] is not None:
            # Normalize to a 0-1 scale
            max_addresses = 1000000 if base_asset == "BTC" else 500000  # Example thresholds
            activity_components.append(min(1.0, metrics["active_addresses"] / max_addresses))
        
        if activity_components:
            metrics["on_chain_activity_score"] = sum(activity_components) / len(activity_components)
        
        return result
    
    except Exception as e:
        log.error(f"Error aggregating on-chain metrics for {symbol}: {e}", exc_info=True)
        result["error"] = str(e)
        return result

async def fetch_glassnode_metrics(asset: str) -> Dict[str, Any]:
    """
    Fetch metrics from Glassnode
    """
    if not GLASSNODE_API_KEY:
        return {"error": "Glassnode API key not configured"}
    
    try:
        # In real implementation, this would call Glassnode API
        # Here we simulate a response
        await asyncio.sleep(0.15)  # Simulate API delay
        
        # Simulated response - metrics will vary by asset
        if asset == "BTC":
            simulated_data = {
                "hash_rate": 150000000000000000000,  # 150 EH/s
                "difficulty": 25046487590033.8,
                "sopr": random.uniform(0.9, 1.1),  # Spent Output Profit Ratio
                "percent_supply_in_profit": random.uniform(60, 90),
                "nvt_ratio": random.uniform(20, 60),
                "puell_multiple": random.uniform(0.8, 4.0),
                "supply_last_active_1y_percent": random.uniform(40, 65)
            }
        elif asset == "ETH":
            simulated_data = {
                "hash_rate": 800000000000000,  # 800 TH/s
                "staking_rate": random.uniform(10, 20),  # Percentage of ETH staked
                "defi_locked_value": random.uniform(15, 50) * 1e9,  # In USD
                "percent_supply_in_profit": random.uniform(50, 85),
                "nvt_ratio": random.uniform(15, 45),
                "supply_last_active_1y_percent": random.uniform(35, 70)
            }
        else:
            # Generic metrics for other assets
            simulated_data = {
                "nvt_ratio": random.uniform(10, 80),
                "percent_supply_in_profit": random.uniform(40, 85)
            }
        
        return simulated_data
    
    except Exception as e:
        log.error(f"Error fetching Glassnode metrics for {asset}: {e}")
        return {"error": str(e)}

async def fetch_cryptoquant_metrics(asset: str) -> Dict[str, Any]:
    """
    Fetch metrics from CryptoQuant
    """
    if not CRYPTOQUANT_API_KEY:
        return {"error": "CryptoQuant API key not configured"}
    
    try:
        # In real implementation, this would call CryptoQuant API
        # Here we simulate a response
        await asyncio.sleep(0.1)  # Simulate API delay
        
        # Simulated response
        simulated_data = {
            "exchange_inflow": random.uniform(1000, 10000),  # Amount of asset flowing into exchanges
            "exchange_outflow": random.uniform(1000, 10000),  # Amount flowing out of exchanges
            "miners_to_exchange": random.uniform(100, 1000),  # Miner transfers to exchanges
            "exchange_reserves": random.uniform(1000000, 3000000),  # Total reserves on exchanges
            "whales_buying": bool(random.getrandbits(1)),  # Boolean indicating if whales are accumulating
            "miner_position_index": random.uniform(-0.5, 1.5)  # Sentiment of miners
        }
        
        return simulated_data
    
    except Exception as e:
        log.error(f"Error fetching CryptoQuant metrics for {asset}: {e}")
        return {"error": str(e)}

async def fetch_chain_txn_metrics(asset: str) -> Dict[str, Any]:
    """
    Fetch blockchain transaction metrics directly from node or API
    """
    try:
        # In real implementation, this would either:
        # 1. Query a blockchain node directly
        # 2. Use a blockchain explorer API
        # Here we simulate a response based on the asset
        await asyncio.sleep(0.2)  # Simulate API delay
        
        # Base metrics simulation
        transaction_count = random.randint(200000, 500000)
        fee_range = (1, 10) if asset == "BTC" else (10, 100) if asset == "ETH" else (0.1, 1)
        
        # Simulated response - values will differ by asset
        simulated_data = {
            "transaction_count": transaction_count,
            "avg_transaction_fee": random.uniform(*fee_range),
            "avg_transaction_value": random.uniform(1000, 10000),
            "active_addresses": random.randint(100000, 1000000),
            "new_addresses": random.randint(5000, 50000),
            "confirmation_time": random.uniform(5, 60),  # minutes
            "mempool_size": random.randint(1000, 20000)
        }
        
        # Add asset-specific metrics
        if asset == "ETH":
            simulated_data["gas_price"] = random.uniform(20, 100)  # Gwei
            simulated_data["contract_calls"] = random.randint(100000, 500000)
            simulated_data["defi_transactions"] = random.randint(50000, 200000)
        elif asset == "BTC":
            simulated_data["segwit_adoption"] = random.uniform(70, 85)  # percentage
            simulated_data["lightning_capacity"] = random.uniform(1000, 5000)  # BTC
        
        return simulated_data
    
    except Exception as e:
        log.error(f"Error fetching blockchain transaction metrics for {asset}: {e}")
        return {"error": str(e)}


#################################################
# AI-DRIVEN STRATEGY GENERATION
#################################################

async def generate_trading_strategy(
    symbol: str,
    price_data: Dict[str, pd.DataFrame],
    ml_predictions: Dict[str, Any],
    sentiment_data: Dict[str, Any] = None,
    onchain_data: Dict[str, Any] = None,
    account_balance: float = DEFAULT_ACCOUNT_BALANCE,
    max_leverage: float = 10.0
) -> Dict[str, Any]:
    """
    Generate a comprehensive trading strategy using GPT-4
    """
    if not openai_client:
        return {
            "error": "OpenAI client not available",
            "trade_direction": "hold",
            "explanation": "GPT-based strategy generation is not available."
        }
    
    # Get the primary timeframe data
    primary_df = price_data.get(PRIMARY_TIMEFRAME)
    
    if primary_df is None or primary_df.empty:
        return {
            "error": "No valid price data for primary timeframe",
            "trade_direction": "hold",
            "explanation": "Cannot generate strategy without price data."
        }
    
    # Prepare the prompt with comprehensive market information
    try:
        # Get the latest data points for each timeframe
        timeframe_data = {}
        for tf, df in price_data.items():
            if not df.empty:
                latest = df.iloc[-1].to_dict()
                timeframe_data[tf] = {
                    "close": latest.get("close"),
                    "open": latest.get("open"),
                    "high": latest.get("high"),
                    "low": latest.get("low"),
                    "volume": latest.get("volume"),
                    "rsi": latest.get("RSI"),
                    "macd": latest.get("MACD"),
                    "signal_line": latest.get("Signal_Line"),
                    "bollinger_upper": latest.get("Bollinger_Upper"),
                    "bollinger_lower": latest.get("Bollinger_Lower"),
                    "sma_50": latest.get("SMA_50"),
                    "sma_200": latest.get("SMA_200")
                }
        
        # Get current price
        current_price = primary_df["close"].iloc[-1]
        
        # Format prediction data
        prediction_info = {
            "lstm_forecast_price": ml_predictions.get("lstm_price"),
            "price_change_predicted": ml_predictions.get("price_change_pct"),
            "volatility_forecast": ml_predictions.get("garch_volatility"),
            "value_at_risk": ml_predictions.get("var95")
        }
        
        # Prepare market context
        market_context = {
            "symbol": symbol,
            "current_price": current_price,
            "predictions": prediction_info,
            "timeframes": timeframe_data,
            "market_phase": ml_predictions.get("market_phase", "unknown"),
            "support_resistance": ml_predictions.get("support_resistance", {})
        }
        
        # Add sentiment data if available
        if sentiment_data and sentiment_data.get("enabled", False):
            sentiment_summary = {
                "overall_sentiment": sentiment_data.get("aggregated", {}).get("sentiment_score"),
                "social_volume": sentiment_data.get("aggregated", {}).get("social_volume"),
                "trending_score": sentiment_data.get("aggregated", {}).get("trending_score"),
                "fear_greed_value": sentiment_data.get("sources", {}).get("fear_greed", {}).get("value")
            }
            market_context["sentiment"] = sentiment_summary
        
        # Add on-chain data if available
        if onchain_data and onchain_data.get("enabled", False) and onchain_data.get("supported", False):
            onchain_summary = {
                "active_addresses": onchain_data.get("metrics", {}).get("active_addresses"),
                "daily_transactions": onchain_data.get("metrics", {}).get("daily_transactions"),
                "exchange_flow": onchain_data.get("metrics", {}).get("net_exchange_flow"),
                "activity_score": onchain_data.get("metrics", {}).get("on_chain_activity_score")
            }
            market_context["onchain"] = onchain_summary
        
        # Add account context
        account_context = {
            "balance_usd": account_balance,
            "max_leverage": max_leverage
        }
        
        # Convert context to JSON for the prompt
        market_context_json = json.dumps(market_context, indent=2)
        account_context_json = json.dumps(account_context, indent=2)
        
        # Create the prompt for the GPT model
        prompt = f"""You are a professional cryptocurrency trading strategy advisor. Generate a trading strategy based on the following market data.

Market Data:
```json
{market_context_json}
```

Account Information:
```json
{account_context_json}
```

Instructions:
1. Analyze the provided market data across multiple timeframes
2. Consider the LSTM forecast, volatility predictions, and technical indicators
3. Factor in sentiment and on-chain metrics if available
4. Determine if there's a potential trading opportunity (long, short, or hold)
5. For trades, suggest entry, stop-loss, and take-profit prices based on support/resistance, volatility, and risk management
6. Recommend appropriate position sizing and leverage considering the account balance and risk

Provide your analysis and recommendation in the following JSON format:
```json
{{
  "trade_direction": "long" | "short" | "hold",
  "optimal_entry": float | null,
  "stop_loss": float | null,
  "take_profit": float | null,
  "leverage": int | null,
  "position_size_usd": float | null,
  "estimated_profit": float | null,
  "confidence_score": float,
  "analysis": {{
    "summary": "Brief summary of the overall trade idea",
    "technical_analysis": "Analysis of technical indicators across timeframes",
    "prediction_analysis": "Interpretation of ML predictions",
    "sentiment_analysis": "Analysis of sentiment data (if available)",
    "onchain_analysis": "Analysis of blockchain metrics (if available)",
    "risk_assessment": "Assessment of trade risks and volatility expectations"
  }}
}}
```

Your response must be a valid JSON object with all fields.
"""
        
        # Call the GPT model
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a cryptocurrency trading strategy advisor that produces structured JSON responses."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Extract and parse the response
        gpt_output = response.choices[0].message.content
        strategy = json.loads(gpt_output)
        
        # Add metadata
        strategy["timestamp"] = datetime.now().isoformat()
        strategy["gpt_model"] = "gpt-4o"
        
        return strategy
    
    except Exception as e:
        log.error(f"Error generating trading strategy for {symbol}: {e}", exc_info=True)
        return {
            "error": str(e),
            "trade_direction": "hold",
            "explanation": "Error occurred during strategy generation.",
            "timestamp": datetime.now().isoformat()
        }


#################################################
# BACKTESTING MODULE
#################################################

class TradeResult:
    """
    Represents the result of a single trade in a backtest
    """
    def __init__(self, entry_price, entry_time, direction, stop_loss, take_profit, leverage=1):
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        
        # These will be set when the trade is closed
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None  # 'tp', 'sl', 'timeout', 'force_close'
        self.profit_pct = None
        self.profit_pips = None
        self.duration = None  # in bars
    
    def close_trade(self, exit_price, exit_time, reason):
        """Close the trade and calculate profit"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        
        # Calculate profit/loss
        if self.direction == 'long':
            self.profit_pips = self.exit_price - self.entry_price
            self.profit_pct = (self.exit_price / self.entry_price - 1) * 100 * self.leverage
        else:  # short
            self.profit_pips = self.entry_price - self.exit_price
            self.profit_pct = (1 - self.exit_price / self.entry_price) * 100 * self.leverage
        
        # Calculate duration in bars
        if isinstance(self.entry_time, pd.Timestamp) and isinstance(exit_time, pd.Timestamp):
            # For timestamp index
            self.duration = (exit_time - self.entry_time).total_seconds()
        else:
            # For integer index
            self.duration = exit_time - self.entry_time
        
        return self.profit_pct
    
    def to_dict(self):
        """Convert trade result to dictionary"""
        return {
            'entry_price': self.entry_price,
            'entry_time': str(self.entry_time),
            'direction': self.direction,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'leverage': self.leverage,
            'exit_price': self.exit_price,
            'exit_time': str(self.exit_time) if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'profit_pips': self.profit_pips,
            'profit_pct': self.profit_pct,
            'duration': self.duration
        }

async def run_backtest(
    df: pd.DataFrame,
    strategy_params: Dict[str, Any],
    max_trades: int = MAX_TRADES_PER_BACKTEST,
    max_bars_per_trade: int = MAX_TRADE_DURATION_BARS
) -> Dict[str, Any]:
    """
    Run a backtest on historical data using the given strategy parameters
    """
    if df.empty or len(df) < 100:
        return {
            "error": "Not enough data for backtesting",
            "success": False
        }
    
    # Extract strategy parameters
    trade_direction = strategy_params.get("trade_direction", "hold")
    if trade_direction == "hold":
        return {
            "message": "No trade direction specified (hold). Skipping backtest.",
            "success": True,
            "trades": [],
            "statistics": None
        }
    
    optimal_entry = strategy_params.get("optimal_entry")
    stop_loss = strategy_params.get("stop_loss")
    take_profit = strategy_params.get("take_profit")
    leverage = strategy_params.get("leverage", 1)
    
    # Validate required parameters
    if None in [optimal_entry, stop_loss, take_profit]:
        return {
            "error": "Missing required strategy parameters (entry, stop-loss, or take-profit)",
            "success": False
        }
    
    # Check if parameters make sense
    if trade_direction == "long" and not (stop_loss < optimal_entry < take_profit):
        return {
            "error": f"Invalid long parameters: stop_loss ({stop_loss}) should be less than entry ({optimal_entry}) and take_profit ({take_profit}) should be greater than entry",
            "success": False
        }
    
    if trade_direction == "short" and not (take_profit < optimal_entry < stop_loss):
        return {
            "error": f"Invalid short parameters: take_profit ({take_profit}) should be less than entry ({optimal_entry}) and stop_loss ({stop_loss}) should be greater than entry",
            "success": False
        }
    
    # Prepare for backtest
    trades = []
    active_trade = None
    
    # Process each bar in the dataframe (skip the first row as we need previous data)
    for i in range(1, len(df)):
        current_bar = df.iloc[i]
        previous_bar = df.iloc[i-1]
        
        # Get current bar data (Open, High, Low, Close)
        current_open = current_bar['open']
        current_high = current_bar['high']
        current_low = current_bar['low']
        current_close = current_bar['close']
        
        # Handle active trade if exists
        if active_trade:
            # Calculate the current duration of the trade
            current_duration = i - active_trade.entry_time if isinstance(active_trade.entry_time, int) else i
            
            # Check for stop loss hit
            if (active_trade.direction == 'long' and current_low <= active_trade.stop_loss) or \
               (active_trade.direction == 'short' and current_high >= active_trade.stop_loss):
                # Stop loss hit
                active_trade.close_trade(active_trade.stop_loss, i, 'sl')
                trades.append(active_trade)
                active_trade = None
            
            # Check for take profit hit
            elif (active_trade.direction == 'long' and current_high >= active_trade.take_profit) or \
                 (active_trade.direction == 'short' and current_low <= active_trade.take_profit):
                # Take profit hit
                active_trade.close_trade(active_trade.take_profit, i, 'tp')
                trades.append(active_trade)
                active_trade = None
            
            # Check for timeout
            elif current_duration >= max_bars_per_trade:
                # Close at current price due to timeout
                active_trade.close_trade(current_close, i, 'timeout')
                trades.append(active_trade)
                active_trade = None
        
        # Look for entry conditions if no active trade
        elif len(trades) < max_trades and not active_trade:
            # Simple entry logic based on price crossing the optimal entry level
            entry_triggered = False
            
            # Long entry: Previous close below entry, current passes through entry
            if trade_direction == 'long' and previous_bar['close'] < optimal_entry and current_low <= optimal_entry <= current_high:
                entry_triggered = True
            
            # Short entry: Previous close above entry, current passes through entry
            elif trade_direction == 'short' and previous_bar['close'] > optimal_entry and current_low <= optimal_entry <= current_high:
                entry_triggered = True
            
            if entry_triggered:
                active_trade = TradeResult(
                    entry_price=optimal_entry,
                    entry_time=i,
                    direction=trade_direction,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage
                )
    
    # Close any open trade at the end of the test period
    if active_trade:
        active_trade.close_trade(df.iloc[-1]['close'], len(df) - 1, 'force_close')
        trades.append(active_trade)
    
    # Calculate backtest statistics
    if trades:
        stats = calculate_backtest_statistics(trades, df)
        return {
            "success": True,
            "trades": [t.to_dict() for t in trades],
            "statistics": stats
        }
    else:
        return {
            "success": True,
            "message": "No trades executed during backtest period",
            "trades": [],
            "statistics": None
        }

def calculate_backtest_statistics(trades: List[TradeResult], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate backtest statistics from a list of trades
    """
    if not trades:
        return None
    
    # Basic trade statistics
    num_trades = len(trades)
    win_trades = [t for t in trades if t.profit_pct > 0]
    loss_trades = [t for t in trades if t.profit_pct <= 0]
    
    num_wins = len(win_trades)
    num_losses = len(loss_trades)
    
    # Win rate
    win_rate = num_wins / num_trades if num_trades > 0 else 0
    
    # Profit statistics
    profit_pcts = [t.profit_pct for t in trades]
    total_profit_pct = sum(profit_pcts)
    avg_profit_pct = total_profit_pct / num_trades if num_trades > 0 else 0
    
    # Average win/loss
    avg_win_pct = sum(t.profit_pct for t in win_trades) / num_wins if num_wins > 0 else 0
    avg_loss_pct = sum(t.profit_pct for t in loss_trades) / num_losses if num_losses > 0 else 0
    
    # Profit factor (gross profit / gross loss)
    gross_profit = sum(t.profit_pct for t in win_trades) if win_trades else 0
    gross_loss = abs(sum(t.profit_pct for t in loss_trades)) if loss_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Max drawdown calculation
    equity_curve = [0]  # Start with 0%
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade.profit_pct)
    
    # Calculate drawdown
    max_equity = 0
    max_drawdown = 0
    for equity in equity_curve:
        max_equity = max(max_equity, equity)
        drawdown = max_equity - equity
        max_drawdown = max(max_drawdown, drawdown)
    
    # Return statistics as dictionary
    return {
        "total_trades": num_trades,
        "winning_trades": num_wins,
        "losing_trades": num_losses,
        "win_rate": round(win_rate, 4),
        "avg_profit_pct": round(avg_profit_pct, 4),
        "avg_win_pct": round(avg_win_pct, 4),
        "avg_loss_pct": round(avg_loss_pct, 4),
        "profit_factor": round(profit_factor, 4),
        "total_profit_pct": round(total_profit_pct, 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "expectancy": round(avg_win_pct * win_rate - abs(avg_loss_pct) * (1 - win_rate), 4),
        "sharpe_ratio": calculate_sharpe_ratio(profit_pcts),
        "strategy_quality_score": calculate_strategy_score(win_rate, profit_factor, num_trades, max_drawdown)
    }

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0):
    """Calculate Sharpe ratio from a list of returns"""
    if len(returns) < 2:
        return 0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    return round(sharpe, 4)

def calculate_strategy_score(win_rate: float, profit_factor: float, num_trades: int, max_drawdown: float) -> float:
    """
    Calculate a normalized score for the strategy quality
    
    Parameters:
    - win_rate: Percentage of winning trades (0.0-1.0)
    - profit_factor: Ratio of gross profit to gross loss
    - num_trades: Number of trades executed
    - max_drawdown: Maximum drawdown percentage
    
    Returns:
    - score: 0.0-1.0 score indicating strategy quality
    """
    # Weights for each component
    win_rate_weight = 0.3
    profit_factor_weight = 0.3
    num_trades_weight = 0.2
    drawdown_weight = 0.2
    
    # Normalize each component to 0-1 scale
    win_rate_score = win_rate  # Already 0-1
    
    # Profit factor normalization (0 to 5+)
    profit_factor_score = min(1.0, profit_factor / 5)
    
    # Number of trades normalization (0 to 50+)
    trades_score = min(1.0, num_trades / 50)
    
    # Drawdown normalization (0% to 50%+)
    # Lower drawdown is better, so invert
    drawdown_score = max(0, 1 - (max_drawdown / 50))
    
    # Weighted sum
    final_score = (
        win_rate_score * win_rate_weight +
        profit_factor_score * profit_factor_weight +
        trades_score * num_trades_weight +
        drawdown_score * drawdown_weight
    )
    
    return round(final_score, 4)


#################################################
# PORTFOLIO MANAGEMENT
#################################################

class Position:
    """
    Represents an open trading position
    """
    def __init__(self, symbol, entry_price, direction, stop_loss, take_profit, position_size, leverage=1):
        self.symbol = symbol
        self.entry_price = entry_price
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size  # in USD
        self.leverage = leverage
        self.entry_time = datetime.now()
        self.last_update_time = self.entry_time
        
        # Track P&L
        self.current_price = entry_price
        self.current_pnl = 0
        self.current_pnl_pct = 0
    
    def update_price(self, current_price):
        """Update the position with the current market price"""
        self.current_price = current_price
        self.last_update_time = datetime.now()
        
        # Calculate P&L
        if self.direction == 'long':
            price_change_pct = (current_price / self.entry_price - 1)
        else:  # short
            price_change_pct = (1 - current_price / self.entry_price)
        
        # Apply leverage to P&L
        self.current_pnl_pct = price_change_pct * 100 * self.leverage
        self.current_pnl = self.position_size * price_change_pct * self.leverage
    
    def should_close(self):
        """Check if the position should be closed (hit stop loss or take profit)"""
        if self.direction == 'long':
            if self.current_price <= self.stop_loss:
                return 'stop_loss'
            if self.current_price >= self.take_profit:
                return 'take_profit'
        else:  # short
            if self.current_price >= self.stop_loss:
                return 'stop_loss'
            if self.current_price <= self.take_profit:
                return 'take_profit'
        
        return None
    
    def to_dict(self):
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            

def initialize_exchange():
    """Initialize the exchange client with proper error handling"""
    try:
        exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'defaultType': 'future',
                'defaultSubType': 'linear'  # Explicitly state linear
            },
            'timeout': 30000,  # 30 seconds timeout
            'rateLimit': 200   # 200ms between requests
        })
        log.info("CCXT Binance Futures (USDM) instance created")
        return exchange
    except Exception as e:
        log.error(f"Error initializing CCXT: {e}", exc_info=True)
        return None

# Initialize exchange client
exchange_client = initialize_exchange()

async def load_exchange_markets(exchange, force=False):
    """Load exchange markets asynchronously"""
    if not exchange:
        return False
    
    # Skip if markets already loaded and force=False
    if not force and exchange.markets:
        return True
    
    try:
        log.info(f"Loading exchange markets for {exchange.id}...")
        markets = await asyncio.to_thread(exchange.load_markets, True)  # Force reload
        if markets:
            log.info(f"Successfully loaded {len(markets)} markets for {exchange.id}")
            return True
        else:
            log.warning(f"Market loading returned empty for {exchange.id}")
            return False
    except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
        log.error(f"Failed to load markets due to network error: {e}")
        return False
    except ccxt.ExchangeError as e:
        log.error(f"Failed to load markets due to exchange error: {e}")
        return False
    except Exception as e:
        log.error(f"Unexpected error loading markets: {e}", exc_info=True)
        return False

async def get_ohlcv_data(symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """Fetch OHLCV data from exchange asynchronously"""
    if not exchange_client:
        raise ConnectionError("Exchange client is not available")
    
    # Ensure markets are loaded
    if not exchange_client.markets:
        loaded = await load_exchange_markets(exchange_client)
        if not loaded:
            raise ConnectionError("Failed to load markets")
    
    try:
        log.debug(f"Fetching OHLCV for {symbol}, timeframe={timeframe}, limit={limit}")
        ohlcv = await asyncio.to_thread(
            exchange_client.fetch_ohlcv, 
            symbol, 
            timeframe=timeframe, 
            limit=limit
        )
        
        if not ohlcv:
            log.warning(f"No OHLCV data returned for {symbol} {timeframe}")
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Ensure correct numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing essential data
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        
        # Tag dataframe with metadata
        df.attrs['symbol'] = symbol
        df.attrs['timeframe'] = timeframe
        
        return df
    
    except ccxt.BadSymbol as e:
        raise ValueError(f"Invalid symbol '{symbol}': {e}")
    except ccxt.RateLimitExceeded as e:
        raise ConnectionAbortedError(f"Rate limit exceeded: {e}")
    except ccxt.NetworkError as e:
        raise ConnectionError(f"Network error: {e}")
    except ccxt.ExchangeError as e:
        raise RuntimeError(f"Exchange error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

async def get_multi_timeframe_data(symbol: str, timeframes: List[str], limit: int = 1000) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple timeframes concurrently"""
    tasks = [get_ohlcv_data(symbol, tf, limit) for tf in timeframes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    data = {}
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            log.error(f"Error fetching {symbol} data for {timeframes[i]}: {result}")
            # Create empty DataFrame as placeholder
            data[timeframes[i]] = pd.DataFrame()
        else:
            data[timeframes[i]] = result
    
    return data

#################################################
# TECHNICAL INDICATOR FUNCTIONS
#################################################

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for the given dataframe"""
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate returns first (needed for other indicators)
    df_copy['returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1)).fillna(0)
    
    # Get symbol for logging if available
    symbol_name = df_copy.attrs.get('symbol', 'Unknown')
    
    # Use TA-Lib if available, otherwise use custom implementations
    if TALIB_AVAILABLE:
        try:
            # RSI
            df_copy['RSI'] = talib.RSI(df_copy['close'], timeperiod=14)
            
            # Moving Averages
            df_copy['SMA_50'] = talib.SMA(df_copy['close'], timeperiod=50)
            df_copy['SMA_200'] = talib.SMA(df_copy['close'], timeperiod=200)
            df_copy['EMA_12'] = talib.EMA(df_copy['close'], timeperiod=12)
            df_copy['EMA_26'] = talib.EMA(df_copy['close'], timeperiod=26)
            
            # MACD
            macd, signal, hist = talib.MACD(df_copy['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df_copy['MACD'] = macd
            df_copy['Signal_Line'] = signal
            df_copy['MACD_Hist'] = hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(df_copy['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            df_copy['Bollinger_Upper'] = upper
            df_copy['Bollinger_Middle'] = middle
            df_copy['Bollinger_Lower'] = lower
            
            # ATR
            df_copy['ATR'] = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(df_copy['high'], df_copy['low'], df_copy['close'], 
                                       fastk_period=14, slowk_period=3, slowk_matype=0, 
                                       slowd_period=3, slowd_matype=0)
            df_copy['Stochastic_K'] = slowk
            df_copy['Stochastic_D'] = slowd
            
            # Williams %R
            df_copy['Williams_%R'] = talib.WILLR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
            
            # ADX
            df_copy['ADX'] = talib.ADX(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
            
            # CCI
            df_copy['CCI'] = talib.CCI(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=20)
            
            # OBV
            df_copy['OBV'] = talib.OBV(df_copy['close'], df_copy['volume'])
            
            # Additional indicators
            df_copy['VWAP'] = (df_copy['close'] * df_copy['volume']).cumsum() / df_copy['volume'].cumsum()
            df_copy['ROC'] = talib.ROC(df_copy['close'], timeperiod=10)  # Rate of Change
            df_copy['MFI'] = talib.MFI(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], timeperiod=14)  # Money Flow Index
            
            log.debug(f"[{symbol_name}] Calculated technical indicators using TA-Lib")
            
        except Exception as e:
            log.error(f"[{symbol_name}] Error calculating TA-Lib indicators: {e}")
            # Fallback to custom implementations
            df_copy = compute_technical_indicators_custom(df_copy)
    else:
        # Use custom implementations
        df_copy = compute_technical_indicators_custom(df_copy)
        
    return df_copy

def compute_technical_indicators_custom(df: pd.DataFrame) -> pd.DataFrame:
    """Custom implementations of technical indicators"""
    df_copy = df.copy()
    symbol_name = df_copy.attrs.get('symbol', 'Unknown')
    
    try:
        # Calculate custom indicators
        
        # RSI
        delta = df_copy['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df_copy['RSI'] = 100 - (100 / (1 + rs)).fillna(50)
        
        # Moving Averages
        df_copy['SMA_50'] = df_copy['close'].rolling(window=50).mean()
        df_copy['SMA_200'] = df_copy['close'].rolling(window=200).mean()
        df_copy['EMA_12'] = df_copy['close'].ewm(span=12, adjust=False).mean()
        df_copy['EMA_26'] = df_copy['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df_copy['MACD'] = df_copy['EMA_12'] - df_copy['EMA_26']
        df_copy['Signal_Line'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
        df_copy['MACD_Hist'] = df_copy['MACD'] - df_copy['Signal_Line']
        
        # Bollinger Bands
        sma = df_copy['close'].rolling(window=20).mean()
        std = df_copy['close'].rolling(window=20).std()
        df_copy['Bollinger_Upper'] = sma + 2 * std
        df_copy['Bollinger_Middle'] = sma
        df_copy['Bollinger_Lower'] = sma - 2 * std
        
        # ATR
        high_low = df_copy['high'] - df_copy['low']
        high_close = abs(df_copy['high'] - df_copy['close'].shift())
        low_close = abs(df_copy['low'] - df_copy['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_copy['ATR'] = tr.rolling(window=14).mean()
        
        # Stochastic Oscillator
        lowest_low = df_copy['low'].rolling(window=14).min()
        highest_high = df_copy['high'].rolling(window=14).max()
        df_copy['Stochastic_K'] = 100 * ((df_copy['close'] - lowest_low) / (highest_high - lowest_low)).fillna(50)
        df_copy['Stochastic_D'] = df_copy['Stochastic_K'].rolling(window=3).mean()
        
        # Williams %R
        df_copy['Williams_%R'] = -100 * ((highest_high - df_copy['close']) / (highest_high - lowest_low)).fillna(-50)
        
        # OBV (On-Balance Volume)
        df_copy['OBV'] = (np.sign(df_copy['close'].diff()) * df_copy['volume']).fillna(0).cumsum()
        
        # ADX (Average Directional Index)
        df_copy['plus_dm'] = df_copy['high'].diff()
        df_copy['minus_dm'] = df_copy['low'].shift().diff(-1)
        df_copy['plus_dm'] = df_copy['plus_dm'].apply(lambda x: x if x > 0 else 0)
        df_copy['minus_dm'] = df_copy['minus_dm'].apply(lambda x: x if x > 0 else 0)
        
        # Calculate true range
        df_copy['plus_di'] = 100 * (df_copy['plus_dm'].ewm(span=14, adjust=False).mean() / df_copy['ATR'])
        df_copy['minus_di'] = 100 * (df_copy['minus_dm'].ewm(span=14, adjust=False).mean() / df_copy['ATR'])
        df_copy['dx'] = 100 * abs(df_copy['plus_di'] - df_copy['minus_di']) / (df_copy['plus_di'] + df_copy['minus_di'])
        df_copy['ADX'] = df_copy['dx'].ewm(span=14, adjust=False).mean()
        
        # CCI (Commodity Channel Index)
        tp = (df_copy['high'] + df_copy['low'] + df_copy['close']) / 3
        ma_tp = tp.rolling(window=20).mean()
        md_tp = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean())
        df_copy['CCI'] = (tp - ma_tp) / (0.015 * md_tp)
        
        # VWAP
        df_copy['VWAP'] = (df_copy['close'] * df_copy['volume']).cumsum() / df_copy['volume'].cumsum()
        
        # Rate of Change
        df_copy['ROC'] = df_copy['close'].pct_change(periods=10) * 100
        
        # Remove temporary columns
        for col in ['plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx']:
            if col in df_copy.columns:
                df_copy.drop(columns=[col], inplace=True)
        
        log.debug(f"[{symbol_name}] Calculated technical indicators using custom implementations")
    
    except Exception as e:
        log.error(f"[{symbol_name}] Error calculating custom indicators: {e}", exc_info=True)
    
    return df_copy

def get_market_phase(df: pd.DataFrame) -> str:
    """
    Determine the market phase (trending up, trending down, or ranging)
    based on price action and indicators
    """
    if df.empty or len(df) < 50:
        return "unknown"
    
    # Get the latest values
    latest = df.iloc[-1]
    
    # Check if necessary indicators exist
    if 'ADX' not in latest or 'SMA_50' not in latest or 'SMA_200' not in latest:
        return "unknown"
    
    # Extract indicator values
    adx = latest['ADX']
    close = latest['close']
    sma50 = latest['SMA_50']
    sma200 = latest['SMA_200']
    
    # Determine market phase
    if adx >= 25:  # Strong trend
        if close > sma50 and sma50 > sma200:
            return "trending_up"
        elif close < sma50 and sma50 < sma200:
            return "trending_down"
        else:
            return "ranging"
    else:  # Weak trend or ranging
        return "ranging"

def find_support_resistance(df: pd.DataFrame, window_size: int = 20, min_touches: int = 2) -> dict:
    """
    Find support and resistance levels using swing highs/lows
    """
    if df.empty or len(df) < window_size:
        return {"support": [], "resistance": []}
    
    supports = []
    resistances = []
    
    # Find local minima/maxima
    for i in range(window_size, len(df) - window_size):
        # Check if this is a local minimum (support)
        if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window_size+1)):
            supports.append(df['low'].iloc[i])
        
        # Check if this is a local maximum (resistance)
        if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window_size+1)):
            resistances.append(df['high'].iloc[i])
    
    # Filter levels that have multiple touches
    current_price = df['close'].iloc[-1]
    
    # Cluster nearby levels
    def cluster_levels(levels, threshold=0.005):
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If this level is close to the previous one, add to current cluster
            if level <= current_cluster[-1] * (1 + threshold):
                current_cluster.append(level)
            else:
                # Start a new cluster
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        # Add the last cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
        
        return clustered
    
    # Cluster and filter
    clustered_supports = cluster_levels(supports)
    clustered_resistances = cluster_levels(resistances)
    
    # Filter levels based on proximity to current price
    relevant_supports = [level for level in clustered_supports if level < current_price]
    relevant_resistances = [level for level in clustered_resistances if level > current_price]
    
    # Sort by distance to current price
    relevant_supports.sort(key=lambda x: current_price - x)
    relevant_resistances.sort(key=lambda x: x - current_price)
    
    return {
        "support": relevant_supports[:3],  # Return top 3 closest supports
        "resistance": relevant_resistances[:3]  # Return top 3 closest resistances
    }

#################################################
# MACHINE LEARNING MODELS
#################################################

# Model cache to avoid retraining for the same symbol/timeframe
model_cache = {}

def get_cached_model(symbol, timeframe):
    """Get a cached model if available"""
    key = f"{symbol}_{timeframe}"
    if key in model_cache:
        model_info = model_cache[key]
        # Check if model is recent enough (less than 1 hour old)
        if time.time() - model_info['timestamp'] < 3600:
            return model_info['model'], model_info['scaler']
    return None, None

def cache_model(symbol, timeframe, model, scaler):
    """Cache a model for future use"""
    key = f"{symbol}_{timeframe}"
    model_cache[key] = {
        'model': model,
        'scaler': scaler,
        'timestamp': time.time()
    }
    # Clean old models from cache if it gets too large
    if len(model_cache) > 100:
        oldest_key = min(model_cache.keys(), key=lambda k: model_cache[k]['timestamp'])
        del model_cache[oldest_key]

def fit_garch_model(returns: pd.Series) -> Optional[float]:
    """
    Fit GARCH(1,1) model and return next period conditional volatility
    """
    if not ML_AVAILABLE:
        return None
    
    valid_returns = returns.dropna() * 100  # Scale for GARCH stability
    if len(valid_returns) < 50:
        return None
    
    try:
        am = arch_model(valid_returns, vol='Garch', p=GARCH_P, q=GARCH_Q, dist='Normal')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(update_freq=0, disp='off', show_warning=False)
        
        if res.convergence_flag == 0:
            forecasts = res.forecast(horizon=1, reindex=False)
            cond_vol_forecast = np.sqrt(forecasts.variance.iloc[-1, 0]) / 100.0
            return float(cond_vol_forecast) if np.isfinite(cond_vol_forecast) else None
        else:
            return None
    except Exception as e:
        log.debug(f"GARCH fitting error: {e}")
        return None

def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> Optional[float]:
    """
    Calculate Value at Risk at specified confidence level
    """
    valid_returns = returns.dropna()
    if len(valid_returns) < 20:
        return None
    
    try:
        var_quantile = (1.0 - confidence_level) * 100.0
        var_value = np.percentile(valid_returns, var_quantile)
        return float(var_value) if np.isfinite(var_value) else None
    except Exception as e:
        log.debug(f"Error calculating VaR: {e}")
        return None

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced features for machine learning models
    """
    if df.empty or len(df) < 30:
        return df
    
    df_feat = df.copy()
    
    try:
        # Price-based features
        df_feat['returns'] = df_feat['close'].pct_change()
        df_feat['log_returns'] = np.log(df_feat['close'] / df_feat['close'].shift(1))
        
        # Volatility features
        df_feat['volatility_5'] = df_feat['returns'].rolling(window=5).std()
        df_feat['volatility_15'] = df_feat['returns'].rolling(window=15).std()
        df_feat['volatility_30'] = df_feat['returns'].rolling(window=30).std()
        
        # Price range features
        df_feat['price_range'] = (df_feat['high'] - df_feat['low']) / df_feat['close']
        df_feat['price_range_ma5'] = df_feat['price_range'].rolling(window=5).mean()
        
        # Volume features
        df_feat['volume_change'] = df_feat['volume'].pct_change()
        df_feat['volume_ma5'] = df_feat['volume'].rolling(window=5).mean()
        df_feat['volume_ma15'] = df_feat['volume'].rolling(window=15).mean()
        df_feat['relative_volume'] = df_feat['volume'] / df_feat['volume_ma15']
        
        # Trend features
        df_feat['close_ma5'] = df_feat['close'].rolling(window=5).mean()
        df_feat['close_ma15'] = df_feat['close'].rolling(window=15).mean()
        df_feat['close_ma30'] = df_feat['close'].rolling(window=30).mean()
        
        # Momentum features
        df_feat['momentum_5'] = df_feat['close'] - df_feat['close'].shift(5)
        df_feat['momentum_15'] = df_feat['close'] - df_feat['close'].shift(15)
        
        # Price distance from moving averages (normalized)
        df_feat['dist_from_ma5'] = (df_feat['close'] - df_feat['close_ma5']) / df_feat['close_ma5']
        df_feat['dist_from_ma15'] = (df_feat['close'] - df_feat['close_ma15']) / df_feat['close_ma15']
        df_feat['dist_from_ma30'] = (df_feat['close'] - df_feat['close_ma30']) / df_feat['close_ma30']
        
        # Moving average crossovers (binary signals)
        df_feat['ma5_above_ma15'] = (df_feat['close_ma5'] > df_feat['close_ma15']).astype(int)
        df_feat['ma5_above_ma30'] = (df_feat['close_ma5'] > df_feat['close_ma30']).astype(int)
        
        # Higher-order features
        df_feat['returns_squared'] = df_feat['returns'] ** 2
        df_feat['volume_returns'] = df_feat['returns'] * df_feat['volume_change']
        
        return df_feat
    
    except Exception as e:
        log.error(f"Error engineering features: {e}", exc_info=True)
        return df

def prepare_lstm_data(data: np.ndarray, time_steps: int, features: int = 1):
    """
    Prepare data for LSTM model
    """
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, 0])  # Predict the close price (first feature)
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, model_type='standard'):
    """
    Build LSTM model with different architectures
    """
    if not ML_AVAILABLE:
        return None
    
    try:
        if model_type == 'standard':
            model = Sequential([
                Input(shape=input_shape),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=1)
            ])
        elif model_type == 'bidirectional':
            model = Sequential([
                Input(shape=input_shape),
                Bidirectional(LSTM(units=50, return_sequences=True)),
                Dropout(0.2),
                Bidirectional(LSTM(units=50, return_sequences=False)),
                Dropout(0.2),
                Dense(units=1)
            ])
        elif model_type == 'gru':
            model = Sequential([
                Input(shape=input_shape),
                GRU(units=50, return_sequences=True),
                Dropout(0.2),
                GRU(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=1)
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model
    
    except Exception as e:
        log.error(f"Error building LSTM model: {e}", exc_info=True)
        return None

def train_lstm_model(model, X_train, y_train, epochs, batch_size):
    """
    Train LSTM model with robust error handling
    """
    if model is None or X_train.size == 0 or y_train.size == 0:
        return None
    
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=max(3, epochs // 5),
        restore_best_weights=True,
        verbose=0
    )
    
    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        return model
    except Exception as e:
        log.error(f"Error during LSTM model training: {e}", exc_info=False)
        return None

def forecast_with_lstm(model, X_input):
    """
    Forecast using LSTM with error handling
    """
    if model is None or not isinstance(model, tf.keras.Model) or not model.built:
        return None
    
    if not isinstance(X_input, np.ndarray):
        log.error(f"Invalid input type: {type(X_input)}. Expected numpy ndarray.")
        return None
    
    if len(X_input.shape) != 3:
        log.error(f"Invalid input shape: {X_input.shape}. Expected 3D array.")
        return None
    
    if not np.all(np.isfinite(X_input)):
        log.error(f"Input data contains non-finite values (NaN or Inf).")
        return None
    
    try:
        # Force prediction on CPU to avoid GPU memory issues
        with tf.device('/cpu:0'):
            forecast = model.predict(X_input, verbose=0)
        
        if forecast is None or not isinstance(forecast, np.ndarray):
            log.warning(f"Model prediction returned unexpected type: {type(forecast)}")
            return None
            
        if not np.all(np.isfinite(forecast)):
            log.warning(f"Prediction contains non-finite values: {forecast}")
            return None
            
        return forecast
    except tf.errors.InvalidArgumentError as e:
        log.error(f"TensorFlow InvalidArgumentError during prediction: {e}", exc_info=False)
        return None
    except Exception as e:
        log.error(f"Unexpected error during prediction: {e}", exc_info=True)
        return None

async def run_price_prediction(symbol: str, timeframes: List[str] = None, lookback: int = 1000):
    """
    Run comprehensive price prediction with LSTM, GARCH, and other models
    """
    if not ML_AVAILABLE:
        return {"error": "ML libraries not available", "predictions": {}}
    
    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES
    
    try:
        # Fetch data for all timeframes
        multi_tf_data = await get_multi_timeframe_data(symbol, timeframes, lookback)
        
        # Initialize results
        prediction_results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "predictions": {}
        }
        
        # Process each timeframe
        for tf, df in multi_tf_data.items():
            if df.empty:
                prediction_results["predictions"][tf] = {"error": f"No data available for timeframe {tf}"}
                continue
            
            # Apply technical indicators and feature engineering
            df_processed = compute_technical_indicators(df)
            df_processed = engineer_features(df_processed)
            
            # Check for cached model
            cached_model, cached_scaler = get_cached_model(symbol, tf)
            
            # LSTM prediction
            lstm_result = await run_lstm_prediction(df_processed, cached_model, cached_scaler, symbol, tf)
            
            # Run GARCH and VaR in parallel
            garch_vol = await asyncio.to_thread(fit_garch_model, df_processed['returns'])
            var95 = await asyncio.to_thread(calculate_var, df_processed['returns'], 0.95)
            
            # Get market phase and support/resistance levels
            market_phase = get_market_phase(df_processed)
            support_resistance = find_support_resistance(df_processed)
            
            # Store all predictions for this timeframe
            timeframe_predictions = {
                "lstm_price": lstm_result.get("forecast_price"),
                "price_change_pct": lstm_result.get("price_change_pct"),
                "forecast_confidence": lstm_result.get("confidence"),
                "garch_volatility": garch_vol,
                "var95": var95,
                "market_phase": market_phase,
                "support_resistance": support_resistance
            }
            
            # Add timeframe results to predictions
            prediction_results["predictions"][tf] = timeframe_predictions
        
        # Add aggregated prediction across timeframes
        prediction_results = aggregate_predictions(prediction_results)
        
        return prediction_results
    
    except Exception as e:
        log.error(f"Error in price prediction for {symbol}: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "predictions": {}
        }

async def run_lstm_prediction(df, cached_model=None, cached_scaler=None, symbol=None, timeframe=None):
    """
    Run LSTM prediction on the given dataframe
    """
    if df.empty or not ML_AVAILABLE:
        return {"error": "No data or ML not available"}
    
    result = {}
    
    try:
        # Extract close prices
        close_prices = df['close'].values.reshape(-1, 1)
        
        # Use cached model and scaler if available
        model = cached_model
        scaler = cached_scaler
        
        # Create new model and scaler if not cached
        if model is None or scaler is None:
            # Scale the data
            scaler = StandardScaler()
            scaled_prices = scaler.fit_transform(close_prices)
            
            # Prepare sequences for LSTM
            X, y = prepare_lstm_data(scaled_prices, LSTM_TIME_STEPS)
            
            if len(X) == 0:
                return {"error": "Not enough data to prepare LSTM sequences"}
            
            # Reshape input for LSTM [samples, time steps, features]
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build and train model
            input_shape = (X_reshaped.shape[1], 1)
            model = build_lstm_model(input_shape)
            
            if model is None:
                return {"error": "Failed to build LSTM model"}
            
            model = await asyncio.to_thread(
                train_lstm_model, 
                model, 
                X_reshaped, 
                y, 
                LSTM_EPOCHS, 
                LSTM_BATCH_SIZE
            )
            
            if model is None:
                return {"error": "Failed to train LSTM model"}
            
            # Cache the model and scaler
            if symbol and timeframe:
                cache_model(symbol, timeframe, model, scaler)
        
        # Prepare input sequence for prediction
        if len(close_prices) < LSTM_TIME_STEPS:
            return {"error": "Not enough data for prediction"}
        
        # Get the latest sequence
        last_sequence = close_prices[-LSTM_TIME_STEPS:]
        scaled_sequence = scaler.transform(last_sequence)
        
        # Reshape for prediction [1, time_steps, features]
        X_pred = scaled_sequence.reshape(1, LSTM_TIME_STEPS, 1)
        
        # Make prediction
        scaled_prediction = await asyncio.to_thread(forecast_with_lstm, model, X_pred)
        
        if scaled_prediction is None:
            return {"error": "LSTM prediction failed"}
        
        # Inverse transform to get the actual prediction
        prediction = scaler.inverse_transform(scaled_prediction)[0, 0]
        
        # Calculate prediction metrics
        current_price = close_prices[-1][0]
        price_change = prediction - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Calculate confidence based on recent model validation
        # This is a simplified approach - in production you might use validation metrics
        confidence = 0.7  # Default confidence score
        
        # Store prediction results
        result = {
            "forecast_price": float(prediction),
            "current_price": float(current_price),
            "price_change": float(price_change),
            "price_change_pct": float(price_change_pct),
            "confidence": confidence
        }
        
        return result
    
    except Exception as e:
        log.error(f"Error in LSTM prediction: {e}", exc_info=True)
        return {"error": str(e)}
    
    finally:
        # Always clear TensorFlow session to prevent memory leaks
        try:
            tf.keras.backend.clear_session()
        except:
            pass

def aggregate_predictions(prediction_results):
    """
    Aggregate predictions from multiple timeframes
    """
    if "predictions" not in prediction_results:
        return prediction_results
    
    timeframe_predictions = prediction_results["predictions"]
    if not timeframe_predictions:
        return prediction_results
    
    # Prioritize specific timeframes
    priority_order = ["1d", "4h", "1h", "15m", "5m"]
    
    # Initialize aggregate values
    agg_prediction = {
        "lstm_price": None,
        "price_change_pct": None,
        "forecast_confidence": None,
        "garch_volatility": None,
        "var95": None,
        "market_phase": None,
        "overall_trend": None,
        "primary_timeframe": None
    }
    
    # First, try to use primary timeframe if available
    if PRIMARY_TIMEFRAME in timeframe_predictions:
        primary_tf_pred = timeframe_predictions[PRIMARY_TIMEFRAME]
        agg_prediction["primary_timeframe"] = PRIMARY_TIMEFRAME
        agg_prediction["lstm_price"] = primary_tf_pred.get("lstm_price")
        agg_prediction["price_change_pct"] = primary_tf_pred.get("price_change_pct")
        agg_prediction["forecast_confidence"] = primary_tf_pred.get("forecast_confidence")
        agg_prediction["garch_volatility"] = primary_tf_pred.get("garch_volatility")
        agg_prediction["var95"] = primary_tf_pred.get("var95")
        agg_prediction["market_phase"] = primary_tf_pred.get("market_phase")
    else:
        # Otherwise, find the first available timeframe based on priority
        for tf in priority_order:
            if tf in timeframe_predictions and "lstm_price" in timeframe_predictions[tf]:
                tf_pred = timeframe_predictions[tf]
                agg_prediction["primary_timeframe"] = tf
                agg_prediction["lstm_price"] = tf_pred.get("lstm_price")
                agg_prediction["price_change_pct"] = tf_pred.get("price_change_pct")
                agg_prediction["forecast_confidence"] = tf_pred.get("forecast_confidence")
                agg_prediction["garch_volatility"] = tf_pred.get("garch_volatility")
                agg_prediction["var95"] = tf_pred.get("var95")
                agg_prediction["market_phase"] = tf_pred.get("market_phase")
                break
    
    # Determine overall trend from multiple timeframes
    trend_votes = {"bullish": 0, "bearish": 0, "neutral": 0}
    for tf, tf_pred in timeframe_predictions.items():
        if "price_change_pct" in tf_pred:
            change_pct = tf_pred["price_change_pct"]
            if change_pct is not None:
                if change_pct > 1.0:
                    trend_votes["bullish"] += 1
                elif change_pct < -1.0:
                    trend_votes["bearish"] += 1
                else:
                    trend_votes["neutral"] += 1
    
    # Determine overall trend based on votes
    if trend_votes["bullish"] > trend_votes["bearish"]:
        agg_prediction["overall_trend"] = "bullish"
    elif trend_votes["bearish"] > trend_votes["bullish"]:
        agg_prediction["overall_trend"] = "bearish"
    else:
        agg_prediction["overall_trend"] = "neutral"
    
    # Add aggregated results to the prediction results
    prediction_results["aggregated"] = agg_prediction
    
    return prediction_results


#################################################
# CORE ANALYSIS FUNCTIONS
#################################################

async def analyze_symbol(
    symbol: str, 
    timeframes: List[str] = None,
    lookback: int = 1000,
    account_balance: float = DEFAULT_ACCOUNT_BALANCE,
    max_leverage: float = 10.0,
    include_sentiment: bool = True,
    include_onchain: bool = True,
    run_backtest: bool = True
):
    """
    Perform comprehensive analysis on a symbol
    """
    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES
    
    analysis_start_time = time.time()
    log.info(f"Starting analysis for {symbol} across timeframes: {timeframes}")
    
    result = {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "timeframes": timeframes,
        "duration_ms": None
    }
    
    try:
        # Run tasks concurrently
        tasks = [
            get_multi_timeframe_data(symbol, timeframes, lookback)
        ]
        
        # Only include sentiment if enabled
        if include_sentiment and ENABLE_SENTIMENT:
            tasks.append(fetch_sentiment_data(symbol))
        else:
            sentiment_data = {"enabled": False}
        
        # Only include on-chain metrics if enabled and supported for this symbol
        if include_onchain and ENABLE_ONCHAIN:
            tasks.append(fetch_onchain_metrics(symbol))
        else:
            onchain_data = {"enabled": False}
        
        # Wait for data fetch tasks to complete
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        price_data = task_results[0]
        if isinstance(price_data, Exception):
            log.error(f"Error fetching price data for {symbol}: {price_data}")
            result["error"] = f"Failed to fetch price data: {str(price_data)}"
            return result
        
        # Check if we have any valid price data
        if all(df.empty for df in price_data.values()):
            result["error"] = f"No valid price data available for {symbol}"
            return result
        
        # Process sentiment data if included
        if include_sentiment and ENABLE_SENTIMENT:
            sentiment_data = task_results[1]
            if isinstance(sentiment_data, Exception):
                log.error(f"Error fetching sentiment data for {symbol}: {sentiment_data}")
                sentiment_data = {"enabled": False, "error": str(sentiment_data)}
        
        # Process on-chain data if included
        if include_onchain and ENABLE_ONCHAIN:
            onchain_data = task_results[-1]
            if isinstance(onchain_data, Exception):
                log.error(f"Error fetching on-chain data for {symbol}: {onchain_data}")
                onchain_data = {"enabled": False, "error": str(onchain_data)}
        
        # Apply technical indicators to all timeframes
        processed_data = {}
        for tf, df in price_data.items():
            if not df.empty:
                processed_data[tf] = compute_technical_indicators(df)
        
        # Run price prediction models
        prediction_results = await run_price_prediction(symbol, timeframes, lookback)
        
        # Generate trading strategy
        strategy = await generate_trading_strategy(
            symbol, 
            processed_data, 
            prediction_results.get("aggregated", {}),
            sentiment_data if include_sentiment and ENABLE_SENTIMENT else None,
            onchain_data if include_onchain and ENABLE_ONCHAIN else None,
            account_balance,
            max_leverage
        )
        
        # Run backtest if requested and we have a tradeable strategy
        backtest_result = None
        if run_backtest and strategy.get("trade_direction") in ["long", "short"]:
            # Use primary timeframe for backtesting
            backtest_df = processed_data.get(PRIMARY_TIMEFRAME)
            if backtest_df is not None and not backtest_df.empty:
                backtest_result = await run_backtest(backtest_df, strategy)
        
        # Compile the final result
        result.update({
            "price_data": {
                tf: {
                    "last_price": float(df["close"].iloc[-1]) if not df.empty else None,
                    "last_updated": df.index[-1].isoformat() if not df.empty else None,
                    "data_points": len(df) if not df.empty else 0
                } for tf, df in price_data.items()
            },
            "predictions": prediction_results,
            "sentiment": sentiment_data if include_sentiment and ENABLE_SENTIMENT else {"enabled": False},
            "onchain": onchain_data if include_onchain and ENABLE_ONCHAIN else {"enabled": False},
            "strategy": strategy,
            "backtest": backtest_result
        })
        
        # Calculate total duration
        analysis_duration = (time.time() - analysis_start_time) * 1000  # Convert to ms
        result["duration_ms"] = round(analysis_duration, 2)
        
        log.info(f"Analysis completed for {symbol} in {analysis_duration:.2f}ms")
        return result
    
    except Exception as e:
        log.error(f"Unexpected error analyzing {symbol}: {e}", exc_info=True)
        result["error"] = f"Analysis failed: {str(e)}"
        result["duration_ms"] = (time.time() - analysis_start_time) * 1000
        return result


async def scan_market(
    filter_params: dict = None,
    max_symbols: int = 20,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT_TASKS,
    timeframes: List[str] = None
):
    """
    Scan the market for trading opportunities based on filter parameters
    """
    if timeframes is None:
        timeframes = DEFAULT_TIMEFRAMES[:2]  # Use fewer timeframes for scanning to improve speed
    
    scan_start_time = time.time()
    log.info(f"Starting market scan with max_symbols={max_symbols}, max_concurrent={max_concurrent}")
    
    if filter_params is None:
        filter_params = {}
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "filter_params": filter_params,
        "max_symbols": max_symbols,
        "total_symbols_checked": 0,
        "opportunities": [],
        "errors": [],
        "duration_seconds": None
    }
    
    try:
        # Ensure markets are loaded
        if not exchange_client.markets:
            loaded = await load_exchange_markets(exchange_client)
            if not loaded:
                raise ConnectionError("Failed to load markets for scanning")
        
        # Get available symbols (focus on USDT futures)
        all_symbols = []
        for symbol_info in exchange_client.markets.values():
            if (symbol_info.get('active') and 
                symbol_info.get('quote') == 'USDT' and 
                symbol_info.get('swap', False) and  # Perpetual swaps/futures
                symbol_info.get('future', False)):
                all_symbols.append(symbol_info['symbol'])
        
        if not all_symbols:
            raise ValueError("No active symbols found for scanning")
        
        result["total_available_symbols"] = len(all_symbols)
        
        # Apply initial filters to symbols if specified
        symbols_to_scan = all_symbols[:max_symbols]  # Limit to max_symbols
        result["symbols_to_scan"] = len(symbols_to_scan)
        
        # Set up semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Wrapper function to use semaphore
        async def analyze_with_semaphore(symbol):
            async with semaphore:
                try:
                    return await analyze_symbol(
                        symbol,
                        timeframes=timeframes,
                        lookback=500,  # Use fewer bars for scanning to improve speed
                        include_sentiment=False,  # Skip sentiment for scanning to improve speed
                        include_onchain=False,  # Skip on-chain for scanning to improve speed
                        run_backtest=True  # Include backtest for verification
                    )
                except Exception as e:
                    log.error(f"Error analyzing {symbol} during scan: {e}")
                    return {"symbol": symbol, "error": str(e)}
        
        # Create tasks for all symbols
        tasks = [analyze_with_semaphore(symbol) for symbol in symbols_to_scan]
        
        # Process results as they complete
        completed = 0
        for task in asyncio.as_completed(tasks):
            analysis = await task
            completed += 1
            
            # Log progress
            if completed % 5 == 0 or completed == len(symbols_to_scan):
                log.info(f"Scan progress: {completed}/{len(symbols_to_scan)} symbols analyzed")
            
            # Check for errors
            if "error" in analysis:
                result["errors"].append({
                    "symbol": analysis.get("symbol", "unknown"),
                    "error": analysis["error"]
                })
                continue
            
            # Check if this is a trading opportunity
            strategy = analysis.get("strategy", {})
            trade_direction = strategy.get("trade_direction")
            confidence_score = strategy.get("confidence_score", 0)
            
            if trade_direction in ["long", "short"] and confidence_score >= 0.6:
                # Check backtest if available
                backtest = analysis.get("backtest", {})
                backtest_score = 0
                
                if backtest and backtest.get("statistics"):
                    stats = backtest["statistics"]
                    backtest_score = stats.get("strategy_quality_score", 0)
                
                # Determine if this is a good opportunity
                if confidence_score >= 0.7 or backtest_score >= 0.7:
                    # Calculate combined score
                    combined_score = 0.7 * confidence_score + 0.3 * backtest_score
                    
                    # Add to opportunities if score is high enough
                    if combined_score >= 0.65:
                        opportunity = {
                            "symbol": analysis["symbol"],
                            "direction": trade_direction,
                            "confidence_score": confidence_score,
                            "backtest_score": backtest_score,
                            "combined_score": combined_score,
                            "entry_price": strategy.get("optimal_entry"),
                            "stop_loss": strategy.get("stop_loss"),
                            "take_profit": strategy.get("take_profit"),
                            "risk_reward_ratio": calculate_risk_reward(
                                trade_direction,
                                strategy.get("optimal_entry"),
                                strategy.get("stop_loss"),
                                strategy.get("take_profit")
                            ),
                            "summary": strategy.get("analysis", {}).get("summary", "")
                        }
                        result["opportunities"].append(opportunity)
        
        # Update final counts
        result["total_symbols_checked"] = completed
        result["opportunities_found"] = len(result["opportunities"])
        
        # Sort opportunities by combined score
        result["opportunities"] = sorted(
            result["opportunities"],
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # Calculate duration
        scan_duration = time.time() - scan_start_time
        result["duration_seconds"] = round(scan_duration, 2)
        
        log.info(f"Market scan completed in {scan_duration:.2f}s. Found {len(result['opportunities'])} opportunities.")
        return result
    
    except Exception as e:
        log.error(f"Error during market scan: {e}", exc_info=True)
        result["error"] = str(e)
        result["duration_seconds"] = time.time() - scan_start_time
        return result

def calculate_risk_reward(direction, entry, stop_loss, take_profit):
    """Calculate risk-reward ratio for a trade setup"""
    if None in [direction, entry, stop_loss, take_profit]:
        return None
    
    if direction == "long":
        if entry <= stop_loss or entry >= take_profit:
            return None
        risk = entry - stop_loss
        reward = take_profit - entry
    else:  # short
        if entry >= stop_loss or entry <= take_profit:
            return None
        risk = stop_loss - entry
        reward = entry - take_profit
    
    if risk <= 0:
        return None
    
    return round(reward / risk, 2)


#################################################
# API MODELS AND ENDPOINTS
#################################################

if API_AVAILABLE:
    # Create FastAPI application
    app = FastAPI(
        title="Crypto Oracle API",
        description="Advanced cryptocurrency analysis and trading strategy generation",
        version="2.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this to your frontend domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models for request/response
    class TimeframeList(BaseModel):
        timeframes: List[str] = DEFAULT_TIMEFRAMES
    
    class AnalysisRequest(BaseModel):
        symbol: str
        timeframes: List[str] = DEFAULT_TIMEFRAMES
        lookback: int = 1000
        account_balance: float = DEFAULT_ACCOUNT_BALANCE
        max_leverage: float = 10.0
        include_sentiment: bool = True
        include_onchain: bool = True
        run_backtest: bool = True
    
    class ScanRequest(BaseModel):
        max_symbols: int = 20
        min_confidence: float = 0.6
        min_backtest_score: float = 0.5
        prefer_direction: Optional[str] = None  # 'long', 'short', or None for both
        max_concurrent: int = DEFAULT_MAX_CONCURRENT_TASKS
        timeframes: List[str] = DEFAULT_TIMEFRAMES[:2]
    
    class PortfolioRequest(BaseModel):
        account_balance: float = DEFAULT_ACCOUNT_BALANCE
        max_positions: int = MAX_POSITIONS
        risk_per_trade: float = DEFAULT_RISK_PER_TRADE
        max_leverage: float = 10.0
    
    @app.get("/", tags=["General"])
    async def root():
        """API root - provides basic system information"""
        return {
            "name": "Crypto Oracle API",
            "version": "2.0.0",
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "technical_analysis": True,
                "machine_learning": ML_AVAILABLE,
                "sentiment_analysis": ENABLE_SENTIMENT,
                "onchain_metrics": ENABLE_ONCHAIN,
                "gpt_strategy": openai_client is not None
            }
        }
    
    @app.get("/tickers", tags=["Data"])
    async def get_tickers():
        """Get available trading pairs"""
        if not exchange_client:
            raise HTTPException(status_code=503, detail="Exchange client not available")
        
        try:
            # Ensure markets are loaded
            if not exchange_client.markets:
                loaded = await load_exchange_markets(exchange_client)
                if not loaded:
                    raise HTTPException(status_code=503, detail="Failed to load markets")
            
            # Filter for active USDT futures
            tickers = []
            for symbol, market in exchange_client.markets.items():
                if (market.get('active', False) and 
                    market.get('quote') == 'USDT' and 
                    market.get('swap', False) and 
                    market.get('future', False)):
                    tickers.append({
                        'symbol': symbol,
                        'base': market.get('base', ''),
                        'quote': market.get('quote', ''),
                        'type': 'perpetual'
                    })
            
            return {
                "success": True,
                "count": len(tickers),
                "tickers": tickers
            }
        
        except Exception as e:
            log.error(f"Error fetching tickers: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to fetch tickers: {str(e)}")
    
    @app.get("/timeframes", tags=["Data"])
    async def get_timeframes():
        """Get available timeframes"""
        return {
            "success": True,
            "timeframes": DEFAULT_TIMEFRAMES,
            "primary_timeframe": PRIMARY_TIMEFRAME
        }
    
    @app.post("/analyze", tags=["Analysis"])
    async def analyze_symbol_endpoint(request: AnalysisRequest):
        """
        Perform comprehensive analysis on a cryptocurrency
        """
        try:
            # Run the analysis
            result = await analyze_symbol(
                request.symbol,
                timeframes=request.timeframes,
                lookback=request.lookback,
                account_balance=request.account_balance,
                max_leverage=request.max_leverage,
                include_sentiment=request.include_sentiment,
                include_onchain=request.include_onchain,
                run_backtest=request.run_backtest
            )
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "symbol": request.symbol,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "data": result
            }
        
        except Exception as e:
            log.error(f"Error in analyze endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    @app.post("/scan", tags=["Analysis"])
    async def scan_market_endpoint(request: ScanRequest):
        """
        Scan the market for trading opportunities
        """
        try:
            # Prepare filter parameters
            filter_params = {
                "min_confidence": request.min_confidence,
                "min_backtest_score": request.min_backtest_score,
                "prefer_direction": request.prefer_direction
            }
            
            # Run the scan
            result = await scan_market(
                filter_params=filter_params,
                max_symbols=request.max_symbols,
                max_concurrent=request.max_concurrent,
                timeframes=request.timeframes
            )
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "data": result
            }
        
        except Exception as e:
            log.error(f"Error in scan endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Market scan failed: {str(e)}")
    
    @app.get("/sentiment/{symbol}", tags=["Data"])
    async def get_sentiment(symbol: str):
        """
        Get sentiment data for a specific cryptocurrency
        """
        if not ENABLE_SENTIMENT:
            return {
                "success": False,
                "error": "Sentiment analysis is disabled",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            sentiment_data = await fetch_sentiment_data(symbol)
            
            if "error" in sentiment_data:
                return {
                    "success": False,
                    "error": sentiment_data["error"],
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "data": sentiment_data
            }
        
        except Exception as e:
            log.error(f"Error fetching sentiment for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to fetch sentiment: {str(e)}")
    
    @app.get("/onchain/{symbol}", tags=["Data"])
    async def get_onchain(symbol: str):
        """
        Get on-chain metrics for a specific cryptocurrency
        """
        if not ENABLE_ONCHAIN:
            return {
                "success": False,
                "error": "On-chain metrics are disabled",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            onchain_data = await fetch_onchain_metrics(symbol)
            
            if "error" in onchain_data:
                return {
                    "success": False,
                    "error": onchain_data["error"],
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "data": onchain_data
            }
        
        except Exception as e:
            log.error(f"Error fetching on-chain metrics for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to fetch on-chain metrics: {str(e)}")
    
    @app.post("/backtest", tags=["Analysis"])
    async def backtest_endpoint(
        symbol: str, 
        timeframe: str = "1h",
        lookback: int = 1000,
        strategy: dict = Body(...)
    ):
        """
        Backtest a trading strategy on historical data
        """
        try:
            # Fetch historical data
            df = await get_ohlcv_data(symbol, timeframe, lookback)
            
            if df.empty:
                return {
                    "success": False,
                    "error": "No historical data available",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Apply technical indicators
            df_processed = compute_technical_indicators(df)
            
            # Run backtest
            result = await run_backtest(df_processed, strategy)
            
            if "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": result
            }
        
        except Exception as e:
            log.error(f"Error in backtest endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
    
    @app.get("/status", tags=["General"])
    async def get_status():
        """
        Get system status and configuration
        """
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "system": {
                "version": "2.0.0",
                "uptime": "N/A",  # Would track actual uptime in production
                "exchange_connected": exchange_client is not None,
                "ml_available": ML_AVAILABLE,
                "sentiment_enabled": ENABLE_SENTIMENT,
                "onchain_enabled": ENABLE_ONCHAIN,
                "gpt_available": openai_client is not None
            },
            "config": {
                "primary_timeframe": PRIMARY_TIMEFRAME,
                "default_timeframes": DEFAULT_TIMEFRAMES,
                "default_account_balance": DEFAULT_ACCOUNT_BALANCE,
                "max_concurrent_tasks": DEFAULT_MAX_CONCURRENT_TASKS,
                "lstm_time_steps": LSTM_TIME_STEPS,
                "lstm_epochs": LSTM_EPOCHS
            },
            "resources": {
                "cached_models": len(model_cache),
                "memory_usage": "N/A"  # Would track actual memory in production
            }
        }
    
    @app.on_event("startup")
    async def startup_event():
        """Run startup tasks when the API starts"""
        log.info("Starting Crypto Oracle API...")
        
        # Load exchange markets
        if exchange_client:
            log.info("Loading exchange markets...")
            await load_exchange_markets(exchange_client, force=True)
        
        # Test OpenAI connection if available
        if openai_client:
            log.info("Testing OpenAI connection...")
            try:
                await asyncio.to_thread(
                    openai_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                log.info("OpenAI connection successful")
            except Exception as e:
                log.error(f"OpenAI connection test failed: {e}")
        
        log.info("Crypto Oracle API startup complete")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run cleanup tasks when the API shuts down"""
        log.info("Shutting down Crypto Oracle API...")
        
        # Clear TensorFlow sessions if ML was used
        if ML_AVAILABLE:
            try:
                tf.keras.backend.clear_session()
                log.info("TensorFlow sessions cleared")
            except:
                pass
        
        log.info("Crypto Oracle API shutdown complete")


#################################################
# PORTFOLIO MANAGEMENT AND POSITION TRACKING
#################################################

class Portfolio:
    """
    Manages a portfolio of trading positions
    """
    def __init__(self, account_balance=DEFAULT_ACCOUNT_BALANCE, max_positions=MAX_POSITIONS, risk_per_trade=DEFAULT_RISK_PER_TRADE):
        self.initial_balance = account_balance
        self.current_balance = account_balance
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        self.positions = {}  # symbol -> Position
        self.closed_positions = []  # List of closed position dictionaries
        self.creation_time = datetime.now()
        self.last_update_time = self.creation_time
    
    def add_position(self, symbol, entry_price, direction, stop_loss, take_profit, position_size, leverage=1):
        """
        Add a new position to the portfolio
        """
        # Check if we're already at max positions
        if len(self.positions) >= self.max_positions:
            return False, "Maximum number of positions reached"
        
        # Check if we already have a position for this symbol
        if symbol in self.positions:
            return False, f"Position already exists for {symbol}"
        
        # Validate position size against available balance
        if position_size > self.current_balance:
            return False, "Position size exceeds available balance"
        
        # Create the position
        position = Position(symbol, entry_price, direction, stop_loss, take_profit, position_size, leverage)
        
        # Add to positions dictionary
        self.positions[symbol] = position
        
        # Update balance
        self.current_balance -= position_size
        self.last_update_time = datetime.now()
        
        return True, position
    
    def update_position(self, symbol, current_price):
        """
        Update a position with the current market price
        """
        if symbol not in self.positions:
            return False, f"No position found for {symbol}"
        
        position = self.positions[symbol]
        position.update_price(current_price)
        self.last_update_time = datetime.now()
        
        # Check if the position should be closed (stop loss or take profit hit)
        close_reason = position.should_close()
        if close_reason:
            return self.close_position(symbol, current_price, close_reason)
        
        return True, position
    
    def close_position(self, symbol, exit_price=None, reason="manual"):
        """
        Close a position and update the portfolio balance
        """
        if symbol not in self.positions:
            return False, f"No position found for {symbol}"
        
        position = self.positions[symbol]
        
        # If exit price not provided, use current price
        if exit_price is None:
            exit_price = position.current_price
        
        # Update position with final price
        position.update_price(exit_price)
        
        # Calculate final P&L and add to balance
        final_pnl = position.current_pnl
        self.current_balance += position.position_size + final_pnl
        
        # Create closed position record
        closed_position = {
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_time": position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "position_size": position.position_size,
            "leverage": position.leverage,
            "pnl": final_pnl,
            "pnl_percent": position.current_pnl_pct,
            "reason": reason
        }
        
        self.closed_positions.append(closed_position)
        
        # Remove from active positions
        del self.positions[symbol]
        self.last_update_time = datetime.now()
        
        return True, closed_position
    
    def get_position(self, symbol):
        """
        Get a specific position
        """
        return self.positions.get(symbol)
    
    def get_positions(self):
        """
        Get all active positions
        """
        return {symbol: pos.to_dict() for symbol, pos in self.positions.items()}
    
    def get_portfolio_stats(self):
        """
        Get overall portfolio statistics
        """
        # Calculate total P&L
        total_pnl = sum(pos.current_pnl for pos in self.positions.values())
        
        # Calculate total position value
        total_position_value = sum(pos.position_size for pos in self.positions.values())
        
        # Calculate realized P&L from closed positions
        realized_pnl = sum(pos["pnl"] for pos in self.closed_positions)
        
        # Calculate total P&L (realized + unrealized)
        total_pnl_with_realized = total_pnl + realized_pnl
        
        # Calculate performance metrics
        total_return_pct = (total_pnl_with_realized / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Current exposure percentage
        exposure_pct = (total_position_value / (self.current_balance + total_position_value)) * 100 if (self.current_balance + total_position_value) > 0 else 0
        
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_position_value": total_position_value,
            "unrealized_pnl": total_pnl,
            "realized_pnl": realized_pnl,
            "total_pnl": total_pnl_with_realized,
            "return_percent": total_return_pct,
            "active_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "exposure_percent": exposure_pct,
            "creation_time": self.creation_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat()
        }
    
    def to_dict(self):
        """
        Convert the portfolio to a dictionary
        """
        return {
            "stats": self.get_portfolio_stats(),
            "active_positions": self.get_positions(),
            "closed_positions": self.closed_positions
        }

class Position:
    """
    Represents an open trading position
    """
    def __init__(self, symbol, entry_price, direction, stop_loss, take_profit, position_size, leverage=1):
        self.symbol = symbol
        self.entry_price = entry_price
        self.direction = direction  # 'long' or 'short'
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size  # in USD
        self.leverage = leverage
        self.entry_time = datetime.now()
        self.last_update_time = self.entry_time
        
        # Track P&L
        self.current_price = entry_price
        self.current_pnl = 0
        self.current_pnl_pct = 0
    
    def update_price(self, current_price):
        """Update the position with the current market price"""
        self.current_price = current_price
        self.last_update_time = datetime.now()
        
        # Calculate P&L
        if self.direction == 'long':
            price_change_pct = (current_price / self.entry_price - 1)
        else:  # short
            price_change_pct = (1 - current_price / self.entry_price)
        
        # Apply leverage to P&L
        self.current_pnl_pct = price_change_pct * 100 * self.leverage
        self.current_pnl = self.position_size * price_change_pct * self.leverage
    
    def should_close(self):
        """Check if the position should be closed (hit stop loss or take profit)"""
        if self.direction == 'long':
            if self.current_price <= self.stop_loss:
                return 'stop_loss'
            if self.current_price >= self.take_profit:
                return 'take_profit'
        else:  # short
            if self.current_price >= self.stop_loss:
                return 'stop_loss'
            if self.current_price <= self.take_profit:
                return 'take_profit'
        
        return None
    
    def to_dict(self):
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'leverage': self.leverage,
            'entry_time': self.entry_time.isoformat(),
            'current_pnl': self.current_pnl,
            'current_pnl_pct': self.current_pnl_pct,
            'last_update_time': self.last_update_time.isoformat()
        }


#################################################
# MAIN EXECUTION
#################################################

def run_api_server(host="0.0.0.0", port=8000):
    """
    Start the FastAPI server
    """
    if not API_AVAILABLE:
        log.error("API libraries not available. Cannot start API server.")
        return
    
    log.info(f"Starting Crypto Oracle API server on {host}:{port}")
    
    # Import uvicorn here to avoid early import errors if not available
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

async def main():
    """
    Main entry point for CLI operation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Oracle - Advanced Trading Analysis System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host address to bind")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a cryptocurrency")
    analyze_parser.add_argument("symbol", help="Symbol to analyze (e.g., BTC/USDT)")
    analyze_parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES, help="Timeframes to analyze")
    analyze_parser.add_argument("--lookback", type=int, default=1000, help="Number of candles to fetch")
    analyze_parser.add_argument("--no-sentiment", action="store_true", help="Disable sentiment analysis")
    analyze_parser.add_argument("--no-onchain", action="store_true", help="Disable on-chain metrics")
    analyze_parser.add_argument("--no-backtest", action="store_true", help="Disable backtesting")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan the market for opportunities")
    scan_parser.add_argument("--max-symbols", type=int, default=20, help="Maximum symbols to scan")
    scan_parser.add_argument("--min-confidence", type=float, default=0.6, help="Minimum confidence score")
    scan_parser.add_argument("--direction", choices=["long", "short"], help="Preferred trade direction")
    scan_parser.add_argument("--concurrent", type=int, default=DEFAULT_MAX_CONCURRENT_TASKS, help="Maximum concurrent tasks")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "api":
        run_api_server(host=args.host, port=args.port)
    
    elif args.command == "analyze":
        # Initialize exchange
        if not exchange_client:
            log.error("Exchange client not available. Cannot analyze.")
            return
        
        # Ensure markets are loaded
        if not exchange_client.markets:
            loaded = await load_exchange_markets(exchange_client)
            if not loaded:
                log.error("Failed to load markets. Cannot analyze.")
                return
        
        # Run analysis
        result = await analyze_symbol(
            args.symbol,
            timeframes=args.timeframes,
            lookback=args.lookback,
            include_sentiment=not args.no_sentiment,
            include_onchain=not args.no_onchain,
            run_backtest=not args.no_backtest
        )
        
        # Print result
        if "error" in result:
            log.error(f"Analysis error: {result['error']}")
        else:
            # Print summary
            print(f"\n=== Analysis Results for {args.symbol} ===")
            
            # Price information
            for tf, data in result["price_data"].items():
                if data["last_price"]:
                    print(f"{tf} Price: {data['last_price']}")
            
            # Prediction summary
            if "aggregated" in result.get("predictions", {}):
                agg = result["predictions"]["aggregated"]
                print(f"\nForecast: {agg.get('lstm_price')}")
                print(f"Predicted Change: {agg.get('price_change_pct')}%")
                print(f"Market Phase: {agg.get('market_phase')}")
                print(f"Overall Trend: {agg.get('overall_trend')}")
            
            # Strategy summary
            strategy = result.get("strategy", {})
            print(f"\nRecommended Action: {strategy.get('trade_direction', 'unknown').upper()}")
            
            if strategy.get("trade_direction") in ["long", "short"]:
                print(f"Entry: {strategy.get('optimal_entry')}")
                print(f"Stop Loss: {strategy.get('stop_loss')}")
                print(f"Take Profit: {strategy.get('take_profit')}")
                print(f"Confidence: {strategy.get('confidence_score', 0) * 100:.1f}%")
                
                # Analysis summary
                if "analysis" in strategy:
                    analysis = strategy["analysis"]
                    print(f"\nSummary: {analysis.get('summary')}")
            
            # Backtest summary
            if result.get("backtest") and result["backtest"].get("statistics"):
                stats = result["backtest"]["statistics"]
                print(f"\nBacktest Results:")
                print(f"Total Trades: {stats.get('total_trades')}")
                print(f"Win Rate: {stats.get('win_rate', 0) * 100:.1f}%")
                print(f"Profit Factor: {stats.get('profit_factor')}")
                print(f"Strategy Score: {stats.get('strategy_quality_score', 0) * 100:.1f}%")
            
            print("\nAnalysis completed successfully!")
    
    elif args.command == "scan":
        # Initialize exchange
        if not exchange_client:
            log.error("Exchange client not available. Cannot scan.")
            return
        
        # Ensure markets are loaded
        if not exchange_client.markets:
            loaded = await load_exchange_markets(exchange_client)
            if not loaded:
                log.error("Failed to load markets. Cannot scan.")
                return
        
        # Run scan
        filter_params = {
            "min_confidence": args.min_confidence,
            "prefer_direction": args.direction
        }
        
        result = await scan_market(
            filter_params=filter_params,
            max_symbols=args.max_symbols,
            max_concurrent=args.concurrent
        )
        
        # Print result
        if "error" in result:
            log.error(f"Scan error: {result['error']}")
        else:
            opportunities = result.get("opportunities", [])
            
            print(f"\n=== Market Scan Results ===")
            print(f"Symbols Checked: {result.get('total_symbols_checked')}")
            print(f"Opportunities Found: {len(opportunities)}")
            
            if opportunities:
                print("\nTop Opportunities:")
                for i, opp in enumerate(opportunities[:10], 1):
                    print(f"\n{i}. {opp['symbol']} - {opp['direction'].upper()}")
                    print(f"   Score: {opp['combined_score']:.2f}")
                    print(f"   Entry: {opp['entry_price']}")
                    print(f"   Risk/Reward: {opp['risk_reward_ratio']}")
                    print(f"   Summary: {opp['summary']}")
            
            print(f"\nScan completed in {result.get('duration_seconds', 0):.1f} seconds")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("crypto_oracle.log")
        ]
    )
    
    # Run main async function
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Program terminated by user")
    except Exception as e:
        log.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)