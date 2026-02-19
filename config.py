"""
Configuration file for the crypto price prediction bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration (Free Tiers)
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')  # Free: 5 calls/min, 500/day
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
FRED_API_KEY = os.getenv('FRED_API_KEY', '')  # Free: 120 calls/min
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')  # Free: 1000 calls/day

# Database Configuration
DATABASE_PATH = "crypto_predictions.db"

# Prediction Settings
CRYPTOCURRENCIES = ['bitcoin', 'ethereum']
PREDICTION_INTERVALS = ['15m', '1h', '4h']
UPDATE_FREQUENCY_MINUTES = 5

# Technical Indicators Settings
TECHNICAL_INDICATORS = {
    'sma_windows': [7, 14, 30, 50],
    'ema_windows': [7, 14, 30],
    'rsi_window': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_window': 20,
    'bollinger_std': 2
}

# Traditional Market Symbols (Yahoo Finance)
TRADITIONAL_MARKETS = {
    'stocks': ['^GSPC', '^IXIC', '^DJI'],  # S&P 500, NASDAQ, Dow Jones
    'bonds': ['^TNX'],  # 10-year Treasury
    'commodities': ['GC=F', 'SI=F'],  # Gold, Silver
    'forex': ['DX-Y.NYB'],  # Dollar Index
    'volatility': ['^VIX']  # VIX
}

# FRED Economic Indicators
FRED_SERIES = {
    'interest_rates': 'FEDFUNDS',
    'inflation': 'CPIAUCSL',
    'gdp': 'GDP',
    'unemployment': 'UNRATE'
}

# Model Configuration
MODEL_SETTINGS = {
    'train_test_split': 0.8,
    'cross_validation_folds': 5,
    'feature_selection_k': 50,
    'xgboost_params': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'retrain_frequency_hours': 24
}

# Rate Limiting (to respect free API limits)
RATE_LIMITS = {
    'coingecko': 30,  # calls per minute
    'alpha_vantage': 5,  # calls per minute
    'fred': 120,  # calls per minute
    'news_api': 1000  # calls per day
} 