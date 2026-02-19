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
# Windows are tuned for 15-minute bars (1 period = 15 minutes):
#   4 periods = 1h, 8 = 2h, 20 = 5h, 48 = 12h
TECHNICAL_INDICATORS = {
    'sma_windows': [4, 8, 20, 48],
    'ema_windows': [4, 8, 20],
    'rsi_window': 7,        # ~1.75h — fast RSI appropriate for 15m momentum
    'macd_fast': 5,
    'macd_slow': 12,
    'macd_signal': 4,
    'bollinger_window': 20, # 5h window
    'bollinger_std': 2
}

# Traditional market and macro data are NOT used for 15m/1h/4h predictions:
# - Traditional markets trade at 1h resolution — too coarse for intraday crypto
# - FRED indicators are monthly/quarterly — irrelevant for sub-4h prediction
# These configs are kept for reference but the collection methods are disabled.
TRADITIONAL_MARKETS = {}
FRED_SERIES = {}

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
} 