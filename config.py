"""
Configuration file for the crypto price prediction bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Binance public REST API — no API key required for market data
# Private endpoints (account, trading) would need BINANCE_API_KEY / BINANCE_API_SECRET

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
        # These are the fallback values used if early stopping fails.
        # Early stopping will search up to 1000 trees and prune back;
        # the value here only matters in edge cases (tiny datasets, API errors).
        'n_estimators': 100,
        'max_depth': 4,          # shallower than before — reduces overfitting on financial noise
        'learning_rate': 0.05,   # slower learning compensated by more trees via early stopping
        'subsample': 0.8,        # row subsampling per tree
        'colsample_bytree': 0.7, # feature subsampling per tree
        'reg_alpha': 0.1,        # L1 regularisation
        'reg_lambda': 1.5,       # L2 regularisation (XGBoost default is 1)
        'random_state': 42
    },
    # Early stopping: search up to 1000 estimators, stop when val loss doesn't
    # improve for this many consecutive rounds.  Last 15% of training data is
    # used as the time-ordered validation split (no shuffling).
    'early_stopping_rounds': 30,
    # Isotonic calibration folds (TimeSeriesSplit).  Must be ≥ 2.
    # Higher = better calibration but slower training.
    'calibration_folds': 3,
    # Fixed production training window (days).  We deliberately do NOT select
    # this from backtest accuracy to avoid test-set selection bias.
    'production_training_days': 365,
    'retrain_frequency_hours': 24
}

# Rate Limiting (to respect free API limits)
RATE_LIMITS = {
    'coingecko': 30,  # calls per minute
    'alpha_vantage': 5,  # calls per minute
} 