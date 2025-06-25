"""
Feature engineering module for crypto price prediction
Computes technical indicators, market correlations, and other ML features
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import logging
import config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        
    def add_technical_indicators(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Add technical analysis indicators to cryptocurrency data
        """
        df = df.copy()
        
        try:
            # Sort by datetime to ensure proper calculation
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Simple Moving Averages
            for window in config.TECHNICAL_INDICATORS['sma_windows']:
                df[f'sma_{window}'] = df[price_col].rolling(window=window).mean()
                df[f'price_sma_{window}_ratio'] = df[price_col] / df[f'sma_{window}']
            
            # Exponential Moving Averages
            for window in config.TECHNICAL_INDICATORS['ema_windows']:
                df[f'ema_{window}'] = df[price_col].ewm(span=window).mean()
                df[f'price_ema_{window}_ratio'] = df[price_col] / df[f'ema_{window}']
            
            # RSI (Relative Strength Index)
            df['rsi'] = ta.momentum.RSIIndicator(
                df[price_col], 
                window=config.TECHNICAL_INDICATORS['rsi_window']
            ).rsi()
            
            # MACD
            macd_indicator = ta.trend.MACD(
                df[price_col],
                window_fast=config.TECHNICAL_INDICATORS['macd_fast'],
                window_slow=config.TECHNICAL_INDICATORS['macd_slow'],
                window_sign=config.TECHNICAL_INDICATORS['macd_signal']
            )
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_histogram'] = macd_indicator.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                df[price_col],
                window=config.TECHNICAL_INDICATORS['bollinger_window'],
                window_dev=config.TECHNICAL_INDICATORS['bollinger_std']
            )
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price momentum and volatility
            df['price_change_1h'] = df[price_col].pct_change(1)
            df['price_change_4h'] = df[price_col].pct_change(4)
            df['price_change_24h'] = df[price_col].pct_change(24)
            
            # Rolling volatility (standard deviation of returns)
            df['volatility_4h'] = df['price_change_1h'].rolling(4).std()
            df['volatility_24h'] = df['price_change_1h'].rolling(24).std()
            
            # Volume indicators (if volume exists)
            if 'volume' in df.columns:
                df['volume_sma_7'] = df['volume'].rolling(7).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma_7']
                df['price_volume_trend'] = df['price_change_1h'] * df['volume']
            
            logger.info(f"Added technical indicators, shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features
        """
        df = df.copy()
        
        try:
            # Ensure datetime is datetime type
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Time features
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['quarter'] = df['datetime'].dt.quarter
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Weekend indicator
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Market session indicators (approximate)
            df['us_market_hours'] = ((df['hour'] >= 14) & (df['hour'] <= 21)).astype(int)  # 14:00-21:00 UTC
            df['asian_market_hours'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)  # 00:00-08:00 UTC
            df['european_market_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)  # 08:00-16:00 UTC
            
            logger.info("Added time-based features")
            
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
        
        return df
    
    def add_market_correlation_features(self, crypto_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on traditional market correlations
        """
        if market_df.empty:
            logger.warning("No market data available for correlation features")
            return crypto_df
        
        df = crypto_df.copy()
        
        try:
            # Prepare market data
            market_df_clean = market_df.copy()
            
            # Ensure datetime column is properly formatted
            market_df_clean['datetime'] = pd.to_datetime(market_df_clean['datetime'], utc=True).dt.tz_localize(None)
            
            market_pivot = market_df_clean.pivot_table(
                index='datetime', 
                columns='symbol', 
                values='Close', 
                aggfunc='last'
            ).ffill()
            
            # Ensure datetime columns are timezone-naive for merging
            crypto_df_clean = crypto_df.copy()
            crypto_df_clean['datetime'] = pd.to_datetime(crypto_df_clean['datetime'], utc=True).dt.tz_localize(None)
            
            market_pivot_clean = market_pivot.copy()
            if hasattr(market_pivot_clean.index.dtype, 'tz') and market_pivot_clean.index.dtype.tz is not None:
                market_pivot_clean.index = market_pivot_clean.index.tz_localize(None)
            
            # Merge with crypto data
            df = crypto_df_clean.merge(market_pivot_clean, left_on='datetime', right_index=True, how='left')
            
            # Forward fill market data to handle timing differences
            market_columns = market_pivot_clean.columns
            df[market_columns] = df[market_columns].ffill()
            
            # Calculate correlations (rolling window)
            window = 24  # 24 hours
            for col in market_columns:
                if col in df.columns:
                    # Rolling correlation
                    df[f'corr_{col.replace("^", "").replace("=F", "").replace("-Y.NYB", "").replace(".", "_")}'] = (
                        df['price'].rolling(window).corr(df[col])
                    )
                    
                    # Price ratio
                    df[f'ratio_{col.replace("^", "").replace("=F", "").replace("-Y.NYB", "").replace(".", "_")}'] = (
                        df['price'] / df[col]
                    )
            
            logger.info("Added market correlation features")
            
        except Exception as e:
            logger.error(f"Error adding market correlation features: {e}")
        
        return df
    
    def add_lagged_features(self, df: pd.DataFrame, target_col: str = 'price', lags: List[int] = [1, 2, 4, 6, 12, 24]) -> pd.DataFrame:
        """
        Add lagged features for time series prediction
        """
        df = df.copy()
        
        try:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
                
                # Lagged price changes
                df[f'price_change_lag_{lag}'] = df['price_change_1h'].shift(lag)
                
                # Lagged RSI
                if 'rsi' in df.columns:
                    df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            
            logger.info(f"Added lagged features for lags: {lags}")
            
        except Exception as e:
            logger.error(f"Error adding lagged features: {e}")
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Create target variables for different prediction horizons
        """
        df = df.copy()
        
        try:
            # Target variables for different horizons
            df['target_1h'] = df[price_col].shift(-1)  # Next hour price
            df['target_1d'] = df[price_col].shift(-24)  # Next day price (24h later)
            df['target_1w'] = df[price_col].shift(-24*7)  # Next week price (7 days later)
            
            # Target returns (percentage change)
            df['target_return_1h'] = (df['target_1h'] / df[price_col] - 1) * 100
            df['target_return_1d'] = (df['target_1d'] / df[price_col] - 1) * 100
            df['target_return_1w'] = (df['target_1w'] / df[price_col] - 1) * 100
            
            # Binary classification targets (up/down)
            df['target_direction_1h'] = (df['target_return_1h'] > 0).astype(int)
            df['target_direction_1d'] = (df['target_return_1d'] > 0).astype(int)
            df['target_direction_1w'] = (df['target_return_1w'] > 0).astype(int)
            
            logger.info("Created target variables for 1h, 1d, 1w horizons")
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
        
        return df
    
    def prepare_features(self, crypto_df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Main feature preparation pipeline
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Add all features
        df = self.add_technical_indicators(crypto_df)
        df = self.add_time_features(df)
        
        if market_df is not None and not market_df.empty:
            df = self.add_market_correlation_features(df, market_df)
        
        df = self.add_lagged_features(df)
        df = self.create_target_variables(df)
        
        # Clean data: replace infinity and extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Replace extreme values (beyond 3 standard deviations) with NaN
        for col in numeric_cols:
            if col not in ['datetime', 'target_1h', 'target_1d', 'target_1w', 
                          'target_return_1h', 'target_return_1d', 'target_return_1w',
                          'target_direction_1h', 'target_direction_1d', 'target_direction_1w']:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if not pd.isna(std_val) and std_val > 0:
                    df.loc[np.abs(df[col] - mean_val) > 3 * std_val, col] = np.nan
        
        # Store feature column names (excluding targets and metadata)
        exclude_cols = ['datetime', 'crypto', 'target_1h', 'target_1d', 'target_1w', 
                       'target_return_1h', 'target_return_1d', 'target_return_1w',
                       'target_direction_1h', 'target_direction_1d', 'target_direction_1w']
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_columns)}")
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def get_feature_importance_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Prepare data for feature importance analysis
        """
        # Remove rows with NaN targets
        clean_df = df.dropna(subset=[target_col])
        
        # Get feature matrix and target
        X = clean_df[self.feature_columns].fillna(0)  # Fill NaN features with 0
        y = clean_df[target_col]
        
        return X, y

if __name__ == "__main__":
    # Test feature engineering
    from data_collector import DataCollector
    
    collector = DataCollector()
    
    # Get sample data
    btc_data = collector.get_crypto_data('bitcoin', days=30)
    market_data = collector.get_traditional_markets_data(days=30)
    
    # Test feature engineering
    fe = FeatureEngineer()
    features_df = fe.prepare_features(btc_data, market_data)
    
    print(f"Original shape: {btc_data.shape}")
    print(f"Features shape: {features_df.shape}")
    print(f"Number of features: {len(fe.feature_columns)}")
    print(f"Features: {fe.feature_columns[:10]}...")  # Show first 10 features 