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
            
            # Note: Comprehensive volume features are added separately in add_volume_based_features()
            
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
    
    def add_cross_asset_correlation_features(self, primary_df: pd.DataFrame, secondary_df: pd.DataFrame, 
                                           primary_name: str, secondary_name: str) -> pd.DataFrame:
        """
        Add cross-asset correlation features between two cryptocurrencies (e.g., Bitcoin and Ethereum)
        This implements the cross-asset spillover effects and correlation analysis.
        
        Args:
            primary_df: Primary crypto dataframe (the one we're adding features to)
            secondary_df: Secondary crypto dataframe (the one we're correlating with)
            primary_name: Name of primary crypto (e.g., 'bitcoin')
            secondary_name: Name of secondary crypto (e.g., 'ethereum')
        """
        if secondary_df.empty:
            logger.warning(f"No {secondary_name} data available for cross-asset correlation")
            return primary_df
        
        df = primary_df.copy()
        
        try:
            # Prepare secondary data for merging
            secondary_clean = secondary_df.copy()
            secondary_clean['datetime'] = pd.to_datetime(secondary_clean['datetime'])
            
            # Create mapping for secondary crypto features
            secondary_features = secondary_clean[['datetime', 'price', 'volume']].copy()
            secondary_features = secondary_features.rename(columns={
                'price': f'{secondary_name}_price',
                'volume': f'{secondary_name}_volume'
            })
            
            # Add technical indicators for secondary crypto
            if 'rsi' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_rsi'] = secondary_clean['rsi']
            if 'macd' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_macd'] = secondary_clean['macd']
            if 'bb_position' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_bb_position'] = secondary_clean['bb_position']
            if 'volatility_24h' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_volatility'] = secondary_clean['volatility_24h']
            
            # Add volume-based indicators for secondary crypto
            if 'obv' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_obv'] = secondary_clean['obv']
            if 'vwap_24h' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_vwap'] = secondary_clean['vwap_24h']
            if 'volume_percentile_7d' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_volume_percentile'] = secondary_clean['volume_percentile_7d']
            if 'money_flow_24h' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_money_flow'] = secondary_clean['money_flow_24h']
            
            # Merge with primary data
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.merge(secondary_features, on='datetime', how='left')
            
            # Forward fill to handle missing data
            secondary_cols = [col for col in df.columns if col.startswith(f'{secondary_name}_')]
            df[secondary_cols] = df[secondary_cols].ffill()
            
            # 1. CORRELATION FEATURES
            correlation_windows = [6, 12, 24, 48, 168]  # 6h, 12h, 1d, 2d, 1w
            for window in correlation_windows:
                if len(df) >= window:
                    # Rolling correlation between prices
                    df[f'corr_{secondary_name}_{window}h'] = (
                        df['price'].rolling(window).corr(df[f'{secondary_name}_price'])
                    )
                    
                    # Rolling correlation between returns
                    if 'price_change_1h' in df.columns:
                        df[f'return_corr_{secondary_name}_{window}h'] = (
                            df['price_change_1h'].rolling(window).corr(
                                df[f'{secondary_name}_price'].pct_change()
                            )
                        )
            
            # Correlation strength (absolute correlation)
            df[f'corr_strength_{secondary_name}'] = df[f'corr_{secondary_name}_24h'].abs()
            
            # Correlation change (momentum in correlation)
            df[f'corr_change_{secondary_name}'] = df[f'corr_{secondary_name}_24h'].diff()
            
            # 2. LEAD-LAG RELATIONSHIPS
            lead_lag_periods = [1, 2, 4, 6, 12, 24]  # 1h to 24h
            for lag in lead_lag_periods:
                # Secondary crypto leading primary (secondary price affecting primary future)
                df[f'{secondary_name}_price_lead_{lag}h'] = df[f'{secondary_name}_price'].shift(lag)
                df[f'{secondary_name}_return_lead_{lag}h'] = df[f'{secondary_name}_price'].pct_change().shift(lag)
                
                # Primary crypto leading secondary (for context)
                df[f'{primary_name}_to_{secondary_name}_lag_{lag}h'] = df['price'].shift(-lag)
            
            # Momentum transfer indicators
            if 'rsi' in df.columns and f'{secondary_name}_rsi' in df.columns:
                df[f'rsi_divergence_{secondary_name}'] = df['rsi'] - df[f'{secondary_name}_rsi']
                # Safer division with larger epsilon and clipping
                rsi_denominator = df[f'{secondary_name}_rsi'] + 1e-3
                df[f'rsi_ratio_{secondary_name}'] = np.clip(df['rsi'] / rsi_denominator, -1000, 1000)
            
            # 3. RATIO ANALYSIS (ETH/BTC is a major trading metric)
            # Use larger epsilon and clip extreme ratios
            price_denominator = df[f'{secondary_name}_price'] + 1e-3
            df[f'{primary_name}_{secondary_name}_ratio'] = np.clip(
                df['price'] / price_denominator, 1e-6, 1e6
            )
            
            # Moving averages of the ratio
            ratio_windows = [7, 14, 30, 50]
            for window in ratio_windows:
                df[f'ratio_sma_{window}_{secondary_name}'] = (
                    df[f'{primary_name}_{secondary_name}_ratio'].rolling(window).mean()
                )
                # Ratio relative to its moving average - safer division
                sma_denominator = df[f'ratio_sma_{window}_{secondary_name}'] + 1e-6
                df[f'ratio_vs_sma_{window}_{secondary_name}'] = np.clip(
                    df[f'{primary_name}_{secondary_name}_ratio'] / sma_denominator, 0.01, 100
                )
            
            # Ratio volatility
            df[f'ratio_volatility_{secondary_name}'] = (
                df[f'{primary_name}_{secondary_name}_ratio'].rolling(24).std()
            )
            
            # Ratio momentum
            df[f'ratio_momentum_24h_{secondary_name}'] = (
                df[f'{primary_name}_{secondary_name}_ratio'].pct_change(24, fill_method=None)
            )
            
            # 4. DIVERGENCE SIGNALS
            # Price divergence (when they move in opposite directions)
            primary_returns = df['price'].pct_change(fill_method=None)
            secondary_returns = df[f'{secondary_name}_price'].pct_change(fill_method=None)
            
            df[f'price_divergence_{secondary_name}'] = (
                (primary_returns > 0) & (secondary_returns < 0) |
                (primary_returns < 0) & (secondary_returns > 0)
            ).astype(int)
            
            # Return spread
            df[f'return_spread_{secondary_name}'] = primary_returns - secondary_returns
            
            # 5. CROSS-MOMENTUM INDICATORS
            # Volume relationship - safer division with clipping
            if 'volume' in df.columns and f'{secondary_name}_volume' in df.columns:
                volume_denominator = df[f'{secondary_name}_volume'] + 1e-3
                df[f'volume_ratio_{secondary_name}'] = np.clip(
                    df['volume'] / volume_denominator, 1e-6, 1e6
                )
                
                # Combined volume momentum
                df[f'combined_volume_momentum_{secondary_name}'] = (
                    (df['volume'].rolling(4).mean() / df['volume'].rolling(24).mean()) *
                    (df[f'{secondary_name}_volume'].rolling(4).mean() / df[f'{secondary_name}_volume'].rolling(24).mean())
                )
            
            # Cross-volatility features
            if f'{secondary_name}_volatility' in df.columns and 'volatility_24h' in df.columns:
                df[f'volatility_spread_{secondary_name}'] = (
                    df['volatility_24h'] - df[f'{secondary_name}_volatility']
                )
                volatility_denominator = df[f'{secondary_name}_volatility'] + 1e-6
                df[f'volatility_ratio_{secondary_name}'] = np.clip(
                    df['volatility_24h'] / volatility_denominator, 1e-3, 1e3
                )
            
            # 6. MARKET REGIME INDICATORS
            # High correlation periods vs divergence periods
            df[f'high_correlation_regime_{secondary_name}'] = (
                df[f'corr_strength_{secondary_name}'] > 0.7
            ).astype(int)
            
            df[f'low_correlation_regime_{secondary_name}'] = (
                df[f'corr_strength_{secondary_name}'] < 0.3
            ).astype(int)
            
            # Correlation breakdown events (when correlation drops significantly)
            df[f'correlation_breakdown_{secondary_name}'] = (
                (df[f'corr_{secondary_name}_24h'].shift(24) > 0.5) & 
                (df[f'corr_{secondary_name}_24h'] < 0.2)
            ).astype(int)
            
            # 7. RELATIVE STRENGTH
            # Which crypto is performing better
            primary_returns_24h = df['price'].pct_change(24, fill_method=None)
            secondary_returns_24h = df[f'{secondary_name}_price'].pct_change(24, fill_method=None)
            
            df[f'relative_strength_{secondary_name}'] = (
                primary_returns_24h - secondary_returns_24h
            )
            
            # Rolling relative strength
            df[f'relative_strength_7d_{secondary_name}'] = (
                df[f'relative_strength_{secondary_name}'].rolling(168).mean()  # 7 days
            )
            
            # 8. CROSS-ASSET VOLUME FEATURES
            if 'volume' in df.columns and f'{secondary_name}_volume' in df.columns:
                # Volume correlation
                df[f'volume_correlation_{secondary_name}_24h'] = (
                    df['volume'].rolling(24).corr(df[f'{secondary_name}_volume'])
                )
                
                # Volume ratio analysis - safer division
                volume_denom = df[f'{secondary_name}_volume'] + 1e-3
                df[f'volume_ratio_{secondary_name}'] = np.clip(
                    df['volume'] / volume_denom, 1e-6, 1e6
                )
                df[f'volume_ratio_sma_14_{secondary_name}'] = (
                    df[f'volume_ratio_{secondary_name}'].rolling(14).mean()
                )
                
                # Cross-volume momentum
                volume_mom_primary = df['volume'].rolling(24).mean() / df['volume'].rolling(168).mean()
                volume_mom_secondary = (
                    df[f'{secondary_name}_volume'].rolling(24).mean() / 
                    df[f'{secondary_name}_volume'].rolling(168).mean()
                )
                df[f'volume_momentum_divergence_{secondary_name}'] = volume_mom_primary - volume_mom_secondary
                
                # Combined volume strength (when both have high volume)
                primary_vol_percentile = df['volume'].rolling(168).rank(pct=True)
                secondary_vol_percentile = df[f'{secondary_name}_volume'].rolling(168).rank(pct=True)
                df[f'combined_high_volume_{secondary_name}'] = (
                    (primary_vol_percentile > 0.8) & (secondary_vol_percentile > 0.8)
                ).astype(int)
                
                # Volume divergence signals (when volumes move opposite directions)
                primary_vol_change = df['volume'].pct_change(fill_method=None)
                secondary_vol_change = df[f'{secondary_name}_volume'].pct_change(fill_method=None)
                df[f'volume_divergence_{secondary_name}'] = (
                    (primary_vol_change > 0) & (secondary_vol_change < 0) |
                    (primary_vol_change < 0) & (secondary_vol_change > 0)
                ).astype(int)
            
            # Cross-OBV features
            if 'obv' in df.columns and f'{secondary_name}_obv' in df.columns:
                df[f'obv_correlation_{secondary_name}_24h'] = (
                    df['obv'].rolling(24).corr(df[f'{secondary_name}_obv'])
                )
                df[f'obv_divergence_{secondary_name}'] = (
                    (df['obv'].diff() > 0) & (df[f'{secondary_name}_obv'].diff() < 0) |
                    (df['obv'].diff() < 0) & (df[f'{secondary_name}_obv'].diff() > 0)
                ).astype(int)
            
            # Cross-VWAP features
            if 'vwap_24h' in df.columns and f'{secondary_name}_vwap' in df.columns:
                df[f'vwap_ratio_{secondary_name}'] = (
                    df['vwap_24h'] / df[f'{secondary_name}_vwap']
                )
                df[f'both_above_vwap_{secondary_name}'] = (
                    (df['price'] > df['vwap_24h']) & 
                    (df[f'{secondary_name}_price'] > df[f'{secondary_name}_vwap'])
                ).astype(int)
            
            # Cross-money flow features
            if 'money_flow_24h' in df.columns and f'{secondary_name}_money_flow' in df.columns:
                df[f'money_flow_correlation_{secondary_name}'] = (
                    df['money_flow_24h'].rolling(24).corr(df[f'{secondary_name}_money_flow'])
                )
                df[f'money_flow_divergence_{secondary_name}'] = (
                    (df['money_flow_24h'] > 0) & (df[f'{secondary_name}_money_flow'] < 0) |
                    (df['money_flow_24h'] < 0) & (df[f'{secondary_name}_money_flow'] > 0)
                ).astype(int)
            
            logger.info(f"Added cross-asset correlation features with {secondary_name}")
            logger.info(f"Added {len([col for col in df.columns if secondary_name in col])} cross-asset features")
            
        except Exception as e:
            logger.error(f"Error adding cross-asset correlation features: {e}")
        
        return df
    
    def add_volume_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive volume-based and market cap features for crypto analysis
        
        Features include:
        - Volume moving averages and ratios
        - On-Balance Volume (OBV) 
        - Volume Weighted Average Price (VWAP)
        - Volume momentum and acceleration
        - Market cap analysis
        - Volume spikes and anomaly detection
        - Price-volume relationships
        """
        if 'volume' not in df.columns and 'market_cap' not in df.columns:
            logger.warning("No volume or market cap data available for volume-based features")
            return df
        
        df = df.copy()
        
        try:
            # ==== VOLUME MOVING AVERAGES AND RATIOS ====
            if 'volume' in df.columns:
                volume_windows = [7, 14, 30, 50]
                for window in volume_windows:
                    # Volume moving averages
                    df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                    df[f'volume_ema_{window}'] = df['volume'].ewm(span=window).mean()
                    
                    # Volume ratio to moving average
                    df[f'volume_ratio_sma_{window}'] = df['volume'] / (df[f'volume_sma_{window}'] + 1e-6)
                    df[f'volume_ratio_ema_{window}'] = df['volume'] / (df[f'volume_ema_{window}'] + 1e-6)
                
                # Volume percentiles (relative volume strength)
                df['volume_percentile_30d'] = df['volume'].rolling(720).rank(pct=True)  # 30 days in hours
                df['volume_percentile_7d'] = df['volume'].rolling(168).rank(pct=True)   # 7 days in hours
                
                # ==== ON-BALANCE VOLUME (OBV) ====
                # OBV accumulates volume based on price direction
                price_change = df['price'].diff()
                volume_direction = np.where(price_change > 0, df['volume'], 
                                          np.where(price_change < 0, -df['volume'], 0))
                df['obv'] = volume_direction.cumsum()
                
                # OBV moving averages and momentum
                df['obv_sma_14'] = df['obv'].rolling(14).mean()
                df['obv_ema_14'] = df['obv'].ewm(span=14).mean()
                df['obv_momentum'] = df['obv'].diff(24)  # 24h OBV change
                df['obv_acceleration'] = df['obv_momentum'].diff()
                
                # OBV relative to price (divergence detection)
                df['obv_price_correlation_24h'] = df['obv'].rolling(24).corr(df['price'])
                df['obv_price_divergence'] = (
                    (df['price'].pct_change(24) > 0) & (df['obv_momentum'] < 0) |
                    (df['price'].pct_change(24) < 0) & (df['obv_momentum'] > 0)
                ).astype(int)
                
                # ==== VOLUME WEIGHTED AVERAGE PRICE (VWAP) ====
                # Calculate VWAP for different periods
                vwap_periods = [24, 168, 720]  # 1 day, 1 week, 1 month
                for period in vwap_periods:
                    # VWAP = Sum(Price * Volume) / Sum(Volume)
                    pv = df['price'] * df['volume']
                    df[f'vwap_{period}h'] = (
                        pv.rolling(period).sum() / df['volume'].rolling(period).sum()
                    )
                    
                    # Price relative to VWAP (above/below VWAP)
                    df[f'price_vs_vwap_{period}h'] = df['price'] / df[f'vwap_{period}h']
                    df[f'above_vwap_{period}h'] = (df['price'] > df[f'vwap_{period}h']).astype(int)
                
                # ==== VOLUME MOMENTUM AND ACCELERATION ====
                # Volume rate of change
                volume_roc_periods = [1, 4, 12, 24, 168]
                for period in volume_roc_periods:
                    df[f'volume_roc_{period}h'] = df['volume'].pct_change(period)
                
                # Volume acceleration (rate of change of rate of change)
                df['volume_acceleration_24h'] = df['volume_roc_24h'].diff()
                
                # Volume momentum indicators
                df['volume_momentum_7d'] = (
                    df['volume'].rolling(24).mean() / df['volume'].rolling(168).mean()
                )
                df['volume_trend_strength'] = df['volume'].rolling(24).apply(
                    lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
                )
                
                # ==== VOLUME SPIKES AND ANOMALIES ====
                # Volume spike detection (volume significantly above average)
                df['volume_spike_2x'] = (df['volume'] > 2 * df['volume_sma_30']).astype(int)
                df['volume_spike_3x'] = (df['volume'] > 3 * df['volume_sma_30']).astype(int)
                
                # Volume drought detection (volume significantly below average)
                df['volume_drought'] = (df['volume'] < 0.5 * df['volume_sma_30']).astype(int)
                
                # Volume volatility
                df['volume_volatility_7d'] = df['volume_roc_1h'].rolling(168).std()
                df['volume_volatility_30d'] = df['volume_roc_1h'].rolling(720).std()
                
                # Z-score of volume (how many standard deviations from mean)
                volume_mean_30d = df['volume'].rolling(720).mean()
                volume_std_30d = df['volume'].rolling(720).std()
                df['volume_zscore_30d'] = (df['volume'] - volume_mean_30d) / (volume_std_30d + 1e-6)
                
                # ==== PRICE-VOLUME RELATIONSHIPS ====
                # Price-volume correlation
                df['price_volume_corr_24h'] = df['price'].rolling(24).corr(df['volume'])
                df['price_volume_corr_7d'] = df['price'].rolling(168).corr(df['volume'])
                
                # Volume-weighted price changes
                df['volume_weighted_return_1h'] = df['price_change_1h'] * df['volume_ratio_sma_14']
                df['volume_weighted_return_24h'] = df['price'].pct_change(24) * df['volume_ratio_sma_14']
                
                # Price impact per unit volume (efficiency measure)
                df['price_impact_per_volume'] = (
                    df['price_change_1h'].abs() / (df['volume_ratio_sma_14'] + 1e-6)
                )
                
                # Volume distribution analysis
                df['high_volume_regime'] = (df['volume_percentile_30d'] > 0.8).astype(int)
                df['low_volume_regime'] = (df['volume_percentile_30d'] < 0.2).astype(int)
            
            # ==== MARKET CAP ANALYSIS ====
            if 'market_cap' in df.columns:
                # Market cap moving averages
                mc_windows = [7, 14, 30]
                for window in mc_windows:
                    df[f'market_cap_sma_{window}'] = df['market_cap'].rolling(window).mean()
                    df[f'market_cap_ratio_sma_{window}'] = df['market_cap'] / df[f'market_cap_sma_{window}']
                
                # Market cap rate of change
                mc_roc_periods = [1, 24, 168]
                for period in mc_roc_periods:
                    df[f'market_cap_roc_{period}h'] = df['market_cap'].pct_change(period)
                
                # Market cap momentum
                df['market_cap_momentum_7d'] = (
                    df['market_cap'].rolling(24).mean() / df['market_cap'].rolling(168).mean()
                )
                
                # Market cap volatility
                df['market_cap_volatility_7d'] = df['market_cap_roc_1h'].rolling(168).std()
                
                # Market cap vs volume relationship
                if 'volume' in df.columns:
                    df['market_cap_volume_ratio'] = df['market_cap'] / (df['volume'] + 1e-6)
                    df['mc_volume_correlation_24h'] = df['market_cap'].rolling(24).corr(df['volume'])
                
                # Market cap percentile
                df['market_cap_percentile_30d'] = df['market_cap'].rolling(720).rank(pct=True)
            
            # ==== ADVANCED VOLUME FEATURES ====
            if 'volume' in df.columns:
                # Volume rate of change acceleration
                df['volume_roc_acceleration'] = df['volume_roc_24h'].diff()
                
                # Volume efficiency (price change per unit of volume)
                df['volume_efficiency'] = (
                    df['price'].pct_change(24).abs() / (df['volume_ratio_sma_30'] + 1e-6)
                )
                
                # Cumulative volume over different periods
                df['cumulative_volume_7d'] = df['volume'].rolling(168).sum()
                df['cumulative_volume_30d'] = df['volume'].rolling(720).sum()
                
                # Volume concentration (what % of recent volume happened in last period)
                df['volume_concentration_24h'] = (
                    df['volume'].rolling(24).sum() / df['cumulative_volume_7d']
                )
                
                # Money flow approximation (price * volume direction)
                price_direction = np.where(df['price'].diff() > 0, 1, 
                                         np.where(df['price'].diff() < 0, -1, 0))
                df['money_flow_raw'] = df['volume'] * price_direction
                df['money_flow_24h'] = df['money_flow_raw'].rolling(24).sum()
                df['money_flow_7d'] = df['money_flow_raw'].rolling(168).sum()
                
                # Volume-based support/resistance
                # High volume periods often indicate support/resistance levels
                df['high_volume_price_level'] = np.where(
                    df['volume_percentile_7d'] > 0.9, df['price'], np.nan
                )
                df['high_volume_price_level'] = df['high_volume_price_level'].ffill()
                df['distance_from_hv_level'] = (
                    (df['price'] - df['high_volume_price_level']) / df['high_volume_price_level']
                )
            
            logger.info("Added comprehensive volume-based features")
            logger.info(f"Volume features added, new shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error adding volume-based features: {e}")
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Create target variables for different prediction horizons
        CRITICAL: This creates targets that look FORWARD in time to avoid data leakage
        """
        df = df.copy()
        
        try:
            # IMPORTANT: For time series prediction, we want to predict FUTURE prices
            # Using shift(-n) means we're looking n periods into the future
            # This is correct as long as during training we drop rows where targets are NaN
            
            # Target variables for different horizons (future prices)
            df['target_1h'] = df[price_col].shift(-1)   # Price 1 hour in the future
            df['target_1d'] = df[price_col].shift(-24)  # Price 24 hours in the future  
            df['target_1w'] = df[price_col].shift(-24*7)  # Price 7 days in the future
            
            # Target returns (percentage change)
            df['target_return_1h'] = (df['target_1h'] / df[price_col] - 1) * 100
            df['target_return_1d'] = (df['target_1d'] / df[price_col] - 1) * 100
            df['target_return_1w'] = (df['target_1w'] / df[price_col] - 1) * 100
            
            # Binary classification targets (up/down)
            df['target_direction_1h'] = (df['target_return_1h'] > 0).astype(int)
            df['target_direction_1d'] = (df['target_return_1d'] > 0).astype(int)
            df['target_direction_1w'] = (df['target_return_1w'] > 0).astype(int)
            
            # CRITICAL: Create future datetime stamps for proper evaluation
            # These represent the actual TIME that our predictions are targeting
            df['target_datetime_1h'] = df['datetime'] + pd.Timedelta(hours=1)
            df['target_datetime_1d'] = df['datetime'] + pd.Timedelta(days=1)
            df['target_datetime_1w'] = df['datetime'] + pd.Timedelta(weeks=1)
            
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
        df = self.add_volume_based_features(df)
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
                       'target_direction_1h', 'target_direction_1d', 'target_direction_1w',
                       'target_datetime_1h', 'target_datetime_1d', 'target_datetime_1w']
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_columns)}")
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def prepare_features_with_cross_asset_correlation(self, crypto_data_dict: Dict[str, pd.DataFrame], 
                                                    market_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Prepare features for multiple cryptocurrencies with cross-asset correlation features
        
        Args:
            crypto_data_dict: Dictionary with crypto names as keys and DataFrames as values
            market_df: Traditional market data for correlation features
            
        Returns:
            Dictionary with enhanced feature DataFrames including cross-asset correlations
        """
        logger.info("Starting cross-asset feature engineering pipeline...")
        
        # First, prepare individual features for each crypto
        prepared_data = {}
        for crypto_name, crypto_df in crypto_data_dict.items():
            if not crypto_df.empty:
                logger.info(f"Preparing individual features for {crypto_name}...")
                prepared_data[crypto_name] = self.prepare_features(crypto_df, market_df)
        
        # Now add cross-asset correlation features for Bitcoin and Ethereum
        if 'bitcoin' in prepared_data and 'ethereum' in prepared_data:
            logger.info("Adding cross-asset correlation features between Bitcoin and Ethereum...")
            
            # Add Ethereum features to Bitcoin data
            prepared_data['bitcoin'] = self.add_cross_asset_correlation_features(
                prepared_data['bitcoin'], 
                prepared_data['ethereum'],
                'bitcoin', 
                'ethereum'
            )
            
            # Add Bitcoin features to Ethereum data  
            prepared_data['ethereum'] = self.add_cross_asset_correlation_features(
                prepared_data['ethereum'],
                prepared_data['bitcoin'], 
                'ethereum',
                'bitcoin'
            )
            
            logger.info("Cross-asset correlation features added successfully")
        else:
            logger.warning("Bitcoin or Ethereum data missing - skipping cross-asset features")
        
        # CRITICAL: Clean data after cross-asset features to prevent training failures
        logger.info("Cleaning cross-asset feature data to prevent infinity values...")
        for crypto_name, crypto_df in prepared_data.items():
            if not crypto_df.empty:
                # Clean infinity and extremely large values
                numeric_cols = crypto_df.select_dtypes(include=[np.number]).columns
                
                # Replace infinity values with NaN
                crypto_df[numeric_cols] = crypto_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
                
                # Replace extremely large values that could cause overflow
                for col in numeric_cols:
                    if col not in ['datetime', 'target_1h', 'target_1d', 'target_1w', 
                                  'target_return_1h', 'target_return_1d', 'target_return_1w',
                                  'target_direction_1h', 'target_direction_1d', 'target_direction_1w']:
                        
                        # Check for extremely large values (beyond reasonable float64 range)
                        max_safe_value = 1e10  # 10 billion - reasonable upper bound
                        crypto_df.loc[crypto_df[col].abs() > max_safe_value, col] = np.nan
                        
                        # Also clean values that are too close to zero in division results
                        # This prevents near-zero denominators from creating unstable features
                        if any(substring in col for substring in ['_ratio', '_vs_', 'divergence', 'correlation']):
                            # For ratio and correlation features, cap extreme values
                            crypto_df.loc[crypto_df[col].abs() > 1000, col] = np.nan
                
                # Fill remaining NaN values with forward fill, then backward fill, then zero
                for col in numeric_cols:
                    if col not in ['datetime', 'target_1h', 'target_1d', 'target_1w', 
                                  'target_return_1h', 'target_return_1d', 'target_return_1w',
                                  'target_direction_1h', 'target_direction_1d', 'target_direction_1w']:
                        # Forward fill
                        crypto_df[col] = crypto_df[col].ffill()
                        # Backward fill
                        crypto_df[col] = crypto_df[col].bfill()
                        # Finally fill any remaining NaN with 0
                        crypto_df[col] = crypto_df[col].fillna(0)
                
                # Final check for any remaining problematic values
                inf_check = np.isinf(crypto_df[numeric_cols]).sum().sum()
                nan_check = np.isnan(crypto_df[numeric_cols]).sum().sum()
                
                logger.info(f"{crypto_name} post-cleaning: {inf_check} inf values, {nan_check} NaN values")
                
                prepared_data[crypto_name] = crypto_df
        
        # Update feature columns for the enhanced datasets
        for crypto_name, crypto_df in prepared_data.items():
            if not crypto_df.empty:
                exclude_cols = ['datetime', 'crypto', 'target_1h', 'target_1d', 'target_1w', 
                               'target_return_1h', 'target_return_1d', 'target_return_1w',
                               'target_direction_1h', 'target_direction_1d', 'target_direction_1w',
                               'target_datetime_1h', 'target_datetime_1d', 'target_datetime_1w']
                
                feature_cols = [col for col in crypto_df.columns if col not in exclude_cols]
                
                logger.info(f"Final {crypto_name} features: {len(feature_cols)} total features")
                logger.info(f"Final {crypto_name} shape: {crypto_df.shape}")
        
        return prepared_data
    
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