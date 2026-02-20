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
            
            # Price momentum and volatility (15-minute data: 1 period = 15 minutes)
            df['price_change_15m'] = df[price_col].pct_change(1)   # 1 period  = 15 min
            df['price_change_1h'] = df[price_col].pct_change(4)    # 4 periods = 1 hour
            df['price_change_4h'] = df[price_col].pct_change(16)   # 16 periods = 4 hours
            df['price_change_1d'] = df[price_col].pct_change(96)   # 96 periods = 1 day

            # Rolling volatility (standard deviation of returns over 15m data)
            df['volatility_1h'] = df['price_change_15m'].rolling(4).std()    # 4 periods  = 1 hour
            df['volatility_4h'] = df['price_change_15m'].rolling(16).std()   # 16 periods = 4 hours
            df['volatility_1d'] = df['price_change_15m'].rolling(96).std()   # 96 periods = 1 day
            
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
            
            # Calculate correlations (rolling window: 96 periods = 1 day in 15m data)
            window = 96
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
    
    def add_lagged_features(self, df: pd.DataFrame, target_col: str = 'price',
                           lags: List[int] = [1, 2, 4, 8, 16]) -> pd.DataFrame:
        """
        Add lagged features for time series prediction.
        With 15-minute data: lags represent 15m, 30m, 1h, 2h, 4h.
        Lags beyond 4h (24, 48, 96 periods) removed — too distant to carry intraday signal.
        """
        df = df.copy()

        try:
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

                # Lagged price changes (using 15m base change)
                if 'price_change_15m' in df.columns:
                    df[f'price_change_lag_{lag}'] = df['price_change_15m'].shift(lag)

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
            if 'volatility_4h' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_volatility'] = secondary_clean['volatility_4h']
            
            # Add volume-based indicators for secondary crypto
            if 'obv' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_obv'] = secondary_clean['obv']
            if 'vwap_1d' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_vwap'] = secondary_clean['vwap_1d']
            if 'volume_percentile_7d' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_volume_percentile'] = secondary_clean['volume_percentile_7d']
            if 'money_flow_1d' in secondary_clean.columns:
                secondary_features[f'{secondary_name}_money_flow'] = secondary_clean['money_flow_1d']
            
            # Merge with primary data
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.merge(secondary_features, on='datetime', how='left')
            
            # Forward fill to handle missing data
            secondary_cols = [col for col in df.columns if col.startswith(f'{secondary_name}_')]
            df[secondary_cols] = df[secondary_cols].ffill()
            
            # 1. CORRELATION FEATURES
            # Windows in 15m periods: 24=6h, 48=12h, 96=1d
            # 2d (192) and 1w (672) removed — too long for 15m/1h/4h targets
            correlation_windows = [24, 48, 96]
            corr_window_labels = ['6h', '12h', '1d']
            for window, label in zip(correlation_windows, corr_window_labels):
                if len(df) >= window:
                    # Rolling correlation between prices
                    df[f'corr_{secondary_name}_{label}'] = (
                        df['price'].rolling(window).corr(df[f'{secondary_name}_price'])
                    )

                    # Rolling correlation between returns
                    if 'price_change_15m' in df.columns:
                        df[f'return_corr_{secondary_name}_{label}'] = (
                            df['price_change_15m'].rolling(window).corr(
                                df[f'{secondary_name}_price'].pct_change()
                            )
                        )
            
            # Correlation strength (absolute correlation) – use 1d window
            df[f'corr_strength_{secondary_name}'] = df[f'corr_{secondary_name}_1d'].abs()

            # Correlation change (momentum in correlation)
            df[f'corr_change_{secondary_name}'] = df[f'corr_{secondary_name}_1d'].diff()

            # 2. LEAD-LAG RELATIONSHIPS
            # Lags in 15m periods: 4=1h, 8=2h, 16=4h
            # 6h/12h/24h lags removed — too long for short-horizon targets
            # NOTE: only LAG secondary into past (positive shift). The previous
            # "primary leading secondary" lines used shift(-lag) = future data leakage.
            lead_lag_periods = [4, 8, 16]
            lead_lag_labels = ['1h', '2h', '4h']
            for lag, label in zip(lead_lag_periods, lead_lag_labels):
                # Secondary crypto lagged relative to primary (past secondary → primary future)
                df[f'{secondary_name}_price_lead_{label}'] = df[f'{secondary_name}_price'].shift(lag)
                df[f'{secondary_name}_return_lead_{label}'] = df[f'{secondary_name}_price'].pct_change().shift(lag)
            
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
            
            # Moving averages of the ratio (in 15m periods: 4=1h, 8=2h, 16=4h, 48=12h)
            ratio_windows = [4, 8, 16, 48]
            for window in ratio_windows:
                df[f'ratio_sma_{window}_{secondary_name}'] = (
                    df[f'{primary_name}_{secondary_name}_ratio'].rolling(window).mean()
                )
                # Ratio relative to its moving average - safer division
                sma_denominator = df[f'ratio_sma_{window}_{secondary_name}'] + 1e-6
                df[f'ratio_vs_sma_{window}_{secondary_name}'] = np.clip(
                    df[f'{primary_name}_{secondary_name}_ratio'] / sma_denominator, 0.01, 100
                )
            
            # Ratio volatility (96 periods = 1 day in 15m data)
            df[f'ratio_volatility_{secondary_name}'] = (
                df[f'{primary_name}_{secondary_name}_ratio'].rolling(96).std()
            )

            # Ratio momentum (96 periods = 1 day in 15m data)
            df[f'ratio_momentum_1d_{secondary_name}'] = (
                df[f'{primary_name}_{secondary_name}_ratio'].pct_change(96, fill_method=None)
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

                # Combined volume momentum (16 periods = 4h, 96 periods = 1d in 15m data)
                df[f'combined_volume_momentum_{secondary_name}'] = (
                    (df['volume'].rolling(16).mean() / df['volume'].rolling(96).mean()) *
                    (df[f'{secondary_name}_volume'].rolling(16).mean() / df[f'{secondary_name}_volume'].rolling(96).mean())
                )
            
            # Cross-volatility features
            if f'{secondary_name}_volatility' in df.columns and 'volatility_4h' in df.columns:
                df[f'volatility_spread_{secondary_name}'] = (
                    df['volatility_4h'] - df[f'{secondary_name}_volatility']
                )
                volatility_denominator = df[f'{secondary_name}_volatility'] + 1e-6
                df[f'volatility_ratio_{secondary_name}'] = np.clip(
                    df['volatility_4h'] / volatility_denominator, 1e-3, 1e3
                )
            
            # 6. MARKET REGIME INDICATORS
            # High correlation periods vs divergence periods
            df[f'high_correlation_regime_{secondary_name}'] = (
                df[f'corr_strength_{secondary_name}'] > 0.7
            ).astype(int)

            df[f'low_correlation_regime_{secondary_name}'] = (
                df[f'corr_strength_{secondary_name}'] < 0.3
            ).astype(int)

            # Correlation breakdown events (96 periods = 1 day in 15m data)
            df[f'correlation_breakdown_{secondary_name}'] = (
                (df[f'corr_{secondary_name}_1d'].shift(96) > 0.5) &
                (df[f'corr_{secondary_name}_1d'] < 0.2)
            ).astype(int)

            # 7. RELATIVE STRENGTH
            # Which crypto is performing better over the past day (96 periods = 1 day in 15m)
            # 7-day rolling mean removed — requires 672 bars and captures too-long a horizon
            primary_returns_1d = df['price'].pct_change(96, fill_method=None)
            secondary_returns_1d = df[f'{secondary_name}_price'].pct_change(96, fill_method=None)

            df[f'relative_strength_{secondary_name}'] = (
                primary_returns_1d - secondary_returns_1d
            )
            
            # 8. CROSS-ASSET VOLUME FEATURES
            if 'volume' in df.columns and f'{secondary_name}_volume' in df.columns:
                # Volume correlation (96 periods = 1 day in 15m data)
                df[f'volume_correlation_{secondary_name}_1d'] = (
                    df['volume'].rolling(96).corr(df[f'{secondary_name}_volume'])
                )

                # Volume ratio analysis - safer division
                volume_denom = df[f'{secondary_name}_volume'] + 1e-3
                df[f'volume_ratio_{secondary_name}'] = np.clip(
                    df['volume'] / volume_denom, 1e-6, 1e6
                )
                df[f'volume_ratio_sma_14_{secondary_name}'] = (
                    df[f'volume_ratio_{secondary_name}'].rolling(14).mean()
                )

                # Volume divergence signals (when volumes move opposite directions)
                primary_vol_change = df['volume'].pct_change(fill_method=None)
                secondary_vol_change = df[f'{secondary_name}_volume'].pct_change(fill_method=None)
                df[f'volume_divergence_{secondary_name}'] = (
                    (primary_vol_change > 0) & (secondary_vol_change < 0) |
                    (primary_vol_change < 0) & (secondary_vol_change > 0)
                ).astype(int)
            
            # Cross-OBV features (96 periods = 1 day in 15m data)
            if 'obv' in df.columns and f'{secondary_name}_obv' in df.columns:
                df[f'obv_correlation_{secondary_name}_1d'] = (
                    df['obv'].rolling(96).corr(df[f'{secondary_name}_obv'])
                )
                df[f'obv_divergence_{secondary_name}'] = (
                    (df['obv'].diff() > 0) & (df[f'{secondary_name}_obv'].diff() < 0) |
                    (df['obv'].diff() < 0) & (df[f'{secondary_name}_obv'].diff() > 0)
                ).astype(int)
            
            # Cross-VWAP features
            if 'vwap_1d' in df.columns and f'{secondary_name}_vwap' in df.columns:
                df[f'vwap_ratio_{secondary_name}'] = (
                    df['vwap_1d'] / df[f'{secondary_name}_vwap']
                )
                df[f'both_above_vwap_{secondary_name}'] = (
                    (df['price'] > df['vwap_1d']) &
                    (df[f'{secondary_name}_price'] > df[f'{secondary_name}_vwap'])
                ).astype(int)

            # Cross-money flow features (96 periods = 1 day in 15m data)
            if 'money_flow_1d' in df.columns and f'{secondary_name}_money_flow' in df.columns:
                df[f'money_flow_correlation_{secondary_name}'] = (
                    df['money_flow_1d'].rolling(96).corr(df[f'{secondary_name}_money_flow'])
                )
                df[f'money_flow_divergence_{secondary_name}'] = (
                    (df['money_flow_1d'] > 0) & (df[f'{secondary_name}_money_flow'] < 0) |
                    (df['money_flow_1d'] < 0) & (df[f'{secondary_name}_money_flow'] > 0)
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
                # Standard TA period counts (independent of timeframe)
                volume_windows = [7, 14, 30, 50]
                for window in volume_windows:
                    # Volume moving averages
                    df[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                    df[f'volume_ema_{window}'] = df['volume'].ewm(span=window).mean()

                    # Volume ratio to moving average
                    df[f'volume_ratio_sma_{window}'] = df['volume'] / (df[f'volume_sma_{window}'] + 1e-6)
                    df[f'volume_ratio_ema_{window}'] = df['volume'] / (df[f'volume_ema_{window}'] + 1e-6)

                # Volume percentile relative to the past 7 days (672 periods in 15m)
                # 30d percentile (2880 periods) removed — too long for intraday targets
                df['volume_percentile_7d'] = df['volume'].rolling(672).rank(pct=True)

                # ==== ON-BALANCE VOLUME (OBV) ====
                price_change = df['price'].diff()
                volume_direction = np.where(price_change > 0, df['volume'],
                                          np.where(price_change < 0, -df['volume'], 0))
                df['obv'] = volume_direction.cumsum()

                # OBV moving averages and momentum
                df['obv_sma_14'] = df['obv'].rolling(14).mean()
                df['obv_ema_14'] = df['obv'].ewm(span=14).mean()
                df['obv_momentum'] = df['obv'].diff(96)   # 96 periods = 1 day in 15m data
                df['obv_acceleration'] = df['obv_momentum'].diff()

                # OBV relative to price (divergence detection, 96 periods = 1 day in 15m)
                df['obv_price_correlation_1d'] = df['obv'].rolling(96).corr(df['price'])
                df['obv_price_divergence'] = (
                    (df['price'].pct_change(96) > 0) & (df['obv_momentum'] < 0) |
                    (df['price'].pct_change(96) < 0) & (df['obv_momentum'] > 0)
                ).astype(int)

                # ==== VOLUME WEIGHTED AVERAGE PRICE (VWAP) ====
                # Only 1d (96 periods) — 7d and 30d VWAP removed (too coarse for intraday)
                vwap_periods = [(96, '1d')]
                for period, label in vwap_periods:
                    pv = df['price'] * df['volume']
                    df[f'vwap_{label}'] = (
                        pv.rolling(period).sum() / df['volume'].rolling(period).sum()
                    )
                    df[f'price_vs_vwap_{label}'] = df['price'] / df[f'vwap_{label}']
                    df[f'above_vwap_{label}'] = (df['price'] > df[f'vwap_{label}']).astype(int)

                # ==== VOLUME MOMENTUM AND ACCELERATION ====
                # 7d period (672) removed — too long for intraday targets
                volume_roc_periods = [(1, '15m'), (4, '1h'), (16, '4h'), (96, '1d')]
                for period, label in volume_roc_periods:
                    df[f'volume_roc_{label}'] = df['volume'].pct_change(period)

                # Volume acceleration
                df['volume_acceleration_1d'] = df['volume_roc_1d'].diff()

                # Volume trend strength via linear correlation over 1 day (96 periods)
                # volume_momentum_7d removed — required 672 bars (7d)
                df['volume_trend_strength'] = df['volume'].rolling(96).apply(
                    lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
                )

                # ==== VOLUME SPIKES AND ANOMALIES ====
                df['volume_spike_2x'] = (df['volume'] > 2 * df['volume_sma_30']).astype(int)
                df['volume_spike_3x'] = (df['volume'] > 3 * df['volume_sma_30']).astype(int)
                df['volume_drought'] = (df['volume'] < 0.5 * df['volume_sma_30']).astype(int)

                # ==== PRICE-VOLUME RELATIONSHIPS ====
                # 7d and 30d volume volatility/zscore removed (require 672–2880 bars)
                df['price_volume_corr_1d'] = df['price'].rolling(96).corr(df['volume'])

                # Volume-weighted price changes (using 15m base change)
                df['volume_weighted_return_15m'] = df['price_change_15m'] * df['volume_ratio_sma_14']
                df['volume_weighted_return_1d'] = df['price'].pct_change(96) * df['volume_ratio_sma_14']

                # Price impact per unit volume
                df['price_impact_per_volume'] = (
                    df['price_change_15m'].abs() / (df['volume_ratio_sma_14'] + 1e-6)
                )

                # Volume distribution analysis (using 7d percentile — 30d removed)
                df['high_volume_regime'] = (df['volume_percentile_7d'] > 0.8).astype(int)
                df['low_volume_regime'] = (df['volume_percentile_7d'] < 0.2).astype(int)
            
            # ==== MARKET CAP ANALYSIS ====
            if 'market_cap' in df.columns:
                # Market cap moving averages
                mc_windows = [7, 14, 30]
                for window in mc_windows:
                    df[f'market_cap_sma_{window}'] = df['market_cap'].rolling(window).mean()
                    df[f'market_cap_ratio_sma_{window}'] = df['market_cap'] / df[f'market_cap_sma_{window}']

                # Market cap rate of change — 7d period removed
                mc_roc_periods = [(1, '15m'), (96, '1d')]
                for period, label in mc_roc_periods:
                    df[f'market_cap_roc_{label}'] = df['market_cap'].pct_change(period)

                # Market cap vs volume relationship (96 = 1 day in 15m)
                # momentum_7d, volatility_7d, percentile_30d removed (require 672–2880 bars)
                if 'volume' in df.columns:
                    df['market_cap_volume_ratio'] = df['market_cap'] / (df['volume'] + 1e-6)
                    df['mc_volume_correlation_1d'] = df['market_cap'].rolling(96).corr(df['volume'])
            
            # ==== ADVANCED VOLUME FEATURES ====
            if 'volume' in df.columns:
                # Volume rate of change acceleration (1d label)
                df['volume_roc_acceleration'] = df['volume_roc_1d'].diff()

                # Volume efficiency (96=1d in 15m)
                df['volume_efficiency'] = (
                    df['price'].pct_change(96).abs() / (df['volume_ratio_sma_30'] + 1e-6)
                )

                # Money flow approximation (96 periods = 1 day in 15m)
                # cumulative_volume_7d/30d, volume_concentration_1d, money_flow_7d removed
                price_direction = np.where(df['price'].diff() > 0, 1,
                                         np.where(df['price'].diff() < 0, -1, 0))
                df['money_flow_raw'] = df['volume'] * price_direction
                df['money_flow_1d'] = df['money_flow_raw'].rolling(96).sum()

                # Volume-based support/resistance
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
    
    def add_momentum_features(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Add momentum features to better capture price movement patterns.
        With 15-minute data: periods are in 15-minute units.
        1h=4, 2h=8, 4h=16, 6h=24, 12h=48, 24h=96
        """
        try:
            logger.info("Adding momentum features...")

            # Price momentum indicators (in 15m periods)
            # 4=1h, 16=4h, 48=12h, 96=24h
            momentum_configs = [(4, '1h'), (16, '4h'), (48, '12h'), (96, '1d')]
            for period, label in momentum_configs:
                df[f'momentum_{label}'] = df[price_col].pct_change(period) * 100
                df[f'momentum_abs_{label}'] = df[f'momentum_{label}'].abs()

            # Defragment the DataFrame to avoid pandas PerformanceWarning
            df = df.copy()

            # Rate of change acceleration (4=1h, 16=4h in 15m)
            df['roc_1h'] = df[price_col].pct_change(4) * 100
            df['roc_4h'] = df[price_col].pct_change(16) * 100
            df['roc_acceleration'] = df['roc_1h'] - df['roc_4h']

            # Moving average momentum (48=12h, 96=1d in 15m)
            # 2d (192 periods) removed — too long relative to 15m/1h/4h targets
            for window, label in [(48, '12h'), (96, '1d')]:
                ma = df[price_col].rolling(window=window).mean()
                df[f'ma_momentum_{label}'] = ((df[price_col] - ma) / ma * 100).fillna(0)

            # Velocity indicators (rate of price change in 15m periods)
            df['velocity_15m'] = df[price_col].diff(1)
            df['velocity_1h'] = df[price_col].diff(4)
            df['velocity_4h'] = df[price_col].diff(16)

            # Momentum strength
            df['momentum_strength'] = (
                df['momentum_1h'].abs() +
                df['momentum_4h'].abs() +
                df['momentum_12h'].abs()
            ) / 3

            # Trend consistency (24=6h, 48=12h, 96=1d in 15m)
            for window, label in [(24, '6h'), (48, '12h'), (96, '1d')]:
                returns = df[price_col].pct_change()
                df[f'trend_consistency_{label}'] = returns.rolling(window=window).apply(
                    lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
                )

            logger.info("Added momentum features")
            
        except Exception as e:
            logger.error(f"Error adding momentum features: {e}")
        
        return df
    
    def add_cvd_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Cumulative Volume Delta (CVD) and OHLC candle features.

        CVD features come from Binance taker_buy_base_vol / taker_sell_base_vol.
        They capture the net directional pressure of aggressive buyers vs sellers
        — the single most informative signal for intraday direction prediction.

        OHLC candle features (body, wicks, close position) are also added here
        because they require open/high/low columns only available from Binance.

        Gracefully no-ops if the required columns are not present.
        """
        df = df.copy()

        # ── CVD features ────────────────────────────────────────────────────────
        has_cvd = ('taker_buy_base_vol' in df.columns and
                   'taker_sell_base_vol' in df.columns)

        if has_cvd:
            # Per-bar volume delta: positive = net buying, negative = net selling
            df['volume_delta'] = df['taker_buy_base_vol'] - df['taker_sell_base_vol']

            # Buy/sell ratio: fraction of volume driven by aggressive buyers (0→1)
            df['buy_sell_ratio'] = df['taker_buy_base_vol'] / (df['volume'] + 1e-10)

            # Rolling CVD (net order-flow pressure over intraday windows)
            # 4=1h, 16=4h, 96=1d in 15m bars
            for window, label in [(4, '1h'), (16, '4h'), (96, '1d')]:
                df[f'cvd_{label}'] = df['volume_delta'].rolling(window).sum()

            # Normalised CVD: scale by total volume so different size bars are comparable
            for window, label in [(4, '1h'), (16, '4h'), (96, '1d')]:
                total = df['volume'].rolling(window).sum()
                df[f'cvd_norm_{label}'] = df[f'cvd_{label}'] / (total + 1e-10)

            # Rolling buy pressure (fraction of volume from buyers)
            for window, label in [(4, '1h'), (16, '4h'), (96, '1d')]:
                df[f'buy_pressure_{label}'] = df['buy_sell_ratio'].rolling(window).mean()

            # CVD momentum: how much CVD changed in the last window
            df['cvd_1h_change'] = df['cvd_1h'].diff(4)
            df['cvd_4h_change'] = df['cvd_4h'].diff(16)

            # Price-CVD divergence: price rising but sell pressure dominant → bearish
            price_chg_1h = df['price'].pct_change(4)
            price_chg_4h = df['price'].pct_change(16)
            df['price_cvd_div_1h'] = (
                ((price_chg_1h > 0) & (df['cvd_1h'] < 0)) |
                ((price_chg_1h < 0) & (df['cvd_1h'] > 0))
            ).astype(int)
            df['price_cvd_div_4h'] = (
                ((price_chg_4h > 0) & (df['cvd_4h'] < 0)) |
                ((price_chg_4h < 0) & (df['cvd_4h'] > 0))
            ).astype(int)

            # Short lags of the key CVD columns (1=15m, 2=30m, 4=1h, 8=2h)
            for lag in [1, 2, 4, 8]:
                df[f'volume_delta_lag_{lag}'] = df['volume_delta'].shift(lag)
                df[f'buy_sell_ratio_lag_{lag}'] = df['buy_sell_ratio'].shift(lag)

            logger.info("Added CVD / volume-delta features")
        else:
            logger.warning("taker_buy/sell_base_vol not found — CVD features skipped")

        # ── Trade-count features ────────────────────────────────────────────────
        if 'num_trades' in df.columns:
            df['num_trades_ratio_1h'] = (
                df['num_trades'].rolling(4).mean() /
                (df['num_trades'].rolling(96).mean() + 1e-10)
            )
            df['num_trades_ratio_4h'] = (
                df['num_trades'].rolling(16).mean() /
                (df['num_trades'].rolling(96).mean() + 1e-10)
            )

        # ── OHLC candle features ────────────────────────────────────────────────
        has_ohlc = all(c in df.columns for c in ['open', 'high', 'low'])

        if has_ohlc:
            eps = df['open'] + 1e-10

            # Candle body: signed (+ = bullish bar, - = bearish bar)
            df['candle_body']  = (df['price'] - df['open']) / eps

            # Total candle range
            df['candle_range'] = (df['high'] - df['low']) / eps

            # Upper and lower wicks
            candle_top    = df[['price', 'open']].max(axis=1)
            candle_bottom = df[['price', 'open']].min(axis=1)
            df['upper_wick'] = (df['high'] - candle_top)    / eps
            df['lower_wick'] = (candle_bottom - df['low'])  / eps

            # Wick ratio: large lower wick → buying at lows (bullish pressure)
            df['wick_ratio'] = df['lower_wick'] / (df['upper_wick'] + 1e-10)

            # Close position within the bar (0 = closed at low, 1 = closed at high)
            candle_range_abs = df['high'] - df['low']
            df['close_position'] = (df['price'] - df['low']) / (candle_range_abs + 1e-10)

            # Short lags for candle body and close position
            for lag in [1, 2, 4]:
                df[f'candle_body_lag_{lag}']     = df['candle_body'].shift(lag)
                df[f'close_position_lag_{lag}']  = df['close_position'].shift(lag)

            logger.info("Added OHLC candle features")
        else:
            logger.warning("open/high/low columns not found — candle features skipped")

        return df

    def add_volatility_features(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Add volatility features to capture market uncertainty and movement patterns.
        With 15-minute data: 1h=4, 4h=16, 12h=48, 1d=96, 2d=192 periods.
        """
        try:
            logger.info("Adding volatility features...")

            # Defragment the DataFrame to avoid pandas PerformanceWarning
            df = df.copy()

            # Rolling volatility (standard deviation of returns)
            returns = df[price_col].pct_change()
            # 24=6h, 48=12h, 96=1d, 192=2d in 15m
            for window, label in [(24, '6h'), (48, '12h'), (96, '1d'), (192, '2d')]:
                df[f'volatility_{label}'] = returns.rolling(window=window).std() * 100

            # Historical volatility (96=1d, 192=2d in 15m)
            # 7d (672 periods) removed — too long for intraday targets
            for window, label in [(96, '1d'), (192, '2d')]:
                df[f'hist_vol_{label}'] = (
                    df[price_col].rolling(window=window).std() /
                    df[price_col].rolling(window=window).mean() * 100
                ).fillna(0)

            # Volatility ratios (short vs long term)
            df['vol_ratio_6h_1d'] = df['volatility_6h'] / (df['volatility_1d'] + 1e-10)
            df['vol_ratio_12h_2d'] = df['volatility_12h'] / (df['volatility_2d'] + 1e-10)

            # Price range indicators (24=6h, 48=12h, 96=1d in 15m)
            high_col = 'high' if 'high' in df.columns else price_col
            low_col = 'low' if 'low' in df.columns else price_col

            for window, label in [(24, '6h'), (48, '12h'), (96, '1d')]:
                rolling_high = df[high_col].rolling(window=window).max()
                rolling_low = df[low_col].rolling(window=window).min()
                df[f'price_range_{label}'] = (
                    (rolling_high - rolling_low) / rolling_low * 100
                ).fillna(0)

            # Volatility breakout indicators
            df['vol_breakout'] = (
                df['volatility_6h'] > df['volatility_1d'].shift(1) * 1.5
            ).astype(int)

            # Average True Range (ATR) approximation
            if 'high' in df.columns and 'low' in df.columns:
                tr = pd.DataFrame({
                    'hl': df['high'] - df['low'],
                    'hc': abs(df['high'] - df[price_col].shift(1)),
                    'lc': abs(df['low'] - df[price_col].shift(1))
                }).max(axis=1)

                for window, label in [(14, '14p'), (96, '1d')]:
                    df[f'atr_{label}'] = tr.rolling(window=window).mean()

            logger.info("Added volatility features")
            
        except Exception as e:
            logger.error(f"Error adding volatility features: {e}")
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, price_col: str = 'price') -> pd.DataFrame:
        """
        Create target variables for different prediction horizons.
        With 15-minute data:
          15m  → shift(-1)   = 1 period ahead
          1h   → shift(-4)   = 4 periods ahead
          4h   → shift(-16)  = 16 periods ahead
        CRITICAL: This creates targets that look FORWARD in time to avoid data leakage
        """
        df = df.copy()

        try:
            # Direction targets only — 1 if price goes UP, 0 if DOWN
            # shift(-N) looks N bars ahead; > current price means UP
            # Use .where(future.notna()) to preserve NaN at the tail so the leakage
            # validator can confirm the shift was applied and training drops those rows.
            future_15m = df[price_col].shift(-1)
            future_1h  = df[price_col].shift(-4)
            future_4h  = df[price_col].shift(-16)
            df['target_direction_15m'] = (future_15m > df[price_col]).where(future_15m.notna())
            df['target_direction_1h']  = (future_1h  > df[price_col]).where(future_1h.notna())
            df['target_direction_4h']  = (future_4h  > df[price_col]).where(future_4h.notna())

            # Future datetime stamps for evaluation alignment
            df['target_datetime_15m'] = df['datetime'] + pd.Timedelta(minutes=15)
            df['target_datetime_1h']  = df['datetime'] + pd.Timedelta(hours=1)
            df['target_datetime_4h']  = df['datetime'] + pd.Timedelta(hours=4)

            logger.info("Created direction target variables for 15m, 1h, 4h horizons")

        except Exception as e:
            logger.error(f"Error creating target variables: {e}")

        return df
    
    def add_funding_rate_features(self, df: pd.DataFrame, funding_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add perpetual funding rate features via merge_asof (backward) so each 15m bar
        receives the most-recent 8h settlement value — the rate actually in effect.
        """
        if funding_df is None or funding_df.empty:
            return df

        try:
            fdf = funding_df.copy()
            fdf['datetime'] = pd.to_datetime(fdf['datetime'])
            fdf = fdf.sort_values('datetime').reset_index(drop=True)

            fdf['funding_rate_rolling_8h']  = fdf['funding_rate'].rolling(3, min_periods=1).mean()
            fdf['funding_rate_std_24h']     = fdf['funding_rate'].rolling(3, min_periods=1).std()
            fdf['funding_rate_cumulative']  = fdf['funding_rate'].cumsum()
            fdf['funding_rate_change']      = fdf['funding_rate'].diff()

            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)

            df = pd.merge_asof(df, fdf, on='datetime', direction='backward')
            logger.info("Added funding rate features")
        except Exception as e:
            logger.error(f"Error adding funding rate features: {e}")

        return df

    def add_open_interest_features(self, df: pd.DataFrame, oi_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add open interest features via merge_asof (backward).
        Renames oi_usd → open_interest; drops oi_volume_usd after deriving oi_volume_ratio.
        """
        if oi_df is None or oi_df.empty:
            return df

        try:
            odf = oi_df.copy()
            odf['datetime'] = pd.to_datetime(odf['datetime'])
            odf = odf.sort_values('datetime').reset_index(drop=True)

            odf['oi_delta']        = odf['oi_usd'].diff()
            odf['oi_trend']        = odf['oi_delta'].diff()   # 2nd derivative / acceleration
            odf['oi_volume_ratio'] = odf['oi_volume_usd'] / (odf['oi_usd'] + 1e-6)

            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)

            df = pd.merge_asof(df, odf, on='datetime', direction='backward')

            df['oi_price_ratio'] = df['oi_usd'] / (df['price'] + 1e-6)
            df = df.rename(columns={'oi_usd': 'open_interest'})
            df = df.drop(columns=['oi_volume_usd'], errors='ignore')

            logger.info("Added open interest features")
        except Exception as e:
            logger.error(f"Error adding open interest features: {e}")

        return df

    def add_fear_greed_features(self, df: pd.DataFrame, fng_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add Fear & Greed Index features via merge_asof (backward).
        Daily values are broadcast to all 15m bars within the day — no future leakage.
        """
        if fng_df is None or fng_df.empty:
            return df

        try:
            fdf = fng_df.copy()
            fdf['datetime'] = pd.to_datetime(fdf['datetime'])
            fdf = fdf.sort_values('datetime').reset_index(drop=True)

            fdf['fear_greed_regime'] = pd.cut(
                fdf['fear_greed_value'],
                bins=[0, 30, 60, 100],
                labels=[0, 1, 2],
                include_lowest=True,
            ).astype(float)
            fdf['fear_greed_change']           = fdf['fear_greed_value'].diff()
            fdf['fear_greed_distance_from_50'] = fdf['fear_greed_value'] - 50

            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)

            df = pd.merge_asof(df, fdf, on='datetime', direction='backward')
            logger.info("Added Fear & Greed features")
        except Exception as e:
            logger.error(f"Error adding Fear & Greed features: {e}")

        return df

    def prepare_features(self, crypto_df: pd.DataFrame, market_df: pd.DataFrame = None, external_data: dict = None) -> pd.DataFrame:
        """
        Main feature preparation pipeline
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Add all features
        df = self.add_technical_indicators(crypto_df)
        df = self.add_time_features(df)
        df = self.add_cvd_features(df)      # Binance taker volumes → CVD + candle features
        df = self.add_lagged_features(df)   # lags price, price_change_15m, rsi
        df = self.add_volume_based_features(df)
        df = self.add_momentum_features(df)  # NEW: Better momentum capture
        df = self.add_volatility_features(df)  # NEW: Volatility patterns

        # External derivative/sentiment data (no future leakage via merge_asof backward)
        external_data = external_data or {}
        df = self.add_funding_rate_features(df, external_data.get('funding_rate'))
        df = self.add_open_interest_features(df, external_data.get('open_interest'))
        df = self.add_fear_greed_features(df, external_data.get('fear_greed'))

        df = self.create_target_variables(df)
        
        # Clean data: replace infinity and extreme values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Replace extreme values (beyond 3 standard deviations) with NaN
        _target_cols_set = {
            'target_direction_15m', 'target_direction_1h', 'target_direction_4h',
        }
        for col in numeric_cols:
            if col not in _target_cols_set:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if not pd.isna(std_val) and std_val > 0:
                    df.loc[np.abs(df[col] - mean_val) > 3 * std_val, col] = np.nan

        # Store feature column names (excluding targets and metadata)
        exclude_cols = [
            'datetime', 'crypto',
            'target_direction_15m', 'target_direction_1h', 'target_direction_4h',
            'target_datetime_15m', 'target_datetime_1h', 'target_datetime_4h',
        ]

        self.feature_columns = [col for col in df.columns if col not in exclude_cols]

        # CRITICAL: Data leakage validation (re-derive feature columns after cross-asset enrichment)
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        self.validate_no_data_leakage(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_columns)}")
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df
    
    def validate_no_data_leakage(self, df: pd.DataFrame):
        """
        Validate that no features use future information (data leakage detection)
        """
        logger.info("🔍 Validating dataset for data leakage...")
        
        # Check 1: Ensure target variables are properly shifted into the future.
        # shift(-n) produces NaN in the LAST n rows (not the first), so a valid
        # value at index 0 is completely correct.  We verify the tail is NaN.
        target_shift_map = {
            'target_direction_15m': 1,   # shift(-1)
            'target_direction_1h':  4,   # shift(-4)
            'target_direction_4h':  16,  # shift(-16)
        }
        for target, shift_n in target_shift_map.items():
            if target in df.columns:
                tail_nans = df[target].iloc[-shift_n:].isna().sum()
                if tail_nans < shift_n:
                    logger.warning(
                        f"⚠️  {target}: expected {shift_n} NaN(s) at tail "
                        f"but only found {tail_nans} — shift may not have been applied!"
                    )
                else:
                    logger.info(f"✅ {target}: tail NaN check passed ({tail_nans}/{shift_n})")
        
        # Check 2: Verify datetime ordering
        if 'datetime' in df.columns:
            is_sorted = df['datetime'].is_monotonic_increasing
            if not is_sorted:
                logger.warning("⚠️  Datetime column is not sorted - this could cause leakage!")
        
        # Check 3: Look for suspicious feature names that might use future data
        suspicious_patterns = ['future', 'next', 'forward', 'ahead', 'target']
        feature_cols = [col for col in df.columns if col in self.feature_columns]
        
        for col in feature_cols:
            col_lower = col.lower()
            for pattern in suspicious_patterns:
                if pattern in col_lower and not col.startswith('lagged_'):
                    logger.warning(f"⚠️  Suspicious feature name '{col}' - may contain future information!")
        
        # Check 4: Validate cross-asset features don't use future timestamps
        cross_asset_features = [col for col in feature_cols if any(crypto in col for crypto in ['bitcoin', 'ethereum'])]
        if cross_asset_features:
            logger.info(f"✅ Found {len(cross_asset_features)} cross-asset features - validating timing...")
            # Cross-asset features should only use same-time or lagged data
            for col in cross_asset_features:
                if 'lead' in col.lower() or 'future' in col.lower():
                    logger.warning(f"⚠️  Cross-asset feature '{col}' may use future data!")
        
        # Check 5: Ensure lagged features actually lag behind
        lagged_features = [col for col in feature_cols if 'lagged' in col.lower()]
        if lagged_features:
            logger.info(f"✅ Found {len(lagged_features)} lagged features - these should be safe")
        
        logger.info("🔍 Data leakage validation complete")
    
    def prepare_features_with_cross_asset_correlation(self, crypto_data_dict: Dict[str, pd.DataFrame],
                                                    market_df: pd.DataFrame = None,
                                                    external_data_by_crypto: dict = None) -> Dict[str, pd.DataFrame]:
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
                ext = (external_data_by_crypto or {}).get(crypto_name, {})
                prepared_data[crypto_name] = self.prepare_features(crypto_df, market_df, ext)
        
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
                    if col not in [
                        'datetime',
                        'target_direction_15m', 'target_direction_1h', 'target_direction_4h',
                    ]:
                        # Check for extremely large values (beyond reasonable float64 range)
                        max_safe_value = 1e10  # 10 billion - reasonable upper bound
                        crypto_df.loc[crypto_df[col].abs() > max_safe_value, col] = np.nan

                        # Also clean values that are too close to zero in division results
                        if any(substring in col for substring in ['_ratio', '_vs_', 'divergence', 'correlation']):
                            crypto_df.loc[crypto_df[col].abs() > 1000, col] = np.nan

                # Fill remaining NaN values with forward fill, then backward fill, then zero.
                # External columns (funding rate, OI, F&G) are excluded from the main loop
                # and zero-filled separately — bfilling them would pull future values backward
                # across the start of history.
                _EXTERNAL_COLS = {
                    'funding_rate', 'funding_rate_rolling_8h', 'funding_rate_std_24h',
                    'funding_rate_cumulative', 'funding_rate_change',
                    'open_interest', 'oi_delta', 'oi_price_ratio', 'oi_trend', 'oi_volume_ratio',
                    'fear_greed_value', 'fear_greed_regime', 'fear_greed_change',
                    'fear_greed_distance_from_50',
                }
                _exclude_fill = {
                    'datetime',
                    'target_direction_15m', 'target_direction_1h', 'target_direction_4h',
                } | _EXTERNAL_COLS
                for col in numeric_cols:
                    if col not in _exclude_fill:
                        # Forward fill
                        crypto_df[col] = crypto_df[col].ffill()
                        # Backward fill
                        crypto_df[col] = crypto_df[col].bfill()
                        # Finally fill any remaining NaN with 0
                        crypto_df[col] = crypto_df[col].fillna(0)

                # Zero-fill external columns only — no bfill to avoid pulling
                # future values backward into the period before data starts.
                for col in _EXTERNAL_COLS:
                    if col in crypto_df.columns:
                        crypto_df[col] = crypto_df[col].fillna(0)
                
                # Final check for any remaining problematic values
                inf_check = np.isinf(crypto_df[numeric_cols]).sum().sum()
                nan_check = np.isnan(crypto_df[numeric_cols]).sum().sum()
                
                logger.info(f"{crypto_name} post-cleaning: {inf_check} inf values, {nan_check} NaN values")
                
                prepared_data[crypto_name] = crypto_df
        
        # Update feature columns for the enhanced datasets
        for crypto_name, crypto_df in prepared_data.items():
            if not crypto_df.empty:
                exclude_cols = [
                    'datetime', 'crypto',
                    'target_direction_15m', 'target_direction_1h', 'target_direction_4h',
                    'target_datetime_15m', 'target_datetime_1h', 'target_datetime_4h',
                ]
                
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