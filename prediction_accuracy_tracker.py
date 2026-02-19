"""
Prediction Accuracy Tracker for Crypto Price Prediction Bot
Tracks and evaluates prediction performance with comprehensive metrics and storage
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
import config

logger = logging.getLogger(__name__)

class PredictionAccuracyTracker:
    def __init__(self):
        self.initialize_evaluation_tables()
        
    def initialize_evaluation_tables(self):
        """Initialize database tables for prediction accuracy tracking"""
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Table for storing prediction evaluations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                crypto TEXT,
                prediction_horizon TEXT,
                predicted_price REAL,
                actual_price REAL,
                absolute_error REAL,
                percent_error REAL,
                squared_error REAL,
                direction_predicted INTEGER,
                direction_actual INTEGER,
                direction_correct INTEGER,
                prediction_timestamp TIMESTAMP,
                evaluation_timestamp TIMESTAMP,
                confidence REAL,
                target_timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        ''')
        
        # Table for storing actual prices at prediction target times
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS actual_prices_at_targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crypto TEXT,
                target_timestamp TIMESTAMP,
                actual_price REAL,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(crypto, target_timestamp)
            )
        ''')
        
        # Table for aggregated accuracy metrics by time period
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accuracy_metrics_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crypto TEXT,
                prediction_horizon TEXT,
                period_start TIMESTAMP,
                period_end TIMESTAMP,
                total_predictions INTEGER,
                mean_absolute_error REAL,
                mean_percent_error REAL,
                root_mean_squared_error REAL,
                direction_accuracy REAL,
                median_absolute_error REAL,
                std_percent_error REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for storing prediction performance over time
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crypto TEXT,
                timestamp TIMESTAMP,
                actual_price REAL,
                predicted_price_15m REAL,
                predicted_price_1h REAL,
                predicted_price_4h REAL,
                error_15m REAL,
                error_1h REAL,
                error_4h REAL,
                percent_error_15m REAL,
                percent_error_1h REAL,
                percent_error_4h REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(crypto, timestamp)
            )
        ''')
        
        # Migration: delete evaluations that were stored with the broken direction-accuracy
        # fallback (direction_predicted was always 1 when start_price was unavailable).
        # Those rows have direction_predicted = 1 AND direction_actual = 1, making direction_correct
        # appear inflated.  We can only reliably identify bad rows by checking whether the
        # originating prediction had current_price = 0 or NULL (the old schema default).
        # Simplest safe approach: delete all evaluations whose linked prediction has no current_price,
        # so they will be re-evaluated correctly next time the bot runs.
        try:
            cursor.execute('''
                DELETE FROM prediction_evaluations
                WHERE prediction_id IN (
                    SELECT id FROM predictions
                    WHERE current_price IS NULL OR current_price = 0
                )
            ''')
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Migration: removed {deleted} evaluations with unreliable direction data "
                            f"(predictions lacked current_price). They will be re-evaluated.")
        except Exception:
            pass  # Table may not exist yet on first run

        conn.commit()
        conn.close()
        logger.info("Prediction accuracy tracking tables initialized")
    
    def store_actual_price_at_target(self, crypto: str, target_timestamp: datetime, actual_price: float):
        """Store actual price at a specific target timestamp for later evaluation"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO actual_prices_at_targets 
                (crypto, target_timestamp, actual_price)
                VALUES (?, ?, ?)
            ''', (crypto, target_timestamp.isoformat(), actual_price))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store actual price for {crypto}: {e}")
    
    def evaluate_prediction(self, prediction_id: int, crypto: str, prediction_horizon: str, 
                          predicted_price: float, actual_price: float, 
                          prediction_timestamp: datetime, target_timestamp: datetime,
                          confidence: float = 0.0) -> Dict:
        """
        Evaluate a single prediction and store the results
        """
        try:
            # predicted_price now stores direction_prob (P(UP), range 0–1), not a price.
            # Price error metrics are meaningless; direction accuracy is the signal.
            absolute_error = 0.0
            percent_error = 0.0
            squared_error = 0.0
            
            # Direction accuracy - compare predicted vs actual price DIRECTION from the price
            # at the time the prediction was made (start_price).
            direction_predicted = None
            direction_actual = None
            direction_correct = 0
            
            # Get the starting price (at prediction time) to calculate direction.
            # Priority: (1) current_price stored with the prediction row,
            #           (2) actual_prices_at_targets near prediction_timestamp,
            #           (3) crypto_data near prediction_timestamp.
            start_price = None
            try:
                conn = sqlite3.connect(config.DATABASE_PATH)
                cursor = conn.cursor()
                
                # 1. Use current_price stored directly in the predictions row (most reliable)
                if prediction_id is not None:
                    cursor.execute(
                        'SELECT current_price FROM predictions WHERE id = ?',
                        (prediction_id,)
                    )
                    row = cursor.fetchone()
                    if row and row[0] and row[0] > 0:
                        start_price = row[0]
                
                # 2. Fall back to actual_prices_at_targets near prediction_timestamp
                if start_price is None:
                    cursor.execute('''
                        SELECT actual_price FROM actual_prices_at_targets
                        WHERE crypto = ? AND 
                              ABS(julianday(target_timestamp) - julianday(?)) < (30.0/1440.0)
                        ORDER BY ABS(julianday(target_timestamp) - julianday(?))
                        LIMIT 1
                    ''', (crypto, prediction_timestamp.isoformat(), prediction_timestamp.isoformat()))
                    row = cursor.fetchone()
                    if row and row[0]:
                        start_price = row[0]
                
                # 3. Fall back to crypto_data table (populated by historical data loads)
                if start_price is None:
                    cursor.execute('''
                        SELECT price FROM crypto_data
                        WHERE crypto = ? AND 
                              ABS(julianday(datetime) - julianday(?)) < (30.0/1440.0)
                        ORDER BY ABS(julianday(datetime) - julianday(?))
                        LIMIT 1
                    ''', (crypto, prediction_timestamp.isoformat(), prediction_timestamp.isoformat()))
                    row = cursor.fetchone()
                    if row and row[0]:
                        start_price = row[0]

                conn.close()
                
                if start_price is not None:
                    # predicted_price is direction_prob (P(UP)) — threshold at 0.5
                    predicted_direction = 1 if predicted_price > 0.5 else 0  # 1=up, 0=down
                    actual_direction = 1 if actual_price > start_price else 0
                    direction_predicted = predicted_direction
                    direction_actual = actual_direction
                    direction_correct = 1 if predicted_direction == actual_direction else 0
                else:
                    # Cannot determine direction without a reliable start price — leave as unknown
                    logger.debug(f"No start price found for {crypto} at {prediction_timestamp}; direction marked unknown")
                    direction_predicted = None
                    direction_actual = None
                    direction_correct = 0
                    
            except Exception as e:
                logger.warning(f"Could not calculate direction accuracy: {e}")
                direction_predicted = None
                direction_actual = None
                direction_correct = 0
            
            # Store evaluation
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prediction_evaluations 
                (prediction_id, crypto, prediction_horizon, predicted_price, actual_price,
                 absolute_error, percent_error, squared_error, direction_predicted, 
                 direction_actual, direction_correct, prediction_timestamp, evaluation_timestamp,
                 confidence, target_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_id, crypto, prediction_horizon, predicted_price, actual_price,
                absolute_error, percent_error, squared_error, direction_predicted,
                direction_actual, direction_correct, prediction_timestamp.isoformat(),
                datetime.now().isoformat(), confidence, target_timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return {
                'absolute_error': absolute_error,
                'percent_error': percent_error,
                'squared_error': squared_error,
                'direction_correct': direction_correct
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate prediction: {e}")
            return {}
    
    def batch_evaluate_mature_predictions(self, data_collector) -> Dict:
        """
        Evaluate predictions based on when they were CREATED, not target times.
        
        Logic:
        - 1H evaluation: Compare current price to predictions made ~1 hour ago
        - 1D evaluation: Compare current price to predictions made ~1 day ago  
        - 1W evaluation: Compare current price to predictions made ~1 week ago
        
        This allows 6 evaluations per hour (every 10 minutes) instead of 1 per hour!
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            evaluations = {}
            evaluated_count = 0
            now = datetime.now()
            
            # Define time windows for when predictions should be evaluated
            evaluation_windows = {
                '15m': {
                    'lookback_time': now - timedelta(minutes=15),
                    'window_size': timedelta(minutes=2),  # ±2 minutes for accurate 15m evaluation
                    'description': '15 minutes ago'
                },
                '1h': {
                    'lookback_time': now - timedelta(hours=1),
                    'window_size': timedelta(minutes=3),  # ±3 minutes for accurate 1-hour evaluation
                    'description': '1 hour ago'
                },
                '4h': {
                    'lookback_time': now - timedelta(hours=4),
                    'window_size': timedelta(minutes=15),  # ±15 minutes for accurate 4-hour evaluation
                    'description': '4 hours ago'
                }
            }
            
            for horizon, window_config in evaluation_windows.items():
                lookback_time = window_config['lookback_time']
                window_size = window_config['window_size']
                
                # Find predictions created around the lookback time that haven't been evaluated
                earliest_time = lookback_time - window_size
                latest_time = lookback_time + window_size
                
                # Evaluate each crypto separately to get the closest prediction for each crypto-horizon combination
                horizon_evaluations = 0
                for crypto in config.CRYPTOCURRENCIES:
                    query = '''
                        SELECT p.id, p.crypto, p.prediction_horizon, p.predicted_price, 
                               p.confidence, p.datetime as target_time, p.created_at,
                               ABS(julianday(p.created_at) - julianday(?)) as time_diff_from_target
                        FROM predictions p
                        LEFT JOIN prediction_evaluations pe ON p.id = pe.prediction_id
                        WHERE pe.id IS NULL 
                        AND p.prediction_horizon = ?
                        AND p.crypto = ?
                        AND p.created_at >= ? 
                        AND p.created_at <= ?
                        ORDER BY time_diff_from_target ASC, p.created_at DESC
                        LIMIT 1
                    '''
                    
                    cursor = conn.cursor()
                    cursor.execute(query, (
                        lookback_time.isoformat(),  # Target time for comparison
                        horizon,
                        crypto,
                        earliest_time.isoformat(), 
                        latest_time.isoformat()
                    ))
                    prediction = cursor.fetchone()
                    
                    if prediction:
                        horizon_evaluations += 1
                        try:
                            pred_id, crypto_name, pred_horizon, predicted_price, confidence, target_time_str, created_at_str, time_diff = prediction
                            
                            created_at = pd.to_datetime(created_at_str)
                            target_time = pd.to_datetime(target_time_str)
                            
                            # FIXED: Get actual price at the TARGET TIME, not current price
                            actual_price_at_target = self.get_actual_price_at_time(
                                crypto, target_time, data_collector, pred_horizon
                            )
                            
                            if actual_price_at_target is None:
                                logger.debug(f"Could not get historical price for {crypto} at {target_time}")
                                continue
                            
                            # Evaluate the prediction using actual price at target time
                            evaluation = self.evaluate_prediction(
                                prediction_id=pred_id,
                                crypto=crypto,
                                prediction_horizon=pred_horizon,
                                predicted_price=predicted_price,
                                actual_price=actual_price_at_target,  # FIXED: Use target time price
                                prediction_timestamp=created_at,  # When prediction was made
                                target_timestamp=target_time,  # FIXED: Use actual target time
                                confidence=confidence
                            )
                            
                            if evaluation:
                                key = f"{crypto}_{pred_horizon}"
                                if key not in evaluations:
                                    evaluations[key] = []
                                evaluations[key].append(evaluation)
                                evaluated_count += 1
                                
                                logger.debug(f"✅ Evaluated {crypto} {pred_horizon} prediction: "
                                           f"predicted ${predicted_price:.2f}, actual ${actual_price_at_target:.2f}, "
                                           f"error {evaluation['percent_error']:.2f}%")
                        
                        except Exception as e:
                            logger.error(f"Failed to evaluate individual prediction: {e}")
                            continue
                
                logger.info(f"Found {horizon_evaluations} {horizon} predictions created {window_config['description']} to evaluate")
                logger.debug(f"  Search window: {earliest_time.strftime('%H:%M')} to {latest_time.strftime('%H:%M')} (±{window_size})")
            
            conn.close()
            
            if evaluated_count > 0:
                logger.info(f"🎯 Successfully evaluated {evaluated_count} mature predictions")
                
                # Log breakdown by type
                for key, evals in evaluations.items():
                    logger.info(f"   - {key}: {len(evals)} evaluations")
            else:
                logger.info("No mature predictions found for evaluation")
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            return {}
    
    def get_actual_price_at_time(self, crypto: str, target_time: datetime, data_collector, horizon: str) -> Optional[float]:
        """
        Get actual price at a specific time using stored historical data
        """
        try:
            # First check if we have it stored in actual_prices_at_targets
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            # Look for actual price within a reasonable window (±6 minutes)
            cursor.execute('''
                SELECT actual_price FROM actual_prices_at_targets
                WHERE crypto = ? AND 
                      ABS(julianday(target_timestamp) - julianday(?)) < (6.0/1440.0)
                ORDER BY ABS(julianday(target_timestamp) - julianday(?))
                LIMIT 1
            ''', (crypto, target_time.isoformat(), target_time.isoformat()))
            
            result = cursor.fetchone()
            if result:
                conn.close()
                return result[0]
            
            # Check historical crypto data stored during collection
            cursor.execute('''
                SELECT price FROM crypto_data
                WHERE crypto = ? AND 
                      ABS(julianday(datetime) - julianday(?)) < (30.0/1440.0)
                ORDER BY ABS(julianday(datetime) - julianday(?))
                LIMIT 1
            ''', (crypto, target_time.isoformat(), target_time.isoformat()))
            
            result = cursor.fetchone()
            if result:
                historical_price = result[0]
                # Store this for future reference
                self.store_actual_price_at_target(crypto, target_time, historical_price)
                conn.close()
                return historical_price
            
            conn.close()
            
            # If no stored data, try to fetch from external API with backoff
            now = datetime.now()
            time_diff_hours = abs((now - target_time).total_seconds()) / 3600
            
            # Only fetch if target time is within reasonable range
            if 1 <= time_diff_hours <= 168:  # Between 1 hour and 1 week ago
                try:
                    # Calculate how many days back to fetch
                    days_back = max(1, int(time_diff_hours / 24) + 1)
                    
                    # Fetch recent historical data
                    historical_data = data_collector.get_crypto_data(crypto, days=days_back)
                    
                    if not historical_data.empty:
                        # Find closest price to target time
                        historical_data['time_diff'] = abs(
                            pd.to_datetime(historical_data['datetime']) - target_time
                        ).dt.total_seconds()
                        
                        closest_idx = historical_data['time_diff'].idxmin()
                        closest_price = historical_data.loc[closest_idx, 'price']
                        closest_time_diff = historical_data.loc[closest_idx, 'time_diff']
                        
                        # Only use if within 2 hours
                        if closest_time_diff <= 7200:  # 2 hours in seconds
                            # Store for future reference
                            self.store_actual_price_at_target(crypto, target_time, closest_price)
                            logger.info(f"Found historical price for {crypto} at {target_time}: ${closest_price:.2f}")
                            return closest_price
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch historical price for {crypto}: {e}")
            
            # If target time is very recent (< 1 hour), can use current price
            if time_diff_hours < 1:
                current_price = data_collector.get_crypto_current_price(crypto)
                if current_price:
                    self.store_actual_price_at_target(crypto, target_time, current_price)
                    return current_price
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get actual price for {crypto} at {target_time}: {e}")
            return None
    
    def update_prediction_timeseries(self, crypto: str, timestamp: datetime, actual_price: float,
                                   predictions: Dict[str, float] = None):
        """
        Update the prediction timeseries table with actual and predicted prices
        Only calculate errors for predictions that have reached their target evaluation time
        Only store prediction values for horizons where sufficient time has passed
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            # Check how long the model has been running by looking at earliest data
            cursor.execute('''
                SELECT MIN(timestamp) FROM prediction_timeseries
                WHERE crypto = ?
            ''', (crypto,))
            result = cursor.fetchone()
            
            if result and result[0]:
                earliest_time = pd.to_datetime(result[0])
                time_running = timestamp - earliest_time
            else:
                # First time storing data for this crypto
                time_running = timedelta(0)
            
            # Only store prediction values for horizons where enough time has passed
            # This prevents showing meaningless prediction traces
            pred_15m = predictions.get('15m') if predictions else None  # Always allow 15m predictions
            pred_1h = predictions.get('1h') if predictions else None    # Always allow 1h predictions
            pred_4h = predictions.get('4h') if predictions and time_running >= timedelta(hours=4) else None
            
            # Direction accuracy is tracked via prediction_evaluations (batch_evaluate_mature_predictions).
            # predicted_price_Xm columns store direction_prob (0–1), not prices, so price-error
            # computations are meaningless here — leave all error columns NULL.
            error_15m = None
            error_1h = None
            error_4h = None
            percent_error_15m = None
            percent_error_1h = None
            percent_error_4h = None
            
            # Store data with properly calculated errors (or NULL if not ready for evaluation)
            # Only store prediction values for horizons where sufficient time has passed
            cursor.execute('''
                INSERT OR REPLACE INTO prediction_timeseries
                (crypto, timestamp, actual_price, predicted_price_15m, predicted_price_1h,
                 predicted_price_4h, error_15m, error_1h, error_4h,
                 percent_error_15m, percent_error_1h, percent_error_4h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                crypto, timestamp.isoformat(), actual_price, pred_15m, pred_1h, pred_4h,
                error_15m, error_1h, error_4h, percent_error_15m, percent_error_1h, percent_error_4h
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update prediction timeseries: {e}")
    
    def calculate_accuracy_metrics(self, crypto: str = None, horizon: str = None, 
                                 days_back: int = 30) -> Dict:
        """
        Calculate comprehensive accuracy metrics for a given crypto and horizon
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Build query
            where_conditions = []
            params = []
            
            if crypto:
                where_conditions.append("crypto = ?")
                params.append(crypto)
            
            if horizon:
                where_conditions.append("prediction_horizon = ?")
                params.append(horizon)
            
            where_conditions.append("evaluation_timestamp >= ?")
            params.append((datetime.now() - timedelta(days=days_back)).isoformat())
            
            where_clause = " AND ".join(where_conditions)
            
            # Deduplicate to avoid counting the same prediction multiple times
            query = f'''
                SELECT *, 
                       ROW_NUMBER() OVER (
                           PARTITION BY target_timestamp, prediction_horizon, crypto
                           ORDER BY evaluation_timestamp DESC
                       ) as rn
                FROM prediction_evaluations
                WHERE {where_clause}
            '''
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Keep only the most recent evaluation for each unique target time + horizon + crypto
            df = df[df['rn'] == 1].drop('rn', axis=1)
            
            if df.empty:
                return {}
            
            # Calculate metrics — direction accuracy is the primary signal
            # (predicted_price column now stores direction_prob; price error fields are 0)
            metrics = {
                'total_predictions': len(df),
                'direction_accuracy': df['direction_correct'].mean(),
                'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0.0,
                # Keep legacy price-error keys at 0 so callers that read them don't crash
                'mean_absolute_error': 0.0,
                'median_absolute_error': 0.0,
                'mean_percent_error': 0.0,
                'median_percent_error': 0.0,
                'std_percent_error': 0.0,
                'root_mean_squared_error': 0.0,
                'min_error': 0.0,
                'max_error': 0.0,
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy metrics: {e}")
            return {}
    
    def get_prediction_timeseries_data(self, crypto: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get timeseries data for plotting actual vs predicted prices
        Uses evaluation data from prediction_evaluations table
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Get most recent evaluation for each unique target_timestamp and horizon
            # This prevents duplicate points on the timeseries plot
            query = '''
                SELECT 
                    target_timestamp as timestamp,
                    evaluation_timestamp,
                    crypto,
                    prediction_horizon,
                    predicted_price,
                    actual_price,
                    absolute_error,
                    percent_error,
                    direction_correct,
                    ROW_NUMBER() OVER (
                        PARTITION BY target_timestamp, prediction_horizon 
                        ORDER BY evaluation_timestamp DESC
                    ) as rn
                FROM prediction_evaluations
                WHERE crypto = ? AND evaluation_timestamp >= ?
            '''
            
            params = [crypto, (datetime.now() - timedelta(days=days_back)).isoformat()]
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Keep only the most recent evaluation for each target_timestamp + horizon
            df = df[df['rn'] == 1].drop('rn', axis=1)
            
            if df.empty:
                return pd.DataFrame(), {}
            
            # Convert timestamps to datetime — use errors='coerce' so bad values become NaT
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=False)
            df['evaluation_timestamp'] = pd.to_datetime(df['evaluation_timestamp'], errors='coerce', utc=False)
            df = df.dropna(subset=['timestamp'])
            
            # Find the most recently evaluated prediction for each horizon
            latest_evaluations = {}
            for horizon in df['prediction_horizon'].unique():
                horizon_data = df[df['prediction_horizon'] == horizon]
                if not horizon_data.empty:
                    # Find most recently evaluated prediction for this horizon
                    latest_eval_idx = horizon_data['evaluation_timestamp'].idxmax()
                    latest_eval_row = horizon_data.loc[latest_eval_idx]
                    latest_evaluations[horizon] = {
                        'target_timestamp': latest_eval_row['timestamp'],
                        'percent_error': latest_eval_row['percent_error']
                    }

            # Now pivot using target_timestamp (for proper x-axis) with no duplicates
            df_pivoted = df.pivot_table(
                index='timestamp',  # target_timestamp for proper x-axis timing
                columns='prediction_horizon', 
                values=['predicted_price', 'actual_price', 'absolute_error', 'percent_error'],
                aggfunc='first'  # Should be unique now since we deduplicated
            )
            
            # Flatten column names
            df_pivoted.columns = [f"{col[0]}_{col[1]}" for col in df_pivoted.columns]
            
            # Add a single actual_price column (take from any horizon since it's the same)
            if 'actual_price_15m' in df_pivoted.columns:
                df_pivoted['actual_price'] = df_pivoted['actual_price_15m']
            elif 'actual_price_1h' in df_pivoted.columns:
                df_pivoted['actual_price'] = df_pivoted['actual_price_1h']
            elif 'actual_price_4h' in df_pivoted.columns:
                df_pivoted['actual_price'] = df_pivoted['actual_price_4h']
            
            # Rename columns to match expected format
            rename_map = {}
            for horizon in ['15m', '1h', '4h']:
                if f'predicted_price_{horizon}' in df_pivoted.columns:
                    rename_map[f'predicted_price_{horizon}'] = f'predicted_price_{horizon}'
                if f'absolute_error_{horizon}' in df_pivoted.columns:
                    rename_map[f'absolute_error_{horizon}'] = f'error_{horizon}'
                if f'percent_error_{horizon}' in df_pivoted.columns:
                    rename_map[f'percent_error_{horizon}'] = f'percent_error_{horizon}'
            
            df_pivoted = df_pivoted.rename(columns=rename_map)
            
            return df_pivoted, latest_evaluations
            
        except Exception as e:
            logger.error(f"Failed to get timeseries data: {e}")
            return pd.DataFrame(), {}
    
    def get_error_distribution_data(self, crypto: str = None, horizon: str = None, 
                                  days_back: int = 30) -> pd.DataFrame:
        """
        Get error distribution data for histogram plotting
        Deduplicates to show only the most recent evaluation for each unique target time
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            where_conditions = []
            params = []
            
            if crypto:
                where_conditions.append("crypto = ?")
                params.append(crypto)
            
            if horizon:
                where_conditions.append("prediction_horizon = ?")
                params.append(horizon)
            
            where_conditions.append("evaluation_timestamp >= ?")
            params.append((datetime.now() - timedelta(days=days_back)).isoformat())
            
            where_clause = " AND ".join(where_conditions)
            
            # Deduplicate to avoid counting the same prediction multiple times
            query = f'''
                SELECT crypto, prediction_horizon, absolute_error, percent_error, 
                       direction_correct, evaluation_timestamp, target_timestamp,
                       ROW_NUMBER() OVER (
                           PARTITION BY target_timestamp, prediction_horizon, crypto
                           ORDER BY evaluation_timestamp DESC
                       ) as rn
                FROM prediction_evaluations
                WHERE {where_clause}
            '''
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Keep only the most recent evaluation for each unique target time + horizon + crypto
            df = df[df['rn'] == 1].drop('rn', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get error distribution data: {e}")
            return pd.DataFrame()

    def get_directional_accuracy_data(self, crypto: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get directional accuracy data for timeseries plotting
        Returns data showing how well predictions predict price direction over time
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Get directional accuracy data with deduplication
            query = '''
                SELECT 
                    target_timestamp as timestamp,
                    prediction_horizon,
                    direction_correct,
                    evaluation_timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY target_timestamp, prediction_horizon 
                        ORDER BY evaluation_timestamp DESC
                    ) as rn
                FROM prediction_evaluations
                WHERE crypto = ? AND evaluation_timestamp >= ?
                ORDER BY target_timestamp
            '''
            
            params = [crypto, (datetime.now() - timedelta(days=days_back)).isoformat()]
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Keep only the most recent evaluation for each target_timestamp + horizon
            df = df[df['rn'] == 1].drop('rn', axis=1)
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert timestamps to datetime — use errors='coerce' so bad values become NaT
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=False)
            df = df.dropna(subset=['timestamp'])

            # Drop rows where direction_correct is NULL (start price was unavailable)
            df = df.dropna(subset=['direction_correct'])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get directional accuracy data: {e}")
            return pd.DataFrame()
    
    def generate_accuracy_report(self, crypto: str = None, days_back: int = 30) -> str:
        """
        Generate a comprehensive accuracy report
        """
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"PREDICTION ACCURACY REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"{'='*80}")
        
        # Overall metrics
        if crypto:
            cryptos = [crypto]
            report.append(f"\nCrypto: {crypto.upper()}")
        else:
            cryptos = config.CRYPTOCURRENCIES
            report.append(f"\nAll Cryptocurrencies")
        
        report.append(f"Period: Last {days_back} days")
        report.append("-" * 80)
        
        for crypto in cryptos:
            # Always show crypto header for clarity
            if crypto == 'bitcoin':
                crypto_emoji = "₿"  # Bitcoin symbol
            elif crypto == 'ethereum':
                crypto_emoji = "♦️ "  # Diamond (ETH is often called digital diamond)
            else:
                crypto_emoji = "📈"  # Default for other cryptos
            
            report.append(f"\nCrypto: {crypto_emoji} {crypto.upper()}")
            report.append("-" * 80)
            
            crypto_has_data = False
            for horizon in config.PREDICTION_INTERVALS:
                metrics = self.calculate_accuracy_metrics(crypto, horizon, days_back)
                
                if metrics and metrics.get('total_predictions', 0) > 0:
                    crypto_has_data = True
                    report.append(f"\n  ⏱️  {horizon.upper()} Predictions:")
                    report.append(f"    Total Evaluated: {metrics['total_predictions']}")
                    report.append(f"    Direction Accuracy: {metrics['direction_accuracy']:.2%}")
                    report.append(f"    Avg Confidence:     {metrics['avg_confidence']:.2%}")
                else:
                    report.append(f"\n  ⏱️  {horizon.upper()} Predictions: No data available")
            
            if not crypto_has_data:
                report.append(f"    📍 No evaluation data available for {crypto} in the last {days_back} days")
        
        report.append(f"\n{'='*80}")
        
        return "\n".join(report)
    
    def _format_time_axis(self, ax, days_back=30):
        """Helper function to format x-axis for time series plots"""
        # Format dates without year and without leading zeros
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%-m/%d'))
        
        # Always label each day for major ticks
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        # Add minor ticks every 6 hours (4 ticks between each day: 6am, 12pm, 6pm, 12am)
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        
        # Rotate labels at 45 degrees and align right
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Configure grid with different styles for major and minor ticks
        ax.grid(True, which='major', alpha=0.6, color='gray', linewidth=0.8)  # Darker major grid
        ax.grid(True, which='minor', alpha=0.45, color='lightgray', linewidth=0.4)  # 25% darker minor grid
        
        # Make major ticks darker and more visible
        ax.tick_params(axis='x', which='major', colors='black', width=1.2, length=6)
        ax.tick_params(axis='x', which='minor', colors='gray', width=0.8, length=3)

        # Limit Y-axis to at most ~8 nice ticks to avoid the "MAXTICKS exceeded" warning
        # that occurs when price range spans thousands of dollars at $1 intervals.
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, prune='both'))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    def plot_prediction_timeseries(self, crypto: str, days_back: int = 30, save_path: str = None):
        """
        Create timeseries plots of actual vs predicted prices
        Uses evaluation data from prediction_evaluations table
        """
        try:
            df, latest_evaluations = self.get_prediction_timeseries_data(crypto, days_back)
            
            if df.empty:
                logger.warning(f"No evaluation data available for {crypto}")
                return
            
            # Check which horizons have evaluation data available
            has_15m_data = 'error_15m' in df.columns and df['error_15m'].notna().any()
            has_1h_data  = 'error_1h'  in df.columns and df['error_1h'].notna().any()
            has_4h_data  = 'error_4h'  in df.columns and df['error_4h'].notna().any()
            
            if not (has_15m_data or has_1h_data or has_4h_data):
                logger.warning(f"No prediction evaluations available for {crypto} yet")
                plt.figure(figsize=(15, 8))
                
                plt.subplot(2, 1, 1)
                if 'actual_price' in df.columns:
                    plt.plot(df.index, df['actual_price'], label='Actual Price', linewidth=2, color='black')
                plt.title(f'{crypto.upper()} - Actual Prices (No Evaluations Yet)')
                plt.xlabel('Time')
                plt.ylabel('Price (USD)')
                plt.legend()
                self._format_time_axis(plt.gca(), days_back)
                
                plt.subplot(2, 1, 2)
                plt.text(0.5, 0.5, 'No prediction evaluations available yet.\n\n' +
                         'Evaluations require waiting for target times:\n' +
                         '• 15m predictions: evaluated 15 minutes after prediction\n' +
                         '• 1h predictions: evaluated 1 hour after prediction\n' +
                         '• 4h predictions: evaluated 4 hours after prediction',
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes, fontsize=11,
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                plt.axis('off')
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Plot saved to {save_path}")
                else:
                    plt.show()
                return
            
            plt.figure(figsize=(15, 10))

            # subplot 1: direction prob signal over time (from prediction_timeseries)
            plt.subplot(2, 2, 1)
            horizon_colors = {'15m': '#1f77b4', '1h': '#ff7f0e', '4h': '#2ca02c'}
            plotted_prob = False
            for horizon_key, col in [('15m', 'predicted_price_15m'), ('1h', 'predicted_price_1h'), ('4h', 'predicted_price_4h')]:
                if col in df.columns:
                    prob_data = df[col].dropna()
                    if not prob_data.empty:
                        plt.plot(prob_data.index, prob_data, label=f'P(UP) {horizon_key}',
                                 alpha=0.7, color=horizon_colors[horizon_key])
                        plotted_prob = True
            if plotted_prob:
                plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                plt.title(f'{crypto.upper()} — P(UP) Signal Over Time')
                plt.ylabel('P(UP) direction probability')
                plt.ylim(0, 1)
                plt.legend()
                self._format_time_axis(plt.gca(), days_back)
            else:
                plt.text(0.5, 0.5, 'No signal data yet', ha='center', va='center',
                         transform=plt.gca().transAxes)
                plt.title(f'{crypto.upper()} — P(UP) Signal (Not Available Yet)')

            # subplot 2: rolling direction accuracy per horizon
            plt.subplot(2, 2, 2)
            direction_data = self.get_directional_accuracy_data(crypto, days_back=days_back)
            if not direction_data.empty:
                # Plot directional accuracy for each horizon
                plotted_any_direction = False
                
                for horizon in ['15m', '1h', '4h']:
                    horizon_data = direction_data[direction_data['prediction_horizon'] == horizon]
                    if not horizon_data.empty:
                        # Calculate rolling accuracy (window of 5 predictions)
                        horizon_data = horizon_data.sort_values('timestamp')
                        rolling_accuracy = horizon_data['direction_correct'].rolling(window=min(5, len(horizon_data)), 
                                                                                   min_periods=1).mean()
                        
                        plt.plot(horizon_data['timestamp'], rolling_accuracy, 
                               label=f'{horizon.upper()} Accuracy (rolling)', 
                               alpha=0.8, linewidth=2)
                        
                        plotted_any_direction = True
                
                if plotted_any_direction:
                    plt.title(f'{crypto.upper()} - Directional Accuracy Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Direction Accuracy (1=Correct, 0=Wrong)')
                    plt.ylim(-0.1, 1.1)  # Set y-limits to show 0-1 range clearly
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Add reference lines
                    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
                    plt.axhline(y=0.0, color='red', linestyle='--', alpha=0.3, linewidth=1)
                    
                    # Calculate and display overall accuracy
                    overall_accuracy = direction_data['direction_correct'].mean()
                    plt.text(0.02, 0.98, f'Overall: {overall_accuracy:.1%}', 
                            transform=plt.gca().transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
                    
                    self._format_time_axis(plt.gca(), days_back)
                else:
                    plt.text(0.5, 0.5, 'No directional accuracy data available yet', 
                             horizontalalignment='center', verticalalignment='center',
                             transform=plt.gca().transAxes)
                    plt.title(f'{crypto.upper()} - Directional Accuracy (Not Available Yet)')
            else:
                plt.text(0.5, 0.5, 'No directional accuracy\ndata available yet',
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.title(f'{crypto.upper()} - Directional Accuracy (Not Available Yet)')

            # subplot 3: overall direction accuracy bar chart per horizon
            plt.subplot(2, 2, 3)
            if not direction_data.empty:
                bar_horizons, bar_accs, bar_counts = [], [], []
                for horizon in ['15m', '1h', '4h']:
                    hd = direction_data[direction_data['prediction_horizon'] == horizon]
                    if not hd.empty:
                        bar_horizons.append(horizon.upper())
                        bar_accs.append(hd['direction_correct'].mean() * 100)
                        bar_counts.append(len(hd))
                if bar_horizons:
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(bar_horizons)]
                    bars = plt.bar(bar_horizons, bar_accs, color=colors, alpha=0.7, edgecolor='black')
                    plt.axhline(50, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Random (50%)')
                    for bar, acc, count in zip(bars, bar_accs, bar_counts):
                        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                                 f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)
                    plt.ylim(0, 110)
                    plt.ylabel('Direction Accuracy (%)')
                    plt.legend()
                    plt.title(f'{crypto.upper()} - Accuracy by Horizon')
                else:
                    plt.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                             transform=plt.gca().transAxes)
                    plt.title(f'{crypto.upper()} - Accuracy by Horizon (Not Available Yet)')
            else:
                plt.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                         transform=plt.gca().transAxes)
                plt.title(f'{crypto.upper()} - Accuracy by Horizon (Not Available Yet)')

            # subplot 4: confidence distribution
            plt.subplot(2, 2, 4)
            try:
                conn_conf = sqlite3.connect(config.DATABASE_PATH)
                conf_query = '''
                    SELECT prediction_horizon, confidence FROM prediction_evaluations
                    WHERE crypto = ? AND evaluation_timestamp >= ?
                '''
                conf_df = pd.read_sql_query(conf_query, conn_conf, params=[
                    crypto, (datetime.now() - timedelta(days=days_back)).isoformat()
                ])
                conn_conf.close()
                if not conf_df.empty:
                    for horizon, color in [('15m', '#1f77b4'), ('1h', '#ff7f0e'), ('4h', '#2ca02c')]:
                        hd = conf_df[conf_df['prediction_horizon'] == horizon]['confidence']
                        if not hd.empty:
                            plt.hist(hd, bins=20, alpha=0.5, label=horizon.upper(), color=color)
                    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
                    plt.xlabel('Model Confidence')
                    plt.ylabel('Frequency')
                    plt.title(f'{crypto.upper()} - Confidence Distribution')
                    plt.legend()
                else:
                    plt.text(0.5, 0.5, 'No confidence data yet', ha='center', va='center',
                             transform=plt.gca().transAxes)
                    plt.title(f'{crypto.upper()} - Confidence Distribution (Not Available Yet)')
            except Exception:
                plt.text(0.5, 0.5, 'Confidence data unavailable', ha='center', va='center',
                         transform=plt.gca().transAxes)
                plt.title(f'{crypto.upper()} - Confidence Distribution')

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create timeseries plot: {e}")
    
    def plot_error_histograms(self, crypto: str = None, days_back: int = 30, save_path: str = None):
        """
        Plot direction accuracy broken down by confidence bucket (calibration chart).
        Higher-confidence predictions should be correct more often — this shows whether that holds.
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            where = "WHERE evaluation_timestamp >= ?"
            params = [(datetime.now() - timedelta(days=days_back)).isoformat()]
            if crypto:
                where += " AND crypto = ?"
                params.append(crypto)

            df = pd.read_sql_query(
                f"SELECT crypto, prediction_horizon, confidence, direction_correct FROM prediction_evaluations {where}",
                conn, params=params
            )
            conn.close()

            if df.empty:
                logger.warning("No evaluation data available for calibration chart")
                return

            horizons = [h for h in ['15m', '1h', '4h'] if h in df['prediction_horizon'].unique()]
            n_horizons = len(horizons)
            if n_horizons == 0:
                return

            plt.figure(figsize=(6 * n_horizons, 5))

            conf_bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
            bin_labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-80%', '>80%']

            for i, horizon in enumerate(horizons):
                plt.subplot(1, n_horizons, i + 1)
                hd = df[df['prediction_horizon'] == horizon].copy()
                if hd.empty:
                    plt.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                             transform=plt.gca().transAxes)
                    plt.title(f'{horizon.upper()} — Calibration')
                    continue

                hd['conf_bin'] = pd.cut(hd['confidence'], bins=conf_bins, labels=bin_labels, right=False)
                grouped = hd.groupby('conf_bin', observed=True)['direction_correct'].agg(['mean', 'count'])
                grouped = grouped.dropna()

                if grouped.empty:
                    plt.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                             transform=plt.gca().transAxes)
                    plt.title(f'{horizon.upper()} — Calibration')
                    continue

                bars = plt.bar(range(len(grouped)), grouped['mean'] * 100, alpha=0.7,
                               edgecolor='black', color='#1f77b4')
                plt.xticks(range(len(grouped)), grouped.index, rotation=30, ha='right')
                plt.axhline(50, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='Random (50%)')
                plt.ylim(0, 110)
                plt.ylabel('Direction Accuracy (%)')
                plt.xlabel('Confidence Bucket')
                plt.title(f'{horizon.upper()} — Accuracy by Confidence')
                plt.legend()

                for bar, (_, row) in zip(bars, grouped.iterrows()):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                             f'{row["mean"]*100:.0f}%\n(n={int(row["count"])})',
                             ha='center', va='bottom', fontsize=8)

                plt.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Calibration chart saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Failed to create calibration chart: {e}")


def main():
    """Test the prediction accuracy tracker"""
    tracker = PredictionAccuracyTracker()
    
    # Generate sample accuracy report
    report = tracker.generate_accuracy_report(days_back=7)
    print(report)

if __name__ == "__main__":
    main() 