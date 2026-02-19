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
            # Calculate error metrics
            absolute_error = abs(predicted_price - actual_price)
            percent_error = (absolute_error / actual_price) * 100 if actual_price != 0 else 0
            squared_error = (predicted_price - actual_price) ** 2
            
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
                    # Calculate direction: did the price go up or down from start_price?
                    predicted_direction = 1 if predicted_price > start_price else 0  # 1=up, 0=down
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
            
            # Only calculate errors for predictions made at the appropriate time in the past
            error_15m = None
            error_1h = None
            error_4h = None
            percent_error_15m = None
            percent_error_1h = None
            percent_error_4h = None
            
            # Get mature predictions from the appropriate time windows
            now = timestamp
            
            # Check for 15m predictions made ~15 minutes ago that can now be evaluated
            if pred_15m is not None:
                fifteen_min_ago = now - timedelta(minutes=15)
                cursor.execute('''
                    SELECT predicted_price_15m FROM prediction_timeseries
                    WHERE crypto = ? AND 
                          ABS(julianday(timestamp) - julianday(?)) < (3.0/1440.0)  -- within 3 minutes
                    ORDER BY ABS(julianday(timestamp) - julianday(?))
                    LIMIT 1
                ''', (crypto, fifteen_min_ago.isoformat(), fifteen_min_ago.isoformat()))
                result = cursor.fetchone()
                if result and result[0] is not None:
                    old_pred_15m = result[0]
                    error_15m = abs(old_pred_15m - actual_price)
                    percent_error_15m = (error_15m / actual_price * 100) if actual_price != 0 else None
            
            # Check for 1h predictions made ~1 hour ago that can now be evaluated
            if pred_1h is not None:
                hour_ago = now - timedelta(hours=1)
                cursor.execute('''
                    SELECT predicted_price_1h FROM prediction_timeseries
                    WHERE crypto = ? AND 
                          ABS(julianday(timestamp) - julianday(?)) < (10.0/1440.0)  -- within 10 minutes
                    ORDER BY ABS(julianday(timestamp) - julianday(?))
                    LIMIT 1
                ''', (crypto, hour_ago.isoformat(), hour_ago.isoformat()))
                result = cursor.fetchone()
                if result and result[0] is not None:
                    old_pred_1h = result[0]
                    error_1h = abs(old_pred_1h - actual_price)
                    percent_error_1h = (error_1h / actual_price * 100) if actual_price != 0 else None
            
            # Check for 4h predictions made ~4 hours ago that can now be evaluated
            if pred_4h is not None:
                four_hours_ago = now - timedelta(hours=4)
                cursor.execute('''
                    SELECT predicted_price_4h FROM prediction_timeseries
                    WHERE crypto = ? AND 
                          ABS(julianday(timestamp) - julianday(?)) < (15.0/1440.0)  -- within 15 minutes
                    ORDER BY ABS(julianday(timestamp) - julianday(?))
                    LIMIT 1
                ''', (crypto, four_hours_ago.isoformat(), four_hours_ago.isoformat()))
                result = cursor.fetchone()
                if result and result[0] is not None:
                    old_pred_4h = result[0]
                    error_4h = abs(old_pred_4h - actual_price)
                    percent_error_4h = (error_4h / actual_price * 100) if actual_price != 0 else None
            
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
            
            # Calculate metrics
            metrics = {
                'total_predictions': len(df),
                'mean_absolute_error': df['absolute_error'].mean(),
                'median_absolute_error': df['absolute_error'].median(),
                'mean_percent_error': df['percent_error'].mean(),
                'median_percent_error': df['percent_error'].median(),
                'std_percent_error': df['percent_error'].std(),
                'root_mean_squared_error': np.sqrt((df['absolute_error'] ** 2).mean()),
                'direction_accuracy': df['direction_correct'].mean(),
                'min_error': df['absolute_error'].min(),
                'max_error': df['absolute_error'].max(),
                'percent_error_percentiles': {
                    '1st': df['percent_error'].quantile(0.01),
                    '10th': df['percent_error'].quantile(0.10),
                    '25th': df['percent_error'].quantile(0.25),
                    '50th': df['percent_error'].quantile(0.50),  # median
                    '75th': df['percent_error'].quantile(0.75),
                    '90th': df['percent_error'].quantile(0.90),
                    '99th': df['percent_error'].quantile(0.99)
                }
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
                    report.append(f"    Total Predictions: {metrics['total_predictions']}")
                    report.append(f"    Mean Absolute Error: ${metrics['mean_absolute_error']:.2f}")
                    report.append(f"    Mean Percent Error: {metrics['mean_percent_error']:.2f}%")
                    report.append(f"    Direction Accuracy: {metrics['direction_accuracy']:.2%}")
                    report.append(f"    RMSE: ${metrics['root_mean_squared_error']:.2f}")
                    report.append(f"    Error Range: ${metrics['min_error']:.2f} - ${metrics['max_error']:.2f}")
                    
                    # Add percentile information
                    percentiles = metrics['percent_error_percentiles']
                    report.append(f"    Percent Error Percentiles:")
                    report.append(f"      1st: {percentiles['1st']:.2f}%   10th: {percentiles['10th']:.2f}%   25th: {percentiles['25th']:.2f}%")
                    report.append(f"     50th: {percentiles['50th']:.2f}%   75th: {percentiles['75th']:.2f}%   90th: {percentiles['90th']:.2f}%   99th: {percentiles['99th']:.2f}%")
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
            
            # Use matplotlib's default color cycle for consistent colors
            
            # Plot actual vs predicted prices
            plt.subplot(2, 2, 1)
            if 'actual_price' in df.columns:
                plt.plot(df.index, df['actual_price'], label='Actual Price', linewidth=2, color='black')
            
            if has_15m_data and 'predicted_price_15m' in df.columns:
                plt.plot(df.index, df['predicted_price_15m'], label='15m Prediction', alpha=0.8, color='#1f77b4')  # blue
            if has_1h_data and 'predicted_price_1h' in df.columns:
                plt.plot(df.index, df['predicted_price_1h'], label='1h Prediction', alpha=0.8, color='#ff7f0e')   # orange
            if has_4h_data and 'predicted_price_4h' in df.columns:
                plt.plot(df.index, df['predicted_price_4h'], label='4h Prediction', alpha=0.8, color='#2ca02c')   # green
            
            plt.title(f'{crypto.upper()} - Actual vs Predicted Prices')
            plt.xlabel('Time')
            plt.ylabel('Price (USD)')
            plt.legend()
            self._format_time_axis(plt.gca(), days_back)
            
            # Plot absolute errors
            plt.subplot(2, 2, 2)
            plotted_any_errors = False
            if has_15m_data and 'error_15m' in df.columns:
                error_data = df['error_15m'].dropna()
                if not error_data.empty:
                    plt.plot(error_data.index, error_data, label='15m Error', alpha=0.8, color='#1f77b4')
                    plotted_any_errors = True
            if has_1h_data and 'error_1h' in df.columns:
                error_data = df['error_1h'].dropna()
                if not error_data.empty:
                    plt.plot(error_data.index, error_data, label='1h Error', alpha=0.8, color='#ff7f0e')
                    plotted_any_errors = True
            if has_4h_data and 'error_4h' in df.columns:
                error_data = df['error_4h'].dropna()
                if not error_data.empty:
                    plt.plot(error_data.index, error_data, label='4h Error', alpha=0.8, color='#2ca02c')
                    plotted_any_errors = True
            
            if plotted_any_errors:
                plt.title(f'{crypto.upper()} - Absolute Errors Over Time')
                plt.xlabel('Time')
                plt.ylabel('Absolute Error (USD)')
                plt.legend()
                self._format_time_axis(plt.gca(), days_back)
            else:
                plt.text(0.5, 0.5, 'No error data available yet', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.title(f'{crypto.upper()} - Absolute Errors (Not Available Yet)')
            
            # Plot percent errors
            plt.subplot(2, 2, 3)
            plotted_any_percent_errors = False
            if has_15m_data and 'percent_error_15m' in df.columns:
                error_data = df['percent_error_15m'].dropna()
                if not error_data.empty:
                    plt.plot(error_data.index, error_data, label='15m Error %', alpha=0.8, color='#1f77b4')
                    plotted_any_percent_errors = True
                    
                    if '15m' in latest_evaluations:
                        latest_info = latest_evaluations['15m']
                        latest_val = latest_info['percent_error']
                        latest_target_time = latest_info['target_timestamp']
                        plt.annotate(f'{latest_val:.2f}%',
                                   xy=(latest_target_time, latest_val),
                                   xytext=(10, 10), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                   fontsize=9, fontweight='bold')
            if has_1h_data and 'percent_error_1h' in df.columns:
                error_data = df['percent_error_1h'].dropna()
                if not error_data.empty:
                    plt.plot(error_data.index, error_data, label='1h Error %', alpha=0.8, color='#ff7f0e')
                    plotted_any_percent_errors = True
                    
                    # Use pre-computed latest evaluation data
                    if '1h' in latest_evaluations:
                        latest_info = latest_evaluations['1h']
                        latest_val = latest_info['percent_error']
                        latest_target_time = latest_info['target_timestamp']
                        plt.annotate(f'{latest_val:.2f}%', 
                                   xy=(latest_target_time, latest_val),
                                   xytext=(10, 10), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                   fontsize=9, fontweight='bold')
            if has_4h_data and 'percent_error_4h' in df.columns:
                error_data = df['percent_error_4h'].dropna()
                if not error_data.empty:
                    plt.plot(error_data.index, error_data, label='4h Error %', alpha=0.8, color='#2ca02c')
                    plotted_any_percent_errors = True
            
            if plotted_any_percent_errors:
                plt.title(f'{crypto.upper()} - Percent Errors Over Time')
                plt.xlabel('Time')
                plt.ylabel('Percent Error (%)')
                plt.legend()
                self._format_time_axis(plt.gca(), days_back)
            else:
                plt.text(0.5, 0.5, 'No percent error data available yet', 
                         horizontalalignment='center', verticalalignment='center',
                         transform=plt.gca().transAxes)
                plt.title(f'{crypto.upper()} - Percent Errors (Not Available Yet)')
            
            # Plot directional accuracy over time
            plt.subplot(2, 2, 4)
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
        Create histogram plots of prediction errors
        """
        try:
            df = self.get_error_distribution_data(crypto, days_back=days_back)
            
            if df.empty:
                logger.warning("No error data available for histogram")
                return
            
            plt.figure(figsize=(15, 10))
            
            # Group by horizon
            horizons = df['prediction_horizon'].unique()
            n_horizons = len(horizons)
            
            for i, horizon in enumerate(horizons):
                horizon_data = df[df['prediction_horizon'] == horizon]
                
                # Percent error histogram
                plt.subplot(2, n_horizons, i + 1)
                
                if not horizon_data.empty:
                    # Calculate appropriate bin edges for percent error
                    percent_errors = horizon_data['percent_error']
                    min_err = percent_errors.min()
                    max_err = percent_errors.max()
                    
                    # Create nice round bin edges
                    range_err = max_err - min_err
                    if range_err < 0.1:
                        bin_width = 0.01  # Very precise for small ranges
                    elif range_err < 1:
                        bin_width = 0.1
                    elif range_err < 5:
                        bin_width = 0.25
                    elif range_err < 10:
                        bin_width = 0.5
                    else:
                        bin_width = 1.0
                    
                    # Round bin edges to nice numbers
                    start_bin = np.floor(min_err / bin_width) * bin_width
                    end_bin = np.ceil(max_err / bin_width) * bin_width
                    bins_percent = np.arange(start_bin, end_bin + bin_width, bin_width)
                    
                    n, bins, patches = plt.hist(percent_errors, bins=bins_percent, alpha=0.7, 
                                              edgecolor='black', align='mid')
                    
                    # Add percentile lines
                    q25 = percent_errors.quantile(0.25)
                    q50 = percent_errors.quantile(0.50)  # median
                    q75 = percent_errors.quantile(0.75)
                    
                    plt.axvline(q25, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
                    plt.axvline(q50, color='red', linestyle='--', alpha=0.8, linewidth=2)
                    plt.axvline(q75, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
                    
                    plt.text(0.98, 0.95, f'Mean: {percent_errors.mean():.2f}%\nStd: {percent_errors.std():.2f}%\nN: {len(percent_errors)}',
                            transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                plt.title(f'{horizon.upper()} - Percent Error Distribution')
                plt.xlabel('Percent Error (%)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Absolute error histogram
                plt.subplot(2, n_horizons, i + 1 + n_horizons)
                
                if not horizon_data.empty:
                    # Calculate appropriate bin edges for absolute error
                    abs_errors = horizon_data['absolute_error']
                    min_err = abs_errors.min()
                    max_err = abs_errors.max()
                    
                    # Create nice round bin edges for dollar amounts
                    range_err = max_err - min_err
                    if range_err < 10:
                        bin_width = 1  # $1 bins for small ranges
                    elif range_err < 100:
                        bin_width = 5  # $5 bins
                    elif range_err < 500:
                        bin_width = 25  # $25 bins
                    elif range_err < 1000:
                        bin_width = 50  # $50 bins
                    else:
                        bin_width = 100  # $100 bins for large ranges
                    
                    # Round bin edges to nice numbers
                    start_bin = np.floor(min_err / bin_width) * bin_width
                    end_bin = np.ceil(max_err / bin_width) * bin_width
                    bins_abs = np.arange(start_bin, end_bin + bin_width, bin_width)
                    
                    n, bins, patches = plt.hist(abs_errors, bins=bins_abs, alpha=0.7, 
                                              edgecolor='black', align='mid')
                    
                    # Add percentile lines
                    q25 = abs_errors.quantile(0.25)
                    q50 = abs_errors.quantile(0.50)  # median
                    q75 = abs_errors.quantile(0.75)
                    
                    plt.axvline(q25, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
                    plt.axvline(q50, color='red', linestyle='--', alpha=0.8, linewidth=2)
                    plt.axvline(q75, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
                    
                    plt.text(0.98, 0.95, f'Mean: ${abs_errors.mean():.0f}\nStd: ${abs_errors.std():.0f}\nN: {len(abs_errors)}',
                            transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                plt.title(f'{horizon.upper()} - Absolute Error Distribution')
                plt.xlabel('Absolute Error (USD)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Histogram saved to {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create error histograms: {e}")


def main():
    """Test the prediction accuracy tracker"""
    tracker = PredictionAccuracyTracker()
    
    # Generate sample accuracy report
    report = tracker.generate_accuracy_report(days_back=7)
    print(report)

if __name__ == "__main__":
    main() 