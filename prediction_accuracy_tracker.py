"""
Prediction Accuracy Tracker for Crypto Price Prediction Bot
Tracks and evaluates prediction performance with comprehensive metrics and storage
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                predicted_price_1h REAL,
                predicted_price_1d REAL,
                predicted_price_1w REAL,
                error_1h REAL,
                error_1d REAL,
                error_1w REAL,
                percent_error_1h REAL,
                percent_error_1d REAL,
                percent_error_1w REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(crypto, timestamp)
            )
        ''')
        
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
            
            # Direction accuracy (simplified - compare predicted vs actual change from prediction time)
            # For this we need the price at prediction time
            direction_predicted = 1 if predicted_price > 0 else 0  # Simplified
            direction_actual = 1 if actual_price > 0 else 0  # Will be properly calculated
            direction_correct = 1 if direction_predicted == direction_actual else 0
            
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
        Evaluate all predictions that have reached their target times
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Get predictions that haven't been evaluated yet
            query = '''
                SELECT p.id, p.crypto, p.prediction_horizon, p.predicted_price, 
                       p.confidence, p.datetime as prediction_time, p.created_at
                FROM predictions p
                LEFT JOIN prediction_evaluations pe ON p.id = pe.prediction_id
                WHERE pe.id IS NULL
                ORDER BY p.created_at DESC
            '''
            
            predictions_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if predictions_df.empty:
                logger.info("No new predictions to evaluate")
                return {}
            
            evaluations = {}
            evaluated_count = 0
            
            for _, pred in predictions_df.iterrows():
                try:
                    prediction_time = pd.to_datetime(pred['prediction_time'])
                    created_at = pd.to_datetime(pred['created_at'])
                    horizon = pred['prediction_horizon']
                    crypto = pred['crypto']
                    
                    # Calculate target time based on horizon
                    if horizon == '1h':
                        target_time = prediction_time + timedelta(hours=1)
                        min_wait = timedelta(hours=1)
                    elif horizon == '1d':
                        target_time = prediction_time + timedelta(days=1)
                        min_wait = timedelta(days=1)
                    elif horizon == '1w':
                        target_time = prediction_time + timedelta(weeks=1)
                        min_wait = timedelta(weeks=1)
                    else:
                        continue
                    
                    # Check if enough time has passed for evaluation
                    now = datetime.now()
                    if now < created_at + min_wait:
                        continue
                    
                    # Get actual price at target time (or closest available)
                    actual_price = self.get_actual_price_at_time(crypto, target_time, data_collector)
                    
                    if actual_price is None:
                        continue
                    
                    # Evaluate the prediction
                    evaluation = self.evaluate_prediction(
                        prediction_id=pred['id'],
                        crypto=crypto,
                        prediction_horizon=horizon,
                        predicted_price=pred['predicted_price'],
                        actual_price=actual_price,
                        prediction_timestamp=prediction_time,
                        target_timestamp=target_time,
                        confidence=pred['confidence']
                    )
                    
                    if evaluation:
                        key = f"{crypto}_{horizon}"
                        if key not in evaluations:
                            evaluations[key] = []
                        evaluations[key].append(evaluation)
                        evaluated_count += 1
                
                except Exception as e:
                    logger.error(f"Failed to evaluate prediction {pred['id']}: {e}")
                    continue
            
            logger.info(f"Evaluated {evaluated_count} predictions")
            return evaluations
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            return {}
    
    def get_actual_price_at_time(self, crypto: str, target_time: datetime, data_collector) -> Optional[float]:
        """
        Get actual price at a specific time, either from database or by fetching
        """
        try:
            # First check if we have it stored
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            # Look for actual price within a reasonable window
            cursor.execute('''
                SELECT actual_price FROM actual_prices_at_targets
                WHERE crypto = ? AND 
                      ABS(julianday(target_timestamp) - julianday(?)) < 0.1
                ORDER BY ABS(julianday(target_timestamp) - julianday(?))
                LIMIT 1
            ''', (crypto, target_time.isoformat(), target_time.isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0]
            
            # If not stored, try to get current price (if target time is recent)
            now = datetime.now()
            if abs((now - target_time).total_seconds()) < 3600:  # Within 1 hour
                current_price = data_collector.get_crypto_current_price(crypto)
                if current_price:
                    # Store for future reference
                    self.store_actual_price_at_target(crypto, target_time, current_price)
                    return current_price
            
            # For historical prices, we'd need to implement historical data fetching
            # For now, return None to indicate we can't evaluate this prediction yet
            return None
            
        except Exception as e:
            logger.error(f"Failed to get actual price for {crypto} at {target_time}: {e}")
            return None
    
    def update_prediction_timeseries(self, crypto: str, timestamp: datetime, actual_price: float,
                                   predictions: Dict[str, float] = None):
        """
        Update the prediction timeseries table with actual and predicted prices
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            # Prepare values
            pred_1h = predictions.get('1h') if predictions else None
            pred_1d = predictions.get('1d') if predictions else None
            pred_1w = predictions.get('1w') if predictions else None
            
            # Calculate errors if we have predictions
            error_1h = abs(pred_1h - actual_price) if pred_1h else None
            error_1d = abs(pred_1d - actual_price) if pred_1d else None
            error_1w = abs(pred_1w - actual_price) if pred_1w else None
            
            percent_error_1h = (error_1h / actual_price * 100) if error_1h and actual_price != 0 else None
            percent_error_1d = (error_1d / actual_price * 100) if error_1d and actual_price != 0 else None
            percent_error_1w = (error_1w / actual_price * 100) if error_1w and actual_price != 0 else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO prediction_timeseries
                (crypto, timestamp, actual_price, predicted_price_1h, predicted_price_1d, 
                 predicted_price_1w, error_1h, error_1d, error_1w, 
                 percent_error_1h, percent_error_1d, percent_error_1w)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                crypto, timestamp.isoformat(), actual_price, pred_1h, pred_1d, pred_1w,
                error_1h, error_1d, error_1w, percent_error_1h, percent_error_1d, percent_error_1w
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
            
            query = f'''
                SELECT * FROM prediction_evaluations
                WHERE {where_clause}
                ORDER BY evaluation_timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
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
                'root_mean_squared_error': np.sqrt(df['squared_error'].mean()),
                'direction_accuracy': df['direction_correct'].mean(),
                'min_error': df['absolute_error'].min(),
                'max_error': df['absolute_error'].max(),
                'error_percentiles': {
                    '25th': df['percent_error'].quantile(0.25),
                    '75th': df['percent_error'].quantile(0.75),
                    '90th': df['percent_error'].quantile(0.90),
                    '95th': df['percent_error'].quantile(0.95)
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy metrics: {e}")
            return {}
    
    def get_prediction_timeseries_data(self, crypto: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get timeseries data for plotting actual vs predicted prices
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            query = '''
                SELECT * FROM prediction_timeseries
                WHERE crypto = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            params = [crypto, (datetime.now() - timedelta(days=days_back)).isoformat()]
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get timeseries data: {e}")
            return pd.DataFrame()
    
    def get_error_distribution_data(self, crypto: str = None, horizon: str = None, 
                                  days_back: int = 30) -> pd.DataFrame:
        """
        Get error distribution data for histogram plotting
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
            
            query = f'''
                SELECT crypto, prediction_horizon, absolute_error, percent_error, 
                       direction_correct, evaluation_timestamp
                FROM prediction_evaluations
                WHERE {where_clause}
                ORDER BY evaluation_timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get error distribution data: {e}")
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
            if len(cryptos) > 1:
                report.append(f"\n📊 {crypto.upper()}")
                report.append("-" * 40)
            
            for horizon in config.PREDICTION_INTERVALS:
                metrics = self.calculate_accuracy_metrics(crypto, horizon, days_back)
                
                if metrics:
                    report.append(f"\n  ⏱️  {horizon.upper()} Predictions:")
                    report.append(f"    Total Predictions: {metrics['total_predictions']}")
                    report.append(f"    Mean Absolute Error: ${metrics['mean_absolute_error']:.2f}")
                    report.append(f"    Mean Percent Error: {metrics['mean_percent_error']:.2f}%")
                    report.append(f"    Direction Accuracy: {metrics['direction_accuracy']:.2%}")
                    report.append(f"    RMSE: ${metrics['root_mean_squared_error']:.2f}")
                    report.append(f"    Error Range: ${metrics['min_error']:.2f} - ${metrics['max_error']:.2f}")
                else:
                    report.append(f"\n  ⏱️  {horizon.upper()} Predictions: No data available")
        
        report.append(f"\n{'='*80}")
        
        return "\n".join(report)
    
    def plot_prediction_timeseries(self, crypto: str, days_back: int = 30, save_path: str = None):
        """
        Create timeseries plots of actual vs predicted prices
        """
        try:
            df = self.get_prediction_timeseries_data(crypto, days_back)
            
            if df.empty:
                logger.warning(f"No timeseries data available for {crypto}")
                return
            
            plt.figure(figsize=(15, 10))
            
            # Plot actual vs predicted prices
            plt.subplot(2, 2, 1)
            plt.plot(df.index, df['actual_price'], label='Actual Price', linewidth=2, color='black')
            if 'predicted_price_1h' in df.columns:
                plt.plot(df.index, df['predicted_price_1h'], label='1h Prediction', alpha=0.7)
            if 'predicted_price_1d' in df.columns:
                plt.plot(df.index, df['predicted_price_1d'], label='1d Prediction', alpha=0.7)
            if 'predicted_price_1w' in df.columns:
                plt.plot(df.index, df['predicted_price_1w'], label='1w Prediction', alpha=0.7)
            
            plt.title(f'{crypto.upper()} - Actual vs Predicted Prices')
            plt.xlabel('Time')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot absolute errors
            plt.subplot(2, 2, 2)
            if 'error_1h' in df.columns:
                plt.plot(df.index, df['error_1h'], label='1h Error', alpha=0.7)
            if 'error_1d' in df.columns:
                plt.plot(df.index, df['error_1d'], label='1d Error', alpha=0.7)
            if 'error_1w' in df.columns:
                plt.plot(df.index, df['error_1w'], label='1w Error', alpha=0.7)
            
            plt.title(f'{crypto.upper()} - Absolute Errors Over Time')
            plt.xlabel('Time')
            plt.ylabel('Absolute Error (USD)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot percent errors
            plt.subplot(2, 2, 3)
            if 'percent_error_1h' in df.columns:
                plt.plot(df.index, df['percent_error_1h'], label='1h Error %', alpha=0.7)
            if 'percent_error_1d' in df.columns:
                plt.plot(df.index, df['percent_error_1d'], label='1d Error %', alpha=0.7)
            if 'percent_error_1w' in df.columns:
                plt.plot(df.index, df['percent_error_1w'], label='1w Error %', alpha=0.7)
            
            plt.title(f'{crypto.upper()} - Percent Errors Over Time')
            plt.xlabel('Time')
            plt.ylabel('Percent Error (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot error distribution (histogram)
            plt.subplot(2, 2, 4)
            error_data = self.get_error_distribution_data(crypto, days_back=days_back)
            if not error_data.empty:
                plt.hist(error_data['percent_error'], bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'{crypto.upper()} - Error Distribution')
                plt.xlabel('Percent Error (%)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
            
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
                plt.hist(horizon_data['percent_error'], bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'{horizon.upper()} - Percent Error Distribution')
                plt.xlabel('Percent Error (%)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # Absolute error histogram
                plt.subplot(2, n_horizons, i + 1 + n_horizons)
                plt.hist(horizon_data['absolute_error'], bins=30, alpha=0.7, edgecolor='black')
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