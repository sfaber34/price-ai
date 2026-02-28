"""
Main Crypto Price Prediction Bot
Orchestrates data collection, feature engineering, model training, and predictions
"""
import sqlite3
import pandas as pd
import schedule
import time
import logging
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import json
import os


def _utcnow() -> datetime:
    """Return current UTC time as timezone-naive datetime (matches Binance timestamps)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

from data_collector import DataCollector, initialize_database
from feature_engineering import FeatureEngineer
from ml_predictor import EnsemblePredictionEngine
from prediction_accuracy_tracker import PredictionAccuracyTracker
from training_pipeline import (
    fetch_data_for_crypto,
    prepare_features_for_crypto,
)
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoPredictionBot:
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.prediction_engine = EnsemblePredictionEngine()
        self.accuracy_tracker = PredictionAccuracyTracker()
        self.is_trained = False
        self.last_training_time = None
        self._bar_cache: Dict[str, pd.DataFrame] = {}  # in-memory 15m bar cache

        # Initialize database
        initialize_database()
        
        logger.info("Crypto Prediction Bot initialized with accuracy tracking")
    
    def collect_all_data(self, days: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect 15m and 1m bar data for training/prediction.

        15m bars are cached in memory: the first call does a full fetch, subsequent
        calls only fetch the last day and append — cutting ~10s down to <1s.

        Returns {crypto: {'15m': df, '1m': df}}.
        """
        logger.info(f"Collecting data for past {days} days...")

        all_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        cutoff = _utcnow() - timedelta(days=days)

        for crypto in config.CRYPTOCURRENCIES:
            crypto_data: Dict[str, pd.DataFrame] = {
                '15m': pd.DataFrame(), '1m': pd.DataFrame(),
            }

            # 15m bars with caching
            try:
                cached = self._bar_cache.get(crypto)
                if cached is not None and not cached.empty:
                    stable_cache = cached.iloc[:-1]
                    fresh = self.data_collector.get_crypto_data(crypto, days=1)
                    if not fresh.empty:
                        combined = pd.concat([stable_cache, fresh], ignore_index=True)
                        combined = combined.drop_duplicates(subset='datetime', keep='last')
                        combined = combined.sort_values('datetime').reset_index(drop=True)
                        combined = combined[combined['datetime'] >= cutoff].reset_index(drop=True)
                        self._bar_cache[crypto] = combined
                        crypto_data['15m'] = combined
                        logger.info(f"Updated {crypto}: {len(combined)} bars (incremental)")
                    else:
                        logger.warning(f"Incremental fetch failed for {crypto}, using cached data")
                        crypto_data['15m'] = stable_cache[stable_cache['datetime'] >= cutoff].reset_index(drop=True)
                else:
                    df = self.data_collector.get_crypto_data(crypto, days=days)
                    if not df.empty:
                        self._bar_cache[crypto] = df
                        crypto_data['15m'] = df
                        logger.info(f"Collected {len(df)} records for {crypto}")
                    else:
                        logger.warning(f"No data collected for {crypto}")
            except Exception as e:
                logger.error(f"Failed to collect data for {crypto}: {e}")
                if crypto in self._bar_cache and not self._bar_cache[crypto].empty:
                    crypto_data['15m'] = self._bar_cache[crypto].iloc[:-1]

            # 1m bars (only 1 day needed for live prediction — intrabar features
            # are per-bar aggregations, only the most recent bar matters)
            try:
                df_1m = self.data_collector.get_crypto_data_1m(crypto, days=1)
                if not df_1m.empty:
                    crypto_data['1m'] = df_1m
                    logger.info(f"Collected {len(df_1m)} 1m bars for {crypto}")
            except Exception as e:
                logger.warning(f"1m data fetch failed for {crypto}: {e}")

            all_data[crypto] = crypto_data

        return all_data

    def prepare_features_for_all_cryptos(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Prepare features using training_pipeline.prepare_features_for_crypto()
        — same function evaluate_calibration.py and train_optimal_models.py use.
        """
        logger.info("Preparing features for all cryptocurrencies...")

        prepared_data = {}
        for crypto, data in raw_data.items():
            df_15m = data.get('15m', pd.DataFrame())
            if df_15m.empty:
                continue
            try:
                features_df = prepare_features_for_crypto(
                    self.feature_engineer, df_15m, data.get('1m'),
                )
                prepared_data[crypto] = features_df
                logger.info(f"{crypto}: {features_df.shape[1]} features, {len(features_df)} bars")
            except Exception as e:
                logger.error(f"Feature preparation failed for {crypto}: {e}")

        return prepared_data
    
    def load_production_models(self) -> bool:
        """
        Load pre-trained production models from train_optimal_models.py
        """
        try:
            # Check if production models exist
            if not os.path.exists('models/production_models.json'):
                logger.warning("No production models found. Run train_optimal_models.py first to train optimal models.")
                return False
            
            # Load model metadata
            with open('models/production_models.json', 'r') as f:
                production_models = json.load(f)
            
            logger.info("Loading pre-trained production models...")
            
            # Load each model
            models_loaded = 0
            for crypto in config.CRYPTOCURRENCIES:
                if crypto in production_models:
                    for horizon in config.PREDICTION_INTERVALS:
                        if horizon in production_models[crypto]:
                            model_info = production_models[crypto][horizon]
                            model_path = model_info['model_path']
                            
                            if os.path.exists(model_path):
                                # Add model to prediction engine and load it
                                model = self.prediction_engine.add_model(crypto, horizon)
                                model.load_model(model_path)
                                
                                models_loaded += 1
                                logger.info(f"✅ Loaded {crypto} {horizon} model (trained on {model_info['training_window']})")
                            else:
                                logger.warning(f"❌ Model file not found: {model_path}")
            
            if models_loaded > 0:
                self.is_trained = True
                self.last_training_time = _utcnow()
                logger.info(f"Successfully loaded {models_loaded} production models")
                return True
            else:
                logger.error("No production models could be loaded")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load production models: {e}")
            return False

    def train_models(self, force_retrain: bool = False):
        """
        Train all prediction models (fallback if no production models available)
        """
        # First try to load pre-trained production models
        if not force_retrain and self.load_production_models():
            logger.info("Using pre-trained production models")
            return
        
        # Check if retraining is needed
        if (not force_retrain and self.is_trained and self.last_training_time and
            _utcnow() - self.last_training_time < timedelta(hours=config.MODEL_SETTINGS['retrain_frequency_hours'])):
            logger.info("Models recently trained, skipping training")
            return
        
        logger.info("Starting model training from scratch...")
        
        # Collect data for training (longer history for better models)
        raw_data = self.collect_all_data(days=90)  # 3 months of data
        
        # Prepare features
        prepared_data = self.prepare_features_for_all_cryptos(raw_data)
        
        if not prepared_data:
            logger.error("No data available for training")
            return
        
        # Train models
        try:
            training_results = self.prediction_engine.train_all_models(prepared_data)
            
            # Log training results
            for model_key, result in training_results.items():
                if 'error' in result:
                    logger.error(f"Training failed for {model_key}: {result['error']}")
                else:
                    clf_acc = result.get('classification_results', {}).get('train_accuracy', 'N/A')
                    logger.info(f"Training completed for {model_key} - Accuracy: {clf_acc}")
            
            # Save models
            self.prediction_engine.save_ensemble('models')
            
            self.is_trained = True
            self.last_training_time = _utcnow()
            logger.info("Model training completed and saved")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def generate_predictions(self):
        """
        Generate predictions for all cryptocurrencies and horizons
        """
        if not self.is_trained:
            logger.warning("Models not trained yet, training first...")
            self.train_models()
            
        logger.info("Generating predictions...")
        
        # Collect recent data for prediction
        raw_data = self.collect_all_data(days=30)  # Last 30 days for prediction
        
        # Prepare features
        prepared_data = self.prepare_features_for_all_cryptos(raw_data)
        
        if not prepared_data:
            logger.error("No data available for prediction")
            return
        
        # Generate predictions
        try:
            predictions = self.prediction_engine.predict_all(prepared_data)
            
            if predictions:
                # Store predictions in database
                self.store_predictions(predictions)
                
                # Update timeseries data for current prices and predictions
                for crypto in config.CRYPTOCURRENCIES:
                    try:
                        current_price = self.data_collector.get_crypto_current_price(crypto)
                        if current_price:
                            # Get current predictions for this crypto
                            current_predictions = {
                                pred['horizon']: pred['direction_prob']
                                for model_key, pred in predictions.items()
                                if pred['crypto'] == crypto
                            }
                            
                            # Update timeseries tracking
                            self.accuracy_tracker.update_prediction_timeseries(
                                crypto=crypto,
                                timestamp=_utcnow(),
                                actual_price=current_price,
                                predictions=current_predictions
                            )

                            # Store actual price at target for future evaluations
                            self.accuracy_tracker.store_actual_price_at_target(
                                crypto=crypto,
                                target_timestamp=_utcnow(),
                                actual_price=current_price
                            )
                    except Exception as e:
                        logger.warning(f"Failed to update timeseries for {crypto}: {e}")
                
                # Display current predictions
                self.display_predictions(predictions)

                # Save candlestick charts
                try:
                    self.plot_candles(raw_data, predictions)
                except Exception as e:
                    logger.warning(f"Candle plot failed: {e}")

                # FIXED: Evaluate mature predictions RIGHT BEFORE displaying evaluation table
                logger.info("Checking for mature predictions to evaluate...")
                try:
                    evaluations = self.accuracy_tracker.batch_evaluate_mature_predictions(self.data_collector)
                    if evaluations:
                        total_evaluated = sum(len(evals) for evals in evaluations.values())
                        logger.info(f"✅ Automatically evaluated {total_evaluated} mature predictions")
                        
                        # Debug: Show what was evaluated
                        for key, evals in evaluations.items():
                            logger.info(f"   - {key}: {len(evals)} evaluations")
                    else:
                        logger.info("No predictions ready for evaluation yet")
                        
                        # Debug: Check how many unevaluated predictions exist
                        try:
                            conn = sqlite3.connect(config.DATABASE_PATH)
                            query = '''
                                SELECT p.crypto, p.prediction_horizon, COUNT(*) as count,
                                       MIN(p.created_at) as oldest, MAX(p.created_at) as newest
                                FROM predictions p
                                LEFT JOIN prediction_evaluations pe ON p.id = pe.prediction_id
                                WHERE pe.id IS NULL
                                GROUP BY p.crypto, p.prediction_horizon
                                ORDER BY p.crypto, p.prediction_horizon
                            '''
                            df = pd.read_sql_query(query, conn)
                            conn.close()
                            
                            if not df.empty:
                                logger.info("📋 Unevaluated predictions waiting:")
                                for _, row in df.iterrows():
                                    oldest_time = pd.to_datetime(row['oldest'])
                                    time_waiting = _utcnow() - oldest_time
                                    logger.info(f"   - {row['crypto']} {row['prediction_horizon']}: {row['count']} predictions "
                                              f"(oldest waiting {time_waiting.total_seconds()/3600:.1f}h)")
                            else:
                                logger.info("   - No unevaluated predictions in database")
                        except Exception as debug_e:
                            logger.warning(f"Debug query failed: {debug_e}")
                            
                except Exception as e:
                    logger.warning(f"Automatic accuracy evaluation failed: {e}")
                
                # Display evaluation of past predictions (now with fresh evaluations)
                self.display_prediction_evaluation()
                
                logger.info(f"Generated {len(predictions)} predictions")
            else:
                logger.warning("No predictions generated")
                
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
    
    def store_predictions(self, predictions: Dict):
        """
        Store predictions in SQLite database
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)

            for model_key, prediction in predictions.items():
                # Convert timestamps to strings - now using target timestamp (when prediction is FOR)
                target_timestamp_str = str(prediction['timestamp'])

                # Insert prediction into database with corrected timestamp logic
                # predicted_price column repurposed: stores raw P(UP) in [0, 1].
                # Direction = 1 (UP) when value >= 0.5, DOWN otherwise.
                # confidence column stores model_confidence (distance from 0.5).
                conn.execute('''
                    INSERT INTO predictions
                    (datetime, crypto, prediction_horizon, predicted_price, current_price, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    target_timestamp_str,
                    str(prediction['crypto']),
                    str(prediction['horizon']),
                    float(prediction.get('direction_prob', 0.5)),   # P(UP)
                    float(prediction.get('current_price', 0)),
                    float(prediction['model_confidence']),
                    _utcnow().isoformat()
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Successfully stored {len(predictions)} predictions in database")
            
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
    
    def display_predictions(self, predictions: Dict):
        """
        Display direction predictions with confidence scores.
        """
        print("\n" + "="*70)
        print(f"DIRECTION PREDICTIONS  —  {_utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("="*70)

        crypto_predictions: Dict[str, list] = {}
        for model_key, prediction in predictions.items():
            crypto = prediction['crypto']
            crypto_predictions.setdefault(crypto, []).append(prediction)

        for crypto, preds in crypto_predictions.items():
            emoji = "₿" if crypto == 'bitcoin' else "♦️ " if crypto == 'ethereum' else "📈"
            print(f"\n{emoji} {crypto.upper()}")
            print("-" * 70)
            print(f"  {'Horizon':>6} | {'Price':>10} | {'Direction':>10} | {'Confidence':>10} | Signal")
            print("-" * 70)

            try:
                live_price = self.data_collector.get_crypto_current_price(crypto) or preds[0]['current_price']
            except Exception:
                live_price = preds[0]['current_price']

            preds.sort(key=lambda x: {'15m': 1}.get(x['horizon'], 99))

            for pred in preds:
                is_up   = bool(pred['predicted_direction'])
                conf    = pred['model_confidence'] * 100
                dir_str = "  UP  " if is_up else " DOWN "
                arrow   = "▲" if is_up else "▼"

                # Signal strength label
                if conf >= 65:
                    signal = "STRONG"
                elif conf >= 58:
                    signal = "MODERATE"
                else:
                    signal = "WEAK"

                print(f"  {pred['horizon'].upper():>6} | "
                      f"${live_price:>9.2f} | "
                      f"{arrow} {dir_str} | "
                      f"{conf:>8.1f}%   | {signal}")

        print("="*70)

    def plot_candles(self, raw_data: Dict[str, pd.DataFrame], predictions: Dict,
                     n_bars: int = 50):
        """
        Save a candlestick chart for each crypto showing recent bars + prediction arrow.
        """
        os.makedirs('candle_plots', exist_ok=True)

        for crypto in config.CRYPTOCURRENCIES:
            df_15m = raw_data.get(crypto, {}).get('15m', pd.DataFrame())
            if df_15m.empty:
                continue

            df = df_15m.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').tail(n_bars).reset_index(drop=True)

            # mplfinance expects a DatetimeIndex and OHLCV columns
            ohlc = df.rename(columns={'price': 'Close', 'open': 'Open',
                                      'high': 'High', 'low': 'Low',
                                      'volume': 'Volume'})
            ohlc = ohlc.set_index('datetime')
            ohlc = ohlc[['Open', 'High', 'Low', 'Close', 'Volume']]

            # Find this crypto's prediction
            pred_info = None
            for _, pred in predictions.items():
                if pred['crypto'] == crypto:
                    pred_info = pred
                    break

            # Title with prediction
            title = f"{crypto.upper()} 15m"
            if pred_info:
                direction = "UP" if pred_info['predicted_direction'] else "DOWN"
                conf = pred_info['model_confidence'] * 100
                arrow = "\u25B2" if pred_info['predicted_direction'] else "\u25BC"
                title += f"  |  Prediction: {arrow} {direction} ({conf:.0f}%)"

            # Color-coded candle style
            mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350',
                                       edge='inherit', wick='inherit',
                                       volume='in')
            style = mpf.make_mpf_style(marketcolors=mc, gridstyle='--',
                                       gridcolor='#e0e0e0')

            timestamp = _utcnow().strftime('%Y%m%d_%H%M')
            filepath = f"candle_plots/{crypto}_{timestamp}.png"

            fig, axes = mpf.plot(ohlc, type='candle', style=style,
                                 volume=True, title=title,
                                 figsize=(14, 7), returnfig=True)

            # Add prediction arrow on the chart
            if pred_info:
                ax = axes[0]
                last_close = ohlc['Close'].iloc[-1]
                price_range = ohlc['High'].max() - ohlc['Low'].min()
                offset = price_range * 0.03
                if pred_info['predicted_direction']:
                    ax.annotate('\u25B2 UP', xy=(len(ohlc) - 1, last_close + offset),
                                fontsize=14, fontweight='bold', color='#26a69a',
                                ha='center')
                else:
                    ax.annotate('\u25BC DOWN', xy=(len(ohlc) - 1, last_close - offset),
                                fontsize=14, fontweight='bold', color='#ef5350',
                                ha='center')

            fig.savefig(filepath, dpi=100, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Candle plot saved: {filepath}")

    def update_current_prices(self):
        """
        Quick update of current prices for monitoring
        """
        logger.info("Updating current prices...")
        
        for crypto in config.CRYPTOCURRENCIES:
            try:
                current_price = self.data_collector.get_crypto_current_price(crypto)
                if current_price:
                    logger.info(f"{crypto}: ${current_price:.2f}")
            except Exception as e:
                logger.error(f"Failed to get current price for {crypto}: {e}")
    
    def display_prediction_evaluation(self):
        """
        Display evaluation of direction prediction accuracy
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            query = '''
                SELECT
                    crypto,
                    prediction_horizon,
                    predicted_price,
                    direction_predicted,
                    direction_actual,
                    direction_correct,
                    confidence,
                    evaluation_timestamp
                FROM (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY crypto, prediction_horizon, target_timestamp
                            ORDER BY evaluation_timestamp DESC
                        ) as rn
                    FROM prediction_evaluations
                    WHERE evaluation_timestamp >= datetime('now', '-7 days')
                )
                WHERE rn = 1
                ORDER BY evaluation_timestamp DESC
            '''
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                print("\n📊 No prediction evaluations available yet")
                print("    Predictions need time to mature before evaluation")
                return

            print("\n" + "="*80)
            print("📊 DIRECTION PREDICTION ACCURACY")
            print("="*80)

            for crypto in config.CRYPTOCURRENCIES:
                crypto_data = df[df['crypto'] == crypto]
                if crypto_data.empty:
                    continue

                crypto_emoji = "₿" if crypto == 'bitcoin' else "♦️ "
                print(f"\n{crypto_emoji} {crypto.upper()}")
                print("-" * 70)
                print(f"{'Time':>4} | {'Predicted':>9} | {'P(UP)':>5} | {'Actual':>6} | {'Correct':>7} | {'Conf':>5} | Evals")
                print("-" * 70)

                for horizon in config.PREDICTION_INTERVALS:
                    hd = crypto_data[crypto_data['prediction_horizon'] == horizon]
                    if hd.empty:
                        continue

                    total_evals = len(hd)
                    dir_accuracy = hd['direction_correct'].mean() * 100
                    latest = hd.iloc[0]

                    # direction_prob is stored in predicted_price column
                    direction_prob = latest['predicted_price']
                    pred_dir = latest['direction_predicted']
                    act_dir  = latest['direction_actual']
                    correct  = latest['direction_correct']
                    conf     = latest['confidence']

                    pred_label = "UP  " if pred_dir == 1 else "DOWN"
                    act_label  = "UP"   if act_dir  == 1 else "DOWN"
                    result     = "✓" if correct else "✗"
                    result_emoji = "🟢" if correct else "🔴"

                    latest_eval_time = pd.to_datetime(latest['evaluation_timestamp'])
                    time_since_eval = _utcnow() - latest_eval_time
                    if time_since_eval < timedelta(minutes=1):
                        eval_age = "just now"
                    elif time_since_eval < timedelta(hours=1):
                        eval_age = f"{int(time_since_eval.total_seconds()/60)}m ago"
                    elif time_since_eval < timedelta(days=1):
                        eval_age = f"{int(time_since_eval.total_seconds()/3600)}h ago"
                    else:
                        eval_age = f"{int(time_since_eval.total_seconds()/86400)}d ago"

                    print(f"  {horizon.upper():>3} | "
                          f"{pred_label:>9} | "
                          f"{direction_prob:>5.2f} | "
                          f"{act_label:>6} | "
                          f"{result_emoji} {result:>4} | "
                          f"{conf*100:>4.0f}% | "
                          f"{total_evals} evals ({dir_accuracy:.0f}% acc), eval'd {eval_age}")

            # Summary
            print("\n" + "="*80)
            print("📊 ACCURACY SUMMARY")
            print("="*80)
            for crypto in config.CRYPTOCURRENCIES:
                crypto_data = df[df['crypto'] == crypto]
                if crypto_data.empty:
                    continue
                crypto_emoji = "₿" if crypto == 'bitcoin' else "♦️ "
                dir_acc = crypto_data['direction_correct'].mean() * 100
                avg_conf = crypto_data['confidence'].mean() * 100
                print(f"{crypto_emoji} {crypto.upper()}")
                print(f"   • Evaluations:        {len(crypto_data)}")
                print(f"   • Direction Accuracy: {dir_acc:.1f}%")
                print(f"   • Avg Confidence:     {avg_conf:.1f}%")
                print()
            print("="*80)

        except Exception as e:
            logger.error(f"Prediction evaluation display failed: {e}")
            print(f"\n❌ Error displaying evaluations: {e}")
    
    def evaluate_and_track_accuracy(self):
        """
        Evaluate mature predictions and update accuracy tracking tables
        """
        try:
            logger.info("Evaluating prediction accuracy...")
            
            # Use the accuracy tracker to evaluate mature predictions
            evaluations = self.accuracy_tracker.batch_evaluate_mature_predictions(self.data_collector)
            
            if evaluations:
                total_evaluated = sum(len(evals) for evals in evaluations.values())
                logger.info(f"Evaluated {total_evaluated} predictions")
            else:
                logger.info("No predictions ready for evaluation")
            
            # Update timeseries data for current prices AND current predictions
            for crypto in config.CRYPTOCURRENCIES:
                try:
                    current_price = self.data_collector.get_crypto_current_price(crypto)
                    if current_price:
                        # Get the most recent predictions for this crypto
                        current_predictions = self.get_current_predictions_for_crypto(crypto)

                        # Store current actual price and predictions for timeseries tracking
                        self.accuracy_tracker.update_prediction_timeseries(
                            crypto=crypto,
                            timestamp=_utcnow(),
                            actual_price=current_price,
                            predictions=current_predictions
                        )

                        # Store actual price at target for future evaluations
                        self.accuracy_tracker.store_actual_price_at_target(
                            crypto=crypto,
                            target_timestamp=_utcnow(),
                            actual_price=current_price
                        )
                except Exception as e:
                    logger.error(f"Failed to update timeseries for {crypto}: {e}")
            
            # Generate and log accuracy report
            report = self.accuracy_tracker.generate_accuracy_report(days_back=7)
            logger.info("Accuracy evaluation completed")
            
            # Optionally print report
            if logger.isEnabledFor(logging.DEBUG):
                print(report)
                
        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")

    def get_current_predictions_for_crypto(self, crypto: str) -> Dict[str, float]:
        """
        Get predictions that are mature enough to be compared with current prices
        Only returns predictions made at appropriate times ago for each horizon
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)

            predictions = {}
            now = _utcnow()
            
            for horizon in config.PREDICTION_INTERVALS:
                # Define time windows for when predictions can be compared to current prices
                if horizon == '15m':
                    # Compare 15m predictions made 5+ minutes ago
                    min_age = timedelta(minutes=5)
                    max_age = timedelta(hours=1)  # Up to 1 hour old
                else:
                    continue
                
                # Get predictions from the appropriate time window
                earliest_time = (now - max_age).isoformat()
                latest_time = (now - min_age).isoformat()
                
                query = '''
                    SELECT predicted_price FROM predictions 
                    WHERE crypto = ? AND prediction_horizon = ?
                    AND created_at >= ? AND created_at <= ?
                    ORDER BY created_at DESC 
                    LIMIT 1
                '''
                cursor = conn.cursor()
                cursor.execute(query, (crypto, horizon, earliest_time, latest_time))
                result = cursor.fetchone()
                
                if result:
                    predictions[horizon] = result[0]
            
            conn.close()
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get evaluatable predictions for {crypto}: {e}")
            return {}
    
    def get_model_performance(self) -> Dict:
        """
        Analyze model performance by comparing predictions with actual prices
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Get predictions from last 24 hours that we can evaluate
            query = '''
                SELECT * FROM predictions 
                WHERE datetime(created_at) >= datetime('now', '-24 hours')
                ORDER BY created_at DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return {"message": "No recent predictions to evaluate"}
            
            # Here you would add logic to compare predictions with actual prices
            # For now, return basic statistics
            # Convert groupby result to JSON-serializable format
            prediction_breakdown = df.groupby(['crypto', 'prediction_horizon']).size()
            breakdown_dict = {f"{crypto}_{horizon}": count for (crypto, horizon), count in prediction_breakdown.items()}
            
            performance = {
                "total_predictions": len(df),
                "prediction_breakdown": breakdown_dict,
                "average_confidence": float(df['confidence'].mean())
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e)}
    
    def run_scheduled_tasks(self):
        """
        Set up and run scheduled tasks with clock-based timing
        """
        logger.info("Setting up clock-based scheduled tasks...")
        
        # Initial training (but not prediction - we'll wait for the boundary)
        logger.info("Running initial training...")
        self.train_models()
        
        # Don't wait for boundaries - start checking immediately
        freq = config.UPDATE_FREQUENCY_MINUTES
        logger.info(f"Starting immediately at {_utcnow().strftime('%H:%M:%S')} UTC - will catch next {freq}-minute boundary")

        # Set up variables for tracking other scheduled tasks
        last_price_update = _utcnow()
        last_model_training = _utcnow()
        last_prediction_run = _utcnow() - timedelta(minutes=freq * 2)  # Initialize to 2 intervals ago

        logger.info(f"Bot started - Predictions every {freq} minutes at clock boundaries")
        logger.info("Checking system clock every second for boundaries...")

        try:
            while True:
                now = _utcnow()
                current_minute = now.minute

                # Check if we're at the configured boundary (e.g. XX:00, XX:15, XX:30, XX:45 for 15m)
                is_boundary = (current_minute % freq == 0)

                # Only run if we're at boundary AND haven't run in the last half-interval (prevent double-runs)
                time_since_last_prediction = (now - last_prediction_run).total_seconds()
                should_run_prediction = is_boundary and time_since_last_prediction > (freq * 60 / 2)
                
                if should_run_prediction:
                    logger.info(f"🎯 BOUNDARY HIT! Running scheduled prediction at {now.strftime('%H:%M:%S')}")
                    self.generate_predictions()
                    last_prediction_run = _utcnow()

                    # Also run other tasks based on their frequency
                    # Update prices every 30 minutes (at XX:00 and XX:30)
                    if current_minute % 30 == 0:
                        self.update_current_prices()
                        last_price_update = _utcnow()

                    # NOTE: Accuracy evaluation now happens automatically in generate_predictions()

                    # Check if model retraining is needed (every N hours)
                    hours_since_training = (_utcnow() - last_model_training).total_seconds() / 3600
                    if hours_since_training >= config.MODEL_SETTINGS['retrain_frequency_hours']:
                        logger.info("Retraining models...")
                        self.train_models()
                        last_model_training = _utcnow()
                
                # Simple: sleep 1 second and check again
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
    
    def run_once(self):
        """
        Run the bot once (for testing)
        """
        logger.info("Running bot once...")
        self.train_models()
        self.generate_predictions()  # Now includes automatic accuracy evaluation
        
        # Generate accuracy report
        accuracy_report = self.accuracy_tracker.generate_accuracy_report(days_back=7)
        print(accuracy_report)
        
        performance = self.get_model_performance()
        print(f"Performance: {json.dumps(performance, indent=2)}")

def main():
    """
    Main entry point
    """
    print("🚀 Crypto Price Prediction Bot")
    print("="*50)
    print("Free ML-powered Bitcoin & Ethereum price predictions")
    print("Horizon: 15 minutes")
    print("="*50)
    
    bot = CryptoPredictionBot()
    
    # Check if user wants to run once or continuously
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        bot.run_once()
    else:
        print(f"\nStarting continuous predictions every {config.UPDATE_FREQUENCY_MINUTES} minutes...")
        print("Press Ctrl+C to stop")
        bot.run_scheduled_tasks()

if __name__ == "__main__":
    main() 