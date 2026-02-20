"""
Main Crypto Price Prediction Bot
Orchestrates data collection, feature engineering, model training, and predictions
"""
import sqlite3
import pandas as pd
import schedule
import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta
from typing import Dict, List
import json
import os

from data_collector import DataCollector, initialize_database
from feature_engineering import FeatureEngineer
from ml_predictor import EnsemblePredictionEngine
from prediction_accuracy_tracker import PredictionAccuracyTracker
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
        
        # Initialize database
        initialize_database()
        
        logger.info("Crypto Prediction Bot initialized with accuracy tracking")
    
    def collect_all_data(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Collect all required data for training/prediction
        """
        logger.info(f"Collecting data for past {days} days...")
        
        data = {}
        
        # Collect crypto data (15m bars for features + 1m bars for intrabar features)
        for crypto in config.CRYPTOCURRENCIES:
            try:
                crypto_data = self.data_collector.get_crypto_data(crypto, days=days)
                if not crypto_data.empty:
                    data[crypto] = crypto_data
                    logger.info(f"Collected {len(crypto_data)} records for {crypto}")
                else:
                    logger.warning(f"No data collected for {crypto}")
            except Exception as e:
                logger.error(f"Failed to collect data for {crypto}: {e}")

            try:
                # Only need ~2 hours of 1m bars for intrabar features (current bar context).
                # Intrabar features are per-bar aggregations, not rolled across bars.
                data_1m = self.data_collector.get_crypto_data_1m(crypto, days=1)
                if not data_1m.empty:
                    data[f'{crypto}_1m'] = data_1m
                    logger.info(f"Collected {len(data_1m)} 1m bars for {crypto}")
            except Exception as e:
                logger.warning(f"1m data fetch failed for {crypto}: {e}")
        
        # Collect traditional market data
        try:
            market_data = self.data_collector.get_traditional_markets_data(days=days)
            if not market_data.empty:
                data['traditional_markets'] = market_data
                logger.info(f"Collected traditional market data: {len(market_data)} records")
        except Exception as e:
            logger.error(f"Failed to collect traditional market data: {e}")
            data['traditional_markets'] = pd.DataFrame()

        # Collect economic indicators
        try:
            econ_data = self.data_collector.get_economic_indicators()
            if not econ_data.empty:
                data['economic_indicators'] = econ_data
                logger.info(f"Collected economic indicators: {len(econ_data)} records")
        except Exception as e:
            logger.error(f"Failed to collect economic indicators: {e}")
            data['economic_indicators'] = pd.DataFrame()

        # External data (Fear & Greed, funding rate, open interest) — fetched in parallel
        # with a hard wall-clock timeout so a slow/hung OKX request can never block
        # prediction cycles.  Cached results return immediately on subsequent calls.
        _EXT_TIMEOUT = 45  # seconds per source before giving up
        ext_tasks = {'fear_greed': (self.data_collector.get_fear_greed, (days,))}
        for crypto in config.CRYPTOCURRENCIES:
            ext_tasks[f'{crypto}_funding_rate']  = (self.data_collector.get_funding_rate,  (crypto, days))
            ext_tasks[f'{crypto}_open_interest'] = (self.data_collector.get_open_interest, (crypto, days))

        with ThreadPoolExecutor(max_workers=len(ext_tasks)) as pool:
            futures = {key: pool.submit(fn, *args) for key, (fn, args) in ext_tasks.items()}
            for key, fut in futures.items():
                try:
                    result = fut.result(timeout=_EXT_TIMEOUT)
                    data[key] = result
                    logger.info(f"Collected {len(result)} {key} records")
                except FuturesTimeout:
                    logger.warning(f"External data '{key}' timed out after {_EXT_TIMEOUT}s — skipping")
                    data[key] = pd.DataFrame()
                except Exception as e:
                    logger.warning(f"External data '{key}' failed: {e} — skipping")
                    data[key] = pd.DataFrame()

        return data
    
    def prepare_features_for_all_cryptos(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Prepare features for all cryptocurrencies with cross-asset correlation features
        """
        logger.info("Preparing features for all cryptocurrencies with cross-asset correlations...")
        
        market_data = raw_data.get('traditional_markets', pd.DataFrame())
        
        # Extract crypto data for cross-asset feature engineering
        crypto_data = {}
        for crypto in config.CRYPTOCURRENCIES:
            if crypto in raw_data and not raw_data[crypto].empty:
                crypto_data[crypto] = raw_data[crypto]
        
        if not crypto_data:
            logger.warning("No crypto data available for feature preparation")
            return {}
        
        # Build per-crypto external data dict (funding rate, OI, Fear & Greed)
        fng_df = raw_data.get('fear_greed', pd.DataFrame())
        external_data_by_crypto = {
            crypto: {
                'funding_rate':  raw_data.get(f'{crypto}_funding_rate', pd.DataFrame()),
                'open_interest': raw_data.get(f'{crypto}_open_interest', pd.DataFrame()),
                'fear_greed':    fng_df,
                'intrabar_1m':   raw_data.get(f'{crypto}_1m', pd.DataFrame()),
            }
            for crypto in config.CRYPTOCURRENCIES
        }

        try:
            # Use the new cross-asset correlation feature preparation
            prepared_data = self.feature_engineer.prepare_features_with_cross_asset_correlation(
                crypto_data,
                market_data,
                external_data_by_crypto=external_data_by_crypto,
            )

            # Log feature counts for each crypto
            for crypto, features_df in prepared_data.items():
                # Count cross-asset features specifically
                cross_asset_features = [col for col in features_df.columns
                                      if any(other_crypto in col for other_crypto in config.CRYPTOCURRENCIES
                                           if other_crypto != crypto)]

                logger.info(f"✅ {crypto}: {features_df.shape[1]} total features "
                           f"({len(cross_asset_features)} cross-asset)")

            return prepared_data

        except Exception as e:
            logger.error(f"Cross-asset feature preparation failed: {e}")

            # Fallback to individual feature preparation
            logger.info("Falling back to individual feature preparation...")
            prepared_data = {}
            for crypto in config.CRYPTOCURRENCIES:
                if crypto in raw_data:
                    try:
                        ext = external_data_by_crypto.get(crypto, {})
                        features_df = self.feature_engineer.prepare_features(
                            raw_data[crypto],
                            market_data,
                            ext,
                        )
                        prepared_data[crypto] = features_df
                        logger.info(f"Features prepared for {crypto}: {features_df.shape}")

                    except Exception as crypto_error:
                        logger.error(f"Feature preparation failed for {crypto}: {crypto_error}")

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
                self.last_training_time = datetime.now()
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
            datetime.now() - self.last_training_time < timedelta(hours=config.MODEL_SETTINGS['retrain_frequency_hours'])):
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
            self.last_training_time = datetime.now()
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
                                timestamp=datetime.now(),
                                actual_price=current_price,
                                predictions=current_predictions
                            )
                            
                            # Store actual price at target for future evaluations
                            self.accuracy_tracker.store_actual_price_at_target(
                                crypto=crypto,
                                target_timestamp=datetime.now(),
                                actual_price=current_price
                            )
                    except Exception as e:
                        logger.warning(f"Failed to update timeseries for {crypto}: {e}")
                
                # Display current predictions
                self.display_predictions(predictions)
                
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
                                    time_waiting = datetime.now() - oldest_time
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
                feature_timestamp_str = str(prediction.get('feature_timestamp', prediction['timestamp']))
                
                # Insert prediction into database with corrected timestamp logic
                # predicted_price column repurposed: stores raw P(UP) in [0, 1].
                # Direction = 1 (UP) when value > 0.5, DOWN otherwise.
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
                    datetime.now().isoformat()
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
        print(f"DIRECTION PREDICTIONS  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

            preds.sort(key=lambda x: {'15m': 1, '1h': 2, '4h': 3}.get(x['horizon'], 99))

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

                for horizon in ['15m', '1h', '4h']:
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
                    time_since_eval = datetime.now() - latest_eval_time
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
                            timestamp=datetime.now(),
                            actual_price=current_price,
                            predictions=current_predictions
                        )
                        
                        # Store actual price at target for future evaluations
                        self.accuracy_tracker.store_actual_price_at_target(
                            crypto=crypto,
                            target_timestamp=datetime.now(),
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
            now = datetime.now()
            
            for horizon in config.PREDICTION_INTERVALS:
                # Define time windows for when predictions can be compared to current prices
                if horizon == '15m':
                    # Compare 15m predictions made 5+ minutes ago
                    min_age = timedelta(minutes=5)
                    max_age = timedelta(hours=1)  # Up to 1 hour old
                elif horizon == '1h':
                    # Compare 1h predictions made 50+ minutes ago
                    min_age = timedelta(minutes=50)
                    max_age = timedelta(hours=4)  # Up to 4 hours old
                elif horizon == '4h':
                    # Compare 4h predictions made 3.5+ hours ago
                    min_age = timedelta(hours=3, minutes=30)
                    max_age = timedelta(hours=24)  # Up to 24 hours old
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
        logger.info(f"Starting immediately at {datetime.now().strftime('%H:%M:%S')} - will catch next {freq}-minute boundary")

        # Set up variables for tracking other scheduled tasks
        last_price_update = datetime.now()
        last_model_training = datetime.now()
        last_prediction_run = datetime.now() - timedelta(minutes=freq * 2)  # Initialize to 2 intervals ago

        logger.info(f"Bot started - Predictions every {freq} minutes at clock boundaries")
        logger.info("Checking system clock every second for boundaries...")

        try:
            while True:
                now = datetime.now()
                current_minute = now.minute

                # Check if we're at the configured boundary (e.g. XX:00, XX:15, XX:30, XX:45 for 15m)
                is_boundary = (current_minute % freq == 0)

                # Only run if we're at boundary AND haven't run in the last half-interval (prevent double-runs)
                time_since_last_prediction = (now - last_prediction_run).total_seconds()
                should_run_prediction = is_boundary and time_since_last_prediction > (freq * 60 / 2)
                
                if should_run_prediction:
                    logger.info(f"🎯 BOUNDARY HIT! Running scheduled prediction at {now.strftime('%H:%M:%S')}")
                    self.generate_predictions()
                    last_prediction_run = now
                    
                    # Also run other tasks based on their frequency
                    # Update prices every 30 minutes (at XX:00 and XX:30)
                    if current_minute % 30 == 0:
                        self.update_current_prices()
                        last_price_update = now
                    
                    # NOTE: Accuracy evaluation now happens automatically in generate_predictions()
                    
                    # Check if model retraining is needed (every N hours)
                    hours_since_training = (now - last_model_training).total_seconds() / 3600
                    if hours_since_training >= config.MODEL_SETTINGS['retrain_frequency_hours']:
                        logger.info("Retraining models...")
                        self.train_models()
                        last_model_training = now
                
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
    print("Horizons: 15 minutes, 1 hour, 4 hours")
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