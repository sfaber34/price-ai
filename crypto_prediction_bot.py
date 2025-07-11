"""
Main Crypto Price Prediction Bot
Orchestrates data collection, feature engineering, model training, and predictions
"""
import sqlite3
import pandas as pd
import schedule
import time
import logging
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
        
        # Collect crypto data
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
        
        try:
            # Use the new cross-asset correlation feature preparation
            prepared_data = self.feature_engineer.prepare_features_with_cross_asset_correlation(
                crypto_data, 
                market_data
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
                        features_df = self.feature_engineer.prepare_features(
                            raw_data[crypto], 
                            market_data
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
                    reg_mse = result.get('regression_results', {}).get('train_mse', 'N/A')
                    clf_acc = result.get('classification_results', {}).get('train_accuracy', 'N/A')
                    logger.info(f"Training completed for {model_key} - MSE: {reg_mse}, Accuracy: {clf_acc}")
            
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
                                pred['horizon']: pred['predicted_price'] 
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
                conn.execute('''
                    INSERT INTO predictions 
                    (datetime, crypto, prediction_horizon, predicted_price, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    target_timestamp_str,  # FIXED: Use target time, not feature time
                    str(prediction['crypto']),
                    str(prediction['horizon']),
                    float(prediction['predicted_price']),
                    float(prediction['model_confidence']),
                    datetime.now().isoformat()  # When the prediction was actually made
                ))
            
            conn.commit()
            conn.close()
            logger.info(f"Successfully stored {len(predictions)} predictions in database")
            
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
    
    def display_predictions(self, predictions: Dict):
        """
        Display predictions in a formatted way
        """
        print("\n" + "="*80)
        print(f"CRYPTO PRICE PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Group predictions by crypto
        crypto_predictions = {}
        for model_key, prediction in predictions.items():
            crypto = prediction['crypto']
            if crypto not in crypto_predictions:
                crypto_predictions[crypto] = []
            crypto_predictions[crypto].append(prediction)
        
        for crypto, preds in crypto_predictions.items():
            # Use distinctive emojis for each crypto
            if crypto == 'bitcoin':
                crypto_emoji = "₿"  # Bitcoin symbol
            elif crypto == 'ethereum':
                crypto_emoji = "♦️ "  # Diamond (ETH is often called digital diamond)
            else:
                crypto_emoji = "📈"  # Default for other cryptos
            
            print(f"\n{crypto_emoji} {crypto.upper()}")
            print("-" * 80)
            print(f"{'Time':>4} | {'Current':>10} | {'Predicted':>10} | {'Return':>8} | {'Dir':>4} | Confidence")
            print("-" * 80)
            
            # Get real-time current price for this crypto
            try:
                real_time_price = self.data_collector.get_crypto_current_price(crypto)
                if real_time_price is None:
                    # Fallback to historical price if real-time fails
                    real_time_price = preds[0]['current_price']
            except Exception as e:
                logger.warning(f"Failed to get real-time price for {crypto}: {e}")
                real_time_price = preds[0]['current_price']
            
            # Sort by horizon
            preds.sort(key=lambda x: {'1h': 1, '1d': 2, '1w': 3}[x['horizon']])
            
            for pred in preds:
                direction_emoji = "🟢" if pred['predicted_direction'] else "🔴"
                confidence_stars = "⭐" * int(pred['model_confidence'] * 5)
                
                # Calculate return based on real-time price vs predicted price
                actual_return = ((pred['predicted_price'] - real_time_price) / real_time_price) * 100
                
                print(f"  {pred['horizon'].upper():>3} | "
                      f"${real_time_price:>9.2f} | "
                      f"${pred['predicted_price']:>9.2f} | "
                      f"{actual_return:>6.2f}% | "
                      f"{direction_emoji:>4} | {confidence_stars}")
        
        print("\n" + "="*80)
    
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
    
    def evaluate_past_predictions(self):
        """
        Evaluate how well past predictions performed against actual prices
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Get predictions that should have matured by now
            query = '''
                SELECT * FROM predictions 
                WHERE datetime(created_at) <= datetime('now', '-1 hours')
                ORDER BY created_at DESC
                LIMIT 100
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return None
            
            # Current real-time prices for comparison
            current_prices = {}
            for crypto in config.CRYPTOCURRENCIES:
                try:
                    current_prices[crypto] = self.data_collector.get_crypto_current_price(crypto)
                except Exception as e:
                    logger.warning(f"Failed to get current price for {crypto}: {e}")
                    current_prices[crypto] = None
            
            # Evaluate predictions by timeframe
            evaluations = {}
            
            for _, pred in df.iterrows():
                crypto = pred['crypto']
                horizon = pred['prediction_horizon']
                predicted_price = pred['predicted_price']
                confidence = pred['confidence']
                created_at = pd.to_datetime(pred['created_at'])
                
                # Calculate time windows for evaluation (more flexible)
                now = datetime.now()
                time_since_created = now - created_at
                
                # Define evaluation windows that match the prediction horizons
                can_evaluate = False
                # FIXED: More flexible evaluation windows that work with 10-minute prediction schedule
                if horizon == '1h' and time_since_created >= timedelta(minutes=50):
                    # Evaluate 1h predictions any time 50+ minutes after creation (was 55-65 minutes)
                    can_evaluate = True
                elif horizon == '1d' and time_since_created >= timedelta(hours=20):
                    # Evaluate 1d predictions any time 20+ hours after creation (was 22-26 hours)
                    can_evaluate = True
                elif horizon == '1w' and time_since_created >= timedelta(days=6):
                    # Evaluate 1w predictions any time 6+ days after creation (was 6-8 days)
                    can_evaluate = True
                
                # Only evaluate if within the time window and we have current price
                if can_evaluate and current_prices.get(crypto):
                    key = f"{crypto}_{horizon}"
                    
                    if key not in evaluations:
                        evaluations[key] = []
                    
                    actual_price = current_prices[crypto]
                    predicted_return = ((predicted_price - actual_price) / actual_price) * 100
                    actual_return = 0  # We're comparing to current price
                    
                    # Direction accuracy
                    predicted_direction = predicted_return > 0
                    # For simplicity, assume current price movement direction
                    # In a real implementation, you'd fetch historical price at target_time
                    
                    evaluation = {
                        'predicted_price': predicted_price,
                        'actual_price': actual_price,
                        'predicted_return': predicted_return,
                        'confidence': confidence,
                        'time_elapsed': time_since_created,
                        'created_at': created_at
                    }
                    
                    evaluations[key].append(evaluation)
            
            return evaluations
            
        except Exception as e:
            logger.error(f"Past prediction evaluation failed: {e}")
            return None
    
    def display_prediction_evaluation(self):
        """
        Display evaluation of prediction accuracy
        """
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            # Get recent prediction evaluations
            query = '''
                SELECT 
                    crypto,
                    prediction_horizon,
                    predicted_price,
                    actual_price,
                    absolute_error,
                    percent_error,
                    direction_correct,
                    confidence,
                    prediction_timestamp,
                    target_timestamp,
                    evaluation_timestamp
                FROM prediction_evaluations 
                WHERE evaluation_timestamp >= datetime('now', '-7 days')
                ORDER BY evaluation_timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                print("\n📊 No prediction evaluations available yet")
                print("    Predictions need time to mature before evaluation")
                return
            
            print("\n" + "="*80)
            print("📊 PREDICTION ACCURACY EVALUATION")
            print("="*80)
            
            # Group by crypto
            for crypto in config.CRYPTOCURRENCIES:
                crypto_data = df[df['crypto'] == crypto]
                
                if crypto_data.empty:
                    continue
                
                # Use distinctive emojis for each crypto
                if crypto == 'bitcoin':
                    crypto_emoji = "₿"
                elif crypto == 'ethereum':
                    crypto_emoji = "♦️ "
                else:
                    crypto_emoji = "📈"
                
                print(f"\n{crypto_emoji} {crypto.upper()} - PREDICTION vs ACTUAL")
                print("-" * 100)
                print(f"{'Time':>4} | {'Predicted':>10} | {'Actual':>10} | {'Error $':>8} | {'% Error':>9} | {'Direction':>9} | {'Result':>6} | Confidence")
                print("-" * 100)
                
                # Get latest evaluation for each horizon
                for horizon in ['1h', '1d', '1w']:
                    horizon_data = crypto_data[crypto_data['prediction_horizon'] == horizon]
                    
                    if not horizon_data.empty:
                        total_evals = len(horizon_data)
                        
                        # Get the most recent evaluation for this horizon
                        latest = horizon_data.iloc[0]
                        latest_eval_time = pd.to_datetime(latest['evaluation_timestamp'])
                        
                        predicted_price = latest['predicted_price']
                        actual_price = latest['actual_price']
                        dollar_error = latest['absolute_error']
                        percent_error = latest['percent_error']
                        direction_correct = latest['direction_correct']
                        confidence = latest['confidence']
                        
                        # Direction indicator
                        direction_emoji = "🟢" if direction_correct else "🔴"
                        direction_text = "✓" if direction_correct else "✗"
                        
                        # Determine if prediction was good based on percent error
                        if percent_error <= 2.0:
                            accuracy_emoji = "🎯"  # Excellent
                        elif percent_error <= 5.0:
                            accuracy_emoji = "✅"  # Good
                        elif percent_error <= 10.0:
                            accuracy_emoji = "⚠️"   # Fair
                        else:
                            accuracy_emoji = "❌"  # Poor
                        
                        confidence_stars = "⭐" * int(confidence * 5) if confidence else ""
                        
                        # Calculate how long ago this evaluation was done
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
                              f"${predicted_price:>9.2f} | "
                              f"${actual_price:>9.2f} | "
                              f"${dollar_error:>7.2f} | "
                              f"{percent_error:>7.2f}% | "
                              f"{direction_emoji} {direction_text:>6} | "
                              f"{accuracy_emoji:>6} | {confidence_stars} ({total_evals} evals, latest {eval_age})")
            
            # Individual crypto summaries
            if not df.empty:
                print("\n" + "="*80)
                print("📊 ACCURACY SUMMARY BY CRYPTOCURRENCY")
                print("="*80)
                
                for crypto in config.CRYPTOCURRENCIES:
                    crypto_data = df[df['crypto'] == crypto]
                    
                    if not crypto_data.empty:
                        avg_error = crypto_data['percent_error'].mean()
                        direction_accuracy = crypto_data['direction_correct'].mean() * 100
                        total_evals = len(crypto_data)
                        
                        # Use distinctive emojis for each crypto
                        if crypto == 'bitcoin':
                            crypto_emoji = "₿"
                        elif crypto == 'ethereum':
                            crypto_emoji = "♦️ "
                        else:
                            crypto_emoji = "📈"
                        
                        print(f"{crypto_emoji} {crypto.upper()}")
                        print(f"   • Total Evaluations: {total_evals}")
                        print(f"   • Average Error: {avg_error:.2f}%")
                        print(f"   • Direction Accuracy: {direction_accuracy:.1f}%")
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
                # FIXED: More flexible evaluation windows that work with 10-minute prediction schedule
                if horizon == '1h':
                    # Compare 1h predictions made 50+ minutes ago (was 55-65 minutes)
                    min_age = timedelta(minutes=50)
                    max_age = timedelta(hours=4)  # Up to 4 hours old
                elif horizon == '1d':
                    # Compare 1d predictions made 20+ hours ago (was 22-26 hours)
                    min_age = timedelta(hours=20)
                    max_age = timedelta(hours=48)  # Up to 48 hours old
                elif horizon == '1w':
                    # Compare 1w predictions made 6+ days ago (was 6-8 days)
                    min_age = timedelta(days=6)
                    max_age = timedelta(days=14)  # Up to 14 days old
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
        logger.info(f"Starting immediately at {datetime.now().strftime('%H:%M:%S')} - will catch next 10-minute boundary")
        
        # Set up variables for tracking other scheduled tasks
        last_price_update = datetime.now()
        last_model_training = datetime.now()
        last_prediction_run = datetime.now() - timedelta(minutes=15)  # Initialize to 15 min ago
        
        logger.info(f"Bot started - Predictions every 10 minutes at clock boundaries (XX:00, XX:10, XX:20, etc.)")
        logger.info("Checking system clock every second for 10-minute boundaries...")
        
        try:
            while True:
                now = datetime.now()
                current_minute = now.minute
                current_second = now.second
                
                # Check if we're at a 10-minute boundary (XX:00, XX:10, XX:20, etc.)
                is_ten_minute_boundary = (current_minute % 10 == 0)
                
                # Only run if we're at boundary AND haven't run in the last 5 minutes (prevent double-runs)
                time_since_last_prediction = (now - last_prediction_run).total_seconds()
                should_run_prediction = is_ten_minute_boundary and time_since_last_prediction > 300
                
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
    print("Horizons: 1 hour, 1 day, 1 week")
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