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
        self.is_trained = False
        self.last_training_time = None
        
        # Initialize database
        initialize_database()
        
        logger.info("Crypto Prediction Bot initialized")
    
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
        Prepare features for all cryptocurrencies
        """
        logger.info("Preparing features for all cryptocurrencies...")
        
        prepared_data = {}
        market_data = raw_data.get('traditional_markets', pd.DataFrame())
        
        for crypto in config.CRYPTOCURRENCIES:
            if crypto in raw_data:
                try:
                    features_df = self.feature_engineer.prepare_features(
                        raw_data[crypto], 
                        market_data
                    )
                    prepared_data[crypto] = features_df
                    logger.info(f"Features prepared for {crypto}: {features_df.shape}")
                    
                except Exception as e:
                    logger.error(f"Feature preparation failed for {crypto}: {e}")
        
        return prepared_data
    
    def train_models(self, force_retrain: bool = False):
        """
        Train all prediction models
        """
        # Check if retraining is needed
        if (not force_retrain and self.is_trained and self.last_training_time and 
            datetime.now() - self.last_training_time < timedelta(hours=config.MODEL_SETTINGS['retrain_frequency_hours'])):
            logger.info("Models recently trained, skipping training")
            return
        
        logger.info("Starting model training...")
        
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
                
                # Display current predictions
                self.display_predictions(predictions)
                
                # Display evaluation of past predictions
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
                # Convert timestamp to string if it's a pandas Timestamp
                timestamp_str = str(prediction['timestamp'])
                
                # Insert prediction into database
                conn.execute('''
                    INSERT INTO predictions 
                    (datetime, crypto, prediction_horizon, predicted_price, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp_str,
                    str(prediction['crypto']),
                    str(prediction['horizon']),
                    float(prediction['predicted_price']),
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
            print(f"\n📈 {crypto.upper()}")
            print("-" * 40)
            
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
                      f"Current: ${real_time_price:>8.2f} | "
                      f"Predicted: ${pred['predicted_price']:>8.2f} | "
                      f"Return: {actual_return:>6.2f}% | "
                      f"{direction_emoji} {confidence_stars}")
        
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
                
                # Define evaluation windows with some tolerance
                can_evaluate = False
                if horizon == '1h' and time_since_created >= timedelta(minutes=45):
                    # Evaluate 1h predictions after 45+ minutes
                    can_evaluate = True
                elif horizon == '1d' and time_since_created >= timedelta(hours=20):
                    # Evaluate 1d predictions after 20+ hours
                    can_evaluate = True
                elif horizon == '1w' and time_since_created >= timedelta(days=5):
                    # Evaluate 1w predictions after 5+ days
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
        Display evaluation of past predictions
        """
        evaluations = self.evaluate_past_predictions()
        
        if not evaluations:
            return
        
        print("\n" + "="*80)
        print("📊 PREDICTION ACCURACY EVALUATION")
        print("="*80)
        print("How well did past predictions perform?")
        
        for crypto in config.CRYPTOCURRENCIES:
            has_data = False
            crypto_evals = {}
            
            # Collect evaluations for this crypto
            for horizon in ['1h', '1d', '1w']:
                key = f"{crypto}_{horizon}"
                if key in evaluations and evaluations[key]:
                    crypto_evals[horizon] = evaluations[key][-1]  # Most recent evaluation
                    has_data = True
            
            if has_data:
                print(f"\n📈 {crypto.upper()} - PREDICTION vs ACTUAL")
                print("-" * 50)
                
                for horizon in ['1h', '1d', '1w']:
                    if horizon in crypto_evals:
                        eval_data = crypto_evals[horizon]
                        
                        # Calculate accuracy metrics
                        predicted_price = eval_data['predicted_price']
                        actual_price = eval_data['actual_price']
                        price_accuracy = abs(predicted_price - actual_price) / actual_price * 100
                        confidence = eval_data['confidence']
                        time_elapsed = eval_data['time_elapsed']
                        
                        # Format time elapsed
                        if time_elapsed.days > 0:
                            time_str = f"{time_elapsed.days}d {time_elapsed.seconds//3600}h"
                        elif time_elapsed.seconds >= 3600:
                            time_str = f"{time_elapsed.seconds//3600}h {(time_elapsed.seconds%3600)//60}m"
                        else:
                            time_str = f"{time_elapsed.seconds//60}m"
                        
                        # Determine if prediction was good
                        if price_accuracy <= 2.0:
                            accuracy_emoji = "🎯"  # Excellent
                        elif price_accuracy <= 5.0:
                            accuracy_emoji = "✅"  # Good
                        elif price_accuracy <= 10.0:
                            accuracy_emoji = "⚠️"   # Fair
                        else:
                            accuracy_emoji = "❌"  # Poor
                        
                        confidence_stars = "⭐" * int(confidence * 5)
                        
                        print(f"  {horizon.upper():>3} | "
                              f"Predicted: ${predicted_price:>8.2f} | "
                              f"Actual: ${actual_price:>8.2f} | "
                              f"Error: {price_accuracy:>5.1f}% | "
                              f"After: {time_str:>6} | "
                              f"{accuracy_emoji} {confidence_stars}")
        
        print("\n" + "="*80)
    
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
        Set up and run scheduled tasks
        """
        logger.info("Setting up scheduled tasks...")
        
        # Schedule tasks
        schedule.every(config.UPDATE_FREQUENCY_MINUTES).minutes.do(self.generate_predictions)
        schedule.every(30).minutes.do(self.update_current_prices)
        schedule.every(config.MODEL_SETTINGS['retrain_frequency_hours']).hours.do(self.train_models)
        
        # Initial training and prediction
        logger.info("Running initial training...")
        self.train_models()
        
        logger.info("Running initial prediction...")
        self.generate_predictions()
        
        # Main loop
        logger.info(f"Bot started - Predictions every {config.UPDATE_FREQUENCY_MINUTES} minutes")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
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
        self.generate_predictions()
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