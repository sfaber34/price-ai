"""
Backtesting script for crypto price prediction models
Trains on historical data, makes predictions, and evaluates against known future prices
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import json
import sys
import argparse

from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from ml_predictor import EnsemblePredictionEngine
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoBacktester:
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.prediction_engine = EnsemblePredictionEngine()
        
    def collect_backtest_data(self, days: int = 180) -> Dict[str, pd.DataFrame]:
        """
        Collect extended historical data for backtesting
        """
        logger.info(f"Collecting {days} days of historical data for backtesting...")
        
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
    
    def prepare_features_for_backtest(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Prepare features for all cryptocurrencies using the feature engineer with cross-asset correlations
        """
        logger.info("Preparing features for backtesting with cross-asset correlations...")
        
        market_data = raw_data.get('traditional_markets', pd.DataFrame())
        
        # Extract crypto data for cross-asset feature engineering
        crypto_data = {}
        for crypto in config.CRYPTOCURRENCIES:
            if crypto in raw_data and not raw_data[crypto].empty:
                crypto_data[crypto] = raw_data[crypto]
        
        if not crypto_data:
            logger.warning("No crypto data available for backtest feature preparation")
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
                
                logger.info(f"🔬 Backtest {crypto}: {features_df.shape[1]} total features "
                           f"({len(cross_asset_features)} cross-asset)")
            
            return prepared_data
            
        except Exception as e:
            logger.error(f"Cross-asset backtest feature preparation failed: {e}")
            
            # Fallback to individual feature preparation
            logger.info("Falling back to individual feature preparation for backtest...")
            prepared_data = {}
            for crypto in config.CRYPTOCURRENCIES:
                if crypto in raw_data and not raw_data[crypto].empty:
                    try:
                        # Prepare features using the same pipeline as the main bot
                        crypto_features = self.feature_engineer.prepare_features(
                            raw_data[crypto],
                            market_data
                        )
                        
                        if not crypto_features.empty:
                            prepared_data[crypto] = crypto_features
                            logger.info(f"Features prepared for {crypto}: {crypto_features.shape}")
                        else:
                            logger.warning(f"No features prepared for {crypto}")
                            
                    except Exception as crypto_error:
                        logger.error(f"Feature preparation failed for {crypto}: {crypto_error}")
            
            return prepared_data
    
    def create_time_splits(self, data: pd.DataFrame, train_days: int = 90, 
                          step_days: int = 7, min_future_days: int = 8) -> List[Tuple[int, int]]:
        """
        Create time-based train/test splits for backtesting
        
        Args:
            data: DataFrame with datetime index
            train_days: Days of data to use for training
            step_days: Days to step forward between tests
            min_future_days: Minimum days needed for future predictions
        
        Returns:
            List of (train_end_idx, test_start_idx) tuples
        """
        data = data.sort_values('datetime').reset_index(drop=True)
        total_records = len(data)
        
        # Calculate indices for splits
        train_records = int(train_days * 24)  # Assuming hourly data
        step_records = int(step_days * 24)
        min_future_records = int(min_future_days * 24)
        
        splits = []
        
        # Start after we have enough training data
        current_train_end = train_records
        
        while current_train_end + min_future_records < total_records:
            train_start = max(0, current_train_end - train_records)
            train_end = current_train_end
            test_start = current_train_end
            
            splits.append((train_start, train_end, test_start))
            current_train_end += step_records
        
        logger.info(f"Created {len(splits)} time splits for backtesting")
        return splits
    
    def evaluate_predictions_at_split(self, data: pd.DataFrame, train_start: int, 
                                    train_end: int, test_start: int, crypto: str) -> Dict:
        """
        Train model on historical data and evaluate predictions against known future
        """
        try:
            # Split data
            train_data = data.iloc[train_start:train_end].copy()
            
            # Minimum data validation
            if len(train_data) < 50:  # Need at least 50 samples for reliable training
                logger.warning(f"Insufficient training data: {len(train_data)} samples (minimum 50 required)")
                return {}
            
            # Check for sufficient price variation
            price_range = train_data['price'].max() - train_data['price'].min()
            if price_range == 0:
                logger.warning("No price variation in training data")
                return {}
            
            # Get the prediction timestamp (start of test period)
            prediction_time = data.iloc[test_start]['datetime']
            
            # Find future prices for evaluation
            future_prices = {}
            for horizon in config.PREDICTION_INTERVALS:
                if horizon == '1h':
                    future_idx = test_start + 1
                elif horizon == '1d':
                    future_idx = test_start + 24
                elif horizon == '1w':
                    future_idx = test_start + (24 * 7)
                else:
                    continue
                
                if future_idx < len(data):
                    future_prices[horizon] = data.iloc[future_idx]['price']
                else:
                    future_prices[horizon] = None
            
            # Train models on truncated data
            results = {}
            for horizon in config.PREDICTION_INTERVALS:
                if future_prices.get(horizon) is None:
                    continue
                
                model_key = f"{crypto}_{horizon}"
                
                # Create or get model
                if model_key not in self.prediction_engine.models:
                    self.prediction_engine.add_model(crypto, horizon)
                
                try:
                    # Train on truncated data with error handling
                    model = self.prediction_engine.models[model_key]
                    
                    # Check target variable distribution before training
                    target_col = f'target_{horizon}'
                    if target_col in train_data.columns:
                        unique_targets = train_data[target_col].nunique()
                        if unique_targets < 2:
                            logger.warning(f"Insufficient target variation for {horizon}: {unique_targets} unique values")
                            continue
                    
                    model.train(train_data)
                    
                    # Make prediction
                    prediction = model.predict(train_data)
                    
                    if prediction:
                        current_price = train_data.iloc[-1]['price']
                        actual_future_price = future_prices[horizon]
                        predicted_price = prediction['predicted_price']
                        
                        # Calculate errors
                        dollar_error = abs(predicted_price - actual_future_price)
                        percent_error = (dollar_error / actual_future_price) * 100
                        
                        # Direction accuracy
                        predicted_direction = predicted_price > current_price
                        actual_direction = actual_future_price > current_price
                        direction_correct = predicted_direction == actual_direction
                        
                        results[horizon] = {
                            'prediction_time': prediction_time,
                            'current_price': current_price,
                            'predicted_price': predicted_price,
                            'actual_price': actual_future_price,
                            'dollar_error': dollar_error,
                            'percent_error': percent_error,
                            'direction_correct': direction_correct,
                            'confidence': prediction['model_confidence'],
                            'horizon': horizon,
                            'training_samples': len(train_data)
                        }
                    
                except Exception as model_error:
                    logger.warning(f"Model training/prediction failed for {crypto}-{horizon}: {model_error}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed at split: {e}")
            return {}
    
    def run_backtest(self, days: int = 180, train_days: int = 90, 
                    step_days: int = 7) -> Dict:
        """
        Run full backtesting process
        """
        logger.info("Starting comprehensive backtest...")
        
        # Collect historical data
        raw_data = self.collect_backtest_data(days)
        
        # Prepare features
        prepared_data = self.prepare_features_for_backtest(raw_data)
        
        # Run backtests for each crypto
        all_results = {}
        
        for crypto in config.CRYPTOCURRENCIES:
            if crypto not in prepared_data:
                logger.warning(f"No prepared data for {crypto}, skipping")
                continue
            
            logger.info(f"Running backtest for {crypto}...")
            
            data = prepared_data[crypto]
            splits = self.create_time_splits(data, train_days, step_days)
            
            crypto_results = []
            
            for i, (train_start, train_end, test_start) in enumerate(splits):
                logger.info(f"Processing split {i+1}/{len(splits)} for {crypto}")
                
                split_results = self.evaluate_predictions_at_split(
                    data, train_start, train_end, test_start, crypto
                )
                
                if split_results:
                    crypto_results.append(split_results)
            
            all_results[crypto] = crypto_results
            logger.info(f"Completed {len(crypto_results)} backtests for {crypto}")
        
        return all_results
    
    def run_training_optimization(self, days: int = 180, runs_per_window: int = 20) -> Dict:
        """
        Run backtest experiment with different training window sizes
        
        Args:
            days: Total days of historical data to collect
            runs_per_window: Number of backtest runs per training window size
        
        Returns:
            Dict with results organized by training window size
        """
        logger.info("Starting training window size experiment...")
        
        # Define training window sizes (in days) - ensuring sufficient data diversity
        training_windows = {
            '1_month': 30,       # 1 month = 720 hours (minimum reliable)
            '6_weeks': 42,       # 6 weeks = 1008 hours (solid foundation)
            '2_months': 60,      # 2 months = 1440 hours (good baseline)
            '10_weeks': 70,      # 10 weeks = 1680 hours (extended range)
            '3_months': 90,      # 3 months = 2160 hours (current bot standard)
            '4_months': 120      # 4 months = 2880 hours (maximum context)
        }
        
        # Collect historical data once
        raw_data = self.collect_backtest_data(days)
        prepared_data = self.prepare_features_for_backtest(raw_data)
        
        # Results organized by training window size
        experiment_results = {}
        
        # Calculate total number of runs across all windows and cryptos
        available_cryptos = [crypto for crypto in config.CRYPTOCURRENCIES if crypto in prepared_data]
        total_runs = len(training_windows) * len(available_cryptos) * runs_per_window
        current_run = 0
        
        # Track progress through training windows
        total_windows = len(training_windows)
        current_window = 0
        
        for window_name, window_days in training_windows.items():
            current_window += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing training window {current_window}/{total_windows}: {window_name.upper()} ({window_days} days)")
            logger.info(f"{'='*60}")
            
            experiment_results[window_name] = {}
            
            for crypto in config.CRYPTOCURRENCIES:
                if crypto not in prepared_data:
                    logger.warning(f"No prepared data for {crypto}, skipping")
                    continue
                
                logger.info(f"Running {window_name} experiment for {crypto}...")
                
                data = prepared_data[crypto]
                
                # Create splits for this window size, step by 1 day to get more samples
                splits = self.create_time_splits(data, window_days, step_days=1, min_future_days=8)
                
                # Limit to requested number of runs
                if len(splits) > runs_per_window:
                    # Take evenly spaced splits across the time range
                    step = len(splits) // runs_per_window
                    splits = splits[::step][:runs_per_window]
                
                crypto_results = []
                
                for i, (train_start, train_end, test_start) in enumerate(splits):
                    current_run += 1
                    logger.info(f"Run {current_run}/{total_runs}: Processing {window_name} split {i+1}/{len(splits)} for {crypto}")
                    
                    split_results = self.evaluate_predictions_at_split(
                        data, train_start, train_end, test_start, crypto
                    )
                    
                    if split_results:
                        crypto_results.append(split_results)
                
                experiment_results[window_name][crypto] = crypto_results
                logger.info(f"Completed {len(crypto_results)} {window_name} backtests for {crypto}")
        
        return experiment_results
    
    def analyze_backtest_results(self, results: Dict) -> Dict:
        """
        Analyze and summarize backtest results
        """
        logger.info("Analyzing backtest results...")
        
        analysis = {}
        
        for crypto, crypto_results in results.items():
            if not crypto_results:
                continue
            
            crypto_analysis = {}
            
            # Analyze by horizon
            for horizon in config.PREDICTION_INTERVALS:
                horizon_data = []
                
                for split_result in crypto_results:
                    if horizon in split_result:
                        horizon_data.append(split_result[horizon])
                
                if horizon_data:
                    # Calculate statistics
                    dollar_errors = [x['dollar_error'] for x in horizon_data]
                    percent_errors = [x['percent_error'] for x in horizon_data]
                    direction_accuracy = [x['direction_correct'] for x in horizon_data]
                    confidences = [x['confidence'] for x in horizon_data]
                    
                    crypto_analysis[horizon] = {
                        'total_predictions': len(horizon_data),
                        'avg_dollar_error': np.mean(dollar_errors),
                        'median_dollar_error': np.median(dollar_errors),
                        'avg_percent_error': np.mean(percent_errors),
                        'median_percent_error': np.median(percent_errors),
                        'direction_accuracy': np.mean(direction_accuracy),
                        'avg_confidence': np.mean(confidences),
                        'max_error': max(percent_errors),
                        'min_error': min(percent_errors)
                    }
            
            analysis[crypto] = crypto_analysis
        
        return analysis
    
    def analyze_training_optimization(self, results: Dict) -> Dict:
        """
        Analyze training window experiment results
        """
        logger.info("Analyzing training window experiment results...")
        
        analysis = {}
        
        for window_name, window_data in results.items():
            window_analysis = {}
            
            for crypto, crypto_results in window_data.items():
                if not crypto_results:
                    continue
                
                crypto_analysis = {}
                
                # Analyze by horizon
                for horizon in config.PREDICTION_INTERVALS:
                    horizon_data = []
                    
                    for split_result in crypto_results:
                        if horizon in split_result:
                            horizon_data.append(split_result[horizon])
                    
                    if horizon_data:
                        # Calculate statistics
                        dollar_errors = [x['dollar_error'] for x in horizon_data]
                        percent_errors = [x['percent_error'] for x in horizon_data]
                        direction_accuracy = [x['direction_correct'] for x in horizon_data]
                        confidences = [x['confidence'] for x in horizon_data]
                        
                        crypto_analysis[horizon] = {
                            'total_predictions': len(horizon_data),
                            'avg_dollar_error': np.mean(dollar_errors),
                            'avg_percent_error': np.mean(percent_errors),
                            'direction_accuracy': np.mean(direction_accuracy),
                            'avg_confidence': np.mean(confidences),
                            'std_percent_error': np.std(percent_errors),
                            'median_percent_error': np.median(percent_errors)
                        }
                
                window_analysis[crypto] = crypto_analysis
            
            analysis[window_name] = window_analysis
        
        return analysis
    
    def display_training_window_results(self, analysis: Dict):
        """
        Display training window experiment results in a comprehensive format
        """
        print("\n" + "="*100)
        print("🧪 TRAINING WINDOW SIZE EXPERIMENT RESULTS")
        print("="*100)
        
        # Create summary table for each crypto and horizon
        for crypto in ['bitcoin', 'ethereum']:
            if crypto == 'bitcoin':
                crypto_emoji = "₿"
            elif crypto == 'ethereum':
                crypto_emoji = "♦️"
            else:
                crypto_emoji = "📈"
            
            print(f"\n{crypto_emoji} {crypto.upper()} - TRAINING WINDOW ANALYSIS")
            print("="*90)
            
            for horizon in config.PREDICTION_INTERVALS:
                print(f"\n📊 {horizon.upper()} PREDICTIONS")
                print("-" * 75)
                print(f"{'Window':>12} | {'Count':>5} | {'Avg %Err':>8} | {'Std %Err':>8} | {'Dir Acc':>7} | {'Conf':>6}")
                print("-" * 75)
                
                for window_name in ['1_month', '6_weeks', '2_months', '10_weeks', '3_months', '4_months']:
                    if (window_name in analysis and 
                        crypto in analysis[window_name] and 
                        horizon in analysis[window_name][crypto]):
                        
                        stats = analysis[window_name][crypto][horizon]
                        
                        # Format window name for display
                        display_name = window_name.replace('_', ' ').title()
                        
                        print(f"{display_name:>12} | "
                              f"{stats['total_predictions']:>5} | "
                              f"{stats['avg_percent_error']:>6.2f}% | "
                              f"{stats['std_percent_error']:>6.2f}% | "
                              f"{stats['direction_accuracy']*100:>5.1f}% | "
                              f"{stats['avg_confidence']*100:>4.0f}%")
        
        # Summary insights
        print("\n" + "="*100)
        print("💡 KEY INSIGHTS")
        print("="*100)
        
        # Find optimal training windows
        for crypto in ['bitcoin', 'ethereum']:
            if crypto == 'bitcoin':
                crypto_emoji = "₿"
            else:
                crypto_emoji = "♦️"
            
            print(f"\n{crypto_emoji} {crypto.upper()} OPTIMAL TRAINING WINDOWS:")
            
            for horizon in config.PREDICTION_INTERVALS:
                best_window = None
                best_error = float('inf')
                
                for window_name in ['1_month', '6_weeks', '2_months', '10_weeks', '3_months', '4_months']:
                    if (window_name in analysis and 
                        crypto in analysis[window_name] and 
                        horizon in analysis[window_name][crypto]):
                        
                        error = analysis[window_name][crypto][horizon]['avg_percent_error']
                        if error < best_error:
                            best_error = error
                            best_window = window_name.replace('_', ' ').title()
                
                if best_window:
                    print(f"  • {horizon.upper():>2}: {best_window} training window (Avg Error: {best_error:.2f}%)")
        
        print("\n" + "="*100)
    
    def display_backtest_results(self, analysis: Dict):
        """
        Display formatted backtest results
        """
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE BACKTEST RESULTS")
        print("="*80)
        
        for crypto, crypto_analysis in analysis.items():
            if crypto == 'bitcoin':
                crypto_emoji = "₿"
            elif crypto == 'ethereum':
                crypto_emoji = "♦️"
            else:
                crypto_emoji = "📈"
            
            print(f"\n{crypto_emoji} {crypto.upper()} BACKTEST ANALYSIS")
            print("-" * 60)
            print(f"{'Horizon':>6} | {'Count':>5} | {'Avg % Err':>9} | {'Dir Acc':>7} | {'Confidence':>10}")
            print("-" * 60)
            
            for horizon, stats in crypto_analysis.items():
                print(f"{horizon.upper():>6} | "
                      f"{stats['total_predictions']:>5} | "
                      f"{stats['avg_percent_error']:>7.2f}% | "
                      f"{stats['direction_accuracy']*100:>5.1f}% | "
                      f"{stats['avg_confidence']*100:>8.1f}%")
        
        print("\n" + "="*80)

def train_production_models(backtester: CryptoBacktester, analysis: Dict, days: int) -> Dict:
    """
    Train final production models using optimal training windows and save them
    """
    import os
    os.makedirs('models', exist_ok=True)
    
    # Find optimal training windows for each crypto/horizon combination
    optimal_windows = {}
    
    for crypto in ['bitcoin', 'ethereum']:
        optimal_windows[crypto] = {}
        
        for horizon in config.PREDICTION_INTERVALS:
            best_window = None
            best_error = float('inf')
            
            for window_name in ['1_month', '6_weeks', '2_months', '10_weeks', '3_months', '4_months']:
                if (window_name in analysis and 
                    crypto in analysis[window_name] and 
                    horizon in analysis[window_name][crypto]):
                    
                    error = analysis[window_name][crypto][horizon]['avg_percent_error']
                    if error < best_error:
                        best_error = error
                        best_window = window_name
            
            if best_window:
                optimal_windows[crypto][horizon] = {
                    'window_name': best_window,
                    'window_days': {
                        '1_month': 30, '6_weeks': 42, '2_months': 60, 
                        '10_weeks': 70, '3_months': 90, '4_months': 120
                    }[best_window],
                    'expected_error': best_error
                }
                print(f"📈 {crypto.upper()} {horizon.upper()}: Using {best_window} window (Error: {best_error:.2f}%)")
    
    # Collect fresh data for training production models
    print("\n📊 Collecting fresh data for production model training...")
    raw_data = backtester.collect_backtest_data(days)
    prepared_data = backtester.prepare_features_for_backtest(raw_data)
    
    # Train and save optimal models
    production_models = {}
    
    for crypto in config.CRYPTOCURRENCIES:
        if crypto not in prepared_data:
            continue
            
        production_models[crypto] = {}
        data = prepared_data[crypto]
        
        for horizon in config.PREDICTION_INTERVALS:
            if crypto in optimal_windows and horizon in optimal_windows[crypto]:
                window_info = optimal_windows[crypto][horizon]
                train_days = window_info['window_days']
                
                print(f"\n🔧 Training {crypto.upper()} {horizon.upper()} model with {window_info['window_name']} window...")
                
                # Use the full recent data for training (last N days)
                recent_data = data.tail(int(train_days * 24)).copy()
                
                # Create model and train
                model_key = f"{crypto}_{horizon}"
                if model_key not in backtester.prediction_engine.models:
                    backtester.prediction_engine.add_model(crypto, horizon)
                
                model = backtester.prediction_engine.models[model_key]
                
                try:
                    model.train(recent_data)
                    
                    # Save the trained model
                    model_filepath = f"models/{crypto}_{horizon}_production.pkl"
                    model.save_model(model_filepath)
                    
                    production_models[crypto][horizon] = {
                        'model_path': model_filepath,
                        'training_window': window_info['window_name'],
                        'training_days': train_days,
                        'training_samples': len(recent_data),
                        'expected_error': window_info['expected_error']
                    }
                    
                    print(f"✅ Saved {crypto} {horizon} model to {model_filepath}")
                    
                except Exception as e:
                    print(f"❌ Failed to train {crypto} {horizon} model: {e}")
    
    # Save production model metadata
    with open('models/production_models.json', 'w') as f:
        json.dump(production_models, f, indent=2)
    
    print(f"\n🎯 Production models metadata saved to models/production_models.json")
    return production_models

def main():
    """
    Run the backtesting script
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crypto Price Prediction Backtester')
    parser.add_argument('--runs', type=int, default=30, 
                       help='Number of runs per training window (default: 30)')
    parser.add_argument('--days', type=int, default=180,
                       help='Days of historical data to collect (default: 180)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (3 runs per window)')
    
    args = parser.parse_args()
    
    # Set runs based on mode
    if args.quick:
        runs_per_window = 1
        print("🚀 Crypto Price Prediction Backtester - QUICK TEST MODE")
    else:
        runs_per_window = args.runs
        print("🚀 Crypto Price Prediction Backtester")
    
    print("="*50)
    print("Training Window Size Experiment")
    print("="*50)
    
    backtester = CryptoBacktester()
    
    # Run training optimization
    print("🧪 Running training window optimization...")
    print(f"Testing 6 different training window sizes with {runs_per_window} runs each")
    print("⚠️  Includes error handling for insufficient data and training failures\n")
    
    experiment_results = backtester.run_training_optimization(
        days=args.days,
        runs_per_window=runs_per_window
    )
    
    # Analyze experiment results
    analysis = backtester.analyze_training_optimization(experiment_results)
    
    # Display comprehensive results
    backtester.display_training_window_results(analysis)
    
    # Save detailed results
    with open('optimal_training_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json_analysis = {}
        for window_name, window_data in analysis.items():
            json_analysis[window_name] = {}
            for crypto, crypto_data in window_data.items():
                json_analysis[window_name][crypto] = {}
                for horizon, stats in crypto_data.items():
                    json_analysis[window_name][crypto][horizon] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in stats.items()
                    }
        
        json.dump(json_analysis, f, indent=2)
    
    # Find and train optimal models for production use
    print("\n🔥 TRAINING PRODUCTION MODELS WITH OPTIMAL WINDOWS...")
    optimal_models = train_production_models(backtester, analysis, args.days)
    
    logger.info("Optimal training results saved to optimal_training_results.json")
    print("\n✅ Optimal model training complete!")
    print("📊 Results saved to optimal_training_results.json")
    print("🤖 Production models saved to models/ directory")

if __name__ == "__main__":
    main() 