"""
Backtesting script for crypto price prediction models
Trains on historical data, makes predictions, and evaluates against known future prices
"""
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
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

        # Collect crypto OHLCV data
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

        # External data — parallel fetch with hard timeout (same pattern as bot)
        _EXT_TIMEOUT = 60  # longer timeout for backtest (not time-critical)
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
                    logger.warning(f"External data '{key}' timed out — skipping")
                    data[key] = pd.DataFrame()
                except Exception as e:
                    logger.warning(f"External data '{key}' failed: {e} — skipping")
                    data[key] = pd.DataFrame()

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
        
        # Build per-crypto external data dict (funding rate, OI, Fear & Greed)
        fng_df = raw_data.get('fear_greed', pd.DataFrame())
        external_data_by_crypto = {
            crypto: {
                'funding_rate':  raw_data.get(f'{crypto}_funding_rate', pd.DataFrame()),
                'open_interest': raw_data.get(f'{crypto}_open_interest', pd.DataFrame()),
                'fear_greed':    fng_df,
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
                        ext = external_data_by_crypto.get(crypto, {})
                        crypto_features = self.feature_engineer.prepare_features(
                            raw_data[crypto],
                            market_data,
                            ext,
                        )

                        if not crypto_features.empty:
                            prepared_data[crypto] = crypto_features
                            logger.info(f"Features prepared for {crypto}: {crypto_features.shape}")
                        else:
                            logger.warning(f"No features prepared for {crypto}")

                    except Exception as crypto_error:
                        logger.error(f"Feature preparation failed for {crypto}: {crypto_error}")

            return prepared_data
    
    def create_time_splits(self, data: pd.DataFrame, train_days: int = 30, 
                          step_days: int = 7, min_future_days: int = 2) -> List[Tuple[int, int]]:
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
        
        # Calculate indices for splits (15-minute intervals: 4 per hour, 96 per day)
        train_records = int(train_days * 24 * 4)
        step_records = int(step_days * 24 * 4)
        min_future_records = int(min_future_days * 24 * 4)
        
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
            
            # Find future prices for evaluation (15-minute intervals: 4 per hour)
            future_prices = {}
            for horizon in config.PREDICTION_INTERVALS:
                if horizon == '15m':
                    future_idx = test_start + 1         # 1 × 15m interval
                elif horizon == '1h':
                    future_idx = test_start + 4         # 4 × 15m intervals
                elif horizon == '4h':
                    future_idx = test_start + 16        # 16 × 15m intervals
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
                    target_col = f'target_direction_{horizon}'
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

                        # Direction evaluation
                        predicted_direction = prediction['predicted_direction']  # 1=UP, 0=DOWN
                        actual_direction = 1 if actual_future_price > current_price else 0
                        direction_correct = int(predicted_direction == actual_direction)

                        results[horizon] = {
                            'prediction_time': prediction_time,
                            'current_price': current_price,
                            'actual_price': actual_future_price,
                            'predicted_direction': predicted_direction,
                            'actual_direction': actual_direction,
                            'direction_correct': direction_correct,
                            'direction_prob': prediction['direction_prob'],
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
        
        # Define training window sizes (in days).
        # Binance provides years of 15m history — use large windows to find what trains best.
        training_windows = {
            '1_month':  30,   # 1 month  =  2 880 15m bars
            '2_months': 60,   # 2 months =  5 760 15m bars
            '3_months': 90,   # 3 months =  8 640 15m bars
            '6_months': 180,  # 6 months = 17 280 15m bars
            '9_months': 270,  # 9 months = 25 920 15m bars
            '1_year':   365,  # 1 year   = 35 040 15m bars
        }
        
        # Collect historical data once — reused by both the experiment and production training
        raw_data = self.collect_backtest_data(days)
        prepared_data = self.prepare_features_for_backtest(raw_data)
        self._last_prepared_data = prepared_data  # stash for caller to reuse
        
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
                splits = self.create_time_splits(data, window_days, step_days=1, min_future_days=2)
                
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
                    direction_accuracy = [x['direction_correct'] for x in horizon_data]
                    confidences = [x['confidence'] for x in horizon_data]

                    crypto_analysis[horizon] = {
                        'total_predictions': len(horizon_data),
                        'direction_accuracy': np.mean(direction_accuracy),
                        'avg_confidence': np.mean(confidences),
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
                        direction_accuracy = [x['direction_correct'] for x in horizon_data]
                        confidences = [x['confidence'] for x in horizon_data]

                        crypto_analysis[horizon] = {
                            'total_predictions': len(horizon_data),
                            'direction_accuracy': np.mean(direction_accuracy),
                            'std_direction_accuracy': np.std(direction_accuracy),
                            'avg_confidence': np.mean(confidences),
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
                print(f"{'Window':>12} | {'Count':>5} | {'Dir Acc':>7} | {'Std Acc':>7} | {'Conf':>6}")
                print("-" * 55)

                for window_name in ['1_month', '2_months', '3_months', '6_months', '9_months', '1_year']:
                    if (window_name in analysis and
                        crypto in analysis[window_name] and
                        horizon in analysis[window_name][crypto]):

                        stats = analysis[window_name][crypto][horizon]
                        display_name = window_name.replace('_', ' ').title()

                        print(f"{display_name:>12} | "
                              f"{stats['total_predictions']:>5} | "
                              f"{stats['direction_accuracy']*100:>5.1f}% | "
                              f"{stats['std_direction_accuracy']*100:>5.1f}% | "
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
                best_accuracy = 0.0

                for window_name in ['1_month', '2_months', '3_months', '6_months', '9_months', '1_year']:
                    if (window_name in analysis and
                        crypto in analysis[window_name] and
                        horizon in analysis[window_name][crypto]):

                        accuracy = analysis[window_name][crypto][horizon]['direction_accuracy']
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_window = window_name.replace('_', ' ').title()

                if best_window:
                    print(f"  • {horizon.upper():>2}: {best_window} training window (Dir Acc: {best_accuracy*100:.1f}%)")
        
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
            print(f"{'Horizon':>6} | {'Count':>5} | {'Dir Acc':>7} | {'Confidence':>10}")
            print("-" * 42)

            for horizon, stats in crypto_analysis.items():
                print(f"{horizon.upper():>6} | "
                      f"{stats['total_predictions']:>5} | "
                      f"{stats['direction_accuracy']*100:>5.1f}% | "
                      f"{stats['avg_confidence']*100:>8.1f}%")
        
        print("\n" + "="*80)

def train_production_models(backtester: CryptoBacktester, analysis: Dict,
                            prepared_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Train final production models on a fixed training window and save them.

    Why a fixed window instead of selecting based on backtest accuracy?
    The window-selection experiment runs ~50 samples per window size.  Picking
    the "best" window from those noisy estimates is test-set selection bias —
    the winner is mostly chosen by luck, and its reported accuracy is inflated.
    Using a fixed 365-day window (config.MODEL_SETTINGS['production_training_days'])
    is unbiased, captures recent market regime, and provides enough data for the
    calibrated XGBoost to learn robust patterns.

    The backtest analysis passed in `analysis` is still printed/saved for
    informational purposes so you can see which windows generalised best.

    `prepared_data` is passed in from the caller so we do NOT re-fetch or
    re-engineer features — that work was already done for the backtest run.
    """
    import os
    os.makedirs('models', exist_ok=True)

    production_window_days = config.MODEL_SETTINGS.get('production_training_days', 365)

    print(f"\n  Fixed training window: {production_window_days} days "
          f"(not selected from backtest — avoids selection bias)")

    production_models = {}

    for crypto in config.CRYPTOCURRENCIES:
        if crypto not in prepared_data:
            logger.warning(f"No prepared data for {crypto}, skipping")
            continue

        production_models[crypto] = {}
        data = prepared_data[crypto]

        # Use the most recent `production_window_days` of data
        n_bars = int(production_window_days * 24 * 4)
        recent_data = data.tail(n_bars).copy()
        print(f"\n  {crypto.upper()}: {len(recent_data)} training samples "
              f"({production_window_days} days × 96 bars/day)")

        for horizon in config.PREDICTION_INTERVALS:
            print(f"\n🔧 Training {crypto.upper()} {horizon.upper()}...")

            model_key = f"{crypto}_{horizon}"
            if model_key not in backtester.prediction_engine.models:
                backtester.prediction_engine.add_model(crypto, horizon)

            model = backtester.prediction_engine.models[model_key]

            try:
                training_info = model.train(recent_data)
                clf_results = training_info.get('classification_results', {})

                model_filepath = f"models/{crypto}_{horizon}_production.pkl"
                model.save_model(model_filepath)

                production_models[crypto][horizon] = {
                    'model_path': model_filepath,
                    'training_window': f'{production_window_days}_days_fixed',
                    'training_days': production_window_days,
                    'training_samples': len(recent_data),
                    'model_type': clf_results.get('model_type', 'unknown'),
                    'cv_accuracy': clf_results.get('cv_accuracy_mean', 0.0),
                    'best_n_estimators': clf_results.get('best_n_estimators', 0),
                }

                cv_acc = clf_results.get('cv_accuracy_mean', 0.0)
                n_est = clf_results.get('best_n_estimators', '?')
                print(f"  ✅ Saved → {model_filepath}  "
                      f"(CV acc: {cv_acc:.3f}, n_estimators: {n_est})")

            except Exception as e:
                print(f"  ❌ Failed to train {crypto} {horizon}: {e}")
                logger.exception(f"Production training failed for {crypto} {horizon}")

    with open('models/production_models.json', 'w') as f:
        json.dump(production_models, f, indent=2)

    print(f"\n🎯 Production models saved → models/production_models.json")
    return production_models

def main():
    """
    Run the backtesting script
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crypto Price Prediction Backtester')
    parser.add_argument('--runs', type=int, default=30, 
                       help='Number of runs per training window (default: 30)')
    parser.add_argument('--days', type=int, default=730,
                       help='Days of history for the backtest experiment (default: 730). '
                            'Production models always train on the most recent '
                            'production_training_days (config, default 365) of this data.')
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
    # Reuse the data that was already fetched and feature-engineered inside the
    # experiment — no second download or feature pass needed.
    prepared_data = backtester._last_prepared_data

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

    # Train production models on fixed window using already-prepared data
    print("\n🔥 TRAINING PRODUCTION MODELS...")
    optimal_models = train_production_models(backtester, analysis, prepared_data)
    
    logger.info("Optimal training results saved to optimal_training_results.json")
    print("\n✅ Optimal model training complete!")
    print("📊 Results saved to optimal_training_results.json")
    print("🤖 Production models saved to models/ directory")

if __name__ == "__main__":
    main() 