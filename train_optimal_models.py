"""
Backtesting script for crypto price prediction models
Trains on historical data, makes predictions, and evaluates against known future prices

All data fetching, feature prep, and model training goes through training_pipeline.py
— the same code evaluate_calibration.py and the live bot use.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import json
import sys
import argparse
import os

import config
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from ml_predictor import CryptoPredictionModel
from training_pipeline import (
    fetch_all_cryptos,
    prepare_all_cryptos,
    train_model,
)

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

    def collect_and_prepare(self, days: int) -> Dict[str, pd.DataFrame]:
        """Fetch data and prepare features using the shared pipeline."""
        raw = fetch_all_cryptos(self.data_collector, days=days)
        return prepare_all_cryptos(self.feature_engineer, raw)

    def create_time_splits(self, data: pd.DataFrame, train_days: int = 30,
                          step_days: int = 7, min_future_days: int = 2) -> List[Tuple[int, int, int]]:
        """
        Create time-based train/test splits for backtesting
        """
        data = data.sort_values('datetime').reset_index(drop=True)
        total_records = len(data)

        # 15-minute intervals: 4 per hour, 96 per day
        train_records = int(train_days * 24 * 4)
        step_records = int(step_days * 24 * 4)
        min_future_records = int(min_future_days * 24 * 4)

        splits = []
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
            train_data = data.iloc[train_start:train_end].copy()

            if len(train_data) < 50:
                logger.warning(f"Insufficient training data: {len(train_data)} samples")
                return {}

            price_range = train_data['price'].max() - train_data['price'].min()
            if price_range == 0:
                logger.warning("No price variation in training data")
                return {}

            prediction_time = data.iloc[test_start]['datetime']

            # Polymarket-style: compare bar close vs bar open of the target bar.
            # The model predicts from data.iloc[test_start - 1] (last training bar).
            # The target for that bar is the *next* bar, which is data.iloc[test_start].
            # Offsets are relative to test_start:
            #   15m → test_start + 0  (the immediate next bar)
            #   1h  → test_start + 3  (4 bars from the prediction row = test_start - 1 + 4)
            #   4h  → test_start + 15 (16 bars from the prediction row)
            future_prices = {}
            future_opens = {}
            _horizon_offsets = {'15m': 0, '1h': 3, '4h': 15}
            for horizon in config.PREDICTION_INTERVALS:
                future_idx = test_start + _horizon_offsets.get(horizon, 0)
                if future_idx < len(data):
                    future_prices[horizon] = data.iloc[future_idx]['price']
                    future_opens[horizon] = data.iloc[future_idx]['open']
                else:
                    future_prices[horizon] = None
                    future_opens[horizon] = None

            results = {}
            for horizon in config.PREDICTION_INTERVALS:
                if future_prices.get(horizon) is None:
                    continue

                target_col = f'target_direction_{horizon}'
                if target_col in train_data.columns:
                    unique_targets = train_data[target_col].nunique()
                    if unique_targets < 2:
                        logger.warning(f"Insufficient target variation for {horizon}")
                        continue

                try:
                    # Train using shared pipeline
                    model = train_model(crypto, horizon, train_data)
                    prediction = model.predict(train_data)

                    if prediction:
                        current_price = train_data.iloc[-1]['price']
                        actual_future_price = future_prices[horizon]
                        actual_future_open = future_opens[horizon]

                        predicted_direction = prediction['predicted_direction']
                        actual_direction = 1 if actual_future_price >= actual_future_open else 0
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
        """Run full backtesting process"""
        logger.info("Starting comprehensive backtest...")

        prepared_data = self.collect_and_prepare(days)

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
        """
        logger.info("Starting training window size experiment...")

        training_windows = {
            '1_month':  30,
            '2_months': 60,
            '3_months': 90,
            '6_months': 180,
            '9_months': 270,
            '1_year':   365,
        }

        # Collect and prepare once — reused for all experiments and production training
        prepared_data = self.collect_and_prepare(days)
        self._last_prepared_data = prepared_data

        experiment_results = {}
        available_cryptos = [c for c in config.CRYPTOCURRENCIES if c in prepared_data]
        total_runs = len(training_windows) * len(available_cryptos) * runs_per_window
        current_run = 0
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
                    continue

                logger.info(f"Running {window_name} experiment for {crypto}...")
                data = prepared_data[crypto]
                splits = self.create_time_splits(data, window_days, step_days=1, min_future_days=2)

                if len(splits) > runs_per_window:
                    step = len(splits) // runs_per_window
                    splits = splits[::step][:runs_per_window]

                crypto_results = []
                for i, (train_start, train_end, test_start) in enumerate(splits):
                    current_run += 1
                    logger.info(f"Run {current_run}/{total_runs}: {window_name} split {i+1}/{len(splits)} for {crypto}")
                    split_results = self.evaluate_predictions_at_split(
                        data, train_start, train_end, test_start, crypto
                    )
                    if split_results:
                        crypto_results.append(split_results)

                experiment_results[window_name][crypto] = crypto_results
                logger.info(f"Completed {len(crypto_results)} {window_name} backtests for {crypto}")

        return experiment_results

    def analyze_backtest_results(self, results: Dict) -> Dict:
        """Analyze and summarize backtest results"""
        analysis = {}
        for crypto, crypto_results in results.items():
            if not crypto_results:
                continue
            crypto_analysis = {}
            for horizon in config.PREDICTION_INTERVALS:
                horizon_data = [sr[horizon] for sr in crypto_results if horizon in sr]
                if horizon_data:
                    crypto_analysis[horizon] = {
                        'total_predictions': len(horizon_data),
                        'direction_accuracy': np.mean([x['direction_correct'] for x in horizon_data]),
                        'avg_confidence': np.mean([x['confidence'] for x in horizon_data]),
                    }
            analysis[crypto] = crypto_analysis
        return analysis

    def analyze_training_optimization(self, results: Dict) -> Dict:
        """Analyze training window experiment results"""
        analysis = {}
        for window_name, window_data in results.items():
            window_analysis = {}
            for crypto, crypto_results in window_data.items():
                if not crypto_results:
                    continue
                crypto_analysis = {}
                for horizon in config.PREDICTION_INTERVALS:
                    horizon_data = [sr[horizon] for sr in crypto_results if horizon in sr]
                    if horizon_data:
                        crypto_analysis[horizon] = {
                            'total_predictions': len(horizon_data),
                            'direction_accuracy': np.mean([x['direction_correct'] for x in horizon_data]),
                            'std_direction_accuracy': np.std([x['direction_correct'] for x in horizon_data]),
                            'avg_confidence': np.mean([x['confidence'] for x in horizon_data]),
                        }
                window_analysis[crypto] = crypto_analysis
            analysis[window_name] = window_analysis
        return analysis

    def display_training_window_results(self, analysis: Dict):
        """Display training window experiment results"""
        print("\n" + "="*100)
        print("TRAINING WINDOW SIZE EXPERIMENT RESULTS")
        print("="*100)

        for crypto in ['bitcoin', 'ethereum']:
            print(f"\n  {crypto.upper()} - TRAINING WINDOW ANALYSIS")
            print("="*90)

            for horizon in config.PREDICTION_INTERVALS:
                print(f"\n  {horizon.upper()} PREDICTIONS")
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

        print("\n" + "="*100)
        print("KEY INSIGHTS")
        print("="*100)

        for crypto in ['bitcoin', 'ethereum']:
            print(f"\n  {crypto.upper()} OPTIMAL TRAINING WINDOWS:")
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
                    print(f"    {horizon.upper():>2}: {best_window} (Dir Acc: {best_accuracy*100:.1f}%)")

        print("\n" + "="*100)

    def display_backtest_results(self, analysis: Dict):
        """Display formatted backtest results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE BACKTEST RESULTS")
        print("="*80)

        for crypto, crypto_analysis in analysis.items():
            print(f"\n  {crypto.upper()} BACKTEST ANALYSIS")
            print("-" * 60)
            print(f"{'Horizon':>6} | {'Count':>5} | {'Dir Acc':>7} | {'Confidence':>10}")
            print("-" * 42)
            for horizon, stats in crypto_analysis.items():
                print(f"{horizon.upper():>6} | "
                      f"{stats['total_predictions']:>5} | "
                      f"{stats['direction_accuracy']*100:>5.1f}% | "
                      f"{stats['avg_confidence']*100:>8.1f}%")

        print("\n" + "="*80)


def train_production_models(prepared_data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Train final production models on a fixed training window and save them.
    Uses training_pipeline.train_model() — same code as evaluate_calibration.
    """
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

        n_bars = int(production_window_days * 24 * 4)
        recent_data = data.tail(n_bars).copy()

        print(f"\n  {crypto.upper()}: {len(recent_data)} training samples "
              f"({production_window_days} days x 96 bars/day)")

        for horizon in config.PREDICTION_INTERVALS:
            print(f"\n  Training {crypto.upper()} {horizon.upper()}...")

            try:
                # Same train_model() that evaluate_calibration uses
                model = train_model(crypto, horizon, recent_data)

                model_filepath = f"models/{crypto}_{horizon}_production.pkl"
                model.save_model(model_filepath)

                clf_results = model.training_history[-1].get('classification_results', {})
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
                print(f"    Saved -> {model_filepath}  "
                      f"(CV acc: {cv_acc:.3f}, n_estimators: {n_est})")

            except Exception as e:
                print(f"    Failed to train {crypto} {horizon}: {e}")
                logger.exception(f"Production training failed for {crypto} {horizon}")

    with open('models/production_models.json', 'w') as f:
        json.dump(production_models, f, indent=2)

    print(f"\n  Production models saved -> models/production_models.json")
    return production_models


def main():
    parser = argparse.ArgumentParser(description='Crypto Price Prediction Backtester')
    parser.add_argument('--runs', type=int, default=30,
                       help='Number of runs per training window (default: 30)')
    parser.add_argument('--days', type=int, default=730,
                       help='Days of history for the backtest experiment (default: 730)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (1 run per window)')
    parser.add_argument('--skip-backtest', action='store_true',
                       help='Skip window-size experiment, go straight to production training')

    args = parser.parse_args()

    production_window_days = config.MODEL_SETTINGS.get('production_training_days', 180)
    backtester = CryptoBacktester()

    # ---- Fast path: skip backtest, just train production models ----
    if args.skip_backtest:
        print("Crypto Model Trainer  (backtest skipped)")
        print("="*50)
        print(f"Fetching {production_window_days}d of data for production training...")

        prepared_data = backtester.collect_and_prepare(days=production_window_days + 5)

        print("\nTRAINING PRODUCTION MODELS...")
        train_production_models(prepared_data)
        print("\nProduction model training complete!")
        print("Production models saved to models/ directory")
        return

    # ---- Full path: window-size experiment + production training ----
    if args.quick:
        runs_per_window = 1
        print("Crypto Price Prediction Backtester - QUICK TEST MODE")
    else:
        runs_per_window = args.runs
        print("Crypto Price Prediction Backtester")

    print("="*50)
    print("Training Window Size Experiment")
    print("="*50)

    print(f"Testing 6 different training window sizes with {runs_per_window} runs each\n")

    experiment_results = backtester.run_training_optimization(
        days=args.days,
        runs_per_window=runs_per_window
    )
    prepared_data = backtester._last_prepared_data

    analysis = backtester.analyze_training_optimization(experiment_results)
    backtester.display_training_window_results(analysis)

    with open('optimal_training_results.json', 'w') as f:
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

    print("\nTRAINING PRODUCTION MODELS...")
    train_production_models(prepared_data)

    logger.info("Optimal training results saved to optimal_training_results.json")
    print("\nOptimal model training complete!")
    print("Results saved to optimal_training_results.json")
    print("Production models saved to models/ directory")

if __name__ == "__main__":
    main()
