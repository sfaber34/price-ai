#!/usr/bin/env python3
"""
Calibration diagnostic for PRODUCTION models.

Loads the trained models from models/ and evaluates them on data that falls
OUTSIDE the training window.  Does NOT train anything — if no production
models exist, it warns and exits.

Usage:
    python3 evaluate_calibration.py
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from ml_predictor import CryptoPredictionModel
from training_pipeline import fetch_data_for_crypto, prepare_features_for_crypto


def calibration_report(probs: np.ndarray, actuals: np.ndarray, horizon: str):
    confidences = np.maximum(probs, 1 - probs)
    predicted   = (probs >= 0.5).astype(int)
    correct     = (predicted == actuals).astype(int)
    n_total     = len(correct)
    overall_acc = correct.mean()
    edge        = overall_acc - 0.50

    sign = '\u25B2' if edge >= 0 else '\u25BC'
    print(f"\n  [{horizon.upper()}]  n={n_total}  "
          f"overall accuracy={overall_acc:.1%}  "
          f"{sign} {abs(edge):.1%} edge vs random")

    pct_high = (confidences >= 0.60).mean()
    print(f"          confidence \u22650.60 on {pct_high:.0%} of predictions")

    bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    labels = ['0.50\u20130.55', '0.55\u20130.60', '0.60\u20130.65',
              '0.65\u20130.70', '0.70\u20130.75', '0.75\u20130.80', '0.80+   ']

    print(f"\n  {'Confidence':>11}  {'n':>5}  {'Accuracy':>8}  {'Edge':>8}  Signal")
    print("  " + "\u2500" * 55)

    any_bucket = False
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        mask = (confidences >= lo) & (confidences < hi)
        n = int(mask.sum())
        if n < 5:
            continue
        any_bucket = True
        acc  = float(correct[mask].mean())
        edge = acc - 0.50
        bar  = '\u2588' * int(abs(edge) * 100) if abs(edge) >= 0.01 else '\u00B7'
        sign = '+' if edge >= 0 else ''
        print(f"  {label}  {n:>5}  {acc:>8.1%}  {sign}{edge:.1%}     {bar}")

    if not any_bucket:
        print("  (all predictions in 0.50\u20130.55 bucket \u2014 model sees no signal)")
        print("  \u2192 model is not producing meaningful probabilities")


def predict_with_loaded_model(model: CryptoPredictionModel, df: pd.DataFrame, horizon: str):
    """
    Run the loaded production model over every row of df.
    Returns (probs, actuals) arrays.
    """
    target_col = f'target_direction_{horizon}'
    clean_df = df.dropna(subset=[target_col]).copy()
    if clean_df.empty:
        return np.array([]), np.array([])

    feature_cols = model.feature_columns
    if not feature_cols:
        raise ValueError("Model has no saved feature_columns")

    missing = [c for c in feature_cols if c not in clean_df.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing, filling with 0")
        for c in missing:
            clean_df[c] = 0.0

    X = clean_df[feature_cols].fillna(0)
    y = clean_df[target_col]
    if y.dtype == bool or y.dtype == object:
        y = y.astype(int)
    y_binary = (y > 0).astype(int)

    selector   = model.feature_selectors[f"{horizon}_selector"]
    classifier = model.models[f"{horizon}_xgb_classifier"]

    X_sel = pd.DataFrame(
        selector.transform(X),
        columns=X.columns[selector.get_support()],
        index=X.index,
    )
    # No StandardScaler step — XGBoost is scale-invariant; the classifier is a
    # CalibratedClassifierCV wrapper that returns calibrated probabilities directly.
    probs = classifier.predict_proba(X_sel)[:, 1]

    return probs, y_binary.values


def main():
    meta_path = 'models/production_models.json'
    if not os.path.exists(meta_path):
        print("\n  ERROR: No production models found.")
        print("  Run  python3 train_optimal_models.py --skip-backtest  first.")
        sys.exit(1)

    with open(meta_path) as f:
        production_models = json.load(f)

    print("\n" + "=" * 60)
    print("  PRODUCTION MODEL CALIBRATION CHECK")
    print("  Evaluates the ACTUAL deployed models on held-out data")
    print("=" * 60)

    collector = DataCollector()
    fe        = FeatureEngineer()

    for crypto in config.CRYPTOCURRENCIES:
        if crypto not in production_models:
            print(f"\n  {crypto.upper()}: no production model, skipping")
            continue

        print("\n" + "\u2500" * 60)
        print(f"  {crypto.upper()}")
        print("\u2500" * 60)

        for horizon in config.PREDICTION_INTERVALS:
            if horizon not in production_models[crypto]:
                print(f"\n  [{horizon.upper()}] no model, skipping")
                continue

            info       = production_models[crypto][horizon]
            model_path = info['model_path']

            if not os.path.exists(model_path):
                print(f"\n  [{horizon.upper()}] model file missing: {model_path}")
                continue

            model = CryptoPredictionModel(crypto, horizon)
            model.load_model(model_path)

            train_days = info.get('training_days', 180)
            hist = model.training_history
            if hist and 'timestamp' in hist[-1]:
                trained_at = hist[-1]['timestamp']
                if isinstance(trained_at, str):
                    trained_at = pd.to_datetime(trained_at)
            else:
                mtime = os.path.getmtime(model_path)
                trained_at = datetime.utcfromtimestamp(mtime)

            train_start = trained_at - pd.Timedelta(days=train_days)

            print(f"\n  Model trained : {trained_at.strftime('%Y-%m-%d %H:%M')} UTC")
            print(f"  Training window: {train_start.strftime('%Y-%m-%d')} \u2192 "
                  f"{trained_at.strftime('%Y-%m-%d')} ({train_days}d)")

            clf = model.models.get(f"{horizon}_xgb_classifier")
            clf_type = type(clf).__name__
            print(f"  Model type    : {clf_type}")

            extra_days = 120
            fetch_days = train_days + extra_days
            print(f"  Fetching {fetch_days}d of data\u2026", end='', flush=True)
            raw = fetch_data_for_crypto(collector, crypto, days=fetch_days)
            df_15m = raw['15m']
            df_1m  = raw['1m']
            if df_15m.empty:
                print(" no data available.")
                continue
            n_1m = len(df_1m) if not df_1m.empty else 0
            print(f" {len(df_15m)} 15m, {n_1m} 1m bars")

            print("  Preparing features\u2026", end='', flush=True)
            df = prepare_features_for_crypto(fe, df_15m, df_1m)
            df = df.sort_values('datetime').reset_index(drop=True)
            print(f" {len(df)} bars")

            test_df = df[df['datetime'] < train_start].copy()
            if len(test_df) < 50:
                print(f"  Only {len(test_df)} pre-training bars available.")
                print("  \u2192 Not enough held-out data for calibration check.")
                continue
            days_out = (test_df['datetime'].iloc[-1] - test_df['datetime'].iloc[0]).total_seconds() / 86400
            print(f"  Held-out: {len(test_df)} bars ({days_out:.1f} days before training window)")

            try:
                probs, actuals = predict_with_loaded_model(model, test_df, horizon)
                if len(probs) < 20:
                    print(f"  Too few valid test samples ({len(probs)})")
                    continue
                print(f"  [PRE-TRAINING evaluation]")
                calibration_report(probs, actuals, horizon)
            except Exception as e:
                print(f"  Evaluation error: {e}")

    print("\n" + "=" * 60)
    print("  HOW TO READ THIS:")
    print("  \u2022 Overall accuracy >52% sustained = genuine edge")
    print("  \u2022 Accuracy rising with confidence = well-calibrated model")
    print("  \u2022 All predictions in 0.50\u20130.55 = model has no signal")
    print("  \u2022 These results reflect the ACTUAL production models")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
