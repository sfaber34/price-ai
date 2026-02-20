#!/usr/bin/env python3
"""
Calibration diagnostic: trains on historical data, evaluates on a held-out period.

Gives thousands of out-of-sample predictions in minutes rather than waiting
weeks for live evaluation. Shows whether:
  1. The model has any directional edge at all
  2. Confidence correlates with accuracy (calibration works)
  3. Which horizons are most predictable

Usage:
    python3 evaluate_calibration.py
"""
import sys
import numpy as np
import pandas as pd
import logging

logging.disable(logging.CRITICAL)  # suppress training noise

import config
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from ml_predictor import CryptoPredictionModel

TRAIN_DAYS = 180   # days used to train the diagnostic model
TEST_DAYS  = 90    # held-out period — model never sees this during training


def calibration_report(probs: np.ndarray, actuals: np.ndarray, horizon: str):
    confidences = np.maximum(probs, 1 - probs)
    predicted   = (probs > 0.5).astype(int)
    correct     = (predicted == actuals).astype(int)
    n_total     = len(correct)
    overall_acc = correct.mean()
    edge        = overall_acc - 0.50

    sign = '▲' if edge >= 0 else '▼'
    print(f"\n  [{horizon.upper()}]  n={n_total}  "
          f"overall accuracy={overall_acc:.1%}  "
          f"{sign} {abs(edge):.1%} edge vs random")

    # Confidence distribution — how often is the model decisive?
    pct_high = (confidences >= 0.60).mean()
    print(f"          confidence ≥0.60 on {pct_high:.0%} of predictions")

    bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
    labels = ['0.50–0.55', '0.55–0.60', '0.60–0.65',
              '0.65–0.70', '0.70–0.75', '0.75–0.80', '0.80+   ']

    print(f"\n  {'Confidence':>11}  {'n':>5}  {'Accuracy':>8}  {'Edge':>8}  Signal")
    print(f"  {'─'*55}")

    any_bucket = False
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        mask = (confidences >= lo) & (confidences < hi)
        n = int(mask.sum())
        if n < 5:
            continue
        any_bucket = True
        acc  = float(correct[mask].mean())
        edge = acc - 0.50
        bar  = '█' * int(abs(edge) * 100) if abs(edge) >= 0.01 else '·'
        sign = '+' if edge >= 0 else ''
        print(f"  {label}  {n:>5}  {acc:>8.1%}  {sign}{edge:.1%}     {bar}")

    if not any_bucket:
        print("  (all predictions in 0.50–0.55 bucket — model sees no signal)")
        print("  → features are not predictive for this horizon")


def main():
    print("\n" + "="*60)
    print("  CALIBRATION DIAGNOSTIC")
    print(f"  Train window : {TRAIN_DAYS} days")
    print(f"  Test window  : {TEST_DAYS} days (held-out, model never sees this)")
    print("  Goal         : does confidence correlate with accuracy?")
    print("="*60)

    collector  = DataCollector()
    fe         = FeatureEngineer()
    total_days = TRAIN_DAYS + TEST_DAYS + 5

    for crypto in config.CRYPTOCURRENCIES:
        print(f"\n{'─'*60}")
        print(f"  {crypto.upper()}")
        print(f"{'─'*60}")

        raw = collector.get_crypto_data(crypto, days=total_days)
        if raw.empty:
            print("  No data available.")
            continue

        print("  Preparing features…", end='', flush=True)
        df = fe.prepare_features(raw)
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f" {len(df)} bars")

        n_train = TRAIN_DAYS * 24 * 4   # 15m bars per day
        n_test  = TEST_DAYS  * 24 * 4

        if len(df) < n_train + 200:
            print(f"  Insufficient data ({len(df)} bars, need ≥{n_train + 200})")
            continue

        train_df = df.iloc[:n_train].copy()
        test_df  = df.iloc[n_train: n_train + n_test].copy()
        print(f"  Train: {len(train_df)} bars  |  Test: {len(test_df)} bars")

        for horizon in config.PREDICTION_INTERVALS:
            target_col = f'target_direction_{horizon}'

            print(f"\n  Training {horizon}…", end='', flush=True)
            model = CryptoPredictionModel(crypto, horizon)
            try:
                model.train(train_df)
            except Exception as e:
                print(f" FAILED: {e}")
                continue
            print(" done", end='', flush=True)

            try:
                # Vectorised evaluation — no bar-by-bar loop needed
                X_test, y_test = model.prepare_data(test_df, target_col)
                if len(X_test) < 20:
                    print(f"\n  [{horizon}] Too few test samples ({len(X_test)})")
                    continue

                selector   = model.feature_selectors[f"{horizon}_selector"]
                scaler     = model.scalers[f"{horizon}_scaler"]
                classifier = model.models[f"{horizon}_xgb_classifier"]

                X_sel = pd.DataFrame(
                    selector.transform(X_test),
                    columns=X_test.columns[selector.get_support()],
                    index=X_test.index,
                )
                X_scaled = pd.DataFrame(
                    scaler.transform(X_sel),
                    columns=X_sel.columns,
                    index=X_sel.index,
                )

                probs   = classifier.predict_proba(X_scaled)[:, 1]
                actuals = (y_test > 0).astype(int).values

                calibration_report(probs, actuals, horizon)

            except Exception as e:
                print(f"\n  [{horizon}] Evaluation error: {e}")

    print("\n" + "="*60)
    print("  HOW TO READ THIS:")
    print("  • Overall accuracy >52% sustained = genuine edge")
    print("  • Accuracy rising with confidence = calibration works")
    print("  • All predictions in 0.50–0.55 = no signal, wrong features")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
