#!/usr/bin/env python3
"""
Calibration diagnostic: trains on historical data, evaluates on a held-out period.

Uses the SAME training_pipeline module as the production bot and trainer —
if you change how data is fetched, features are built, or models are trained,
it automatically takes effect here too.

Usage:
    python3 evaluate_calibration.py
"""
import numpy as np
import logging

logging.disable(logging.CRITICAL)  # suppress training noise

import config
from data_collector import DataCollector
from feature_engineering import FeatureEngineer
from training_pipeline import (
    fetch_data_for_crypto,
    prepare_features_for_crypto,
    train_model,
    predict_proba_batch,
)

TRAIN_DAYS = 180   # days used to train the diagnostic model
TEST_DAYS  = 90    # held-out period — model never sees this during training


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
    print(f"  {'\u2500'*55}")

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
        print("  \u2192 features are not predictive for this horizon")


def main():
    print("\n" + "="*60)
    print("  CALIBRATION DIAGNOSTIC")
    print(f"  Train window : {TRAIN_DAYS} days")
    print(f"  Test window  : {TEST_DAYS} days (held-out, model never sees this)")
    print("  Goal         : does confidence correlate with accuracy?")
    print("="*60)

    collector = DataCollector()
    fe        = FeatureEngineer()
    total_days = TRAIN_DAYS + TEST_DAYS + 5

    for crypto in config.CRYPTOCURRENCIES:
        print(f"\n{'\u2500'*60}")
        print(f"  {crypto.upper()}")
        print(f"{'\u2500'*60}")

        # Fetch data — same function the bot and trainer use
        print("  Fetching data\u2026", end='', flush=True)
        raw = fetch_data_for_crypto(collector, crypto, days=total_days)
        df_15m = raw['15m']
        df_1m  = raw['1m']
        if df_15m.empty:
            print(" no data available.")
            continue
        print(f" {len(df_15m)} 15m bars, {len(df_1m)} 1m bars")

        # Prepare features — same function the bot and trainer use
        print("  Preparing features\u2026", end='', flush=True)
        df = prepare_features_for_crypto(fe, df_15m, df_1m)
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f" {len(df)} bars, {df.shape[1]} columns")

        n_train = TRAIN_DAYS * 24 * 4
        n_test  = TEST_DAYS  * 24 * 4

        if len(df) < n_train + 200:
            print(f"  Insufficient data ({len(df)} bars, need \u2265{n_train + 200})")
            continue

        train_df = df.iloc[:n_train].copy()
        test_df  = df.iloc[n_train: n_train + n_test].copy()
        print(f"  Train: {len(train_df)} bars  |  Test: {len(test_df)} bars")

        for horizon in config.PREDICTION_INTERVALS:
            print(f"\n  Training {horizon}\u2026", end='', flush=True)
            try:
                # Train — same function the bot and trainer use
                model = train_model(crypto, horizon, train_df)
            except Exception as e:
                print(f" FAILED: {e}")
                continue
            print(" done", end='', flush=True)

            try:
                # Predict — same function used everywhere
                probs, actuals = predict_proba_batch(model, test_df, horizon)
                if len(probs) < 20:
                    print(f"\n  [{horizon}] Too few test samples ({len(probs)})")
                    continue
                calibration_report(probs, actuals, horizon)
            except Exception as e:
                print(f"\n  [{horizon}] Evaluation error: {e}")

    print("\n" + "="*60)
    print("  HOW TO READ THIS:")
    print("  \u2022 Overall accuracy >52% sustained = genuine edge")
    print("  \u2022 Accuracy rising with confidence = calibration works")
    print("  \u2022 All predictions in 0.50\u20130.55 = no signal, wrong features")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
