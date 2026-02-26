"""
Feature correlation analysis: which existing features correlate most with
the NEW Polymarket-style target (next candle green/red) vs the OLD target
(next close > current close)?
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, '/home/buidl-guidl/price-ai')

from data_collector import DataCollector, initialize_database
from feature_engineering import FeatureEngineer

# ── 1. Fetch data and prepare features ──────────────────────────────────────
print("=" * 90)
print("FEATURE CORRELATION ANALYSIS: NEW (green/red candle) vs OLD (price up/down) TARGET")
print("=" * 90)

initialize_database()
collector = DataCollector()
fe = FeatureEngineer()

print("\nFetching 90 days of BTC 15m data from Binance...")
btc_data = collector.get_crypto_data('bitcoin', days=90)
print(f"  Raw data shape: {btc_data.shape}")
print(f"  Columns: {list(btc_data.columns)}")
print(f"  Date range: {btc_data['datetime'].min()} to {btc_data['datetime'].max()}")

print("\nRunning feature engineering pipeline...")
features = fe.prepare_features(btc_data, external_data={})
print(f"  Features shape: {features.shape}")

# ── 2. Create targets ───────────────────────────────────────────────────────
# NEW target: next candle green (close >= open) = 1, red = 0
next_close = features['price'].shift(-1)
next_open = features['open'].shift(-1)
new_target = (next_close >= next_open).astype(float)
new_target[next_close.isna()] = np.nan

# OLD target: next close > current close = 1, else 0
old_target = (features['price'].shift(-1) > features['price']).astype(float)
old_target[next_close.isna()] = np.nan

print(f"\nNEW target (next candle green): {new_target.sum():.0f} green / {(new_target == 0).sum():.0f} red  "
      f"({new_target.mean():.3f} green rate)")
print(f"OLD target (next close > now):   {old_target.sum():.0f} up   / {(old_target == 0).sum():.0f} down "
      f"({old_target.mean():.3f} up rate)")

# Agreement between targets
both_valid = new_target.notna() & old_target.notna()
agreement = (new_target[both_valid] == old_target[both_valid]).mean()
print(f"\nTarget agreement rate: {agreement:.3f}  (how often both targets agree)")

# ── 3. Identify feature columns ─────────────────────────────────────────────
exclude_cols = {
    'datetime', 'crypto',
    'target_direction_15m', 'target_datetime_15m',
    # Also exclude raw OHLC since they're absolute prices, not features
    'price', 'open', 'high', 'low',
}

feature_cols = [c for c in features.columns
                if c not in exclude_cols
                and features[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

print(f"\nTotal feature columns to analyze: {len(feature_cols)}")

# ── 4. Compute correlations ────────────────────────────────────────────────
results = []

for col in feature_cols:
    series = features[col]

    # For NEW target
    valid_new = series.notna() & new_target.notna() & np.isfinite(series)
    if valid_new.sum() > 30:
        x = series[valid_new].values
        y = new_target[valid_new].values
        # Point-biserial correlation (equivalent to Pearson with binary y)
        corr_new, pval_new = stats.pointbiserialr(y, x)
        # Also Spearman
        spearman_new, spearman_pval_new = stats.spearmanr(x, y)
    else:
        corr_new, pval_new = np.nan, np.nan
        spearman_new, spearman_pval_new = np.nan, np.nan

    # For OLD target
    valid_old = series.notna() & old_target.notna() & np.isfinite(series)
    if valid_old.sum() > 30:
        x = series[valid_old].values
        y = old_target[valid_old].values
        corr_old, pval_old = stats.pointbiserialr(y, x)
        spearman_old, spearman_pval_old = stats.spearmanr(x, y)
    else:
        corr_old, pval_old = np.nan, np.nan
        spearman_old, spearman_pval_old = np.nan, np.nan

    results.append({
        'feature': col,
        'corr_NEW': corr_new,
        'pval_NEW': pval_new,
        'spearman_NEW': spearman_new,
        'sp_pval_NEW': spearman_pval_new,
        'corr_OLD': corr_old,
        'pval_OLD': pval_old,
        'spearman_OLD': spearman_old,
        'sp_pval_OLD': spearman_pval_old,
    })

df_results = pd.DataFrame(results)
df_results['abs_corr_NEW'] = df_results['corr_NEW'].abs()
df_results['abs_corr_OLD'] = df_results['corr_OLD'].abs()
df_results['abs_spearman_NEW'] = df_results['spearman_NEW'].abs()
df_results['abs_spearman_OLD'] = df_results['spearman_OLD'].abs()
df_results['sig_NEW'] = df_results['pval_NEW'] < 0.05
df_results['sig_OLD'] = df_results['pval_OLD'] < 0.05
df_results['sig_sp_NEW'] = df_results['sp_pval_NEW'] < 0.05
df_results['sig_sp_OLD'] = df_results['sp_pval_OLD'] < 0.05

# ── 5. Print top 50 features for NEW target ─────────────────────────────────
print("\n" + "=" * 90)
print("TOP 50 FEATURES BY |CORRELATION| WITH NEW TARGET (next candle green/red)")
print("=" * 90)
top_new = df_results.sort_values('abs_corr_NEW', ascending=False).head(50)
print(f"\n{'Rank':<5} {'Feature':<40} {'Corr':<9} {'p-value':<12} {'Sig?':<5} {'Spearman':<9}")
print("-" * 90)
for i, (_, row) in enumerate(top_new.iterrows(), 1):
    sig = "***" if row['pval_NEW'] < 0.001 else ("**" if row['pval_NEW'] < 0.01 else ("*" if row['pval_NEW'] < 0.05 else ""))
    print(f"{i:<5} {row['feature']:<40} {row['corr_NEW']:>+.5f}  {row['pval_NEW']:<12.2e} {sig:<5} {row['spearman_NEW']:>+.5f}")

# ── 6. Print top 50 features for OLD target ──────────────────────────────────
print("\n" + "=" * 90)
print("TOP 50 FEATURES BY |CORRELATION| WITH OLD TARGET (next close > current close)")
print("=" * 90)
top_old = df_results.sort_values('abs_corr_OLD', ascending=False).head(50)
print(f"\n{'Rank':<5} {'Feature':<40} {'Corr':<9} {'p-value':<12} {'Sig?':<5} {'Spearman':<9}")
print("-" * 90)
for i, (_, row) in enumerate(top_old.iterrows(), 1):
    sig = "***" if row['pval_OLD'] < 0.001 else ("**" if row['pval_OLD'] < 0.01 else ("*" if row['pval_OLD'] < 0.05 else ""))
    print(f"{i:<5} {row['feature']:<40} {row['corr_OLD']:>+.5f}  {row['pval_OLD']:<12.2e} {sig:<5} {row['spearman_OLD']:>+.5f}")

# ── 7. Comparison table ─────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("COMPARISON TABLE: Feature correlations with NEW vs OLD target")
print("=" * 110)

# Union of top 50 from each
top_features = set(top_new['feature'].tolist()) | set(top_old['feature'].tolist())
df_compare = df_results[df_results['feature'].isin(top_features)].copy()
df_compare['max_abs'] = df_compare[['abs_corr_NEW', 'abs_corr_OLD']].max(axis=1)
df_compare = df_compare.sort_values('max_abs', ascending=False)

# Determine which target each feature is more useful for
def better_for(row):
    if row['abs_corr_NEW'] > row['abs_corr_OLD'] * 1.2:
        return "NEW >>>"
    elif row['abs_corr_NEW'] > row['abs_corr_OLD']:
        return "NEW >"
    elif row['abs_corr_OLD'] > row['abs_corr_NEW'] * 1.2:
        return "<<< OLD"
    elif row['abs_corr_OLD'] > row['abs_corr_NEW']:
        return "< OLD"
    else:
        return "EQUAL"

df_compare['better_for'] = df_compare.apply(better_for, axis=1)

print(f"\n{'Feature':<40} {'Corr NEW':>9} {'Sig':>4} {'Corr OLD':>9} {'Sig':>4} {'Better For':>10}")
print("-" * 110)
for _, row in df_compare.iterrows():
    sig_n = "*" if row['sig_NEW'] else ""
    sig_o = "*" if row['sig_OLD'] else ""
    print(f"{row['feature']:<40} {row['corr_NEW']:>+.5f} {sig_n:>4} {row['corr_OLD']:>+.5f} {sig_o:>4}  {row['better_for']:>10}")

# ── 8. Summary statistics ───────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SUMMARY STATISTICS")
print("=" * 90)

n_sig_new = df_results['sig_NEW'].sum()
n_sig_old = df_results['sig_OLD'].sum()
n_total = len(df_results)

print(f"\nTotal features analyzed: {n_total}")
print(f"Features with significant correlation (p<0.05):")
print(f"  NEW target (green/red candle): {n_sig_new} ({n_sig_new/n_total*100:.1f}%)")
print(f"  OLD target (price up/down):    {n_sig_old} ({n_sig_old/n_total*100:.1f}%)")

print(f"\nMean |correlation| across all features:")
print(f"  NEW target: {df_results['abs_corr_NEW'].mean():.5f}")
print(f"  OLD target: {df_results['abs_corr_OLD'].mean():.5f}")

print(f"\nMax |correlation| across all features:")
print(f"  NEW target: {df_results['abs_corr_NEW'].max():.5f}  ({df_results.loc[df_results['abs_corr_NEW'].idxmax(), 'feature']})")
print(f"  OLD target: {df_results['abs_corr_OLD'].max():.5f}  ({df_results.loc[df_results['abs_corr_OLD'].idxmax(), 'feature']})")

# Features that are significant for NEW but not OLD (uniquely useful for new target)
new_only = df_results[df_results['sig_NEW'] & ~df_results['sig_OLD']].sort_values('abs_corr_NEW', ascending=False)
print(f"\nFeatures significant ONLY for NEW target ({len(new_only)}):")
for _, row in new_only.head(20).iterrows():
    print(f"  {row['feature']:<40} corr_NEW={row['corr_NEW']:>+.5f}  corr_OLD={row['corr_OLD']:>+.5f}")

# Features that are significant for OLD but not NEW
old_only = df_results[~df_results['sig_NEW'] & df_results['sig_OLD']].sort_values('abs_corr_OLD', ascending=False)
print(f"\nFeatures significant ONLY for OLD target ({len(old_only)}):")
for _, row in old_only.head(20).iterrows():
    print(f"  {row['feature']:<40} corr_NEW={row['corr_NEW']:>+.5f}  corr_OLD={row['corr_OLD']:>+.5f}")

# Features significant for BOTH
both_sig = df_results[df_results['sig_NEW'] & df_results['sig_OLD']].sort_values('abs_corr_NEW', ascending=False)
print(f"\nFeatures significant for BOTH targets ({len(both_sig)}):")
for _, row in both_sig.head(20).iterrows():
    print(f"  {row['feature']:<40} corr_NEW={row['corr_NEW']:>+.5f}  corr_OLD={row['corr_OLD']:>+.5f}")

# ── 9. Rank comparison: how different are the orderings? ─────────────────────
print("\n" + "=" * 90)
print("RANK COMPARISON: Do the two targets prefer different features?")
print("=" * 90)
df_results['rank_NEW'] = df_results['abs_corr_NEW'].rank(ascending=False)
df_results['rank_OLD'] = df_results['abs_corr_OLD'].rank(ascending=False)
df_results['rank_diff'] = (df_results['rank_NEW'] - df_results['rank_OLD']).abs()

rank_corr, rank_pval = stats.spearmanr(df_results['abs_corr_NEW'].fillna(0),
                                         df_results['abs_corr_OLD'].fillna(0))
print(f"\nSpearman correlation between feature importance rankings: {rank_corr:.3f} (p={rank_pval:.2e})")
print("  1.0 = identical rankings, 0.0 = no relationship, -1.0 = reversed")

biggest_rank_changes = df_results.sort_values('rank_diff', ascending=False).head(20)
print(f"\nFeatures with BIGGEST rank difference (most differently ranked):")
print(f"{'Feature':<40} {'Rank NEW':>9} {'Rank OLD':>9} {'Diff':>6}")
print("-" * 70)
for _, row in biggest_rank_changes.iterrows():
    print(f"{row['feature']:<40} {row['rank_NEW']:>9.0f} {row['rank_OLD']:>9.0f} {row['rank_diff']:>6.0f}")

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
