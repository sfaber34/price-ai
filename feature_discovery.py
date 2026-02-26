#!/usr/bin/env python3
"""
Feature Discovery Script for 15-minute Bitcoin Next-Bar Green/Red Prediction.

Fetches 180 days of 15m data and 30 days of 1m data from Binance,
computes novel features across 8 categories, and measures correlation
with the target: whether the NEXT 15m bar is green (close >= open).
"""

import requests
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import time
import warnings
from datetime import datetime, timedelta, timezone
from collections import OrderedDict

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────

BASE = "https://api.binance.com"

def fetch_klines(symbol, interval, start_ms, end_ms, limit=1000):
    """Fetch klines from Binance with pagination."""
    url = f"{BASE}/api/v3/klines"
    all_data = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", 10))
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise
        data = r.json()
        if not data:
            break
        all_data.extend(data)
        current_start = data[-1][0] + 1  # next ms after last candle open
        time.sleep(0.15)  # respect rate limits

    return all_data


def build_df(raw, prefix=""):
    """Convert raw Binance klines to DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def fetch_funding_rates(symbol="BTCUSDT", limit_days=180):
    """Fetch funding rate history from Binance futures."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_data = []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - limit_days * 86400 * 1000

    current_start = start_ms
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "limit": 1000,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(10)
                continue
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            all_data.extend(data)
            current_start = data[-1]["fundingTime"] + 1
            time.sleep(0.2)
        except Exception as e:
            print(f"  Funding rate fetch error: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    return df


# ─────────────────────────────────────────────────────────────
# FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_features_15m_only(df15):
    """Compute features using only 15m data (for the full 180-day window)."""
    feat = pd.DataFrame(index=df15.index)

    close = df15["close"]
    open_ = df15["open"]
    high = df15["high"]
    low = df15["low"]
    volume = df15["volume"]
    taker_buy = df15["taker_buy_base"]
    trades = df15["trades"]

    # ── Category 3: Volatility & Range ──
    bar_range = high - low
    bar_range_pct = bar_range / close
    feat["range_pct"] = bar_range_pct
    feat["range_vs_avg_10"] = bar_range / bar_range.rolling(10).mean()
    feat["range_vs_avg_20"] = bar_range / bar_range.rolling(20).mean()
    feat["range_vs_avg_50"] = bar_range / bar_range.rolling(50).mean()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    atr_50 = tr.rolling(50).mean()
    feat["atr_ratio_14_50"] = atr_14 / atr_50

    # Bollinger squeeze
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    bb_width = (2 * std_20) / sma_20
    feat["bb_width"] = bb_width
    feat["bb_squeeze"] = bb_width / bb_width.rolling(50).mean()
    feat["bb_position"] = (close - (sma_20 - std_20)) / (2 * std_20)  # 0=lower, 1=upper

    # Realized vol (std of returns)
    ret = close.pct_change()
    feat["realized_vol_10"] = ret.rolling(10).std()
    feat["realized_vol_20"] = ret.rolling(20).std()
    feat["vol_ratio_10_50"] = ret.rolling(10).std() / ret.rolling(50).std()

    # ── Category 4: Price Level Features ──
    feat["dist_round_100"] = (close % 100) / 100  # position within $100 band
    feat["dist_round_500"] = (close % 500) / 500
    feat["dist_round_1000"] = (close % 1000) / 1000

    # Distance from 24h high/low (96 bars)
    h24_high = high.rolling(96).max()
    h24_low = low.rolling(96).min()
    h24_range = h24_high - h24_low
    feat["dist_24h_high_pct"] = (h24_high - close) / h24_range.replace(0, np.nan)
    feat["dist_24h_low_pct"] = (close - h24_low) / h24_range.replace(0, np.nan)

    # Distance from 1h high/low (4 bars)
    h1_high = high.rolling(4).max()
    h1_low = low.rolling(4).min()
    h1_range = h1_high - h1_low
    feat["dist_1h_high_pct"] = (h1_high - close) / h1_range.replace(0, np.nan)
    feat["dist_1h_low_pct"] = (close - h1_low) / h1_range.replace(0, np.nan)

    # Session VWAP (rolling 96 bars ~ 24h)
    cum_vol = volume.rolling(96).sum()
    cum_vp = (close * volume).rolling(96).sum()
    vwap_24h = cum_vp / cum_vol
    feat["dist_vwap_24h_pct"] = (close - vwap_24h) / close

    # ── Category 5: Cross-timeframe patterns ──
    # Hourly direction (look back 4 bars)
    close_4ago = close.shift(3)
    feat["hourly_direction"] = (close > close_4ago).astype(float)

    # 4h direction (look back 16 bars)
    close_16ago = close.shift(15)
    feat["4h_direction"] = (close > close_16ago).astype(float)

    # Current bar direction
    feat["current_bar_green"] = (close >= open_).astype(float)

    # Multi-timeframe alignment
    bar_green = (close >= open_).astype(float)
    feat["alignment_4bar"] = bar_green.rolling(4).mean()  # fraction of last 4 bars green
    feat["alignment_8bar"] = bar_green.rolling(8).mean()
    feat["alignment_16bar"] = bar_green.rolling(16).mean()

    # ── Category 6: Mean Reversion Signals ──
    # Consecutive same-color candles
    green = (close >= open_).astype(int)
    streak = pd.Series(0, index=df15.index, dtype=int)
    for i in range(1, len(green)):
        if green.iloc[i] == green.iloc[i-1]:
            streak.iloc[i] = streak.iloc[i-1] + (1 if green.iloc[i] == 1 else -1)
        else:
            streak.iloc[i] = 1 if green.iloc[i] == 1 else -1
    feat["streak_length"] = streak

    # Simpler streak: count consecutive green or red
    feat["consecutive_green"] = 0.0
    feat["consecutive_red"] = 0.0
    cg = np.zeros(len(green))
    cr = np.zeros(len(green))
    for i in range(1, len(green)):
        if green.iloc[i] == 1:
            cg[i] = cg[i-1] + 1
            cr[i] = 0
        else:
            cr[i] = cr[i-1] + 1
            cg[i] = 0
    feat["consecutive_green"] = cg
    feat["consecutive_red"] = cr

    # Cumulative return over last N bars
    feat["cum_return_4"] = close.pct_change(4)
    feat["cum_return_8"] = close.pct_change(8)
    feat["cum_return_16"] = close.pct_change(16)
    feat["cum_return_32"] = close.pct_change(32)

    # Distance from EMAs
    ema_8 = compute_ema(close, 8)
    ema_21 = compute_ema(close, 21)
    ema_50 = compute_ema(close, 50)
    feat["dist_ema_8_pct"] = (close - ema_8) / close
    feat["dist_ema_21_pct"] = (close - ema_21) / close
    feat["dist_ema_50_pct"] = (close - ema_50) / close

    # Z-score of recent returns
    feat["return_zscore_20"] = (ret - ret.rolling(20).mean()) / ret.rolling(20).std()
    feat["return_zscore_50"] = (ret - ret.rolling(50).mean()) / ret.rolling(50).std()

    # RSI
    rsi_14 = compute_rsi(close, 14)
    feat["rsi_14"] = rsi_14
    feat["rsi_7"] = compute_rsi(close, 7)
    feat["rsi_distance_50"] = rsi_14 - 50  # how far from neutral

    # ── Category 7: Volume Patterns ──
    feat["vol_ratio_20"] = volume / volume.rolling(20).mean()
    feat["vol_ratio_50"] = volume / volume.rolling(50).mean()
    feat["unusual_volume"] = (volume > volume.rolling(20).mean() + 2 * volume.rolling(20).std()).astype(float)

    # Buy volume percentage
    buy_pct = taker_buy / volume.replace(0, np.nan)
    feat["buy_vol_pct"] = buy_pct
    feat["buy_vol_pct_ma5"] = buy_pct.rolling(5).mean()
    feat["buy_vol_pct_trend"] = buy_pct - buy_pct.rolling(10).mean()

    # Volume delta (taker buy - taker sell)
    sell_vol = volume - taker_buy
    vol_delta = taker_buy - sell_vol
    feat["vol_delta_pct"] = vol_delta / volume.replace(0, np.nan)
    feat["vol_delta_pct_ma5"] = feat["vol_delta_pct"].rolling(5).mean()
    feat["vol_delta_pct_ma10"] = feat["vol_delta_pct"].rolling(10).mean()

    # Trades per unit volume (proxy for order size)
    feat["trades_per_vol"] = trades / volume.replace(0, np.nan)
    feat["trades_per_vol_ratio"] = feat["trades_per_vol"] / feat["trades_per_vol"].rolling(20).mean()

    # ── Category 8: Calendar / Regime ──
    ts = df15["open_time"]
    feat["hour_of_day"] = ts.dt.hour
    feat["minute_of_hour"] = ts.dt.minute
    feat["day_of_week"] = ts.dt.dayofweek
    feat["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)

    # Minutes since last hourly boundary
    feat["min_since_hour"] = ts.dt.minute

    # Near major market opens (UTC times)
    hour = ts.dt.hour
    minute = ts.dt.minute
    time_in_min = hour * 60 + minute

    # NYSE open ~14:30 UTC, London ~08:00 UTC, Tokyo ~00:00 UTC
    feat["near_nyse_open"] = ((time_in_min - 870).abs() <= 30).astype(float)  # 14:30
    feat["near_london_open"] = ((time_in_min - 480).abs() <= 30).astype(float)  # 08:00
    feat["near_tokyo_open"] = ((time_in_min <= 30) | ((1440 - time_in_min) <= 30)).astype(float)

    # Volatility regime
    vol_20 = ret.rolling(20).std()
    vol_long = ret.rolling(200).std()
    feat["vol_regime"] = vol_20 / vol_long

    # Body ratio
    body = (close - open_).abs()
    feat["body_ratio"] = body / bar_range.replace(0, np.nan)

    # Upper/lower wick ratios
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
    feat["upper_wick_ratio"] = upper_wick / bar_range.replace(0, np.nan)
    feat["lower_wick_ratio"] = lower_wick / bar_range.replace(0, np.nan)

    # Close position within bar
    feat["close_position"] = (close - low) / bar_range.replace(0, np.nan)

    # Momentum
    feat["return_1bar"] = ret
    feat["return_2bar"] = close.pct_change(2)
    feat["return_3bar"] = close.pct_change(3)

    # MACD-like
    ema_12 = compute_ema(close, 12)
    ema_26 = compute_ema(close, 26)
    macd = ema_12 - ema_26
    signal = compute_ema(macd, 9)
    feat["macd_hist"] = macd - signal
    feat["macd_hist_change"] = feat["macd_hist"].diff()

    return feat


def compute_microstructure_features(df15, df1m):
    """
    Compute microstructure features from 1-minute data, aligned to 15m bars.
    Only uses data available at bar close (no future leakage).
    """
    # We need to map each 15m bar to its constituent 1m bars
    feat = pd.DataFrame(index=df15.index)

    # Pre-compute 1m returns and metrics
    df1m = df1m.copy()
    df1m["return"] = df1m["close"].pct_change()
    df1m["sell_vol"] = df1m["volume"] - df1m["taker_buy_base"]
    df1m["vol_delta"] = df1m["taker_buy_base"] - df1m["sell_vol"]
    df1m["buy_pct"] = df1m["taker_buy_base"] / df1m["volume"].replace(0, np.nan)
    df1m["mid"] = (df1m["high"] + df1m["low"]) / 2

    # Also compute 1m RSI
    df1m["rsi_14"] = compute_rsi(df1m["close"], 14)

    # For each 15m bar, find the matching 1m bars
    # 15m bar open_time to close_time contains 15 one-minute bars
    results = {
        "cvd_1min": [], "cvd_3min": [], "cvd_5min": [], "cvd_10min": [], "cvd_15min": [],
        "vol_accel": [],
        "imbalance_1min": [], "imbalance_3min": [], "imbalance_5min": [],
        "micro_mom_1min": [], "micro_mom_2min": [], "micro_mom_3min": [], "micro_mom_5min": [],
        "vwap_position": [],
        "rsi_1m_at_close": [],
        "roc_5min": [],
        "slope_5min": [],
        "vol_first_half_vs_second": [],
        "realized_vol_1m": [],
    }

    # Build a lookup: for each 15m bar, get its 1m bars
    df1m_indexed = df1m.set_index("open_time").sort_index()

    count = 0
    for idx, row in df15.iterrows():
        bar_start = row["open_time"]
        bar_end = row["close_time"]

        # Get 1m bars within this 15m window
        mask = (df1m_indexed.index >= bar_start) & (df1m_indexed.index <= bar_end)
        m1 = df1m_indexed.loc[mask]

        if len(m1) < 10:
            for k in results:
                results[k].append(np.nan)
            continue

        # CVD (cumulative volume delta) in last N minutes
        for n, key in [(1, "cvd_1min"), (3, "cvd_3min"), (5, "cvd_5min"),
                       (10, "cvd_10min"), (15, "cvd_15min")]:
            tail = m1.tail(n)
            cvd = tail["vol_delta"].sum()
            # Normalize by total volume
            total_v = tail["volume"].sum()
            results[key].append(cvd / total_v if total_v > 0 else 0)

        # Volume acceleration: compare volume in last 5 min vs first 5 min, weighted by delta sign
        first5 = m1.head(5)
        last5 = m1.tail(5)
        buy_accel = last5["taker_buy_base"].sum() - first5["taker_buy_base"].sum()
        sell_accel = last5["sell_vol"].sum() - first5["sell_vol"].sum()
        total = m1["volume"].sum()
        results["vol_accel"].append((buy_accel - sell_accel) / total if total > 0 else 0)

        # Trade imbalance ratio over last N minutes
        for n, key in [(1, "imbalance_1min"), (3, "imbalance_3min"), (5, "imbalance_5min")]:
            tail = m1.tail(n)
            tv = tail["volume"].sum()
            results[key].append(tail["taker_buy_base"].sum() / tv if tv > 0 else 0.5)

        # Micro-momentum: price change in last N minutes
        closes = m1["close"].values
        for n, key in [(1, "micro_mom_1min"), (2, "micro_mom_2min"),
                       (3, "micro_mom_3min"), (5, "micro_mom_5min")]:
            if len(closes) > n:
                mom = (closes[-1] - closes[-1 - n]) / closes[-1 - n]
            else:
                mom = 0
            results[key].append(mom)

        # VWAP position within bar
        vwap = (m1["close"] * m1["volume"]).sum() / m1["volume"].sum() if m1["volume"].sum() > 0 else m1["close"].mean()
        bar_range = m1["high"].max() - m1["low"].min()
        if bar_range > 0:
            results["vwap_position"].append((vwap - m1["low"].min()) / bar_range)
        else:
            results["vwap_position"].append(0.5)

        # 1m RSI at bar close
        results["rsi_1m_at_close"].append(m1["rsi_14"].iloc[-1] if not np.isnan(m1["rsi_14"].iloc[-1]) else 50)

        # 5-minute rate of change
        if len(closes) >= 6:
            results["roc_5min"].append((closes[-1] - closes[-6]) / closes[-6])
        else:
            results["roc_5min"].append(0)

        # Slope of 1m prices in last 5 minutes (linear regression)
        if len(closes) >= 5:
            y = closes[-5:]
            x = np.arange(5)
            slope = np.polyfit(x, y, 1)[0]
            results["slope_5min"].append(slope / closes[-1])  # normalize
        else:
            results["slope_5min"].append(0)

        # Volume first half vs second half
        mid_point = len(m1) // 2
        vol_first = m1.iloc[:mid_point]["volume"].sum()
        vol_second = m1.iloc[mid_point:]["volume"].sum()
        results["vol_first_half_vs_second"].append(
            vol_second / vol_first if vol_first > 0 else 1.0
        )

        # Realized volatility of 1m returns within bar
        m1_rets = m1["return"].dropna()
        results["realized_vol_1m"].append(m1_rets.std() if len(m1_rets) > 2 else 0)

        count += 1
        if count % 2000 == 0:
            print(f"  Processed {count}/{len(df15)} bars for microstructure features...")

    for k, v in results.items():
        feat[k] = v

    return feat


# ─────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_correlations(features_df, target, feature_names=None):
    """Compute Pearson and Spearman correlations with p-values."""
    results = []
    if feature_names is None:
        feature_names = features_df.columns.tolist()

    for fname in feature_names:
        col = features_df[fname]
        valid = col.notna() & target.notna() & np.isfinite(col)
        if valid.sum() < 100:
            continue

        x = col[valid].values
        y = target[valid].values

        # Remove any remaining infinities
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]

        if len(x) < 100:
            continue

        try:
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
        except Exception:
            continue

        results.append({
            "feature": fname,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "n_samples": len(x),
            "abs_pearson": abs(pr),
            "abs_spearman": abs(sr),
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    now_ms = int(time.time() * 1000)

    # ── Fetch 180 days of 15m data ──
    days_15m = 180
    start_15m = now_ms - days_15m * 86400 * 1000
    print(f"Fetching {days_15m} days of 15m BTCUSDT data...")
    raw_15m = fetch_klines("BTCUSDT", "15m", start_15m, now_ms)
    df15 = build_df(raw_15m)
    print(f"  Got {len(df15)} 15m candles ({df15['open_time'].min()} to {df15['open_time'].max()})")

    # ── Fetch 30+ days of 1m data ──
    days_1m = 35  # a bit extra for warmup
    start_1m = now_ms - days_1m * 86400 * 1000
    print(f"\nFetching {days_1m} days of 1m BTCUSDT data (this takes a while)...")
    raw_1m = fetch_klines("BTCUSDT", "1m", start_1m, now_ms)
    df1m = build_df(raw_1m)
    print(f"  Got {len(df1m)} 1m candles ({df1m['open_time'].min()} to {df1m['open_time'].max()})")

    # ── Fetch funding rates ──
    print("\nFetching funding rates...")
    df_funding = fetch_funding_rates("BTCUSDT", days_15m)
    print(f"  Got {len(df_funding)} funding rate entries")

    # ── Compute target: is NEXT bar green? ──
    df15["next_green"] = (df15["close"].shift(-1) >= df15["open"].shift(-1)).astype(float)
    # Drop last row (no next bar)
    df15 = df15.iloc[:-1].reset_index(drop=True)

    print(f"\nTarget distribution: {df15['next_green'].mean():.4f} green rate "
          f"({df15['next_green'].sum():.0f} green, {(1-df15['next_green']).sum():.0f} red)")

    # ── Compute 15m-only features (full 180 days) ──
    print("\nComputing 15m-based features...")
    feat_15m = compute_features_15m_only(df15)
    print(f"  Computed {len(feat_15m.columns)} features from 15m data")

    # ── Add funding rate features ──
    if len(df_funding) > 0:
        df_funding = df_funding.sort_values("fundingTime")
        # Forward-fill funding rate to each 15m bar
        df15["_ts"] = df15["open_time"]
        merged = pd.merge_asof(
            df15[["_ts"]].sort_values("_ts"),
            df_funding[["fundingTime", "fundingRate"]].rename(columns={"fundingTime": "_ts"}),
            on="_ts",
            direction="backward"
        )
        feat_15m["funding_rate"] = merged["fundingRate"].values
        feat_15m["funding_rate_sign"] = np.sign(merged["fundingRate"].values)
        feat_15m["funding_rate_abs"] = merged["fundingRate"].abs().values
        print(f"  Added funding rate features")

    # ── Compute microstructure features (1m data window) ──
    # Only for 15m bars that overlap with our 1m data
    min_1m_time = df1m["open_time"].min()
    max_1m_time = df1m["open_time"].max()
    mask_micro = (df15["open_time"] >= min_1m_time) & (df15["close_time"] <= max_1m_time)
    df15_micro = df15[mask_micro].reset_index(drop=True)
    print(f"\nComputing microstructure features for {len(df15_micro)} bars with 1m data overlap...")

    feat_micro = compute_microstructure_features(df15_micro, df1m)
    print(f"  Computed {len(feat_micro.columns)} microstructure features")

    # ──────────────────────────────────────────────────────
    # ANALYSIS 1: 15m-only features (180 days)
    # ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("ANALYSIS 1: 15m-BASED FEATURES (180 days, ~{} samples)".format(len(df15)))
    print("="*80)

    target_15m = df15["next_green"]
    corr_15m = analyze_correlations(feat_15m, target_15m)
    corr_15m = corr_15m.sort_values("abs_spearman", ascending=False)

    print(f"\n{'Feature':<35} {'Pearson r':>10} {'p-value':>12} {'Spearman r':>10} {'p-value':>12} {'N':>8}")
    print("-"*90)
    for _, row in corr_15m.head(60).iterrows():
        sig = ""
        if row["abs_spearman"] > 0.03 and row["spearman_p"] < 0.01:
            sig = " ***"
        elif row["abs_spearman"] > 0.02 and row["spearman_p"] < 0.05:
            sig = " **"
        elif row["spearman_p"] < 0.05:
            sig = " *"
        print(f"{row['feature']:<35} {row['pearson_r']:>10.5f} {row['pearson_p']:>12.2e} "
              f"{row['spearman_r']:>10.5f} {row['spearman_p']:>12.2e} {row['n_samples']:>8.0f}{sig}")

    # ──────────────────────────────────────────────────────
    # ANALYSIS 2: Microstructure features (30 days)
    # ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("ANALYSIS 2: MICROSTRUCTURE FEATURES (30 days, ~{} samples)".format(len(df15_micro)))
    print("="*80)

    target_micro = df15_micro["next_green"].reset_index(drop=True)
    corr_micro = analyze_correlations(feat_micro, target_micro)
    corr_micro = corr_micro.sort_values("abs_spearman", ascending=False)

    print(f"\n{'Feature':<35} {'Pearson r':>10} {'p-value':>12} {'Spearman r':>10} {'p-value':>12} {'N':>8}")
    print("-"*90)
    for _, row in corr_micro.iterrows():
        sig = ""
        if row["abs_spearman"] > 0.03 and row["spearman_p"] < 0.01:
            sig = " ***"
        elif row["abs_spearman"] > 0.02 and row["spearman_p"] < 0.05:
            sig = " **"
        elif row["spearman_p"] < 0.05:
            sig = " *"
        print(f"{row['feature']:<35} {row['pearson_r']:>10.5f} {row['pearson_p']:>12.2e} "
              f"{row['spearman_r']:>10.5f} {row['spearman_p']:>12.2e} {row['n_samples']:>8.0f}{sig}")

    # ──────────────────────────────────────────────────────
    # COMBINED: All features on the overlapping window
    # ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("COMBINED ANALYSIS: ALL FEATURES ON OVERLAPPING WINDOW")
    print("="*80)

    # Get 15m features for the micro window
    feat_15m_window = feat_15m.loc[mask_micro].reset_index(drop=True)
    all_feat = pd.concat([feat_15m_window, feat_micro], axis=1)

    # Remove duplicate columns if any
    all_feat = all_feat.loc[:, ~all_feat.columns.duplicated()]

    corr_all = analyze_correlations(all_feat, target_micro)
    corr_all = corr_all.sort_values("abs_spearman", ascending=False)

    # Highlight promising features
    print("\n*** PROMISING FEATURES (|Spearman r| > 0.03 AND p < 0.01) ***\n")
    promising = corr_all[(corr_all["abs_spearman"] > 0.03) & (corr_all["spearman_p"] < 0.01)]
    if len(promising) > 0:
        print(f"{'Feature':<35} {'Pearson r':>10} {'p-value':>12} {'Spearman r':>10} {'p-value':>12} {'N':>8}")
        print("-"*90)
        for _, row in promising.iterrows():
            print(f"{row['feature']:<35} {row['pearson_r']:>10.5f} {row['pearson_p']:>12.2e} "
                  f"{row['spearman_r']:>10.5f} {row['spearman_p']:>12.2e} {row['n_samples']:>8.0f}")
    else:
        print("  None found at this strict threshold.")

    print("\n*** MODERATELY PROMISING (|Spearman r| > 0.02 AND p < 0.05) ***\n")
    moderate = corr_all[(corr_all["abs_spearman"] > 0.02) & (corr_all["spearman_p"] < 0.05)]
    if len(moderate) > 0:
        print(f"{'Feature':<35} {'Pearson r':>10} {'p-value':>12} {'Spearman r':>10} {'p-value':>12} {'N':>8}")
        print("-"*90)
        for _, row in moderate.iterrows():
            print(f"{row['feature']:<35} {row['pearson_r']:>10.5f} {row['pearson_p']:>12.2e} "
                  f"{row['spearman_r']:>10.5f} {row['spearman_p']:>12.2e} {row['n_samples']:>8.0f}")
    else:
        print("  None found at this threshold either.")

    # ──────────────────────────────────────────────────────
    # TOP 20 + COMBINATION ANALYSIS
    # ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TOP 20 FEATURES BY ABSOLUTE SPEARMAN CORRELATION")
    print("="*80)

    top20 = corr_all.head(20)
    print(f"\n{'Rank':<6} {'Feature':<35} {'Spearman r':>10} {'p-value':>12} {'N':>8}")
    print("-"*75)
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        print(f"{i:<6} {row['feature']:<35} {row['spearman_r']:>10.5f} {row['spearman_p']:>12.2e} {row['n_samples']:>8.0f}")

    # ── Inter-feature correlation matrix for top 20 ──
    print("\n" + "="*80)
    print("INTER-FEATURE CORRELATIONS (TOP 20)")
    print("Finding low-redundancy combinations...")
    print("="*80)

    top20_names = top20["feature"].tolist()
    top20_data = all_feat[top20_names].dropna()

    if len(top20_data) > 100:
        inter_corr = top20_data.corr(method="spearman")

        # Find pairs with low mutual correlation but both correlated with target
        print(f"\nBest orthogonal pairs (both |r_target| > 0.015, mutual |r| < 0.3):\n")
        pairs = []
        for i in range(len(top20_names)):
            for j in range(i+1, len(top20_names)):
                f1, f2 = top20_names[i], top20_names[j]
                mutual = abs(inter_corr.loc[f1, f2])
                r1 = top20.loc[top20["feature"]==f1, "abs_spearman"].values[0]
                r2 = top20.loc[top20["feature"]==f2, "abs_spearman"].values[0]
                if mutual < 0.3 and r1 > 0.015 and r2 > 0.015:
                    pairs.append((f1, f2, r1, r2, mutual))

        pairs.sort(key=lambda x: -(x[2] + x[3]))
        for f1, f2, r1, r2, mutual in pairs[:20]:
            print(f"  {f1:<30} (r={r1:.4f}) + {f2:<30} (r={r2:.4f})  [mutual={mutual:.3f}]")

        # Suggest best diverse feature set
        print("\n" + "="*80)
        print("SUGGESTED DIVERSE FEATURE SET")
        print("(Greedy selection: pick top feature, then add features with <0.3 correlation to all selected)")
        print("="*80)

        selected = [top20_names[0]]
        for candidate in top20_names[1:]:
            if candidate not in top20_data.columns:
                continue
            max_corr_with_selected = max(
                abs(inter_corr.loc[candidate, s]) for s in selected
                if s in inter_corr.columns and candidate in inter_corr.columns
            )
            if max_corr_with_selected < 0.4:
                selected.append(candidate)
            if len(selected) >= 10:
                break

        print(f"\nSelected {len(selected)} diverse features:")
        for i, f in enumerate(selected, 1):
            r_val = top20.loc[top20["feature"]==f, "spearman_r"].values[0]
            p_val = top20.loc[top20["feature"]==f, "spearman_p"].values[0]
            print(f"  {i}. {f:<35} r={r_val:.5f}  p={p_val:.2e}")

    # ──────────────────────────────────────────────────────
    # FULL RANKED LIST
    # ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("FULL RANKED LIST (ALL {} FEATURES)".format(len(corr_all)))
    print("="*80)
    print(f"\n{'Rank':<6} {'Feature':<35} {'Spearman r':>10} {'p-value':>12} {'Pearson r':>10} {'N':>8}")
    print("-"*85)
    for i, (_, row) in enumerate(corr_all.iterrows(), 1):
        sig = ""
        if row["abs_spearman"] > 0.03 and row["spearman_p"] < 0.01:
            sig = " ***"
        elif row["abs_spearman"] > 0.02 and row["spearman_p"] < 0.05:
            sig = " **"
        elif row["spearman_p"] < 0.05:
            sig = " *"
        print(f"{i:<6} {row['feature']:<35} {row['spearman_r']:>10.5f} {row['spearman_p']:>12.2e} "
              f"{row['pearson_r']:>10.5f} {row['n_samples']:>8.0f}{sig}")

    print("\n\nDone! Feature discovery complete.")


if __name__ == "__main__":
    main()
