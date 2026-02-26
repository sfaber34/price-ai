"""
Statistical analysis: Is 15-minute Bitcoin candle color predictable or random?
Fetches ~180 days of 15m BTCUSDT klines from Binance and runs comprehensive tests.
"""

import requests
import time
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────

def fetch_klines(symbol="BTCUSDT", interval="15m", days=180):
    """Fetch kline data from Binance, paginating as needed."""
    url = "https://api.binance.com/api/v3/klines"
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    all_klines = []
    current_start = start_ms
    limit = 1000  # Binance max per request

    print(f"Fetching {days} days of {interval} {symbol} data from Binance...")
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_klines.extend(data)
        # Move start past the last candle we received
        current_start = data[-1][0] + 1
        if len(data) < limit:
            break
        time.sleep(0.15)  # rate-limit politeness

    print(f"Fetched {len(all_klines)} candles.")
    return all_klines


def build_dataframe(raw):
    """Convert raw Binance klines into a clean DataFrame."""
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Derived columns
    df["green"] = (df["close"] >= df["open"]).astype(int)  # 1=green, 0=red
    df["body"] = df["close"] - df["open"]
    df["body_pct"] = df["body"] / df["open"] * 100
    df["abs_body_pct"] = df["body_pct"].abs()
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]
    df["hour"] = df["open_time"].dt.hour
    df["dow"] = df["open_time"].dt.dayofweek  # 0=Mon
    return df


# ─────────────────────────────────────────────
# 2. ANALYSIS HELPERS
# ─────────────────────────────────────────────

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def sig_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "(not significant)"


def chi2_test_proportion(count, total, expected_p):
    """One-sample binomial test via normal approximation."""
    observed_p = count / total
    se = np.sqrt(expected_p * (1 - expected_p) / total)
    z = (observed_p - expected_p) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return observed_p, z, p_val


# ─────────────────────────────────────────────
# 3. ANALYSES
# ─────────────────────────────────────────────

def analysis_base_rate(df):
    section("BASE RATE: Green vs Red")
    n = len(df)
    n_green = df["green"].sum()
    n_red = n - n_green
    pct_green = n_green / n * 100
    # Binomial test against 50%
    p_val = stats.binomtest(n_green, n, 0.5).pvalue
    print(f"  Total candles : {n}")
    print(f"  Green         : {n_green}  ({pct_green:.2f}%)")
    print(f"  Red           : {n_red}  ({100 - pct_green:.2f}%)")
    print(f"  Binomial test vs 50%: p = {p_val:.6f} {sig_label(p_val)}")
    print(f"  Effect size   : {pct_green - 50:.2f} percentage points from 50%")


def analysis_autocorrelation(df):
    section("AUTOCORRELATION: Does bar N predict bar N+1? (Lags 1-10)")
    series = df["green"].values.astype(float)
    mean = series.mean()
    centered = series - mean
    var = np.sum(centered ** 2)
    n = len(series)

    print(f"  {'Lag':>4}  {'Autocorr':>10}  {'p-value':>10}  {'Sig':>15}")
    print(f"  {'-'*45}")
    for lag in range(1, 11):
        ac = np.sum(centered[:-lag] * centered[lag:]) / var
        # Bartlett SE for white noise
        se = 1.0 / np.sqrt(n)
        z = ac / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f"  {lag:>4}  {ac:>10.5f}  {p:>10.6f}  {sig_label(p):>15}")


def analysis_streaks(df):
    section("STREAK ANALYSIS: More streaks than random?")
    series = df["green"].values
    n = len(series)
    p = series.mean()

    # Count runs (consecutive same-color sequences)
    changes = np.sum(series[1:] != series[:-1])
    runs = changes + 1

    # Under iid Bernoulli(p), expected runs and variance
    # Wald-Wolfowitz runs test
    n1 = series.sum()      # number of greens
    n0 = n - n1             # number of reds
    expected_runs = 1 + 2 * n1 * n0 / n
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1))
    sd_runs = np.sqrt(var_runs)
    z = (runs - expected_runs) / sd_runs
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    # Streak length distribution
    streak_lengths = []
    current_len = 1
    for i in range(1, n):
        if series[i] == series[i - 1]:
            current_len += 1
        else:
            streak_lengths.append(current_len)
            current_len = 1
    streak_lengths.append(current_len)

    counter = Counter(streak_lengths)
    max_streak = max(streak_lengths)
    mean_streak = np.mean(streak_lengths)

    print(f"  Total candles      : {n}")
    print(f"  Number of runs     : {runs}")
    print(f"  Expected (random)  : {expected_runs:.1f}")
    print(f"  Z-score            : {z:.4f}")
    print(f"  p-value            : {p_val:.6f} {sig_label(p_val)}")
    if z < 0:
        print(f"  Interpretation     : FEWER runs than random -> tendency to STREAK")
    else:
        print(f"  Interpretation     : MORE runs than random -> tendency to ALTERNATE")
    print(f"\n  Mean streak length : {mean_streak:.2f}")
    print(f"  Max streak length  : {max_streak}")
    print(f"\n  Streak length distribution (top 10):")
    for length, count in sorted(counter.items())[:10]:
        print(f"    Length {length:>3}: {count:>5} occurrences")


def analysis_time_of_day(df):
    section("TIME-OF-DAY PATTERNS (UTC hours)")
    overall_p = df["green"].mean()
    hourly = df.groupby("hour")["green"].agg(["mean", "count"])

    # Chi-squared test: are hourly green rates equal?
    observed_green = df.groupby("hour")["green"].sum().values
    observed_red = df.groupby("hour").apply(lambda g: len(g) - g["green"].sum()).values
    chi2, p_val = stats.chisquare(observed_green, f_exp=hourly["count"].values * overall_p)

    print(f"  Overall green rate: {overall_p:.4f}")
    print(f"  Chi-squared test (are all hours equal?): chi2={chi2:.2f}, p={p_val:.6f} {sig_label(p_val)}")
    print(f"\n  {'Hour':>6} {'Green%':>8} {'Count':>7} {'Deviation':>10}")
    print(f"  {'-'*35}")
    for hour in range(24):
        row = hourly.loc[hour]
        dev = (row["mean"] - overall_p) * 100
        marker = " <--" if abs(dev) > 2 else ""
        print(f"  {hour:>6} {row['mean']*100:>7.2f}% {int(row['count']):>7} {dev:>+9.2f}pp{marker}")


def analysis_day_of_week(df):
    section("DAY-OF-WEEK PATTERNS")
    overall_p = df["green"].mean()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby("dow")["green"].agg(["mean", "count"])

    observed_green = df.groupby("dow")["green"].sum().values
    chi2, p_val = stats.chisquare(observed_green, f_exp=daily["count"].values * overall_p)

    print(f"  Chi-squared test: chi2={chi2:.2f}, p={p_val:.6f} {sig_label(p_val)}")
    print(f"\n  {'Day':>6} {'Green%':>8} {'Count':>7} {'Deviation':>10}")
    print(f"  {'-'*35}")
    for dow in range(7):
        row = daily.loc[dow]
        dev = (row["mean"] - overall_p) * 100
        print(f"  {days[dow]:>6} {row['mean']*100:>7.2f}% {int(row['count']):>7} {dev:>+9.2f}pp")


def analysis_volatility_regimes(df):
    section("VOLATILITY REGIMES: Is color more predictable when vol is high?")
    # Use rolling 96-bar (1 day) standard deviation of returns as volatility
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["vol_20"] = df["ret"].rolling(96).std()
    df_clean = df.dropna(subset=["vol_20"]).copy()

    # Split into terciles
    q33 = df_clean["vol_20"].quantile(0.333)
    q66 = df_clean["vol_20"].quantile(0.666)

    low = df_clean[df_clean["vol_20"] <= q33]
    mid = df_clean[(df_clean["vol_20"] > q33) & (df_clean["vol_20"] <= q66)]
    high = df_clean[df_clean["vol_20"] > q66]

    # For each regime, check lag-1 autocorrelation
    print(f"  {'Regime':<10} {'N':>7} {'Green%':>8} {'Lag1-AC':>9} {'AC p-val':>10}")
    print(f"  {'-'*48}")
    for label, subset in [("Low vol", low), ("Mid vol", mid), ("High vol", high)]:
        s = subset["green"].values.astype(float)
        green_pct = s.mean() * 100
        if len(s) > 10:
            centered = s - s.mean()
            var = np.sum(centered**2)
            if var > 0:
                ac = np.sum(centered[:-1] * centered[1:]) / var
            else:
                ac = 0
            se = 1 / np.sqrt(len(s))
            p = 2 * (1 - stats.norm.cdf(abs(ac / se)))
        else:
            ac, p = 0, 1
        print(f"  {label:<10} {len(s):>7} {green_pct:>7.2f}% {ac:>9.5f} {p:>10.6f} {sig_label(p)}")

    # Also: is the conditional probability of "next candle same color" different across regimes?
    print(f"\n  Conditional P(next same color) by regime:")
    for label, subset in [("Low vol", low), ("Mid vol", mid), ("High vol", high)]:
        s = subset["green"].values
        same = np.sum(s[1:] == s[:-1]) / (len(s) - 1)
        print(f"    {label:<10}: {same:.4f}  (0.50 = random)")


def analysis_volume_predicts_next(df):
    section("VOLUME: Does high/low volume predict NEXT candle color?")
    df = df.copy()
    df["vol_z"] = (df["volume"] - df["volume"].rolling(96).mean()) / df["volume"].rolling(96).std()
    df["next_green"] = df["green"].shift(-1)
    df_clean = df.dropna(subset=["vol_z", "next_green"]).copy()

    q25 = df_clean["vol_z"].quantile(0.25)
    q75 = df_clean["vol_z"].quantile(0.75)

    low_vol = df_clean[df_clean["vol_z"] <= q25]
    high_vol = df_clean[df_clean["vol_z"] >= q75]

    p_low = low_vol["next_green"].mean()
    p_high = high_vol["next_green"].mean()
    p_all = df_clean["next_green"].mean()

    # Two-proportion z-test
    n1, n2 = len(low_vol), len(high_vol)
    p_pool = (low_vol["next_green"].sum() + high_vol["next_green"].sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z = (p_high - p_low) / se if se > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"  Low volume  (bottom 25%): next green = {p_low:.4f}  (n={n1})")
    print(f"  High volume (top 25%)   : next green = {p_high:.4f}  (n={n2})")
    print(f"  Overall                 : next green = {p_all:.4f}")
    print(f"  Difference              : {(p_high - p_low)*100:.2f}pp")
    print(f"  Z-test: z={z:.4f}, p={p_val:.6f} {sig_label(p_val)}")

    # Also: logistic regression of vol_z -> next_green
    from scipy.optimize import minimize_scalar
    # Point-biserial correlation as simple effect size
    corr, p_corr = stats.pointbiserialr(df_clean["next_green"].values.astype(int),
                                          df_clean["vol_z"].values)
    print(f"  Point-biserial corr(vol_z, next_green): r={corr:.5f}, p={p_corr:.6f} {sig_label(p_corr)}")


def analysis_body_size_predicts_next(df):
    section("BODY SIZE: Does a big candle predict next candle color?")
    df = df.copy()
    df["next_green"] = df["green"].shift(-1)
    df_clean = df.dropna(subset=["next_green", "abs_body_pct"]).copy()

    q75 = df_clean["abs_body_pct"].quantile(0.75)
    big = df_clean[df_clean["abs_body_pct"] >= q75]
    small = df_clean[df_clean["abs_body_pct"] < q75]

    # Split big candles by color
    big_green = big[big["green"] == 1]
    big_red = big[big["green"] == 0]

    print(f"  After BIG GREEN candles (top 25% body): next green = {big_green['next_green'].mean():.4f}  (n={len(big_green)})")
    print(f"  After BIG RED candles   (top 25% body): next green = {big_red['next_green'].mean():.4f}  (n={len(big_red)})")
    print(f"  After SMALL candles     (bottom 75%)  : next green = {small['next_green'].mean():.4f}  (n={len(small)})")

    # Momentum vs mean reversion for big candles
    # If momentum: big green -> next green > 0.5; big red -> next green < 0.5
    # If mean reversion: opposite
    for label, subset in [("Big green", big_green), ("Big red", big_red)]:
        n = len(subset)
        k = int(subset["next_green"].sum())
        res = stats.binomtest(k, n, 0.5)
        expected = "momentum" if (label == "Big green" and subset["next_green"].mean() > 0.5) or \
                                  (label == "Big red" and subset["next_green"].mean() < 0.5) else "mean-reversion"
        print(f"  {label}: {subset['next_green'].mean():.4f} -> {expected}, binomial p={res.pvalue:.6f} {sig_label(res.pvalue)}")


def analysis_wicks(df):
    section("WICK PATTERNS: Do wicks predict next candle color?")
    df = df.copy()
    df["next_green"] = df["green"].shift(-1)
    df["upper_wick_ratio"] = df["upper_wick"] / df["range"].replace(0, np.nan)
    df["lower_wick_ratio"] = df["lower_wick"] / df["range"].replace(0, np.nan)
    df_clean = df.dropna(subset=["next_green", "upper_wick_ratio", "lower_wick_ratio"]).copy()

    # Big upper wick = selling pressure -> predicts red?
    q75_upper = df_clean["upper_wick_ratio"].quantile(0.75)
    big_upper = df_clean[df_clean["upper_wick_ratio"] >= q75_upper]

    # Big lower wick = buying pressure -> predicts green?
    q75_lower = df_clean["lower_wick_ratio"].quantile(0.75)
    big_lower = df_clean[df_clean["lower_wick_ratio"] >= q75_lower]

    rest = df_clean[(df_clean["upper_wick_ratio"] < q75_upper) & (df_clean["lower_wick_ratio"] < q75_lower)]

    p_all = df_clean["next_green"].mean()

    print(f"  Overall next-green rate  : {p_all:.4f}")
    for label, subset, expected_dir in [
        ("Big upper wick (top 25%)", big_upper, "red (selling)"),
        ("Big lower wick (top 25%)", big_lower, "green (buying)"),
        ("Neither", rest, "baseline"),
    ]:
        p = subset["next_green"].mean()
        n = len(subset)
        res = stats.binomtest(int(subset["next_green"].sum()), n, p_all)
        print(f"  {label:<28}: next green = {p:.4f} (n={n}), "
              f"vs baseline p={res.pvalue:.6f} {sig_label(res.pvalue)}  [expect: {expected_dir}]")

    # Correlation
    corr_upper, p_upper = stats.pointbiserialr(df_clean["next_green"].values.astype(int),
                                                 df_clean["upper_wick_ratio"].values)
    corr_lower, p_lower = stats.pointbiserialr(df_clean["next_green"].values.astype(int),
                                                 df_clean["lower_wick_ratio"].values)
    print(f"\n  Correlation: upper_wick_ratio vs next_green: r={corr_upper:.5f}, p={p_upper:.6f} {sig_label(p_upper)}")
    print(f"  Correlation: lower_wick_ratio vs next_green: r={corr_lower:.5f}, p={p_lower:.6f} {sig_label(p_lower)}")


def analysis_big_moves(df):
    section("AFTER BIG MOVES (>1%): Mean reversion or momentum?")
    df = df.copy()
    df["ret_pct"] = (df["close"] - df["open"]) / df["open"] * 100
    df["next_green"] = df["green"].shift(-1)
    df_clean = df.dropna(subset=["next_green"]).copy()

    big_up = df_clean[df_clean["ret_pct"] > 1.0]
    big_down = df_clean[df_clean["ret_pct"] < -1.0]
    normal = df_clean[(df_clean["ret_pct"] >= -1.0) & (df_clean["ret_pct"] <= 1.0)]

    overall_p = df_clean["next_green"].mean()

    print(f"  Overall next-green rate: {overall_p:.4f}")
    print(f"\n  After big UP moves (>+1%):")
    p = big_up["next_green"].mean()
    n = len(big_up)
    res = stats.binomtest(int(big_up["next_green"].sum()), n, overall_p)
    direction = "MOMENTUM" if p > overall_p else "MEAN REVERSION"
    print(f"    Next green: {p:.4f} (n={n}), p={res.pvalue:.6f} {sig_label(res.pvalue)} -> {direction}")

    print(f"\n  After big DOWN moves (<-1%):")
    p = big_down["next_green"].mean()
    n = len(big_down)
    res = stats.binomtest(int(big_down["next_green"].sum()), n, overall_p)
    direction = "MEAN REVERSION" if p > overall_p else "MOMENTUM"
    print(f"    Next green: {p:.4f} (n={n}), p={res.pvalue:.6f} {sig_label(res.pvalue)} -> {direction}")

    print(f"\n  After normal moves (|ret| <= 1%):")
    p = normal["next_green"].mean()
    n = len(normal)
    print(f"    Next green: {p:.4f} (n={n})")

    # Also check 2-candle and 3-candle lookahead
    print(f"\n  Multi-bar follow-through after big UP moves:")
    for ahead in [1, 2, 3, 4]:
        col = f"green_plus{ahead}"
        df_clean[col] = df_clean["green"].shift(-ahead)
    df2 = df_clean.dropna(subset=[f"green_plus{i}" for i in range(1, 5)])
    big_up2 = df2[df2["ret_pct"] > 1.0]
    if len(big_up2) > 0:
        for ahead in range(1, 5):
            p = big_up2[f"green_plus{ahead}"].mean()
            print(f"    Bar +{ahead}: green rate = {p:.4f}")


def analysis_transition_matrix(df):
    section("TRANSITION MATRIX: P(next color | current color)")
    s = df["green"].values
    # 2x2 transition counts
    gg = np.sum((s[:-1] == 1) & (s[1:] == 1))
    gr = np.sum((s[:-1] == 1) & (s[1:] == 0))
    rg = np.sum((s[:-1] == 0) & (s[1:] == 1))
    rr = np.sum((s[:-1] == 0) & (s[1:] == 0))

    p_g_given_g = gg / (gg + gr) if (gg + gr) > 0 else 0
    p_g_given_r = rg / (rg + rr) if (rg + rr) > 0 else 0

    # Chi-squared test for independence
    table = np.array([[gg, gr], [rg, rr]])
    chi2, p_val, dof, expected = stats.chi2_contingency(table)

    print(f"  Transition counts:")
    print(f"    Green -> Green: {gg}    Green -> Red: {gr}")
    print(f"    Red   -> Green: {rg}    Red   -> Red: {rr}")
    print(f"\n  P(Green | prev Green) = {p_g_given_g:.4f}")
    print(f"  P(Green | prev Red)   = {p_g_given_r:.4f}")
    print(f"  Difference            = {(p_g_given_g - p_g_given_r)*100:.2f}pp")
    print(f"\n  Chi-squared independence test: chi2={chi2:.4f}, p={p_val:.6f} {sig_label(p_val)}")


def analysis_combined_signal(df):
    section("COMBINED SIGNAL CHECK: Multiple features -> next color")
    df = df.copy()
    df["next_green"] = df["green"].shift(-1)
    df["ret_pct"] = (df["close"] - df["open"]) / df["open"] * 100
    df["vol_z"] = (df["volume"] - df["volume"].rolling(96).mean()) / df["volume"].rolling(96).std()
    df["upper_wick_ratio"] = df["upper_wick"] / df["range"].replace(0, np.nan)
    df["lower_wick_ratio"] = df["lower_wick"] / df["range"].replace(0, np.nan)
    df_clean = df.dropna(subset=["next_green", "vol_z", "upper_wick_ratio", "lower_wick_ratio"]).copy()

    features = ["green", "ret_pct", "vol_z", "abs_body_pct", "upper_wick_ratio", "lower_wick_ratio"]
    print(f"\n  Point-biserial correlations with NEXT candle color:")
    print(f"  {'Feature':<22} {'Correlation':>12} {'p-value':>12} {'Sig':>15}")
    print(f"  {'-'*63}")

    results = []
    for feat in features:
        corr, p = stats.pointbiserialr(df_clean["next_green"].values.astype(int),
                                         df_clean[feat].values)
        results.append((feat, corr, p))
        print(f"  {feat:<22} {corr:>12.6f} {p:>12.6f} {sig_label(p):>15}")

    # Sort by absolute correlation
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Ranked by |correlation|:")
    for feat, corr, p in results:
        print(f"    {feat:<22}: r={corr:+.6f}  (p={p:.6f})")


# ─────────────────────────────────────────────
# 4. CONCLUSION
# ─────────────────────────────────────────────

def conclusion():
    section("CONCLUSION")
    print("""
  Summary of findings:

  The analyses above test whether 15-minute Bitcoin candle color is
  predictable using various features. For each test, we report a p-value
  (probability of seeing the result if candle colors were truly random).

  Key decision criteria:
  - p < 0.05 : statistically significant (but may not be practically useful)
  - |effect|  : even significant effects may be too small to trade profitably
  - Multiple testing: with ~15+ tests, some p < 0.05 results are expected
    by chance alone (Bonferroni correction: need p < 0.003 for true significance)

  In general, 15-minute candle color is expected to be very close to random
  (efficient market hypothesis). Any small deviations are typically:
  1. Too small to overcome transaction costs
  2. Not stable over time (regime-dependent)
  3. Potentially spurious from multiple testing

  Look at the results above to see if any effect is both:
  - Statistically significant (p < 0.003 after Bonferroni correction)
  - Practically meaningful (effect size > 2 percentage points)
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  15-MINUTE BITCOIN CANDLE COLOR: PREDICTABLE OR RANDOM?")
    print("  Comprehensive Statistical Analysis")
    print(f"  Run date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    raw = fetch_klines(days=180)
    df = build_dataframe(raw)
    print(f"  Date range: {df['open_time'].iloc[0]} to {df['open_time'].iloc[-1]}")

    analysis_base_rate(df)
    analysis_autocorrelation(df)
    analysis_transition_matrix(df)
    analysis_streaks(df)
    analysis_time_of_day(df)
    analysis_day_of_week(df)
    analysis_volatility_regimes(df)
    analysis_volume_predicts_next(df)
    analysis_body_size_predicts_next(df)
    analysis_wicks(df)
    analysis_big_moves(df)
    analysis_combined_signal(df)
    conclusion()
