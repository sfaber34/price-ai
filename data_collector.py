"""
Data collection module for crypto price prediction bot
Fetches OHLCV + taker volume data from Binance.US public REST API.
"""
import requests
import pandas as pd
import time
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    # Cache TTLs in seconds — matched to each source's update frequency.
    # These are fetched once per TTL window; prediction cycles reuse the cache.
    _CACHE_TTL = {
        'fear_greed':    86400,  # daily — refreshes at midnight
        'funding_rate':  28800,  # 8-hour settlements
        'open_interest':  3600,  # hourly snapshots
    }

    def __init__(self):
        self.session = requests.Session()
        self.last_api_call = {}
        self._cache: dict = {}   # key → {'data': DataFrame, 'fetched_at': float}
        
    def _rate_limit(self, api_name: str, calls_per_minute: int):
        """Simple rate limiting to respect free API limits"""
        current_time = time.time()
        if api_name in self.last_api_call:
            time_diff = current_time - self.last_api_call[api_name]
            min_interval = 60 / calls_per_minute
            if time_diff < min_interval:
                sleep_time = min_interval - time_diff
                logger.info(f"Rate limiting {api_name}: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self.last_api_call[api_name] = current_time

    # Binance spot REST endpoints (OHLCV).
    # Primary:  api.binance.us  — US exchange, works from US IPs (no VPN)
    # Fallback: api.binance.com — global exchange, works from non-US IPs (VPN on)
    _BINANCE_BASE        = 'https://api.binance.us/api/v3'
    _BINANCE_BASE_GLOBAL = 'https://api.binance.com/api/v3'
    _BINANCE_SYMBOLS     = {'bitcoin': 'BTCUSDT', 'ethereum': 'ETHUSDT'}

    # Binance global futures REST endpoints — geo-blocked in the US without a VPN.
    # Used as the primary source for funding rate + open interest (2400 req/min,
    # no pagination issues).  Falls back to OKX automatically if unavailable.
    _BINANCE_FUTURES_BASE = 'https://fapi.binance.com'

    # OKX public REST endpoints — fallback when Binance futures is unavailable
    _OKX_BASE     = 'https://www.okx.com/api/v5'
    _OKX_INST_IDS = {'bitcoin': 'BTC-USD-SWAP', 'ethereum': 'ETH-USD-SWAP'}
    _OKX_CCY      = {'bitcoin': 'BTC', 'ethereum': 'ETH'}

    def get_crypto_data(self, crypto_id: str, days: int = 365) -> pd.DataFrame:
        """
        Fetch 15-minute OHLCV bars from Binance (no API key required).

        Key advantage over yfinance:
          - No 60-day history cap — years of data available.
          - Includes taker_buy_base_vol and taker_sell_base_vol per bar,
            which are the raw inputs for Volume Delta / CVD — the most
            informative short-term directional feature available.

        Returned columns:
            datetime, crypto,
            open, high, low, price (=close),
            volume, quote_volume, num_trades,
            taker_buy_base_vol, taker_sell_base_vol
        """
        if crypto_id not in self._BINANCE_SYMBOLS:
            logger.error(f"Unknown crypto_id '{crypto_id}'. Supported: {list(self._BINANCE_SYMBOLS)}")
            return pd.DataFrame()

        symbol   = self._BINANCE_SYMBOLS[crypto_id]
        end_ms   = int(datetime.now().timestamp() * 1000)
        start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_klines: list = []
        current_start = start_ms

        # Try .us first (no VPN), fall back to .com (VPN on, non-US exit)
        base_urls = [self._BINANCE_BASE, self._BINANCE_BASE_GLOBAL]
        active_base = self._BINANCE_BASE  # will be updated on first successful call

        logger.info(f"Fetching {days}d of 15m bars for {crypto_id} from Binance…")

        while current_start < end_ms:
            fetched = False
            for base in base_urls:
                try:
                    resp = self.session.get(
                        f"{base}/klines",
                        params={
                            'symbol':    symbol,
                            'interval':  '15m',
                            'startTime': current_start,
                            'endTime':   end_ms,
                            'limit':     1000,
                        },
                        timeout=10,
                    )
                    resp.raise_for_status()
                    klines = resp.json()

                    if base != active_base:
                        logger.info(f"Switched OHLCV source to {base} for {crypto_id}")
                        active_base = base
                        # Once we know which base works, only use that for remaining pages
                        base_urls = [base]

                    if not klines:
                        fetched = True   # empty = reached end, exit outer loop cleanly
                        break

                    all_klines.extend(klines)
                    current_start = klines[-1][6] + 1

                    if len(klines) < 1000:
                        fetched = True   # last page
                        break

                    time.sleep(0.05)
                    fetched = True
                    break

                except Exception as e:
                    if base == base_urls[-1]:
                        logger.error(f"Binance fetch error for {crypto_id} ({base}): {e}")
                    continue

            if not fetched:
                break

        if not all_klines:
            logger.error(f"No data returned from Binance for {crypto_id}")
            return pd.DataFrame()

        # Binance kline layout (positional):
        #  0 open_time  1 open  2 high  3 low  4 close  5 volume
        #  6 close_time  7 quote_vol  8 num_trades
        #  9 taker_buy_base_vol  10 taker_buy_quote_vol  11 ignore
        kline_cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'num_trades',
            'taker_buy_base_vol', 'taker_buy_quote_vol', '_ignore',
        ]
        df = pd.DataFrame(all_klines, columns=kline_cols)

        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume',
                    'quote_volume', 'taker_buy_base_vol']:
            df[col] = pd.to_numeric(df[col])
        df['num_trades'] = pd.to_numeric(df['num_trades'])

        # Taker sell = total − taker buy
        df['taker_sell_base_vol'] = df['volume'] - df['taker_buy_base_vol']
        df = df.rename(columns={'close': 'price'})
        df['crypto'] = crypto_id

        keep = [
            'datetime', 'crypto',
            'open', 'high', 'low', 'price',
            'volume', 'quote_volume', 'num_trades',
            'taker_buy_base_vol', 'taker_sell_base_vol',
        ]
        df = (df[keep]
              .drop_duplicates('datetime')
              .sort_values('datetime')
              .reset_index(drop=True))

        logger.info(
            f"Collected {len(df)} 15m bars for {crypto_id} "
            f"({df['datetime'].min().date()} → {df['datetime'].max().date()})"
        )
        return df

    def get_crypto_data_1m(self, crypto_id: str, days: int = 7) -> pd.DataFrame:
        """
        Fetch 1-minute OHLCV bars from Binance for intrabar feature engineering.

        Same columns as get_crypto_data but at 1m resolution.  Only used to
        compute within-bar CVD trajectory and price momentum features — the
        15m bar is still the primary prediction unit.

        For training (180 days): ~260 API calls, ~30-40 seconds.
        For live inference (2-7 days): 3-10 API calls, near-instant.
        """
        if crypto_id not in self._BINANCE_SYMBOLS:
            return pd.DataFrame()

        symbol   = self._BINANCE_SYMBOLS[crypto_id]
        end_ms   = int(datetime.now().timestamp() * 1000)
        start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_klines: list = []
        current_start = start_ms
        base_urls = [self._BINANCE_BASE, self._BINANCE_BASE_GLOBAL]
        active_base = self._BINANCE_BASE

        logger.info(f"Fetching {days}d of 1m bars for {crypto_id} from Binance…")

        while current_start < end_ms:
            fetched = False
            for base in base_urls:
                try:
                    resp = self.session.get(
                        f"{base}/klines",
                        params={
                            'symbol':    symbol,
                            'interval':  '1m',
                            'startTime': current_start,
                            'endTime':   end_ms,
                            'limit':     1000,
                        },
                        timeout=10,
                    )
                    resp.raise_for_status()
                    klines = resp.json()

                    if base != active_base:
                        active_base = base
                        base_urls = [base]

                    if not klines:
                        fetched = True
                        break

                    all_klines.extend(klines)
                    current_start = klines[-1][6] + 1

                    if len(klines) < 1000:
                        fetched = True
                        break

                    time.sleep(0.05)
                    fetched = True
                    break

                except Exception as e:
                    if base == base_urls[-1]:
                        logger.error(f"Binance 1m fetch error for {crypto_id}: {e}")
                    continue

            if not fetched:
                break

        if not all_klines:
            logger.warning(f"No 1m data returned for {crypto_id}")
            return pd.DataFrame()

        kline_cols = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'num_trades',
            'taker_buy_base_vol', 'taker_buy_quote_vol', '_ignore',
        ]
        df = pd.DataFrame(all_klines, columns=kline_cols)
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume',
                    'quote_volume', 'taker_buy_base_vol']:
            df[col] = pd.to_numeric(df[col])
        df['num_trades'] = pd.to_numeric(df['num_trades'])
        df['taker_sell_base_vol'] = df['volume'] - df['taker_buy_base_vol']
        df = df.rename(columns={'close': 'price'})
        df['crypto'] = crypto_id

        keep = [
            'datetime', 'crypto',
            'open', 'high', 'low', 'price',
            'volume', 'num_trades',
            'taker_buy_base_vol', 'taker_sell_base_vol',
        ]
        df = (df[keep]
              .drop_duplicates('datetime')
              .sort_values('datetime')
              .reset_index(drop=True))

        logger.info(
            f"Collected {len(df)} 1m bars for {crypto_id} "
            f"({df['datetime'].min().date()} → {df['datetime'].max().date()})"
        )
        return df

    def get_traditional_markets_data(self, days: int = 30) -> pd.DataFrame:
        """
        Disabled: traditional market data (stocks, bonds, forex) comes at 1h resolution
        and is forward-filled to match 15m crypto bars, producing 4 identical rows per hour.
        Rolling correlations over such synthetic data are spurious and hurt model accuracy.
        Returns empty DataFrame so the rest of the pipeline is unaffected.
        """
        logger.info("Traditional market data collection disabled for 15m predictions (too coarse)")
        return pd.DataFrame()

    def get_economic_indicators(self) -> pd.DataFrame:
        """
        Disabled: FRED economic indicators (Fed Funds rate, CPI, GDP, unemployment) are
        published monthly/quarterly and have no predictive signal at 15m/1h/4h horizons.
        Returns empty DataFrame so the rest of the pipeline is unaffected.
        """
        logger.info("Economic indicator collection disabled for 15m predictions (monthly/quarterly frequency)")
        return pd.DataFrame()

    def get_crypto_current_price(self, crypto_id: str) -> Optional[float]:
        """Get the latest trade price from Binance ticker (no API key required)."""
        if crypto_id not in self._BINANCE_SYMBOLS:
            logger.error(f"Unknown crypto_id '{crypto_id}'")
            return None
        try:
            resp = self.session.get(
                f"{self._BINANCE_BASE}/ticker/price",
                params={'symbol': self._BINANCE_SYMBOLS[crypto_id]},
                timeout=5,
            )
            resp.raise_for_status()
            return float(resp.json()['price'])
        except Exception as e:
            logger.error(f"Binance price fetch failed for {crypto_id}: {e}")
            return None

    def _cache_get(self, key: str, ttl_key: str) -> pd.DataFrame:
        """Return cached DataFrame if still fresh, else empty DataFrame."""
        entry = self._cache.get(key)
        if entry and (time.time() - entry['fetched_at']) < self._CACHE_TTL[ttl_key]:
            age_min = (time.time() - entry['fetched_at']) / 60
            logger.info(f"Cache hit [{key}] — {age_min:.0f}m old, TTL {self._CACHE_TTL[ttl_key]//60}m")
            return entry['data']
        return pd.DataFrame()

    def _cache_set(self, key: str, df: pd.DataFrame):
        self._cache[key] = {'data': df, 'fetched_at': time.time()}

    def _okx_get(self, url: str, params: dict, timeout: int = 10, max_retries: int = 4) -> dict:
        """
        GET an OKX endpoint with automatic retry and exponential backoff on HTTP 429.
        Raises for any other non-2xx status.
        """
        for attempt in range(max_retries):
            resp = self.session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = 2.0 * (attempt + 1)   # 2s, 4s, 6s, 8s
                logger.warning(
                    f"OKX rate limit (429) — waiting {wait:.0f}s before retry "
                    f"{attempt + 1}/{max_retries} [{url.split('/')[-1]}]"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        # Final attempt after all waits
        resp = self.session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def get_funding_rate(self, crypto_id: str, days: int = 90) -> pd.DataFrame:
        """
        Fetch perpetual funding rate history from OKX (8-hour settlements).

        Returns columns: datetime (tz-naive UTC), funding_rate (float).
        Returns empty DataFrame on any failure.
        """
        if crypto_id not in self._BINANCE_SYMBOLS:
            logger.error(f"Unknown crypto_id '{crypto_id}' for funding rate")
            return pd.DataFrame()

        cache_key = f'funding_rate_{crypto_id}_{days}'
        cached = self._cache_get(cache_key, 'funding_rate')
        if not cached.empty:
            return cached

        symbol   = self._BINANCE_SYMBOLS[crypto_id]
        start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        end_ms   = int(datetime.now().timestamp() * 1000)

        # ── Primary: Binance futures (requires VPN from US) ──────────────────
        df = self._get_funding_rate_binance(symbol, start_ms, end_ms)

        # ── Fallback: OKX ────────────────────────────────────────────────────
        if df.empty and crypto_id in self._OKX_INST_IDS:
            df = self._get_funding_rate_okx(self._OKX_INST_IDS[crypto_id], start_ms)

        if df.empty:
            logger.warning(f"No funding rate data available for {crypto_id}")
            return pd.DataFrame()

        logger.info(f"Collected {len(df)} funding rate records for {crypto_id}")
        self._cache_set(cache_key, df)
        return df

    def _get_funding_rate_binance(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """
        Fetch funding rate history from Binance futures, paginating forward in 1000-record
        chunks until end_ms is covered (needed for training windows > ~4 months).
        """
        all_rows: list = []
        current_start = start_ms
        try:
            while True:
                resp = self.session.get(
                    f"{self._BINANCE_FUTURES_BASE}/fapi/v1/fundingRate",
                    params={'symbol': symbol, 'startTime': current_start,
                            'endTime': end_ms, 'limit': 1000},
                    timeout=10,
                )
                resp.raise_for_status()
                rows = resp.json()
                if not rows:
                    break
                all_rows.extend(rows)
                if len(rows) < 1000:
                    break
                current_start = int(rows[-1]['fundingTime']) + 1
                time.sleep(0.05)
        except Exception as e:
            logger.warning(f"Binance futures funding rate unavailable for {symbol}: {e}")
            return pd.DataFrame()
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df['datetime']     = pd.to_datetime(df['fundingTime'].astype(int), unit='ms')
        df['funding_rate'] = pd.to_numeric(df['fundingRate'])
        df = (df[['datetime', 'funding_rate']]
              .drop_duplicates('datetime')
              .sort_values('datetime')
              .reset_index(drop=True))
        logger.info(f"Binance futures funding rate: {len(df)} records for {symbol}")
        return df

    def _get_funding_rate_okx(self, inst_id: str, start_ms: int) -> pd.DataFrame:
        """Fetch funding rate history from OKX (fallback, paginated)."""
        all_rows: list = []
        after: Optional[str] = None
        logger.info(f"OKX fallback: fetching funding rate for {inst_id}…")
        try:
            while True:
                params: dict = {'instId': inst_id, 'limit': 100}
                if after is not None:
                    params['after'] = after
                result = self._okx_get(f"{self._OKX_BASE}/public/funding-rate-history", params=params)
                if result.get('code') != '0' or not result.get('data'):
                    break
                data = result['data']
                all_rows.extend(data)
                oldest_ts = int(data[-1]['fundingTime'])
                if oldest_ts < start_ms or len(data) < 100:
                    break
                after = str(oldest_ts)
                time.sleep(0.12)
        except Exception as e:
            logger.error(f"OKX funding rate fallback error: {e}")
            return pd.DataFrame()
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        df['datetime']     = pd.to_datetime(df['fundingTime'].astype(int), unit='ms')
        df['funding_rate'] = pd.to_numeric(df['fundingRate'])
        df = df[df['datetime'] >= pd.to_datetime(start_ms, unit='ms')]
        return (df[['datetime', 'funding_rate']]
                .drop_duplicates('datetime')
                .sort_values('datetime')
                .reset_index(drop=True))

    def get_open_interest(self, crypto_id: str, days: int = 90) -> pd.DataFrame:
        """
        Fetch hourly open interest from Binance futures (primary) or OKX (fallback).

        Returns columns: datetime (tz-naive UTC), oi_usd (float), oi_volume_usd (float).
        Returns empty DataFrame on any failure.
        """
        if crypto_id not in self._BINANCE_SYMBOLS:
            logger.error(f"Unknown crypto_id '{crypto_id}' for open interest")
            return pd.DataFrame()

        cache_key = f'open_interest_{crypto_id}_{days}'
        cached = self._cache_get(cache_key, 'open_interest')
        if not cached.empty:
            return cached

        symbol   = self._BINANCE_SYMBOLS[crypto_id]
        start_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        end_ms   = int(datetime.now().timestamp() * 1000)

        # ── Primary: Binance futures ──────────────────────────────────────────
        df = self._get_open_interest_binance(symbol, start_ms, end_ms)

        # ── Fallback: OKX ────────────────────────────────────────────────────
        if df.empty and crypto_id in self._OKX_CCY:
            df = self._get_open_interest_okx(self._OKX_CCY[crypto_id], start_ms)

        if df.empty:
            logger.warning(f"No open interest data available for {crypto_id}")
            return pd.DataFrame()

        logger.info(f"Collected {len(df)} open interest records for {crypto_id}")
        self._cache_set(cache_key, df)
        return df

    def _get_open_interest_binance(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """
        Fetch hourly OI history from Binance futures/data/openInterestHist.
        Binance only retains the last 30 days — cap startTime accordingly.
        Limit=500 per call; pages forward using startTime until end_ms is covered.
        """
        _30d_ms = int((datetime.now() - timedelta(days=29)).timestamp() * 1000)
        current_start = max(start_ms, _30d_ms)   # Binance 400s on anything older
        all_rows: list = []
        try:
            while True:
                resp = self.session.get(
                    f"{self._BINANCE_FUTURES_BASE}/futures/data/openInterestHist",
                    params={
                        'symbol':    symbol,
                        'period':    '1h',
                        'limit':     500,
                        'startTime': current_start,
                        'endTime':   end_ms,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                rows = resp.json()
                if not rows:
                    break
                all_rows.extend(rows)
                if len(rows) < 500:
                    break
                current_start = int(rows[-1]['timestamp']) + 1
                time.sleep(0.05)
        except Exception as e:
            logger.warning(f"Binance futures open interest unavailable for {symbol}: {e}")
            return pd.DataFrame()

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df['datetime']     = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df['oi_usd']       = pd.to_numeric(df['sumOpenInterestValue'])
        # Binance OI hist doesn't include volume — fill with 0 so downstream code works
        df['oi_volume_usd'] = 0.0

        df = (df[['datetime', 'oi_usd', 'oi_volume_usd']]
              .drop_duplicates('datetime')
              .sort_values('datetime')
              .reset_index(drop=True))
        logger.info(f"Binance futures open interest: {len(df)} records for {symbol}")
        return df

    def _get_open_interest_okx(self, ccy: str, start_ms: int) -> pd.DataFrame:
        """Fetch open interest from OKX (fallback, paginated)."""
        all_rows: list = []
        after: Optional[str] = None
        logger.info(f"OKX fallback: fetching open interest for {ccy}…")
        try:
            while True:
                params: dict = {'ccy': ccy, 'period': '1H', 'limit': 100}
                if after is not None:
                    params['after'] = after
                result = self._okx_get(
                    f"{self._OKX_BASE}/rubik/stat/contracts/open-interest-volume",
                    params=params,
                )
                if result.get('code') != '0' or not result.get('data'):
                    break
                data = result['data']
                all_rows.extend(data)
                oldest_ts = int(data[0][0])
                if oldest_ts <= start_ms or len(data) < 100:
                    break
                after = str(oldest_ts)
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"OKX open interest fallback error: {e}")
            return pd.DataFrame()
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows, columns=['ts_ms', 'oi_usd', 'oi_volume_usd'])
        df['datetime']      = pd.to_datetime(df['ts_ms'].astype(int), unit='ms')
        df['oi_usd']        = pd.to_numeric(df['oi_usd'])
        df['oi_volume_usd'] = pd.to_numeric(df['oi_volume_usd'])
        df = df[df['datetime'] >= pd.to_datetime(start_ms, unit='ms')]
        return (df[['datetime', 'oi_usd', 'oi_volume_usd']]
                .drop_duplicates('datetime')
                .sort_values('datetime')
                .reset_index(drop=True))

    def get_fear_greed(self, days: int = 90) -> pd.DataFrame:
        """
        Fetch the Fear & Greed Index from alternative.me (daily, single call).

        Returns columns: datetime (tz-naive UTC, midnight), fear_greed_value (int).
        Returns empty DataFrame on any failure.
        """
        cache_key = f'fear_greed_{days}'
        cached = self._cache_get(cache_key, 'fear_greed')
        if not cached.empty:
            return cached

        try:
            resp = self.session.get(
                f"https://api.alternative.me/fng/?limit={days}",
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()

            if 'data' not in result or not result['data']:
                logger.warning("No Fear & Greed data returned from alternative.me")
                return pd.DataFrame()

            df = pd.DataFrame(result['data'])
            df['datetime']          = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df['fear_greed_value']  = pd.to_numeric(df['value'])

            df = (df[['datetime', 'fear_greed_value']]
                  .drop_duplicates('datetime')
                  .sort_values('datetime')
                  .reset_index(drop=True))

            logger.info(f"Collected {len(df)} Fear & Greed records")
            self._cache_set(cache_key, df)
            return df

        except Exception as e:
            logger.error(f"Fear & Greed fetch error: {e}")
            return pd.DataFrame()


def initialize_database():
    """Initialize SQLite database for storing collected data"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crypto_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TIMESTAMP,
            crypto TEXT,
            price REAL,
            market_cap REAL,
            volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traditional_markets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TIMESTAMP,
            symbol TEXT,
            open_price REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS economic_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TIMESTAMP,
            indicator TEXT,
            value REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datetime TIMESTAMP,
            crypto TEXT,
            prediction_horizon TEXT,
            predicted_price REAL,
            current_price REAL,
            confidence REAL,
            actual_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Migration: add current_price column to existing predictions tables that pre-date this change
    try:
        cursor.execute('ALTER TABLE predictions ADD COLUMN current_price REAL')
        conn.commit()
        logger.info("Migration: added current_price column to predictions table")
    except Exception:
        pass  # Column already exists — expected on re-runs
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

if __name__ == "__main__":
    # Initialize database
    initialize_database()
    
    # Test data collection
    collector = DataCollector()
    
    # Test crypto OHLCV data
    btc_data = collector.get_crypto_data('bitcoin', days=7)
    eth_data = collector.get_crypto_data('ethereum', days=7)

    # Test traditional market data
    market_data = collector.get_traditional_markets_data(days=7)

    # Test economic indicators
    econ_data = collector.get_economic_indicators()

    # Test OKX funding rate
    btc_fr  = collector.get_funding_rate('bitcoin',  days=7)
    eth_fr  = collector.get_funding_rate('ethereum', days=7)

    # Test OKX open interest
    btc_oi  = collector.get_open_interest('bitcoin',  days=7)
    eth_oi  = collector.get_open_interest('ethereum', days=7)

    # Test Fear & Greed
    fng     = collector.get_fear_greed(days=7)

    print(f"BTC data shape:         {btc_data.shape}")
    print(f"ETH data shape:         {eth_data.shape}")
    print(f"Market data shape:      {market_data.shape}")
    print(f"Economic data shape:    {econ_data.shape}")
    print(f"BTC funding rate shape: {btc_fr.shape}")
    print(f"ETH funding rate shape: {eth_fr.shape}")
    print(f"BTC open interest shape:{btc_oi.shape}")
    print(f"ETH open interest shape:{eth_oi.shape}")
    print(f"Fear & Greed shape:     {fng.shape}") 