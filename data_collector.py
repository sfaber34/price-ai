"""
Data collection module for crypto price prediction bot
Collects data from free APIs: CoinGecko, Yahoo Finance, Alpha Vantage, FRED
"""
import requests
import pandas as pd
import yfinance as yf
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
    def __init__(self):
        self.session = requests.Session()
        self.last_api_call = {}
        
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

    # Binance.US public REST endpoints — no API key required
    # (api.binance.com returns HTTP 451 for US IPs; binance.us is the US-accessible endpoint)
    _BINANCE_BASE    = 'https://api.binance.us/api/v3'
    _BINANCE_SYMBOLS = {'bitcoin': 'BTCUSDT', 'ethereum': 'ETHUSDT'}

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

        logger.info(f"Fetching {days}d of 15m bars for {crypto_id} from Binance…")

        while current_start < end_ms:
            try:
                resp = self.session.get(
                    f"{self._BINANCE_BASE}/klines",
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

                if not klines:
                    break

                all_klines.extend(klines)
                current_start = klines[-1][6] + 1   # advance past last bar's close_time

                if len(klines) < 1000:
                    break   # reached present, no more pages

                time.sleep(0.05)    # stay well under 1200 req/min limit

            except Exception as e:
                logger.error(f"Binance fetch error for {crypto_id}: {e}")
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
    
    # Test crypto data collection
    btc_data = collector.get_crypto_data('bitcoin', days=7)
    eth_data = collector.get_crypto_data('ethereum', days=7)

    # Test traditional market data
    market_data = collector.get_traditional_markets_data(days=7)
    
    # Test economic indicators
    econ_data = collector.get_economic_indicators()
    
    print(f"BTC data shape: {btc_data.shape}")
    print(f"ETH data shape: {eth_data.shape}")
    print(f"Market data shape: {market_data.shape}")
    print(f"Economic data shape: {econ_data.shape}") 