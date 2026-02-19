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

    def get_crypto_data(self, crypto_id: str, days: int = 30) -> pd.DataFrame:
        """
        Get cryptocurrency data at 15-minute intervals.
        Primary source: yfinance (supports 15m intervals up to 60 days back).
        Fallback: CoinGecko hourly data resampled to 15m.
        """
        # yfinance only supports 15m data for up to 60 days
        days = min(days, 59)

        crypto_symbols = {
            'bitcoin': 'BTC-USD',
            'ethereum': 'ETH-USD'
        }

        # Primary: yfinance with 15-minute intervals
        if crypto_id in crypto_symbols:
            try:
                symbol = crypto_symbols[crypto_id]
                ticker = yf.Ticker(symbol)

                end_date = datetime.now()
                # Yahoo Finance limits 15m interval to the last 60 days
                effective_days = min(days, 59)
                start_date = end_date - timedelta(days=effective_days)

                hist = ticker.history(start=start_date, end=end_date, interval='15m')

                if not hist.empty:
                    df = hist.reset_index()
                    df = df.rename(columns={
                        'Datetime': 'datetime',
                        'Close': 'price',
                        'Volume': 'volume'
                    })
                    df['crypto'] = crypto_id
                    df['market_cap'] = df['price'] * 1000000  # Approximation
                    df = df[['datetime', 'crypto', 'price', 'market_cap', 'volume']]

                    # Ensure timezone-naive datetimes
                    if hasattr(df['datetime'].dtype, 'tz') and df['datetime'].dt.tz is not None:
                        df['datetime'] = df['datetime'].dt.tz_localize(None)

                    logger.info(f"Collected {len(df)} 15m records for {crypto_id} from Yahoo Finance")
                    return df

            except Exception as e:
                logger.warning(f"Yahoo Finance 15m failed for {crypto_id}: {e}")

        # Fallback: CoinGecko hourly data (resample to 15m via ffill)
        try:
            self._rate_limit('coingecko', config.RATE_LIMITS['coingecko'])

            simple_url = f"{config.COINGECKO_BASE_URL}/simple/price"
            simple_params = {
                'ids': crypto_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }

            simple_response = self.session.get(simple_url, params=simple_params)
            if simple_response.status_code == 200:
                url = f"{config.COINGECKO_BASE_URL}/coins/{crypto_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': min(days, 30),
                    'interval': 'hourly'
                }

                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()

                    df = pd.DataFrame({
                        'timestamp': [item[0] for item in data['prices']],
                        'price': [item[1] for item in data['prices']],
                        'market_cap': [item[1] for item in data['market_caps']],
                        'volume': [item[1] for item in data['total_volumes']]
                    })

                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['crypto'] = crypto_id
                    df = df.drop('timestamp', axis=1)

                    # Resample hourly CoinGecko data to 15-minute intervals via ffill
                    df = df.set_index('datetime')
                    df_15m = df.resample('15min').ffill()
                    df_15m = df_15m.reset_index()

                    logger.info(f"Collected {len(df_15m)} 15m records for {crypto_id} from CoinGecko (resampled)")
                    return df_15m

        except Exception as e:
            logger.warning(f"CoinGecko fallback failed for {crypto_id}: {e}")

        logger.error(f"All data sources failed for {crypto_id}")
        return pd.DataFrame()

    def get_traditional_markets_data(self, days: int = 30) -> pd.DataFrame:
        """
        Get traditional market data using yfinance (free)
        """
        all_data = []
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Collect all traditional market data
            all_symbols = []
            for category, symbols in config.TRADITIONAL_MARKETS.items():
                all_symbols.extend(symbols)
            
            for symbol in all_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    # Yahoo Finance limits 15m data to the last 60 days.
                    # Traditional markets are auxiliary features, so use 1h interval
                    # which supports up to 730 days of history.
                    hist = ticker.history(start=start_date, end=end_date, interval='1h')
                    
                    if not hist.empty:
                        df = hist.reset_index()
                        df['symbol'] = symbol
                        df['datetime'] = df['Datetime']
                        # Ensure timezone-naive datetimes
                        if hasattr(df['datetime'].dtype, 'tz') and df['datetime'].dt.tz is not None:
                            df['datetime'] = df['datetime'].dt.tz_localize(None)
                        df = df[['datetime', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        all_data.append(df)
                        
                except Exception as e:
                    logger.warning(f"Could not fetch {symbol}: {e}")
                    continue
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"Collected traditional market data for {len(all_symbols)} symbols")
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error collecting traditional market data: {e}")
            return pd.DataFrame()

    def get_economic_indicators(self) -> pd.DataFrame:
        """
        Get economic indicators from FRED API (requires free API key)
        """
        if not config.FRED_API_KEY:
            logger.warning("FRED API key not provided, skipping economic indicators")
            return pd.DataFrame()
        
        all_data = []
        
        try:
            for indicator_name, series_id in config.FRED_SERIES.items():
                self._rate_limit('fred', config.RATE_LIMITS['fred'])
                
                params = {
                    'series_id': series_id,
                    'api_key': config.FRED_API_KEY,
                    'file_type': 'json',
                    'observation_start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                    'observation_end': datetime.now().strftime('%Y-%m-%d')
                }
                
                response = self.session.get(config.FRED_BASE_URL, params=params)
                response.raise_for_status()
                
                data = response.json()
                observations = data.get('observations', [])
                
                if observations:
                    df = pd.DataFrame(observations)
                    df['indicator'] = indicator_name
                    df['datetime'] = pd.to_datetime(df['date'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df[['datetime', 'indicator', 'value']].dropna()
                    all_data.append(df)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"Collected economic indicators: {list(config.FRED_SERIES.keys())}")
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error collecting economic indicators: {e}")
            return pd.DataFrame()

    def get_crypto_current_price(self, crypto_id: str) -> Optional[float]:
        """Get current price for quick updates using Yahoo Finance as primary source"""
        # First try Yahoo Finance (more reliable, no rate limits)
        try:
            crypto_symbols = {
                'bitcoin': 'BTC-USD',
                'ethereum': 'ETH-USD'
            }
            
            if crypto_id in crypto_symbols:
                symbol = crypto_symbols[crypto_id]
                ticker = yf.Ticker(symbol)
                
                # Get the latest price
                hist = ticker.history(period='1d', interval='1m')
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
                    return float(latest_price)
        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {crypto_id}: {e}")
        
        # Fallback to CoinGecko if Yahoo Finance fails
        try:
            self._rate_limit('coingecko', config.RATE_LIMITS['coingecko'])
            
            url = f"{config.COINGECKO_BASE_URL}/simple/price"
            params = {
                'ids': crypto_id,
                'vs_currencies': 'usd'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get(crypto_id, {}).get('usd')
            
        except Exception as e:
            logger.error(f"Error getting current price for {crypto_id}: {e}")
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