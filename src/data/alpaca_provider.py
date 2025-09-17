"""
Alpaca Markets Data Provider
High-quality market data with real-time capabilities
"""
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
import logging
import requests
import time

try:
    from ..utils.logging_config import get_logger
except ImportError:
    from utils.logging_config import get_logger

from .duckdb_manager import get_duckdb_manager


class AlpacaDataProvider:
    """
    Alpaca Markets data provider for high-quality market data
    Supports both paper and live trading data feeds
    """
    
    def __init__(
        self, 
        api_key: str = None, 
        secret_key: str = None,
        paper: bool = True,
        data_feed: str = "iex"
    ):
        """
        Initialize Alpaca data provider
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key  
            paper: Use paper trading environment
            data_feed: Data feed ('iex', 'sip')
        """
        self.api_key = api_key or "YOUR_ALPACA_API_KEY"
        self.secret_key = secret_key or "YOUR_ALPACA_SECRET_KEY"
        self.paper = paper
        self.data_feed = data_feed
        
        # API endpoints
        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
        
        self.logger = get_logger("alpaca_provider")
        self.duckdb_manager = get_duckdb_manager()
        
        # Rate limiting - CONSERVATIVE for daily data
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 request per second max (very conservative)
        
        self.logger.info(f"Alpaca Provider initialized - Paper: {paper}, Feed: {data_feed}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
    
    def _rate_limit(self):
        """Rate limiting to avoid API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1Day",
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch stock data from Alpaca - DAILY DATA ONLY
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date  
            interval: Data interval (forced to 1Day for API conservation)
            limit: Maximum records
            
        Returns:
            DataFrame with OHLCV data
        """
        # SAFETY: Force daily intervals only to avoid API rate limits
        if interval != "1Day":
            self.logger.warning(f"Forcing interval from {interval} to 1Day to avoid API limits")
            interval = "1Day"
        try:
            # Check cache first
            cached_data = self._get_cached_data(symbol, interval, start_date, end_date)
            if not cached_data.empty and not self._needs_refresh(symbol, interval):
                self.logger.debug(f"Using cached data for {symbol}")
                return cached_data
            
            # Fetch fresh data
            self._rate_limit()
            
            # Build API request
            endpoint = f"{self.data_url}/v2/stocks/{symbol}/bars"
            
            params = {
                "timeframe": interval,
                "limit": limit,
                "feed": self.data_feed
            }
            
            if start_date:
                params["start"] = start_date.isoformat()
            if end_date:
                params["end"] = end_date.isoformat()
            
            response = requests.get(
                endpoint,
                headers=self._get_headers(),
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                self.logger.error(f"Alpaca API error {response.status_code}: {response.text}")
                return cached_data if not cached_data.empty else pd.DataFrame()
            
            data = response.json()
            
            if "bars" not in data or not data["bars"]:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data["bars"])
            
            # Standardize columns
            df = self._standardize_data(df, symbol)
            
            # Cache the data
            if not df.empty:
                self._cache_data(df, symbol, interval)
                self.logger.info(f"Fetched {len(df)} records for {symbol} {interval}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return cached_data if not cached_data.empty else pd.DataFrame()
    
    def _standardize_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize Alpaca data format"""
        if df.empty:
            return df
        
        # Rename columns to match expected format
        column_mapping = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low', 
            'c': 'close',
            'v': 'volume',
            'n': 'trade_count',
            'vw': 'vwap'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add missing columns
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']
        
        if 'vwap' not in df.columns:
            df['vwap'] = None
            
        if 'trade_count' not in df.columns:
            df['trade_count'] = None
        
        # Filter to required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'vwap', 'trade_count']
        df = df[required_cols]
        
        return df
    
    def _cache_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """Cache data in DuckDB"""
        try:
            success = self.duckdb_manager.store_market_data(
                symbol=symbol,
                data=df,
                interval=interval,
                data_source="alpaca"
            )
            
            if success:
                self.logger.debug(f"Cached {len(df)} records for {symbol}")
            else:
                self.logger.warning(f"Failed to cache data for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error caching data: {e}")
    
    def _get_cached_data(
        self, 
        symbol: str, 
        interval: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get cached data from DuckDB"""
        try:
            return self.duckdb_manager.get_market_data(
                symbol=symbol,
                interval=self._convert_interval(interval),
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            self.logger.error(f"Error getting cached data: {e}")
            return pd.DataFrame()
    
    def _convert_interval(self, alpaca_interval: str) -> str:
        """Convert Alpaca interval to standard format"""
        interval_mapping = {
            "1Min": "1m",
            "5Min": "5m", 
            "15Min": "15m",
            "30Min": "30m",
            "1Hour": "1h",
            "1Day": "1d"
        }
        return interval_mapping.get(alpaca_interval, alpaca_interval.lower())
    
    def _needs_refresh(self, symbol: str, interval: str) -> bool:
        """Check if cached data needs refresh"""
        try:
            summary = self.duckdb_manager.get_data_summary()
            
            # Find matching record
            for record in summary.get("summary_by_symbol", []):
                if (record["symbol"] == symbol and 
                    record["interval"] == self._convert_interval(interval) and
                    record["data_source"] == "alpaca"):
                    
                    # Check age based on interval
                    newest = pd.to_datetime(record["newest"])
                    now = pd.Timestamp.now(tz='UTC')
                    age = now - newest
                    
                    # Refresh thresholds
                    thresholds = {
                        "1m": timedelta(minutes=5),
                        "5m": timedelta(minutes=15), 
                        "15m": timedelta(hours=1),
                        "30m": timedelta(hours=2),
                        "1h": timedelta(hours=4),
                        "1d": timedelta(hours=12)
                    }
                    
                    threshold = thresholds.get(self._convert_interval(interval), timedelta(hours=1))
                    return age > threshold
            
            # No cached data found
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking refresh need: {e}")
            return True
    
    def get_multiple_stocks(
        self, 
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1Day"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks"""
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Fetching {symbol}...")
            data = self.get_stock_data(symbol, start_date, end_date, interval)
            if not data.empty:
                results[symbol] = data
            time.sleep(0.1)  # Small delay between symbols
        
        return results
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol - using daily data only"""
        try:
            # Use daily data only to be conservative with API
            data = self.get_stock_data(symbol, interval="1Day", limit=1)
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return None


# Global instance
alpaca_provider: Optional[AlpacaDataProvider] = None


def get_alpaca_provider() -> AlpacaDataProvider:
    """Get global Alpaca provider instance"""
    global alpaca_provider
    if alpaca_provider is None:
        alpaca_provider = AlpacaDataProvider()
    return alpaca_provider