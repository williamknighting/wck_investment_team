"""
Alpaca Market Data Provider implementing MarketDataProvider interface
"""
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import logging

from ..services.market_data_service import MarketDataProvider
from .alpaca_provider import get_alpaca_provider

try:
    from ..utils.logging_config import get_logger
except ImportError:
    from utils.logging_config import get_logger


class AlpacaMarketProvider(MarketDataProvider):
    """
    Alpaca provider implementing MarketDataProvider interface
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        """
        Initialize Alpaca market provider
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading environment
        """
        self.alpaca = get_alpaca_provider()
        if api_key:
            self.alpaca.api_key = api_key
        if secret_key:
            self.alpaca.secret_key = secret_key
        self.alpaca.paper = paper
        
        self.logger = get_logger("alpaca_market_provider")
        
        # Supported intervals and periods - CONSERVATIVE: Daily data only to avoid API limits
        self._intervals = ["1Day"]  # Only daily data to be gentle on Alpaca API
        self._periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
        
        self.logger.info("Alpaca Market Provider initialized")
    
    def fetch_stock_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data from Alpaca - DAILY DATA ONLY
        
        Args:
            symbol: Stock ticker
            period: Time period 
            interval: Data interval (forced to 1d for API conservation)
            
        Returns:
            DataFrame with OHLCV data, indexed by timestamp
        """
        # SAFETY: Force daily intervals only to protect against API rate limits
        if interval != "1d":
            self.logger.warning(f"Forcing interval from {interval} to 1d to avoid API limits")
            interval = "1d"
        try:
            # Convert period to start/end dates
            end_date = datetime.now(timezone.utc)
            start_date = self._period_to_start_date(period, end_date)
            
            # Convert interval to Alpaca format
            alpaca_interval = self._convert_interval_to_alpaca(interval)
            
            # Fetch data
            df = self.alpaca.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=alpaca_interval
            )
            
            if df.empty:
                return df
            
            # Set timestamp as index
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            
            # Standardize column names for service layer
            column_mapping = {
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adj_close': 'Adj Close',
                'vwap': 'VWAP',
                'trade_count': 'Trade Count'
            }
            df = df.rename(columns=column_mapping)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol"""
        return self.alpaca.get_latest_price(symbol)
    
    def get_available_intervals(self) -> List[str]:
        """Get list of supported intervals - DAILY ONLY for API conservation"""
        return ["1d"]  # Only daily to be conservative with Alpaca API
    
    def get_available_periods(self) -> List[str]:
        """Get list of supported periods"""
        return self._periods
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            # Basic market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
            now = datetime.now(timezone.utc)
            
            # Convert to ET
            et_offset = timedelta(hours=-5)  # EST, adjust for DST if needed
            et_time = now + et_offset
            
            # Check if weekday
            if et_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check market hours (9:30 AM - 4:00 PM ET)
            market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= et_time <= market_close
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    def _period_to_start_date(self, period: str, end_date: datetime) -> datetime:
        """Convert period string to start date"""
        period_mapping = {
            "1d": timedelta(days=1),
            "2d": timedelta(days=2),
            "5d": timedelta(days=5),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "5y": timedelta(days=1825),
            "ytd": timedelta(days=(end_date.timetuple().tm_yday - 1))
        }
        
        delta = period_mapping.get(period, timedelta(days=365))
        return end_date - delta
    
    def _convert_interval_to_alpaca(self, interval: str) -> str:
        """Convert standard interval to Alpaca format"""
        interval_mapping = {
            "1m": "1Min",
            "5m": "5Min", 
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "1d": "1Day"
        }
        return interval_mapping.get(interval, "1Day")


def create_alpaca_provider(api_key: str = None, secret_key: str = None, paper: bool = True) -> AlpacaMarketProvider:
    """
    Factory function to create Alpaca provider
    
    Args:
        api_key: Alpaca API key
        secret_key: Alpaca secret key
        paper: Use paper trading environment
        
    Returns:
        AlpacaMarketProvider instance
    """
    return AlpacaMarketProvider(api_key, secret_key, paper)