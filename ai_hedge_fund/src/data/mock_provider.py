"""
Mock Data Provider for Testing
Generates realistic mock data when API keys aren't available
"""
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import numpy as np

from ..services.market_data_service import MarketDataProvider
from .duckdb_manager import get_duckdb_manager

try:
    from ..utils.logging_config import get_logger
except ImportError:
    from utils.logging_config import get_logger


class MockMarketProvider(MarketDataProvider):
    """
    Mock provider for testing when real API keys aren't available
    Generates realistic stock data
    """
    
    def __init__(self):
        """Initialize mock provider"""
        self.logger = get_logger("mock_provider")
        self.duckdb_manager = get_duckdb_manager()
        
        # Mock price data for common stocks
        self.base_prices = {
            "SPY": 430.0,
            "AAPL": 175.0,
            "NVDA": 460.0,
            "TSLA": 250.0,
            "MSFT": 360.0,
            "GOOGL": 135.0,
            "AMZN": 145.0
        }
        
        self.logger.info("Mock Market Provider initialized for testing")
    
    def fetch_stock_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Generate mock stock data
        
        Args:
            symbol: Stock ticker
            period: Time period
            interval: Data interval
            
        Returns:
            DataFrame with mock OHLCV data
        """
        try:
            # Check cache first
            cached_data = self._get_cached_data(symbol, interval, period)
            if not cached_data.empty:
                self.logger.debug(f"Using cached mock data for {symbol}")
                return cached_data
            
            # Generate new mock data
            self.logger.info(f"Generating mock data for {symbol} {period} {interval}")
            
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            days = self._period_to_days(period)
            start_date = end_date - timedelta(days=days)
            
            # Generate date range - ALWAYS DAILY for conservative API usage
            # Force all intervals to daily to match production behavior
            if interval != "1d":
                self.logger.info(f"Mock provider forcing {interval} to daily for consistency")
            date_range = pd.bdate_range(start=start_date, end=end_date)
            
            if len(date_range) == 0:
                return pd.DataFrame()
            
            # Generate realistic price data
            df = self._generate_price_data(symbol, date_range)
            
            # Cache the data
            self._cache_data(df, symbol, interval)
            
            self.logger.info(f"Generated {len(df)} mock records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error generating mock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _generate_price_data(self, symbol: str, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate realistic OHLCV data"""
        base_price = self.base_prices.get(symbol, 100.0)
        num_periods = len(date_range)
        
        # Generate random walk for closing prices
        returns = np.random.normal(0.0005, 0.02, num_periods)  # ~0.05% daily return, 2% volatility
        price_multipliers = np.exp(np.cumsum(returns))
        closes = base_price * price_multipliers
        
        # Generate OHLV data based on closes
        data = []
        for i, (timestamp, close) in enumerate(zip(date_range, closes)):
            # Previous close for gap calculation
            prev_close = closes[i-1] if i > 0 else close
            
            # Generate realistic OHLV
            gap = np.random.normal(0, 0.005)  # Small overnight gap
            open_price = prev_close * (1 + gap)
            
            daily_range = abs(np.random.normal(0, 0.015))  # Daily range ~1.5%
            high = max(open_price, close) * (1 + daily_range/2)
            low = min(open_price, close) * (1 - daily_range/2)
            
            # Volume (random but realistic)
            base_volume = 50_000_000 if symbol == "SPY" else 25_000_000
            volume = int(np.random.lognormal(np.log(base_volume), 0.5))
            
            data.append({
                'timestamp': timestamp,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume,
                'Adj Close': round(close, 2),  # Simplified
                'VWAP': round((high + low + close) / 3, 2),
                'Trade Count': int(volume / 100)  # Mock trade count
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        return df
    
    def _period_to_days(self, period: str) -> int:
        """Convert period string to number of days"""
        period_mapping = {
            "1d": 1,
            "2d": 2, 
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825
        }
        return period_mapping.get(period, 365)
    
    def _cache_data(self, df: pd.DataFrame, symbol: str, interval: str):
        """MOCK DATA - DO NOT CACHE TO DUCKDB (production database is for real data only)"""
        # INTENTIONALLY DO NOTHING - Mock data should never be persisted to DuckDB
        # DuckDB is reserved for real Alpaca data only
        self.logger.debug(f"Mock data generated for {symbol} (NOT cached - mock data stays in memory only)")
    
    def _get_cached_data(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Mock provider has no caching - always return empty to force fresh generation"""
        # Mock data is never cached to DuckDB, always generate fresh
        return pd.DataFrame()
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from mock data"""
        try:
            data = self.fetch_stock_data(symbol, period="1d", interval="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception:
            return self.base_prices.get(symbol, 100.0)
    
    def get_available_intervals(self) -> List[str]:
        """Get supported intervals - DAILY ONLY to match Alpaca conservative approach"""
        return ["1d"]
    
    def get_available_periods(self) -> List[str]:
        """Get supported periods"""
        return ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
    
    def is_market_open(self) -> bool:
        """Mock market hours check"""
        now = datetime.now(timezone.utc)
        
        # Convert to ET (approximate)
        et_time = now - timedelta(hours=5)  # EST, adjust for DST
        
        # Check if weekday and market hours (9:30 AM - 4:00 PM ET)
        if et_time.weekday() >= 5:
            return False
            
        market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= et_time <= market_close


def create_mock_provider() -> MockMarketProvider:
    """Factory function to create mock provider"""
    return MockMarketProvider()