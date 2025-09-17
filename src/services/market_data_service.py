"""
Market Data Service - Abstract interface for market data providers
This allows easy swapping between yfinance, IBKR, Alpha Vantage, etc.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd


class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    @abstractmethod
    def fetch_stock_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a symbol
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            period: Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval ("1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo")
            
        Returns:
            DataFrame with OHLCV data, indexed by date
        """
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol"""
        pass
    
    @abstractmethod
    def get_available_intervals(self) -> List[str]:
        """Get list of supported intervals"""
        pass
    
    @abstractmethod
    def get_available_periods(self) -> List[str]:
        """Get list of supported periods"""
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        pass


class MarketDataService:
    """
    Market Data Service - Main interface for all market data operations
    This is what the research agent will interact with
    """
    
    def __init__(self, provider: MarketDataProvider):
        """
        Initialize with a specific market data provider
        
        Args:
            provider: Market data provider instance (YfinanceProvider, IBKRProvider, etc.)
        """
        self.provider = provider
    
    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "1y", 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get stock data - main method for research agents
        
        Args:
            symbol: Stock ticker
            period: Time period 
            interval: Data interval
            
        Returns:
            Standardized DataFrame with columns: Open, High, Low, Close, Volume
        """
        df = self.provider.fetch_stock_data(symbol, period, interval)
        return self._standardize_dataframe(df)
    
    def get_multiple_stocks(
        self, 
        symbols: List[str], 
        period: str = "1y", 
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks
        
        Args:
            symbols: List of stock tickers
            period: Time period
            interval: Data interval
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.get_stock_data(symbol, period, interval)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                
        return results
    
    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest prices for multiple symbols
        
        Args:
            symbols: List of stock tickers
            
        Returns:
            Dictionary mapping symbol to latest price
        """
        prices = {}
        for symbol in symbols:
            try:
                price = self.provider.get_latest_price(symbol)
                if price is not None:
                    prices[symbol] = price
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                
        return prices
    
    def get_intraday_data(
        self, 
        symbol: str, 
        interval: str = "5m", 
        days: int = 1
    ) -> pd.DataFrame:
        """
        Get intraday data (convenience method)
        
        Args:
            symbol: Stock ticker
            interval: Intraday interval ("1m", "5m", "15m", "30m", "1h")
            days: Number of days of intraday data
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        period_map = {
            1: "1d",
            2: "2d", 
            5: "5d",
            7: "7d"
        }
        period = period_map.get(days, f"{days}d")
        
        return self.get_stock_data(symbol, period, interval)
    
    def get_daily_data(
        self, 
        symbol: str, 
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Get daily data (convenience method)
        
        Args:
            symbol: Stock ticker  
            period: Time period
            
        Returns:
            DataFrame with daily OHLCV data
        """
        return self.get_stock_data(symbol, period, "1d")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        return self.provider.is_market_open()
    
    def get_supported_intervals(self) -> List[str]:
        """Get list of supported intervals"""
        return self.provider.get_available_intervals()
    
    def get_supported_periods(self) -> List[str]:
        """Get list of supported periods"""
        return self.provider.get_available_periods()
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame format across all providers
        
        Args:
            df: Raw DataFrame from provider
            
        Returns:
            Standardized DataFrame with consistent column names and format
        """
        if df.empty:
            return df
        
        # Ensure standard column names (title case)
        column_mapping = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj_close': 'Adj Close'
        }
        
        # Apply mapping for lowercase columns
        df = df.rename(columns=column_mapping)
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by date index
        if hasattr(df.index, 'sort_values'):
            df = df.sort_index()
        
        return df


# Singleton instance - will be initialized with provider
market_data_service: Optional[MarketDataService] = None


def get_market_data_service() -> MarketDataService:
    """Get the global market data service instance"""
    global market_data_service
    if market_data_service is None:
        raise RuntimeError("Market data service not initialized. Call initialize_market_data_service() first.")
    return market_data_service


def initialize_market_data_service(provider: MarketDataProvider) -> MarketDataService:
    """
    Initialize the global market data service with a provider
    
    Args:
        provider: Market data provider instance
        
    Returns:
        MarketDataService instance
    """
    global market_data_service
    market_data_service = MarketDataService(provider)
    return market_data_service