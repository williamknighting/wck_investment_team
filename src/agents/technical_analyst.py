"""
Technical Analysis Agent
Calculates all technical indicators and metrics for trading strategies
Uses DuckDB data only - NO API calls for performance
"""
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..services.market_data_service import get_market_data_service
from ..utils.logging_config import get_logger


@dataclass
class TechnicalMetrics:
    """Container for all technical analysis metrics"""
    symbol: str
    timestamp: datetime
    
    # Core indicators
    moving_averages: Dict[str, float]
    momentum_indicators: Dict[str, float] 
    volatility_indicators: Dict[str, float]
    volume_indicators: Dict[str, float]
    pattern_metrics: Dict[str, Any]
    setup_scores: Dict[str, float]
    risk_metrics: Dict[str, float]


class SimpleTechnicalAnalyst:
    """
    Simple technical analysis agent - no AutoGen overhead
    Calculates all technical indicators from market data
    """
    
    def __init__(self):
        self.logger = get_logger("simple_technical_analyst")
        self.metrics_cache = {}
        self.logger.info("Simple Technical Analyst initialized")
    
    def calculate_all_metrics(self, symbol: str, period: str = "1y", interval: str = "1d") -> TechnicalMetrics:
        """Calculate all technical metrics for a symbol using DuckDB data ONLY"""
        
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache first (1 hour expiry)
        if cache_key in self.metrics_cache:
            cached_metrics = self.metrics_cache[cache_key]
            if (datetime.now(timezone.utc) - cached_metrics.timestamp).seconds < 3600:
                return cached_metrics
        
        # Get data from DuckDB ONLY - NO API CALLS
        from ..data.duckdb_manager import get_duckdb_manager
        from datetime import timedelta
        
        duckdb_manager = get_duckdb_manager()
        
        # Convert period to start_date (get last year of data)
        end_date = datetime.now(timezone.utc)
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
            
        # Convert interval format for DuckDB (it stores as "1Day", not "1d")
        duckdb_interval = interval
        if interval == "1d":
            duckdb_interval = "1Day"
        elif interval == "1h":
            duckdb_interval = "1Hour"
        elif interval == "1m":
            duckdb_interval = "1Min"
            
        df = duckdb_manager.get_market_data(
            symbol=symbol, 
            interval=duckdb_interval,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise ValueError(f"No data available for {symbol} in DuckDB")
        
        # Calculate all metrics
        metrics = TechnicalMetrics(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            moving_averages=self._calculate_moving_averages(df),
            momentum_indicators=self._calculate_momentum_indicators(df),
            volatility_indicators=self._calculate_volatility_indicators(df),
            volume_indicators=self._calculate_volume_indicators(df),
            pattern_metrics=self._calculate_pattern_metrics(df),
            setup_scores=self._calculate_setup_scores(df),
            risk_metrics=self._calculate_risk_metrics(df)
        )
        
        # Cache results
        self.metrics_cache[cache_key] = metrics
        
        self.logger.info(f"Calculated technical metrics for {symbol}")
        return metrics
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all moving average indicators"""
        current_price = float(df['Close'].iloc[-1])
        
        # Simple Moving Averages
        sma_10 = df['Close'].rolling(window=10).mean().iloc[-1] if len(df) >= 10 else current_price
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_price
        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else current_price
        sma_200 = df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else current_price
        
        # Exponential Moving Averages
        ema_10 = df['Close'].ewm(span=10).mean().iloc[-1] if len(df) >= 10 else current_price
        ema_20 = df['Close'].ewm(span=20).mean().iloc[-1] if len(df) >= 20 else current_price
        ema_65 = df['Close'].ewm(span=65).mean().iloc[-1] if len(df) >= 65 else current_price
        
        # Moving Average Angles (slope in degrees)
        ma_angle_10 = self._calculate_ma_angle(df['Close'], 10) if len(df) >= 15 else 0.0
        ma_angle_20 = self._calculate_ma_angle(df['Close'], 20) if len(df) >= 25 else 0.0
        
        # Trend Intensity (MA13/MA65 ratio)
        ema_13 = df['Close'].ewm(span=13).mean().iloc[-1] if len(df) >= 13 else current_price
        trend_intensity = float(ema_13 / ema_65) if ema_65 > 0 else 1.0
        
        # MA Alignment
        ma_alignment = sma_10 > sma_20 > sma_50
        
        return {
            'sma_10': float(sma_10),
            'sma_20': float(sma_20), 
            'sma_50': float(sma_50),
            'sma_200': float(sma_200),
            'ema_10': float(ema_10),
            'ema_20': float(ema_20),
            'ema_65': float(ema_65),
            'ma_angle_10': ma_angle_10,
            'ma_angle_20': ma_angle_20,
            'trend_intensity': trend_intensity,
            'ma_alignment': ma_alignment,
            'current_price': current_price
        }
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum and performance metrics"""
        # Price returns for different periods
        returns = {
            'gain_1d': self._safe_pct_change(df['Close'], 1),
            'gain_5d': self._safe_pct_change(df['Close'], 5),
            'gain_22d': self._safe_pct_change(df['Close'], 22),  # 1 month
            'gain_67d': self._safe_pct_change(df['Close'], 67),  # 3 months
            'gain_126d': self._safe_pct_change(df['Close'], 126), # 6 months
            'gain_252d': self._safe_pct_change(df['Close'], 252)  # 1 year
        }
        
        # RSI
        rsi_14 = self._calculate_rsi(df['Close'], 14)
        
        # Consecutive up/down days
        consecutive_up, consecutive_down = self._calculate_consecutive_days(df['Close'])
        
        # Distance from 52-week high
        high_52w = df['High'].rolling(window=min(252, len(df))).max().iloc[-1]
        current_price = df['Close'].iloc[-1]
        distance_from_52w_high = float((current_price - high_52w) / high_52w * 100)
        
        return {
            **returns,
            'rsi_14': float(rsi_14),
            'consecutive_up_days': consecutive_up,
            'consecutive_down_days': consecutive_down,
            'distance_from_52w_high': distance_from_52w_high
        }
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics"""
        # Average Daily Range (ADR) 
        df = df.copy()  # Avoid SettingWithCopyWarning
        df['daily_range'] = (df['High'] - df['Low']) / df['Close'] * 100
        adr_20 = df['daily_range'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else 0.0
        
        # Average True Range (ATR)
        atr_20 = self._calculate_atr(df, 20)
        
        # Range contraction
        current_range = df['daily_range'].iloc[-1] if len(df) > 0 else 0.0
        avg_range_20 = df['daily_range'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_range
        range_contraction = float(current_range / avg_range_20) if avg_range_20 > 0 else 1.0
        
        return {
            'adr_20': float(adr_20),
            'atr_20': float(atr_20), 
            'range_contraction': range_contraction,
            'current_daily_range': float(current_range)
        }
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based metrics"""
        current_volume = int(df['Volume'].iloc[-1])
        current_price = df['Close'].iloc[-1]
        
        # Average volumes
        avg_volume_20 = df['Volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_volume
        
        # Volume ratios
        volume_ratio_20 = float(current_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0
        volume_surge = volume_ratio_20 > 2.0
        
        # Dollar volume
        current_dollar_volume = current_volume * current_price
        
        return {
            'current_volume': current_volume,
            'avg_volume_20d': float(avg_volume_20),
            'volume_ratio': volume_ratio_20,
            'volume_surge': volume_surge,
            'current_dollar_volume': float(current_dollar_volume)
        }
    
    def _calculate_pattern_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price pattern metrics"""
        # Extension from moving averages
        current_price = df['Close'].iloc[-1]
        sma_10 = df['Close'].rolling(window=10).mean().iloc[-1] if len(df) >= 10 else current_price
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_price
        
        extension_from_10ma = float((current_price - sma_10) / sma_10 * 100)
        extension_from_20ma = float((current_price - sma_20) / sma_20 * 100) 
        
        # Total extension over 5 days
        total_extension_5d = self._safe_pct_change(df['Close'], 5)
        
        return {
            'extension_from_10ma': extension_from_10ma,
            'extension_from_20ma': extension_from_20ma,
            'total_extension_5d': total_extension_5d
        }
    
    def _calculate_setup_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Qullamaggie setup scores"""
        # Breakout Score (1-5)
        breakout_score = self._score_breakout_setup(df)
        
        # Consolidation Quality (0-1)
        consolidation_quality = self._score_consolidation_quality(df)
        
        # Parabolic extension
        parabolic_extension = abs(self._safe_pct_change(df['Close'], 5))
        
        return {
            'breakout_score': breakout_score,
            'consolidation_quality': consolidation_quality,
            'parabolic_extension': parabolic_extension
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk management metrics"""
        current_price = df['Close'].iloc[-1]
        atr_20 = self._calculate_atr(df, 20)
        
        # Stop distances
        stop_distance_atr = float(atr_20 / current_price * 100) if current_price > 0 else 0.0
        stop_distance_2atr = stop_distance_atr * 2
        
        return {
            'atr_20': float(atr_20),
            'stop_distance_atr': stop_distance_atr,
            'stop_distance_2atr': stop_distance_2atr,
            'current_price': float(current_price)
        }
    
    # Helper methods
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _safe_pct_change(self, series: pd.Series, periods: int) -> float:
        """Safely calculate percentage change"""
        if len(series) <= periods:
            return 0.0
        return float(series.pct_change(periods).iloc[-1] * 100)
    
    def _calculate_ma_angle(self, prices: pd.Series, window: int) -> float:
        """Calculate moving average angle in degrees"""
        if len(prices) < window + 5:
            return 0.0
        
        ma = prices.rolling(window=window).mean()
        recent_ma = ma.iloc[-5:].values  # Last 5 days
        
        if len(recent_ma) < 2:
            return 0.0
        
        # Calculate slope
        x = np.arange(len(recent_ma))
        slope = np.polyfit(x, recent_ma, 1)[0]
        
        # Convert to angle in degrees
        angle = np.arctan(slope / recent_ma[-1]) * 180 / np.pi if recent_ma[-1] != 0 else 0
        return float(angle)
    
    def _calculate_consecutive_days(self, prices: pd.Series) -> Tuple[int, int]:
        """Calculate consecutive up and down days"""
        changes = prices.diff() > 0
        
        consecutive_up = 0
        consecutive_down = 0
        
        # Count consecutive ups from the end
        for i in range(len(changes) - 1, -1, -1):
            if pd.isna(changes.iloc[i]):
                break
            if changes.iloc[i]:
                consecutive_up += 1
            else:
                break
        
        # Count consecutive downs from the end
        if consecutive_up == 0:
            for i in range(len(changes) - 1, -1, -1):
                if pd.isna(changes.iloc[i]):
                    break
                if not changes.iloc[i]:
                    consecutive_down += 1
                else:
                    break
        
        return consecutive_up, consecutive_down
    
    def _score_breakout_setup(self, df: pd.DataFrame) -> float:
        """Score breakout setup quality (1-5)"""
        score = 1.0
        
        if len(df) < 50:
            return score
        
        # Check volume surge
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
        if current_volume > avg_volume * 1.5:
            score += 1.0
        
        # Check MA alignment
        current_price = df['Close'].iloc[-1]
        sma_10 = df['Close'].rolling(window=10).mean().iloc[-1]
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        if current_price > sma_10 > sma_20:
            score += 1.0
        
        # Check if breaking out of range
        recent_high = df['High'].iloc[-20:-1].max() if len(df) > 20 else df['High'].max()
        if current_price > recent_high:
            score += 1.0
        
        return min(5.0, score)
    
    def _score_consolidation_quality(self, df: pd.DataFrame) -> float:
        """Score consolidation tightness (0-1)"""
        if len(df) < 20:
            return 0.0
        
        # Calculate range tightness over last 20 days
        ranges = (df['High'] - df['Low']) / df['Close'] * 100
        avg_range = ranges.iloc[-20:].mean()
        
        # Score based on tightness (lower range = higher quality)
        if avg_range < 2.0:
            return 1.0
        elif avg_range < 3.0:
            return 0.8
        elif avg_range < 4.0:
            return 0.6
        elif avg_range < 5.0:
            return 0.4
        else:
            return 0.2