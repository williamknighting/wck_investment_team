"""
Technical Analysis Agent for AI Hedge Fund System
Calculates technical indicators and metrics for trading strategies
"""
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
import pandas as pd
import numpy as np
from dataclasses import dataclass

try:
    # Try relative imports first (works in package context)
    from ..base_agent import BaseHedgeFundAgent, AgentCapability
    from ...data.market_data import fetch_stock_data, get_latest_price
    from ...utils.logging_config import get_logger
except ImportError:
    # Fall back to absolute imports (works in script context)
    from agents.base_agent import BaseHedgeFundAgent, AgentCapability
    from data.market_data import fetch_stock_data, get_latest_price
    from utils.logging_config import get_logger


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


class TechnicalAnalystAgent(BaseHedgeFundAgent):
    """
    Agent responsible for calculating technical indicators and metrics
    Provides standardized technical analysis for all trading strategies
    """
    
    def __init__(self, **kwargs):
        system_message = """You are a Technical Analysis Agent for an AI hedge fund.

Your responsibilities:
1. Calculate all technical indicators from OHLCV price data
2. Provide moving averages, momentum, volatility, and volume metrics
3. Detect price patterns and chart setups
4. Score trading setups using quantitative methods
5. Calculate risk management metrics (ATR stops, position sizing)
6. Serve as the single source of truth for all technical calculations

Technical capabilities:
- Moving Averages: SMA, EMA for multiple periods
- Momentum: RSI, gains/returns, consecutive day counts
- Volatility: ATR, ADR, range analysis, Bollinger Bands
- Volume: Volume ratios, VWAP, dollar volume
- Patterns: Higher lows/highs, consolidations, breakouts
- Setups: Qullamaggie breakout/EP/parabolic scoring
- Risk: Stop distances, position sizing calculations

Always provide accurate, well-tested calculations for strategy agents."""
        
        super().__init__(
            name="technical_analyst_agent", 
            system_message=system_message,
            capabilities=[
                AgentCapability.MARKET_ANALYSIS,
                AgentCapability.RESEARCH
            ],
            **kwargs
        )
    
    def _initialize(self) -> None:
        """Initialize technical analysis agent"""
        self.metrics_cache = {}  # Cache calculated metrics
        self.calculation_methods = self._setup_calculation_methods()
        
        self.logger.info("Technical Analysis Agent initialized")
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data"""
        message_type = data.get("type", "general")
        
        if message_type in ["calculate_all_metrics", "get_indicators"]:
            return "symbol" in data
        elif message_type in ["get_moving_averages", "get_momentum", "get_volatility"]:
            return "symbol" in data
        
        return True
    
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """Process technical analysis requests"""
        try:
            message_type = message.get("type", "general")
            
            if message_type == "calculate_all_metrics":
                return self._calculate_all_metrics(message)
            elif message_type == "get_moving_averages":
                return self._get_moving_averages(message)
            elif message_type == "get_momentum_indicators":
                return self._get_momentum_indicators(message)
            elif message_type == "get_volatility_indicators":
                return self._get_volatility_indicators(message)
            elif message_type == "get_volume_indicators":
                return self._get_volume_indicators(message)
            elif message_type == "get_pattern_metrics":
                return self._get_pattern_metrics(message)
            elif message_type == "get_setup_scores":
                return self._get_setup_scores(message)
            elif message_type == "get_risk_metrics":
                return self._get_risk_metrics(message)
            else:
                return self._general_response(message)
        
        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return {
                "type": "error",
                "error": str(e),
                "agent": self.name
            }
    
    def _calculate_all_metrics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate all technical metrics for a symbol"""
        symbol = message.get("symbol")
        period = message.get("period", "1y")
        interval = message.get("interval", "1d")
        force_refresh = message.get("force_refresh", False)
        
        cache_key = f"{symbol}_{period}_{interval}"
        
        # Check cache first
        if not force_refresh and cache_key in self.metrics_cache:
            cached_metrics = self.metrics_cache[cache_key]
            # Check if cache is fresh (less than 1 hour old)
            if (datetime.now(timezone.utc) - cached_metrics.timestamp).seconds < 3600:
                return {
                    "type": "technical_metrics",
                    "symbol": symbol,
                    "metrics": cached_metrics,
                    "from_cache": True,
                    "agent": self.name
                }
        
        # Fetch stock data
        df = fetch_stock_data(symbol, period, interval)
        
        if df.empty:
            return {
                "type": "error",
                "error": f"No data available for {symbol}",
                "agent": self.name
            }
        
        # Calculate all metrics
        try:
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
            
            self.log_activity("calculated_all_metrics", level="info",
                             symbol=symbol, metrics_count=7)
            
            return {
                "type": "technical_metrics",
                "symbol": symbol,
                "metrics": metrics,
                "from_cache": False,
                "agent": self.name
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {symbol}: {e}")
            return {
                "type": "error", 
                "error": f"Calculation failed: {str(e)}",
                "agent": self.name
            }
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all moving average indicators"""
        current_price = float(df['close'].iloc[-1])
        
        # Simple Moving Averages
        sma_10 = df['close'].rolling(window=10).mean().iloc[-1] if len(df) >= 10 else current_price
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_price
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else current_price
        sma_200 = df['close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else current_price
        
        # Exponential Moving Averages
        ema_10 = df['close'].ewm(span=10).mean().iloc[-1] if len(df) >= 10 else current_price
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1] if len(df) >= 20 else current_price
        ema_65 = df['close'].ewm(span=65).mean().iloc[-1] if len(df) >= 65 else current_price
        
        # Moving Average Angles (slope in degrees)
        ma_angle_10 = self._calculate_ma_angle(df['close'], 10) if len(df) >= 15 else 0.0
        ma_angle_20 = self._calculate_ma_angle(df['close'], 20) if len(df) >= 25 else 0.0
        
        # Trend Intensity (MA13/MA65 ratio)
        ema_13 = df['close'].ewm(span=13).mean().iloc[-1] if len(df) >= 13 else current_price
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
            'gain_1d': self._safe_pct_change(df['close'], 1),
            'gain_5d': self._safe_pct_change(df['close'], 5),
            'gain_22d': self._safe_pct_change(df['close'], 22),  # 1 month
            'gain_67d': self._safe_pct_change(df['close'], 67),  # 3 months
            'gain_126d': self._safe_pct_change(df['close'], 126), # 6 months
            'gain_252d': self._safe_pct_change(df['close'], 252)  # 1 year
        }
        
        # RSI
        rsi_14 = self._calculate_rsi(df['close'], 14)
        
        # Consecutive up/down days
        consecutive_up, consecutive_down = self._calculate_consecutive_days(df['close'])
        
        # Distance from 52-week high
        high_52w = df['high'].rolling(window=min(252, len(df))).max().iloc[-1]
        current_price = df['close'].iloc[-1]
        distance_from_52w_high = float((current_price - high_52w) / high_52w * 100)
        
        # Days since 52-week high
        days_since_52w_high = self._days_since_high(df['high'], min(252, len(df)))
        
        return {
            **returns,
            'rsi_14': float(rsi_14),
            'consecutive_up_days': consecutive_up,
            'consecutive_down_days': consecutive_down,
            'distance_from_52w_high': distance_from_52w_high,
            'days_since_52w_high': days_since_52w_high
        }
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics"""
        # Average Daily Range (ADR)
        df['daily_range'] = (df['high'] - df['low']) / df['close'] * 100
        adr_20 = df['daily_range'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else 0.0
        
        # Average True Range (ATR)
        atr_20 = self._calculate_atr(df, 20)
        
        # Range contraction
        current_range = df['daily_range'].iloc[-1] if len(df) > 0 else 0.0
        avg_range_20 = df['daily_range'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_range
        range_contraction = float(current_range / avg_range_20) if avg_range_20 > 0 else 1.0
        
        # Historical volatility
        returns = df['close'].pct_change().dropna()
        hist_vol_20 = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) if len(returns) >= 20 else 0.0
        
        return {
            'adr_20': float(adr_20),
            'atr_20': float(atr_20), 
            'range_contraction': range_contraction,
            'historical_volatility_20d': float(hist_vol_20),
            'current_daily_range': float(current_range)
        }
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-based metrics"""
        current_volume = int(df['volume'].iloc[-1])
        current_price = df['close'].iloc[-1]
        
        # Average volumes
        avg_volume_20 = df['volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_volume
        avg_volume_50 = df['volume'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else current_volume
        
        # Volume ratios
        volume_ratio_20 = float(current_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0
        volume_surge = volume_ratio_20 > 2.0  # Volume surge if >2x average
        
        # Dollar volume
        df['dollar_volume'] = df['volume'] * df['close']
        dollar_volume_20d = df['dollar_volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else 0.0
        current_dollar_volume = current_volume * current_price
        
        # VWAP (Volume Weighted Average Price)
        vwap = self._calculate_vwap(df)
        distance_from_vwap = float((current_price - vwap) / vwap * 100) if vwap > 0 else 0.0
        
        return {
            'current_volume': current_volume,
            'avg_volume_20d': float(avg_volume_20),
            'volume_ratio': volume_ratio_20,
            'volume_surge': volume_surge,
            'dollar_volume_20d': float(dollar_volume_20d),
            'current_dollar_volume': float(current_dollar_volume),
            'vwap': float(vwap),
            'distance_from_vwap': distance_from_vwap
        }
    
    def _calculate_pattern_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price pattern metrics"""
        # Higher lows and lower highs count
        higher_lows = self._count_higher_lows(df, lookback=20)
        lower_highs = self._count_lower_highs(df, lookback=20)
        
        # Consolidation analysis
        consolidation_days = self._calculate_consolidation_days(df)
        
        # Gap analysis
        gap_percent = self._calculate_gap_percent(df)
        
        # Extension from moving averages
        current_price = df['close'].iloc[-1]
        sma_10 = df['close'].rolling(window=10).mean().iloc[-1] if len(df) >= 10 else current_price
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_price
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else current_price
        
        extension_from_10ma = float((current_price - sma_10) / sma_10 * 100)
        extension_from_20ma = float((current_price - sma_20) / sma_20 * 100) 
        extension_from_50ma = float((current_price - sma_50) / sma_50 * 100)
        
        # Total extension over 5 days
        total_extension_5d = self._safe_pct_change(df['close'], 5)
        
        return {
            'higher_lows_count': higher_lows,
            'lower_highs_count': lower_highs,
            'consolidation_days': consolidation_days,
            'gap_percent': gap_percent,
            'extension_from_10ma': extension_from_10ma,
            'extension_from_20ma': extension_from_20ma,
            'extension_from_50ma': extension_from_50ma,
            'total_extension_5d': total_extension_5d
        }
    
    def _calculate_setup_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Qullamaggie setup scores"""
        # Breakout Score (1-5)
        breakout_score = self._score_breakout_setup(df)
        
        # Consolidation Quality (0-1)
        consolidation_quality = self._score_consolidation_quality(df)
        
        # Episodic Pivot qualification
        ep_gap_qualified = abs(self._calculate_gap_percent(df)) > 10.0
        ep_volume_qualified = self._check_ep_volume_qualification(df)
        ep_no_recent_rally = self._check_no_recent_rally(df)
        
        # Parabolic extension
        parabolic_extension = abs(self._safe_pct_change(df['close'], 5))
        exhaustion_signal = self._detect_exhaustion_signal(df)
        short_score = self._score_parabolic_short(df)
        
        return {
            'breakout_score': breakout_score,
            'consolidation_quality': consolidation_quality,
            'ep_gap_qualified': ep_gap_qualified,
            'ep_volume_qualified': ep_volume_qualified,
            'ep_no_recent_rally': ep_no_recent_rally,
            'parabolic_extension': parabolic_extension,
            'exhaustion_signal': exhaustion_signal,
            'short_score': short_score
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk management metrics"""
        current_price = df['close'].iloc[-1]
        atr_20 = self._calculate_atr(df, 20)
        
        # Stop distances
        stop_distance_atr = float(atr_20 / current_price * 100) if current_price > 0 else 0.0
        stop_distance_2atr = stop_distance_atr * 2
        
        # Volatility-based stop
        volatility_20d = df['close'].pct_change().rolling(window=20).std().iloc[-1] if len(df) >= 20 else 0.02
        volatility_stop_distance = float(volatility_20d * 2 * 100)  # 2 standard deviations
        
        return {
            'atr_20': float(atr_20),
            'stop_distance_atr': stop_distance_atr,
            'stop_distance_2atr': stop_distance_2atr,
            'volatility_stop_distance': volatility_stop_distance,
            'current_price': float(current_price)
        }
    
    # Helper methods for calculations
    
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
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return float(vwap)
    
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
    
    def _days_since_high(self, highs: pd.Series, window: int) -> int:
        """Calculate days since highest high in window"""
        if len(highs) < window:
            window = len(highs)
        
        recent_highs = highs.iloc[-window:]
        max_high_idx = recent_highs.idxmax()
        current_idx = highs.index[-1]
        
        return len(highs.loc[max_high_idx:current_idx]) - 1
    
    def _count_higher_lows(self, df: pd.DataFrame, lookback: int = 20) -> int:
        """Count higher lows in lookback period"""
        if len(df) < lookback:
            return 0
        
        lows = df['low'].iloc[-lookback:]
        higher_lows = 0
        
        for i in range(1, len(lows)):
            if lows.iloc[i] > lows.iloc[i-1]:
                higher_lows += 1
        
        return higher_lows
    
    def _count_lower_highs(self, df: pd.DataFrame, lookback: int = 20) -> int:
        """Count lower highs in lookback period"""
        if len(df) < lookback:
            return 0
        
        highs = df['high'].iloc[-lookback:]
        lower_highs = 0
        
        for i in range(1, len(highs)):
            if highs.iloc[i] < highs.iloc[i-1]:
                lower_highs += 1
        
        return lower_highs
    
    def _calculate_consolidation_days(self, df: pd.DataFrame) -> int:
        """Calculate days in current consolidation"""
        if len(df) < 10:
            return 0
        
        # Simple consolidation detection: range < 5% for consecutive days
        df['daily_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        
        consolidation_days = 0
        for i in range(len(df) - 1, -1, -1):
            if df['daily_range_pct'].iloc[i] < 5.0:  # Less than 5% daily range
                consolidation_days += 1
            else:
                break
        
        return consolidation_days
    
    def _calculate_gap_percent(self, df: pd.DataFrame) -> float:
        """Calculate gap from previous close"""
        if len(df) < 2:
            return 0.0
        
        current_open = df['open'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        gap_percent = (current_open - prev_close) / prev_close * 100
        return float(gap_percent)
    
    def _score_breakout_setup(self, df: pd.DataFrame) -> float:
        """Score breakout setup quality (1-5)"""
        score = 1.0
        
        if len(df) < 50:
            return score
        
        # Check volume surge
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        if current_volume > avg_volume * 1.5:
            score += 1.0
        
        # Check MA alignment
        current_price = df['close'].iloc[-1]
        sma_10 = df['close'].rolling(window=10).mean().iloc[-1]
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1]
        if current_price > sma_10 > sma_20:
            score += 1.0
        
        # Check consolidation before breakout
        consolidation_days = self._calculate_consolidation_days(df[:-1])  # Exclude current day
        if 5 <= consolidation_days <= 30:
            score += 1.0
        
        # Check if breaking out of range
        recent_high = df['high'].iloc[-20:-1].max() if len(df) > 20 else df['high'].max()
        if current_price > recent_high:
            score += 1.0
        
        return min(5.0, score)
    
    def _score_consolidation_quality(self, df: pd.DataFrame) -> float:
        """Score consolidation tightness (0-1)"""
        if len(df) < 20:
            return 0.0
        
        # Calculate range tightness over last 20 days
        ranges = (df['high'] - df['low']) / df['close'] * 100
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
    
    def _check_ep_volume_qualification(self, df: pd.DataFrame) -> bool:
        """Check EP volume qualification (placeholder - needs intraday data)"""
        # This would need first 20-minute volume data
        # For now, check if current volume > 2x average
        if len(df) < 20:
            return False
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        
        return current_volume > avg_volume * 2.0
    
    def _check_no_recent_rally(self, df: pd.DataFrame) -> bool:
        """Check if no recent rally in last 3 months"""
        if len(df) < 67:  # 3 months
            return True
        
        # Check if stock hasn't rallied >50% in last 3 months
        three_month_return = self._safe_pct_change(df['close'], 67)
        return three_month_return < 50.0
    
    def _detect_exhaustion_signal(self, df: pd.DataFrame) -> bool:
        """Detect parabolic exhaustion signals"""
        if len(df) < 10:
            return False
        
        # Check for shooting star or doji patterns
        recent_candles = df.iloc[-5:]
        
        for _, candle in recent_candles.iterrows():
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            
            # Shooting star: small body, long upper shadow
            if total_range > 0 and body_size / total_range < 0.3 and upper_shadow / total_range > 0.6:
                return True
        
        return False
    
    def _score_parabolic_short(self, df: pd.DataFrame) -> float:
        """Score parabolic short setup (1-5)"""
        score = 1.0
        
        if len(df) < 10:
            return score
        
        # Check for parabolic move (>20% in 5 days)
        five_day_return = abs(self._safe_pct_change(df['close'], 5))
        if five_day_return > 20:
            score += 1.5
        
        # Check for exhaustion
        if self._detect_exhaustion_signal(df):
            score += 1.0
        
        # Check volume exhaustion (decreasing volume on up days)
        if len(df) >= 5:
            recent_volume = df['volume'].iloc[-5:].mean()
            prev_volume = df['volume'].iloc[-10:-5].mean()
            if recent_volume < prev_volume:
                score += 1.0
        
        # Check extension from MA
        current_price = df['close'].iloc[-1]
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_price
        extension = (current_price - sma_20) / sma_20 * 100
        if extension > 25:  # >25% extended from 20-day MA
            score += 0.5
        
        return min(5.0, score)
    
    def _setup_calculation_methods(self) -> Dict[str, callable]:
        """Setup mapping of calculation methods"""
        return {
            'moving_averages': self._calculate_moving_averages,
            'momentum_indicators': self._calculate_momentum_indicators,
            'volatility_indicators': self._calculate_volatility_indicators,
            'volume_indicators': self._calculate_volume_indicators,
            'pattern_metrics': self._calculate_pattern_metrics,
            'setup_scores': self._calculate_setup_scores,
            'risk_metrics': self._calculate_risk_metrics
        }
    
    # Individual metric getters
    
    def _get_moving_averages(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get only moving average indicators"""
        symbol = message.get("symbol")
        df = fetch_stock_data(symbol, message.get("period", "1y"), message.get("interval", "1d"))
        
        if df.empty:
            return {"type": "error", "error": f"No data for {symbol}", "agent": self.name}
        
        return {
            "type": "moving_averages",
            "symbol": symbol,
            "indicators": self._calculate_moving_averages(df),
            "agent": self.name
        }
    
    def _get_momentum_indicators(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get only momentum indicators"""
        symbol = message.get("symbol")
        df = fetch_stock_data(symbol, message.get("period", "1y"), message.get("interval", "1d"))
        
        if df.empty:
            return {"type": "error", "error": f"No data for {symbol}", "agent": self.name}
        
        return {
            "type": "momentum_indicators", 
            "symbol": symbol,
            "indicators": self._calculate_momentum_indicators(df),
            "agent": self.name
        }
    
    def _get_volatility_indicators(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get only volatility indicators"""
        symbol = message.get("symbol")
        df = fetch_stock_data(symbol, message.get("period", "1y"), message.get("interval", "1d"))
        
        if df.empty:
            return {"type": "error", "error": f"No data for {symbol}", "agent": self.name}
        
        return {
            "type": "volatility_indicators",
            "symbol": symbol, 
            "indicators": self._calculate_volatility_indicators(df),
            "agent": self.name
        }
    
    def _get_volume_indicators(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get only volume indicators"""
        symbol = message.get("symbol")
        df = fetch_stock_data(symbol, message.get("period", "1y"), message.get("interval", "1d"))
        
        if df.empty:
            return {"type": "error", "error": f"No data for {symbol}", "agent": self.name}
        
        return {
            "type": "volume_indicators",
            "symbol": symbol,
            "indicators": self._calculate_volume_indicators(df),
            "agent": self.name
        }
    
    def _get_pattern_metrics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get only pattern metrics"""
        symbol = message.get("symbol")
        df = fetch_stock_data(symbol, message.get("period", "1y"), message.get("interval", "1d"))
        
        if df.empty:
            return {"type": "error", "error": f"No data for {symbol}", "agent": self.name}
        
        return {
            "type": "pattern_metrics",
            "symbol": symbol,
            "metrics": self._calculate_pattern_metrics(df),
            "agent": self.name
        }
    
    def _get_setup_scores(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get only setup scores"""
        symbol = message.get("symbol") 
        df = fetch_stock_data(symbol, message.get("period", "1y"), message.get("interval", "1d"))
        
        if df.empty:
            return {"type": "error", "error": f"No data for {symbol}", "agent": self.name}
        
        return {
            "type": "setup_scores",
            "symbol": symbol,
            "scores": self._calculate_setup_scores(df),
            "agent": self.name
        }
    
    def _get_risk_metrics(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get only risk management metrics"""
        symbol = message.get("symbol")
        df = fetch_stock_data(symbol, message.get("period", "1y"), message.get("interval", "1d"))
        
        if df.empty:
            return {"type": "error", "error": f"No data for {symbol}", "agent": self.name}
        
        return {
            "type": "risk_metrics",
            "symbol": symbol,
            "metrics": self._calculate_risk_metrics(df),
            "agent": self.name
        }
    
    def _general_response(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general requests"""
        return {
            "type": "general_response",
            "message": "Technical Analysis Agent ready. Available methods: calculate_all_metrics, get_moving_averages, get_momentum_indicators, get_volatility_indicators, get_volume_indicators, get_pattern_metrics, get_setup_scores, get_risk_metrics",
            "available_methods": list(self.calculation_methods.keys()),
            "agent": self.name
        }