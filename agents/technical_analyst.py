"""
Technical Analyst Agent for AI Hedge Fund System
Reactive technical analysis with conversation-driven metrics calculation
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from agents.base_agent import BaseHedgeFundAgent

# Import the existing technical analyst functionality
import sys
sys.path.append('..')
from src.agents.technical_analyst import SimpleTechnicalAnalyst, TechnicalMetrics


class TechnicalAnalystAgent(BaseHedgeFundAgent):
    """
    Technical Analyst Agent with AutoGen integration
    Reactive agent that responds to conversation requests for technical analysis
    """
    
    def __init__(self, name: str = "technical_analyst", description: str = "Technical analysis specialist", **kwargs):
        """Initialize Technical Analyst Agent"""
        super().__init__(
            name=name,
            description=description,
            system_message=self._get_system_message(),
            **kwargs
        )
    
    def _get_system_message(self) -> str:
        """Get system message for Technical Analyst"""
        return """You are a Technical Analyst for an AI hedge fund. Your role is to:

1. ANALYZE price action, volume, and technical indicators
2. PROVIDE objective technical metrics and assessments
3. RESPOND to requests for specific technical analysis
4. CALCULATE momentum, volatility, and trend indicators
5. IDENTIFY key support/resistance levels and patterns

You are data-driven and objective. Provide clear, actionable technical insights based on quantitative analysis.
Always include specific metrics, timeframes, and confidence levels in your analysis."""
    
    def _initialize(self) -> None:
        """Initialize the technical analyst with specialized tools"""
        self.analyst = SimpleTechnicalAnalyst()
        self.conversation_context = {}
        self.custom_indicators = {}
        self._initialized_at = datetime.now(timezone.utc)
        self.logger.info("Technical Analyst Agent initialized - Reactive mode active")
    
    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process incoming messages and provide reactive technical analysis
        
        Args:
            message: The message/request from another agent
            context: Optional context including symbol, timeframe, specific indicators
            
        Returns:
            Dict containing technical analysis results or custom metrics
        """
        try:
            # Store conversation context for follow-up questions
            if context:
                self.conversation_context.update(context)
            
            # Parse message intent and provide appropriate analysis
            if "comprehensive" in message.lower() or "full analysis" in message.lower():
                return self._provide_comprehensive_analysis(context)
            elif "custom" in message.lower() or "calculate" in message.lower():
                return self._calculate_custom_metrics(message, context)
            elif "bollinger" in message.lower():
                return self._calculate_bollinger_bands(context)
            elif "macd" in message.lower():
                return self._calculate_macd(context)
            elif "support" in message.lower() or "resistance" in message.lower():
                return self._calculate_support_resistance(context)
            elif "trend" in message.lower():
                return self._analyze_trend_strength(context)
            elif "momentum" in message.lower():
                return self._analyze_momentum(context)
            elif "volume" in message.lower():
                return self._analyze_volume_patterns(context)
            elif "risk" in message.lower() or "atr" in message.lower():
                return self._calculate_risk_metrics(context)
            elif "signal" in message.lower() or "setup" in message.lower():
                return self._generate_trading_signals(context)
            else:
                return self._provide_standard_analysis(context)
                
        except Exception as e:
            self.logger.error(f"Error processing technical analysis message: {e}")
            return {
                "type": "error",
                "message": str(e),
                "agent": self.agent_name
            }
    
    def _provide_comprehensive_analysis(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Provide comprehensive technical analysis using existing SimpleTechnicalAnalyst
        
        Args:
            context: Context with symbol and parameters
            
        Returns:
            Complete technical analysis package
        """
        symbol = self._extract_symbol(context)
        period = context.get("period", "1y") if context else "1y"
        interval = context.get("interval", "1d") if context else "1d"
        
        try:
            # Use existing comprehensive analysis
            metrics = self.analyst.calculate_all_metrics(symbol, period, interval)
            
            # Format for conversation
            analysis_result = {
                "type": "comprehensive_technical_analysis",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "agent": self.agent_name,
                "metrics": {
                    "moving_averages": metrics.moving_averages,
                    "momentum_indicators": metrics.momentum_indicators,
                    "volatility_indicators": metrics.volatility_indicators,
                    "volume_indicators": metrics.volume_indicators,
                    "pattern_metrics": metrics.pattern_metrics,
                    "setup_scores": metrics.setup_scores,
                    "risk_metrics": metrics.risk_metrics
                },
                "summary": self._generate_conversational_summary(metrics),
                "formatted_for_strategies": self._format_for_strategy_agents(metrics),
                "data_source": "DuckDB"
            }
            
            self.logger.info(f"Comprehensive technical analysis completed for {symbol}")
            return analysis_result
            
        except Exception as e:
            return self._handle_analysis_error(symbol, str(e))
    
    def _provide_standard_analysis(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Provide standard technical analysis (alias for comprehensive)"""
        return self._provide_comprehensive_analysis(context)
    
    def _calculate_custom_metrics(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculate custom metrics based on conversation requests
        
        Args:
            message: Message containing custom metric requests
            context: Context with symbol and parameters
            
        Returns:
            Custom calculated metrics
        """
        symbol = self._extract_symbol(context)
        
        try:
            # Get market data for custom calculations
            df = self._get_market_data(symbol, context)
            
            custom_metrics = {}
            
            # Parse message for specific metric requests
            if "price ratio" in message.lower():
                custom_metrics.update(self._calculate_price_ratios(df))
            if "momentum score" in message.lower():
                custom_metrics.update(self._calculate_momentum_score(df))
            if "volatility rank" in message.lower():
                custom_metrics.update(self._calculate_volatility_rank(df))
            if "strength index" in message.lower():
                custom_metrics.update(self._calculate_strength_index(df))
            if "breakout probability" in message.lower():
                custom_metrics.update(self._calculate_breakout_probability(df))
            
            return {
                "type": "custom_metrics",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "requested_metrics": self._extract_metric_requests(message),
                "custom_calculations": custom_metrics,
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"Custom metrics error: {str(e)}")
    
    def _calculate_bollinger_bands(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate Bollinger Bands with conversational formatting"""
        symbol = self._extract_symbol(context)
        
        try:
            df = self._get_market_data(symbol, context)
            
            # Bollinger Bands parameters
            period = context.get("bb_period", 20) if context else 20
            std_dev = context.get("bb_std", 2.0) if context else 2.0
            
            # Calculate Bollinger Bands
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            current_price = float(df['Close'].iloc[-1])
            current_upper = float(upper_band.iloc[-1])
            current_lower = float(lower_band.iloc[-1])
            current_middle = float(sma.iloc[-1])
            
            # Calculate position within bands
            band_position = (current_price - current_lower) / (current_upper - current_lower)
            
            # Squeeze detection
            band_width = (current_upper - current_lower) / current_middle
            avg_band_width = ((upper_band - lower_band) / sma).rolling(window=20).mean().iloc[-1]
            squeeze_ratio = band_width / avg_band_width if avg_band_width > 0 else 1.0
            
            return {
                "type": "bollinger_bands",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "parameters": {"period": period, "std_dev": std_dev},
                "bands": {
                    "upper": current_upper,
                    "middle": current_middle,
                    "lower": current_lower,
                    "current_price": current_price
                },
                "analysis": {
                    "band_position": band_position,
                    "band_width": band_width,
                    "squeeze_ratio": squeeze_ratio,
                    "squeeze_detected": squeeze_ratio < 0.8,
                    "position_signal": self._interpret_bb_position(band_position)
                },
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"Bollinger Bands error: {str(e)}")
    
    def _calculate_macd(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate MACD with signal interpretation"""
        symbol = self._extract_symbol(context)
        
        try:
            df = self._get_market_data(symbol, context)
            
            # MACD parameters
            fast = context.get("macd_fast", 12) if context else 12
            slow = context.get("macd_slow", 26) if context else 26
            signal = context.get("macd_signal", 9) if context else 9
            
            # Calculate MACD
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            current_macd = float(macd_line.iloc[-1])
            current_signal = float(signal_line.iloc[-1])
            current_histogram = float(histogram.iloc[-1])
            
            # Detect crossovers
            previous_macd = float(macd_line.iloc[-2]) if len(macd_line) > 1 else current_macd
            previous_signal = float(signal_line.iloc[-2]) if len(signal_line) > 1 else current_signal
            
            bullish_crossover = previous_macd <= previous_signal and current_macd > current_signal
            bearish_crossover = previous_macd >= previous_signal and current_macd < current_signal
            
            return {
                "type": "macd_analysis",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "parameters": {"fast": fast, "slow": slow, "signal": signal},
                "values": {
                    "macd": current_macd,
                    "signal": current_signal,
                    "histogram": current_histogram
                },
                "signals": {
                    "bullish_crossover": bullish_crossover,
                    "bearish_crossover": bearish_crossover,
                    "momentum_direction": "bullish" if current_histogram > 0 else "bearish",
                    "histogram_trend": self._analyze_histogram_trend(histogram)
                },
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"MACD error: {str(e)}")
    
    def _calculate_support_resistance(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        symbol = self._extract_symbol(context)
        
        try:
            df = self._get_market_data(symbol, context)
            
            # Find support and resistance levels using pivot points
            lookback = context.get("sr_lookback", 20) if context else 20
            
            support_levels = []
            resistance_levels = []
            
            # Simple pivot point detection
            for i in range(lookback, len(df) - lookback):
                # Resistance: high is higher than surrounding highs
                if all(df['High'].iloc[i] >= df['High'].iloc[i-j] for j in range(1, lookback+1)) and \
                   all(df['High'].iloc[i] >= df['High'].iloc[i+j] for j in range(1, lookback+1)):
                    resistance_levels.append({
                        "level": float(df['High'].iloc[i]),
                        "date": df.index[i].isoformat(),
                        "strength": self._calculate_level_strength(df, df['High'].iloc[i], "resistance")
                    })
                
                # Support: low is lower than surrounding lows
                if all(df['Low'].iloc[i] <= df['Low'].iloc[i-j] for j in range(1, lookback+1)) and \
                   all(df['Low'].iloc[i] <= df['Low'].iloc[i+j] for j in range(1, lookback+1)):
                    support_levels.append({
                        "level": float(df['Low'].iloc[i]),
                        "date": df.index[i].isoformat(),
                        "strength": self._calculate_level_strength(df, df['Low'].iloc[i], "support")
                    })
            
            # Sort by proximity to current price
            current_price = float(df['Close'].iloc[-1])
            support_levels.sort(key=lambda x: abs(x["level"] - current_price))
            resistance_levels.sort(key=lambda x: abs(x["level"] - current_price))
            
            return {
                "type": "support_resistance",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
                "support_levels": support_levels[:5],  # Top 5 closest
                "resistance_levels": resistance_levels[:5],  # Top 5 closest
                "nearest_support": support_levels[0] if support_levels else None,
                "nearest_resistance": resistance_levels[0] if resistance_levels else None,
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"Support/Resistance error: {str(e)}")
    
    def _analyze_trend_strength(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze trend strength using multiple indicators"""
        symbol = self._extract_symbol(context)
        
        try:
            # Get comprehensive metrics
            metrics = self.analyst.calculate_all_metrics(symbol)
            
            # Combine multiple trend indicators
            ma_alignment_score = 1.0 if metrics.moving_averages["ma_alignment"] else 0.0
            trend_intensity = metrics.moving_averages["trend_intensity"]
            ma_angle_strength = abs(metrics.moving_averages["ma_angle_20"]) / 45.0  # Normalize to 45 degrees
            
            # Volume confirmation
            volume_confirmation = min(metrics.volume_indicators["volume_ratio"] / 1.5, 1.0)
            
            # Overall trend strength (0-1)
            trend_strength = (ma_alignment_score * 0.3 + 
                            min(abs(trend_intensity - 1.0) * 2, 1.0) * 0.3 +
                            min(ma_angle_strength, 1.0) * 0.2 +
                            volume_confirmation * 0.2)
            
            return {
                "type": "trend_strength_analysis",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "trend_strength": trend_strength,
                "trend_direction": "bullish" if trend_intensity > 1.0 else "bearish",
                "components": {
                    "ma_alignment": ma_alignment_score,
                    "trend_intensity": trend_intensity,
                    "angle_strength": ma_angle_strength,
                    "volume_confirmation": volume_confirmation
                },
                "interpretation": self._interpret_trend_strength(trend_strength),
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"Trend analysis error: {str(e)}")
    
    def _analyze_momentum(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        symbol = self._extract_symbol(context)
        
        try:
            metrics = self.analyst.calculate_all_metrics(symbol)
            
            momentum = metrics.momentum_indicators
            
            return {
                "type": "momentum_analysis",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "momentum_indicators": {
                    "rsi_14": momentum["rsi_14"],
                    "recent_performance": {
                        "1_day": momentum["gain_1d"],
                        "5_day": momentum["gain_5d"],
                        "22_day": momentum["gain_22d"]
                    },
                    "distance_from_52w_high": momentum["distance_from_52w_high"]
                },
                "momentum_signals": {
                    "rsi_signal": self._interpret_rsi(momentum["rsi_14"]),
                    "short_term_momentum": "strong" if momentum["gain_5d"] > 5 else "weak" if momentum["gain_5d"] < -5 else "neutral",
                    "medium_term_momentum": "strong" if momentum["gain_22d"] > 20 else "weak" if momentum["gain_22d"] < -20 else "neutral"
                },
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"Momentum analysis error: {str(e)}")
    
    def _analyze_volume_patterns(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze volume patterns and anomalies"""
        symbol = self._extract_symbol(context)
        
        try:
            df = self._get_market_data(symbol, context)
            metrics = self.analyst.calculate_all_metrics(symbol)
            
            volume = metrics.volume_indicators
            
            # Additional volume analysis
            volume_trend = self._calculate_volume_trend(df)
            volume_profile = self._analyze_volume_profile(df)
            
            return {
                "type": "volume_analysis",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "volume_metrics": {
                    "current_volume": volume["current_volume"],
                    "average_volume_20d": volume["avg_volume_20d"],
                    "volume_ratio": volume["volume_ratio"],
                    "volume_surge": volume["volume_surge"],
                    "dollar_volume": volume["current_dollar_volume"]
                },
                "volume_patterns": {
                    "volume_trend": volume_trend,
                    "volume_profile": volume_profile,
                    "accumulation_distribution": self._calculate_accumulation_distribution(df)
                },
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"Volume analysis error: {str(e)}")
    
    def _calculate_risk_metrics(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate risk management metrics"""
        symbol = self._extract_symbol(context)
        
        try:
            metrics = self.analyst.calculate_all_metrics(symbol)
            risk = metrics.risk_metrics
            volatility = metrics.volatility_indicators
            
            return {
                "type": "risk_metrics",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "risk_indicators": {
                    "atr_20": risk["atr_20"],
                    "stop_distance_1atr": risk["stop_distance_atr"],
                    "stop_distance_2atr": risk["stop_distance_2atr"],
                    "daily_range": volatility["current_daily_range"],
                    "average_daily_range": volatility["adr_20"]
                },
                "position_sizing": {
                    "suggested_stop_levels": {
                        "tight": risk["current_price"] - risk["atr_20"],
                        "normal": risk["current_price"] - (risk["atr_20"] * 1.5),
                        "wide": risk["current_price"] - (risk["atr_20"] * 2.0)
                    },
                    "risk_per_share_1atr": risk["atr_20"],
                    "risk_per_share_2atr": risk["atr_20"] * 2
                },
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"Risk metrics error: {str(e)}")
    
    def _generate_trading_signals(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate trading signals based on technical analysis"""
        symbol = self._extract_symbol(context)
        
        try:
            metrics = self.analyst.calculate_all_metrics(symbol)
            
            # Compile signals from various indicators
            signals = []
            confidence_scores = []
            
            # Moving average signals
            if metrics.moving_averages["ma_alignment"]:
                signals.append("Bullish MA alignment")
                confidence_scores.append(0.8)
            
            # RSI signals
            rsi = metrics.momentum_indicators["rsi_14"]
            if rsi < 30:
                signals.append("RSI oversold - potential buy")
                confidence_scores.append(0.7)
            elif rsi > 70:
                signals.append("RSI overbought - potential sell")
                confidence_scores.append(0.7)
            
            # Volume confirmation
            if metrics.volume_indicators["volume_surge"]:
                signals.append("Volume surge confirmation")
                confidence_scores.append(0.6)
            
            # Momentum signals
            if metrics.momentum_indicators["gain_22d"] > 20:
                signals.append("Strong 22-day momentum")
                confidence_scores.append(0.8)
            
            # Overall signal strength
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return {
                "type": "trading_signals",
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "signals": signals,
                "overall_confidence": overall_confidence,
                "signal_strength": "strong" if overall_confidence > 0.7 else "moderate" if overall_confidence > 0.5 else "weak",
                "recommendation": self._generate_recommendation(signals, overall_confidence),
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_analysis_error(symbol, f"Signal generation error: {str(e)}")
    
    def _format_for_strategy_agents(self, metrics: TechnicalMetrics) -> Dict[str, Any]:
        """
        Format technical data specifically for strategy agents
        
        Args:
            metrics: Technical metrics from SimpleTechnicalAnalyst
            
        Returns:
            Structured data optimized for strategy agent consumption
        """
        return {
            "price_data": {
                "current": metrics.moving_averages["current_price"],
                "sma_20": metrics.moving_averages["sma_20"],
                "sma_50": metrics.moving_averages["sma_50"],
                "ema_20": metrics.moving_averages["ema_20"],
                "trend_intensity": metrics.moving_averages["trend_intensity"]
            },
            "momentum_data": {
                "rsi": metrics.momentum_indicators["rsi_14"],
                "gain_22d": metrics.momentum_indicators["gain_22d"],
                "gain_5d": metrics.momentum_indicators["gain_5d"],
                "distance_52w_high": metrics.momentum_indicators["distance_from_52w_high"]
            },
            "volume_data": {
                "volume_ratio": metrics.volume_indicators["volume_ratio"],
                "volume_surge": metrics.volume_indicators["volume_surge"],
                "avg_volume": metrics.volume_indicators["avg_volume_20d"]
            },
            "volatility_data": {
                "atr": metrics.volatility_indicators["atr_20"],
                "daily_range": metrics.volatility_indicators["current_daily_range"],
                "range_contraction": metrics.volatility_indicators["range_contraction"]
            },
            "setup_scores": metrics.setup_scores,
            "quality_flags": {
                "ma_alignment": metrics.moving_averages["ma_alignment"],
                "volume_confirmed": metrics.volume_indicators["volume_ratio"] > 1.2,
                "momentum_positive": metrics.momentum_indicators["gain_22d"] > 0,
                "low_volatility": metrics.volatility_indicators["range_contraction"] < 0.8
            }
        }
    
    # Helper methods for conversation processing
    
    def _extract_symbol(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Extract symbol from context or conversation history"""
        if context and "symbol" in context:
            return context["symbol"]
        elif "symbol" in self.conversation_context:
            return self.conversation_context["symbol"]
        else:
            return "SPY"  # Default fallback
    
    def _get_market_data(self, symbol: str, context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Get market data from DuckDB"""
        period = context.get("period", "1y") if context else "1y"
        interval = context.get("interval", "1d") if context else "1d"
        
        # Convert period to date range
        end_date = datetime.now(timezone.utc)
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=365)
        
        # Get data from DuckDB
        df = self.db.get_market_data(
            symbol=symbol,
            interval="1d",  # Force daily for consistency
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            raise ValueError(f"No market data available for {symbol}")
        
        return df
    
    def _generate_conversational_summary(self, metrics: TechnicalMetrics) -> Dict[str, Any]:
        """Generate human-readable summary for conversations"""
        ma = metrics.moving_averages
        momentum = metrics.momentum_indicators
        volume = metrics.volume_indicators
        
        return {
            "current_price": ma["current_price"],
            "trend_direction": "bullish" if ma["ma_alignment"] else "bearish",
            "trend_strength": "strong" if abs(ma["trend_intensity"] - 1.0) > 0.1 else "weak",
            "momentum_signal": self._interpret_rsi(momentum["rsi_14"]),
            "recent_performance": {
                "5_day": f"{momentum['gain_5d']:+.1f}%",
                "22_day": f"{momentum['gain_22d']:+.1f}%"
            },
            "volume_status": "high" if volume["volume_ratio"] > 1.5 else "normal",
            "distance_from_52w_high": f"{momentum['distance_from_52w_high']:.1f}%",
            "overall_setup": "bullish" if ma["ma_alignment"] and momentum["gain_22d"] > 10 else "bearish" if not ma["ma_alignment"] and momentum["gain_22d"] < -10 else "neutral"
        }
    
    def _handle_analysis_error(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """Handle analysis errors gracefully"""
        self.logger.error(f"Technical analysis error for {symbol}: {error_msg}")
        return {
            "type": "error",
            "symbol": symbol,
            "message": error_msg,
            "agent": self.agent_name,
            "suggestion": "Try checking if market data is available for this symbol or adjust the time period"
        }
    
    # Additional helper methods for custom metrics
    
    def _calculate_price_ratios(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various price ratios"""
        current_price = df['Close'].iloc[-1]
        high_52w = df['High'].rolling(window=min(252, len(df))).max().iloc[-1]
        low_52w = df['Low'].rolling(window=min(252, len(df))).min().iloc[-1]
        
        return {
            "price_to_52w_high": current_price / high_52w if high_52w > 0 else 1.0,
            "price_to_52w_low": current_price / low_52w if low_52w > 0 else 1.0,
            "52w_range_position": (current_price - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5
        }
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate composite momentum score"""
        gain_5d = df['Close'].pct_change(5).iloc[-1] * 100 if len(df) > 5 else 0
        gain_22d = df['Close'].pct_change(22).iloc[-1] * 100 if len(df) > 22 else 0
        
        # Normalize to 0-1 scale
        momentum_score = min(max((gain_22d + 20) / 40, 0), 1)  # -20% to +20% maps to 0-1
        
        return {
            "momentum_score": momentum_score,
            "gain_5d": gain_5d,
            "gain_22d": gain_22d
        }
    
    def _calculate_volatility_rank(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility rank"""
        current_atr = self._calculate_atr(df, 20)
        atr_series = df['High'].rolling(20).apply(lambda x: self._calculate_atr(df.loc[x.index], 20))
        
        volatility_rank = (atr_series <= current_atr).mean() if len(atr_series) > 0 else 0.5
        
        return {
            "volatility_rank": volatility_rank,
            "current_atr": current_atr
        }
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate ATR helper method"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI values"""
        if rsi > 70:
            return "overbought"
        elif rsi < 30:
            return "oversold"
        else:
            return "neutral"
    
    def _interpret_bb_position(self, position: float) -> str:
        """Interpret Bollinger Band position"""
        if position > 0.8:
            return "near_upper_band"
        elif position < 0.2:
            return "near_lower_band"
        else:
            return "middle_range"
    
    def _analyze_histogram_trend(self, histogram: pd.Series) -> str:
        """Analyze MACD histogram trend"""
        if len(histogram) < 3:
            return "insufficient_data"
        
        recent_values = histogram.iloc[-3:].values
        if all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
            return "increasing"
        elif all(recent_values[i] < recent_values[i-1] for i in range(1, len(recent_values))):
            return "decreasing"
        else:
            return "mixed"
    
    def _calculate_level_strength(self, df: pd.DataFrame, level: float, level_type: str) -> float:
        """Calculate support/resistance level strength"""
        # Simple strength calculation based on touches
        touches = 0
        tolerance = level * 0.02  # 2% tolerance
        
        if level_type == "resistance":
            touches = ((df['High'] >= level - tolerance) & (df['High'] <= level + tolerance)).sum()
        else:
            touches = ((df['Low'] >= level - tolerance) & (df['Low'] <= level + tolerance)).sum()
        
        return min(touches / 5.0, 1.0)  # Normalize to 0-1
    
    def _interpret_trend_strength(self, strength: float) -> str:
        """Interpret trend strength value"""
        if strength > 0.8:
            return "very_strong"
        elif strength > 0.6:
            return "strong"
        elif strength > 0.4:
            return "moderate"
        elif strength > 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> str:
        """Calculate volume trend"""
        recent_volume = df['Volume'].iloc[-5:].mean()
        previous_volume = df['Volume'].iloc[-10:-5].mean()
        
        if recent_volume > previous_volume * 1.2:
            return "increasing"
        elif recent_volume < previous_volume * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile"""
        avg_volume = df['Volume'].mean()
        volume_std = df['Volume'].std()
        
        return {
            "average_volume": float(avg_volume),
            "volume_volatility": float(volume_std / avg_volume) if avg_volume > 0 else 0.0,
            "volume_consistency": "high" if volume_std / avg_volume < 0.5 else "low"
        }
    
    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution Line"""
        if len(df) < 2:
            return 0.0
        
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * df['Volume']).cumsum()
        
        return float(ad_line.iloc[-1])
    
    def _calculate_strength_index(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate custom strength index"""
        # Combine multiple strength factors
        price_strength = self._calculate_price_ratios(df)["52w_range_position"]
        momentum_strength = self._calculate_momentum_score(df)["momentum_score"]
        volume_strength = min(df['Volume'].iloc[-1] / df['Volume'].mean(), 2.0) / 2.0
        
        overall_strength = (price_strength * 0.4 + momentum_strength * 0.4 + volume_strength * 0.2)
        
        return {
            "overall_strength": overall_strength,
            "price_component": price_strength,
            "momentum_component": momentum_strength,
            "volume_component": volume_strength
        }
    
    def _calculate_breakout_probability(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate breakout probability based on technical factors"""
        # Volume increase
        volume_factor = min(df['Volume'].iloc[-1] / df['Volume'].mean(), 3.0) / 3.0
        
        # Range contraction
        recent_ranges = (df['High'] - df['Low']).iloc[-10:]
        avg_range = recent_ranges.mean()
        current_range = recent_ranges.iloc[-1]
        contraction_factor = 1.0 - min(current_range / avg_range, 1.0)
        
        # Proximity to resistance
        high_20d = df['High'].iloc[-20:].max()
        current_price = df['Close'].iloc[-1]
        proximity_factor = current_price / high_20d if high_20d > 0 else 0.5
        
        breakout_probability = (volume_factor * 0.4 + contraction_factor * 0.3 + proximity_factor * 0.3)
        
        return {
            "breakout_probability": breakout_probability,
            "volume_factor": volume_factor,
            "contraction_factor": contraction_factor,
            "proximity_factor": proximity_factor
        }
    
    def _extract_metric_requests(self, message: str) -> List[str]:
        """Extract specific metric requests from message"""
        requests = []
        keywords = ["price ratio", "momentum score", "volatility rank", "strength index", "breakout probability"]
        
        for keyword in keywords:
            if keyword in message.lower():
                requests.append(keyword)
        
        return requests
    
    def _generate_recommendation(self, signals: List[str], confidence: float) -> str:
        """Generate trading recommendation based on signals"""
        if confidence > 0.7 and len(signals) >= 3:
            return "STRONG_BUY" if any("bullish" in s.lower() or "buy" in s.lower() for s in signals) else "STRONG_SELL"
        elif confidence > 0.5:
            return "BUY" if any("bullish" in s.lower() or "buy" in s.lower() for s in signals) else "SELL"
        else:
            return "HOLD"