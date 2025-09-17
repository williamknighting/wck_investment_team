"""
Fund Head Agent for AI Hedge Fund System
Lead decision maker who analyzes market regimes and makes final investment calls
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
import json

from .base_agent import BaseHedgeFundAgent, AgentCapability, AgentStatus
from ..models.market_state import MarketState, MarketCondition, MarketRegime, MarketTrend, VolatilityRegime
from ..utils.logging_config import get_logger


class FundHeadAgent(BaseHedgeFundAgent):
    """
    Fund Head Agent - Lead decision maker for the hedge fund
    Analyzes market regimes and makes final investment decisions based on team input
    """
    
    def __init__(self, **kwargs):
        system_message = """You are the Fund Head of an AI hedge fund.

Your responsibilities:
1. Analyze current market conditions using technical and fundamental indicators
2. Classify market regimes (bull, bear, sideways, high volatility, etc.)
3. Assess regime stability and probability of regime changes
4. Provide market context for trading strategies
5. Monitor key market indicators and alert on significant changes

Key indicators to monitor:
- SPY price and moving averages (20, 50, 200 day)
- VIX levels and volatility regime
- Market breadth (advance/decline, new highs/lows)
- Interest rates and yield curve
- Sector rotation patterns
- Momentum and trend strength

Response format should include:
- Current regime classification with confidence score
- Supporting evidence and conflicting indicators
- Regime change probability assessment
- Trading implications and strategy recommendations

Be concise but thorough in your analysis. Focus on actionable insights."""
        
        super().__init__(
            name="fund_head_agent",
            system_message=system_message,
            capabilities=[
                AgentCapability.MARKET_ANALYSIS,
                AgentCapability.RESEARCH
            ],
            **kwargs
        )
    
    def _initialize(self) -> None:
        """Initialize market regime agent"""
        self.market_state = MarketState()
        self.indicators_cache = {}
        self.regime_history = []
        self.alert_thresholds = {
            "vix_high": 30,
            "vix_extreme": 40,
            "trend_change_threshold": 0.7,
            "volatility_spike": 2.0
        }
        
        self.logger.info("Fund Head Agent initialized")
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data for market analysis"""
        if not isinstance(data, dict):
            return False
        
        # Check for required fields based on message type
        message_type = data.get("type", "general")
        
        if message_type == "analyze_market":
            return "market_data" in data
        elif message_type == "classify_regime":
            return "conditions" in data
        elif message_type == "update_conditions":
            return "market_data" in data
        
        return True
    
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """Process market regime analysis requests"""
        try:
            message_type = message.get("type", "general")
            
            if message_type == "analyze_market":
                return self._analyze_market_conditions(message)
            elif message_type == "classify_regime":
                return self._classify_market_regime(message)
            elif message_type == "update_conditions":
                return self._update_market_conditions(message)
            elif message_type == "get_regime_status":
                return self._get_regime_status()
            elif message_type == "check_alerts":
                return self._check_market_alerts()
            else:
                return self._general_market_analysis(message)
        
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                "type": "error",
                "error": str(e),
                "agent": self.name
            }
    
    def _analyze_market_conditions(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market conditions"""
        market_data = message.get("market_data", {})
        
        # Create market conditions
        conditions = MarketCondition(
            spy_price=Decimal(str(market_data.get("spy_price", 0))) if market_data.get("spy_price") else None,
            spy_change_pct=market_data.get("spy_change_pct"),
            vix_level=Decimal(str(market_data.get("vix_level", 0))) if market_data.get("vix_level") else None,
            rsi_14d=market_data.get("rsi_14d"),
            ten_year_yield=Decimal(str(market_data.get("ten_year_yield", 0))) if market_data.get("ten_year_yield") else None,
            put_call_ratio=market_data.get("put_call_ratio")
        )
        
        # Analyze trends
        self._analyze_market_trends(conditions, market_data)
        
        # Update market state
        self.market_state.update_conditions(conditions)
        
        # Generate analysis
        analysis = {
            "type": "market_analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conditions": {
                "spy_price": float(conditions.spy_price) if conditions.spy_price else None,
                "spy_change_pct": conditions.spy_change_pct,
                "vix_level": float(conditions.vix_level) if conditions.vix_level else None,
                "market_trend": conditions.market_trend.value,
                "volatility_regime": conditions.volatility_regime.value,
                "trend_strength": conditions.trend_strength
            },
            "regime_classification": self.market_state.regime.primary_regime.value,
            "confidence": self.market_state.regime.confidence_score,
            "analysis": self._generate_market_commentary(conditions),
            "alerts": self._check_condition_alerts(conditions),
            "agent": self.name
        }
        
        self.log_activity("market_analysis_completed", level="info", 
                         spy_price=float(conditions.spy_price) if conditions.spy_price else None,
                         vix_level=float(conditions.vix_level) if conditions.vix_level else None,
                         regime=self.market_state.regime.primary_regime.value)
        
        return analysis
    
    def _analyze_market_trends(self, conditions: MarketCondition, market_data: Dict[str, Any]) -> None:
        """Analyze market trends and momentum"""
        # Simple trend analysis based on moving averages
        spy_price = conditions.spy_price
        if spy_price:
            ma_20 = market_data.get("ma_20")
            ma_50 = market_data.get("ma_50")
            ma_200 = market_data.get("ma_200")
            
            if ma_20:
                conditions.spy_above_ma20 = spy_price > Decimal(str(ma_20))
            if ma_50:
                conditions.spy_above_ma50 = spy_price > Decimal(str(ma_50))
            if ma_200:
                conditions.spy_above_ma200 = spy_price > Decimal(str(ma_200))
            
            # Determine trend
            trend_score = 0
            if conditions.spy_above_ma20:
                trend_score += 1
            if conditions.spy_above_ma50:
                trend_score += 1
            if conditions.spy_above_ma200:
                trend_score += 1
            
            # Set trend based on score and price change
            if trend_score >= 3 and conditions.spy_change_pct and conditions.spy_change_pct > 1:
                conditions.market_trend = MarketTrend.STRONG_UP
                conditions.trend_strength = 0.8
            elif trend_score >= 2 and conditions.spy_change_pct and conditions.spy_change_pct > 0:
                conditions.market_trend = MarketTrend.MODERATE_UP
                conditions.trend_strength = 0.6
            elif trend_score <= 0 and conditions.spy_change_pct and conditions.spy_change_pct < -1:
                conditions.market_trend = MarketTrend.STRONG_DOWN
                conditions.trend_strength = 0.8
            elif trend_score <= 1 and conditions.spy_change_pct and conditions.spy_change_pct < 0:
                conditions.market_trend = MarketTrend.MODERATE_DOWN
                conditions.trend_strength = 0.6
            else:
                conditions.market_trend = MarketTrend.SIDEWAYS
                conditions.trend_strength = 0.3
        
        # Volatility regime analysis
        if conditions.vix_level:
            if conditions.vix_level > 40:
                conditions.volatility_regime = VolatilityRegime.EXTREME
            elif conditions.vix_level > 30:
                conditions.volatility_regime = VolatilityRegime.HIGH
            elif conditions.vix_level > 20:
                conditions.volatility_regime = VolatilityRegime.ELEVATED
            elif conditions.vix_level < 15:
                conditions.volatility_regime = VolatilityRegime.LOW
            else:
                conditions.volatility_regime = VolatilityRegime.NORMAL
        
        # Momentum analysis
        if conditions.rsi_14d:
            if conditions.rsi_14d > 70:
                conditions.momentum_score = 0.8
            elif conditions.rsi_14d > 60:
                conditions.momentum_score = 0.6
            elif conditions.rsi_14d < 30:
                conditions.momentum_score = -0.8
            elif conditions.rsi_14d < 40:
                conditions.momentum_score = -0.6
            else:
                conditions.momentum_score = 0.0
    
    def _classify_market_regime(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Classify current market regime"""
        # Use market state's classification logic
        regime_classification = self.market_state.classify_regime()
        
        # Update market state
        self.market_state.update_regime(regime_classification)
        
        return {
            "type": "regime_classification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "primary_regime": regime_classification.primary_regime.value,
            "confidence_score": regime_classification.confidence_score,
            "regime_probabilities": regime_classification.regime_probabilities,
            "recommended_strategies": regime_classification.recommended_strategies,
            "risk_level": regime_classification.risk_level,
            "position_sizing_multiplier": regime_classification.position_sizing_multiplier,
            "regime_change_probability": regime_classification.regime_change_probability,
            "agent": self.name
        }
    
    def _update_market_conditions(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Update market conditions with new data"""
        return self._analyze_market_conditions(message)
    
    def _get_regime_status(self) -> Dict[str, Any]:
        """Get current regime status"""
        return {
            "type": "regime_status",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_regime": {
                "primary": self.market_state.regime.primary_regime.value,
                "confidence": self.market_state.regime.confidence_score,
                "change_probability": self.market_state.regime.regime_change_probability
            },
            "market_conditions": {
                "trend": self.market_state.current_conditions.market_trend.value,
                "volatility": self.market_state.current_conditions.volatility_regime.value,
                "trend_strength": self.market_state.current_conditions.trend_strength
            },
            "active_alerts": self.market_state.active_alerts,
            "trading_recommendations": self.market_state.get_trading_recommendations(),
            "agent": self.name
        }
    
    def _check_market_alerts(self) -> Dict[str, Any]:
        """Check for market alerts and warnings"""
        alerts = []
        conditions = self.market_state.current_conditions
        
        # VIX alerts
        if conditions.vix_level:
            if conditions.vix_level > self.alert_thresholds["vix_extreme"]:
                alerts.append({
                    "type": "volatility_extreme",
                    "message": f"VIX at extreme level: {conditions.vix_level}",
                    "severity": "high"
                })
            elif conditions.vix_level > self.alert_thresholds["vix_high"]:
                alerts.append({
                    "type": "volatility_elevated",
                    "message": f"VIX at elevated level: {conditions.vix_level}",
                    "severity": "medium"
                })
        
        # Trend change alerts
        if self.market_state.regime.regime_change_probability > self.alert_thresholds["trend_change_threshold"]:
            alerts.append({
                "type": "regime_change",
                "message": f"High probability of regime change: {self.market_state.regime.regime_change_probability:.2%}",
                "severity": "medium"
            })
        
        # Update market state alerts
        self.market_state.active_alerts = [alert["message"] for alert in alerts]
        
        return {
            "type": "market_alerts",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alerts": alerts,
            "alert_count": len(alerts),
            "agent": self.name
        }
    
    def _generate_market_commentary(self, conditions: MarketCondition) -> str:
        """Generate human-readable market commentary"""
        commentary_parts = []
        
        # Trend commentary
        if conditions.market_trend == MarketTrend.STRONG_UP:
            commentary_parts.append("Market showing strong upward momentum")
        elif conditions.market_trend == MarketTrend.MODERATE_UP:
            commentary_parts.append("Market in moderate uptrend")
        elif conditions.market_trend == MarketTrend.STRONG_DOWN:
            commentary_parts.append("Market in strong downtrend")
        elif conditions.market_trend == MarketTrend.MODERATE_DOWN:
            commentary_parts.append("Market showing moderate weakness")
        else:
            commentary_parts.append("Market trending sideways")
        
        # Volatility commentary
        if conditions.volatility_regime == VolatilityRegime.EXTREME:
            commentary_parts.append("extreme volatility environment")
        elif conditions.volatility_regime == VolatilityRegime.HIGH:
            commentary_parts.append("elevated volatility conditions")
        elif conditions.volatility_regime == VolatilityRegime.LOW:
            commentary_parts.append("low volatility environment")
        
        # VIX specific commentary
        if conditions.vix_level:
            if conditions.vix_level > 30:
                commentary_parts.append(f"VIX at {conditions.vix_level} indicates fear in markets")
            elif conditions.vix_level < 15:
                commentary_parts.append(f"VIX at {conditions.vix_level} suggests complacency")
        
        return ". ".join(commentary_parts) + "."
    
    def _check_condition_alerts(self, conditions: MarketCondition) -> List[Dict[str, str]]:
        """Check conditions for alert triggers"""
        alerts = []
        
        # Major trend change
        if len(self.regime_history) > 0:
            last_trend = self.regime_history[-1].get("trend")
            if last_trend and last_trend != conditions.market_trend.value:
                alerts.append({
                    "type": "trend_change",
                    "message": f"Market trend changed from {last_trend} to {conditions.market_trend.value}"
                })
        
        # Extreme conditions
        if conditions.vix_level and conditions.vix_level > 40:
            alerts.append({
                "type": "extreme_volatility",
                "message": f"Extreme volatility detected: VIX at {conditions.vix_level}"
            })
        
        return alerts
    
    def _general_market_analysis(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general market analysis requests"""
        content = message.get("content", "")
        
        if "regime" in content.lower():
            return self._get_regime_status()
        elif "alert" in content.lower():
            return self._check_market_alerts()
        else:
            return {
                "type": "general_response",
                "message": "Market regime analysis available. Use 'analyze_market', 'classify_regime', or 'get_regime_status' message types.",
                "current_regime": self.market_state.regime.primary_regime.value,
                "confidence": self.market_state.regime.confidence_score,
                "agent": self.name
            }
    
    def make_investment_decision(self, symbol: str, agent_analyses: List[Dict[str, Any]], market_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make final investment decision as Fund Head
        
        Args:
            symbol: Symbol being analyzed
            agent_analyses: List of analyses from strategy agents
            market_context: Additional market context
            
        Returns:
            Final investment decision
        """
        
        print(f"\n🏛️ FUND HEAD DECISION - {symbol}")
        print("=" * 60)
        print(f"Fund Head: AI Hedge Fund Leader | Symbol: {symbol}")
        
        # Phase 1: Review market regime
        print(f"\n📋 PHASE 1: Market Regime Assessment")
        current_regime = self.market_state.regime.primary_regime.value
        confidence = self.market_state.regime.confidence_score
        print(f"   Current Market Regime: {current_regime} (confidence: {confidence:.1f})")
        
        # Phase 2: Review strategy analyses
        print(f"\n📊 PHASE 2: Strategy Team Reports")
        if not agent_analyses:
            print("   ❌ No strategy analyses available")
            return {
                "symbol": symbol,
                "decision": "Hold",
                "position_size": "None",
                "reasoning": "No strategy team input available",
                "fund_head": "Fund Head Agent",
                "market_regime": current_regime,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        recommendations = []
        for analysis in agent_analyses:
            agent_name = analysis.get("agent", "Unknown")
            assessment = analysis.get("assessment", {})
            recommendation = assessment.get("recommendation", "Hold")
            recommendations.append(recommendation)
            print(f"   • {agent_name}: {recommendation}")
        
        # Phase 3: Fund head decision logic
        print(f"\n👔 PHASE 3: Fund Head Decision")
        
        # Count recommendations
        buy_votes = sum(1 for r in recommendations if r in ["Strong Buy", "Buy"])
        sell_votes = sum(1 for r in recommendations if r in ["Strong Sell", "Sell"])
        hold_votes = len(recommendations) - buy_votes - sell_votes
        
        # Simple decision logic based on market regime and team consensus
        if current_regime == "Bull" and buy_votes > sell_votes:
            if buy_votes >= len(recommendations) * 0.6:
                decision = "Buy"
                position_size = "Medium"
                reasoning = f"Bull market + team consensus ({buy_votes}/{len(recommendations)} buy votes)"
            else:
                decision = "Hold"
                position_size = "Small"
                reasoning = f"Bull market but mixed signals ({buy_votes} buy, {sell_votes} sell)"
        elif current_regime == "Bear":
            decision = "Hold" 
            position_size = "None"
            reasoning = f"Bear market - staying defensive regardless of team input"
        else:  # Choppy/Neutral
            if buy_votes > len(recommendations) * 0.8:
                decision = "Buy"
                position_size = "Small"
                reasoning = f"Choppy market - only strong consensus ({buy_votes}/{len(recommendations)})"
            else:
                decision = "Hold"
                position_size = "None"
                reasoning = f"Choppy market - insufficient consensus"
        
        print(f"   Decision: {decision}")
        print(f"   Position Size: {position_size}")
        print(f"   Reasoning: {reasoning}")
        
        return {
            "symbol": symbol,
            "decision": decision,
            "position_size": position_size,
            "reasoning": reasoning,
            "fund_head": "Fund Head Agent",
            "market_regime": current_regime,
            "regime_confidence": confidence,
            "team_votes": {
                "buy": buy_votes,
                "sell": sell_votes, 
                "hold": hold_votes
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }