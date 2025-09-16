"""
Risk Manager Agent for AI Hedge Fund System
Monitors portfolio risk and enforces risk limits
"""
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from decimal import Decimal
import json

from ..base_agent import BaseHedgeFundAgent, AgentCapability
from ...models.portfolio import Portfolio
from ...models.position import Position
from ...models.trade import TradeSignal, Order
from ...utils.logging_config import get_logger, log_risk_alert


class RiskManagerAgent(BaseHedgeFundAgent):
    """
    Agent responsible for monitoring and managing portfolio risk
    Enforces risk limits and provides risk assessments
    """
    
    def __init__(self, **kwargs):
        system_message = """You are a Risk Manager Agent for an AI hedge fund.

Your primary responsibilities:
1. Monitor portfolio-level risk metrics (VaR, drawdown, concentration)
2. Enforce position size limits and sector allocation limits
3. Assess individual trade risks before execution
4. Monitor stop losses and risk-adjusted position sizing
5. Generate risk alerts and recommend risk mitigation actions
6. Perform stress testing and scenario analysis

Key risk metrics to monitor:
- Portfolio Value at Risk (VaR)
- Maximum drawdown
- Position concentration risk
- Sector/industry concentration
- Correlation risk between positions
- Leverage and exposure limits
- Stop loss compliance
- Volatility-adjusted position sizing

Risk assessment framework:
- Evaluate each trade signal for risk/reward
- Check position size against portfolio limits
- Assess correlation with existing positions
- Validate stop loss and profit targets
- Consider market regime in risk assessment

Always prioritize capital preservation while allowing for appropriate risk-taking."""
        
        super().__init__(
            name="risk_manager_agent",
            system_message=system_message,
            capabilities=[
                AgentCapability.RISK_ASSESSMENT,
                AgentCapability.PORTFOLIO_MANAGEMENT
            ],
            **kwargs
        )
    
    def _initialize(self) -> None:
        """Initialize risk manager agent"""
        # Load risk limits from config
        from ...config.settings import settings
        
        self.risk_limits = {
            "max_position_size_pct": 0.05,  # 5% max per position
            "max_sector_allocation_pct": 0.25,  # 25% max per sector
            "max_portfolio_var_pct": 0.02,  # 2% daily VaR
            "max_drawdown_pct": 0.15,  # 15% max drawdown
            "max_correlation": 0.7,  # 70% max correlation
            "min_liquidity_score": 0.7,  # 70% min liquidity
            "max_leverage": 1.0,  # No leverage initially
            "stop_loss_buffer_pct": 0.01  # 1% buffer for stop losses
        }
        
        # Override with config if available
        if hasattr(settings, 'risk_limits'):
            self.risk_limits.update(settings.risk_limits)
        
        self.active_alerts = []
        self.risk_metrics_history = []
        
        self.logger.info("Risk Manager Agent initialized")
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data for risk analysis"""
        if not isinstance(data, dict):
            return False
        
        message_type = data.get("type", "general")
        
        if message_type == "assess_signal_risk":
            return "signal" in data and "portfolio" in data
        elif message_type == "check_portfolio_risk":
            return "portfolio" in data
        elif message_type == "validate_trade":
            return "trade" in data and "portfolio" in data
        elif message_type == "calculate_position_size":
            return "signal" in data and "portfolio" in data
        
        return True
    
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """Process risk management requests"""
        try:
            message_type = message.get("type", "general")
            
            if message_type == "assess_signal_risk":
                return self._assess_signal_risk(message)
            elif message_type == "check_portfolio_risk":
                return self._check_portfolio_risk(message)
            elif message_type == "validate_trade":
                return self._validate_trade(message)
            elif message_type == "calculate_position_size":
                return self._calculate_position_size(message)
            elif message_type == "stress_test":
                return self._perform_stress_test(message)
            elif message_type == "get_risk_limits":
                return self._get_risk_limits()
            else:
                return self._general_risk_assessment(message)
        
        except Exception as e:
            self.logger.error(f"Error processing risk message: {e}")
            return {
                "type": "error",
                "error": str(e),
                "agent": self.name
            }
    
    def _assess_signal_risk(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of a trading signal"""
        signal_data = message.get("signal", {})
        portfolio_data = message.get("portfolio", {})
        
        # Extract signal information
        symbol = signal_data.get("symbol")
        side = signal_data.get("side")
        signal_strength = signal_data.get("signal_strength", 0)
        stop_loss = signal_data.get("stop_loss")
        profit_target = signal_data.get("profit_target")
        
        # Create mock portfolio if needed (in real implementation, would use actual portfolio)
        portfolio = self._create_portfolio_from_data(portfolio_data)
        
        # Calculate risk metrics
        risk_assessment = {
            "symbol": symbol,
            "overall_risk_score": 0.0,
            "risk_factors": [],
            "risk_mitigation": [],
            "recommended_position_size": 0,
            "max_loss_estimate": 0.0,
            "risk_reward_ratio": None,
            "approved": False
        }
        
        # Position size risk
        current_price = signal_data.get("current_price", 100)  # Mock price
        proposed_value = signal_data.get("position_value", portfolio.metrics.total_value * Decimal('0.02'))
        
        position_size_pct = float(proposed_value / portfolio.metrics.total_value) if portfolio.metrics.total_value > 0 else 0
        
        if position_size_pct > self.risk_limits["max_position_size_pct"]:
            risk_assessment["risk_factors"].append(f"Position size {position_size_pct:.2%} exceeds limit")
            risk_assessment["overall_risk_score"] += 0.3
        
        # Stop loss analysis
        if stop_loss and current_price:
            if side == "buy":
                stop_loss_risk = (current_price - stop_loss) / current_price
            else:
                stop_loss_risk = (stop_loss - current_price) / current_price
            
            risk_assessment["max_loss_estimate"] = float(proposed_value * Decimal(str(abs(stop_loss_risk))))
            
            if abs(stop_loss_risk) > 0.10:  # 10% stop loss
                risk_assessment["risk_factors"].append("Stop loss risk exceeds 10%")
                risk_assessment["overall_risk_score"] += 0.2
            
            # Risk/reward ratio
            if profit_target:
                if side == "buy":
                    reward = (profit_target - current_price) / current_price
                else:
                    reward = (current_price - profit_target) / current_price
                
                if abs(stop_loss_risk) > 0:
                    risk_assessment["risk_reward_ratio"] = reward / abs(stop_loss_risk)
                
                if risk_assessment["risk_reward_ratio"] and risk_assessment["risk_reward_ratio"] < 1.5:
                    risk_assessment["risk_factors"].append("Risk/reward ratio below 1.5:1")
                    risk_assessment["overall_risk_score"] += 0.2
        
        # Signal strength risk
        if signal_strength < 0.6:
            risk_assessment["risk_factors"].append("Low signal confidence")
            risk_assessment["overall_risk_score"] += 0.2
        
        # Portfolio concentration
        existing_position = portfolio.get_position(symbol)
        if existing_position and existing_position.is_open:
            risk_assessment["risk_factors"].append("Adding to existing position")
            risk_assessment["overall_risk_score"] += 0.1
        
        # Calculate recommended position size
        risk_assessment["recommended_position_size"] = self._calculate_optimal_position_size(
            portfolio, symbol, current_price, stop_loss, signal_strength
        )
        
        # Final approval
        risk_assessment["approved"] = risk_assessment["overall_risk_score"] < 0.5
        
        if not risk_assessment["approved"]:
            risk_assessment["risk_mitigation"].append("Reduce position size")
            if not stop_loss:
                risk_assessment["risk_mitigation"].append("Set stop loss")
        
        self.log_activity("signal_risk_assessed", level="info",
                         symbol=symbol, risk_score=risk_assessment["overall_risk_score"],
                         approved=risk_assessment["approved"])
        
        return {
            "type": "signal_risk_assessment",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "assessment": risk_assessment,
            "agent": self.name
        }
    
    def _check_portfolio_risk(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall portfolio risk metrics"""
        portfolio_data = message.get("portfolio", {})
        portfolio = self._create_portfolio_from_data(portfolio_data)
        
        risk_summary = portfolio.get_risk_summary()
        alerts = []
        
        # Check limits
        if risk_summary["gross_exposure"] > self.risk_limits["max_leverage"]:
            alerts.append({
                "type": "leverage_exceeded",
                "message": f"Gross exposure {risk_summary['gross_exposure']:.2%} exceeds limit",
                "severity": "high"
            })
        
        if risk_summary["max_drawdown"] > self.risk_limits["max_drawdown_pct"]:
            alerts.append({
                "type": "drawdown_exceeded",
                "message": f"Drawdown {risk_summary['max_drawdown']:.2%} exceeds limit",
                "severity": "critical"
            })
        
        if risk_summary["largest_position"] > self.risk_limits["max_position_size_pct"]:
            alerts.append({
                "type": "position_size_exceeded",
                "message": f"Largest position {risk_summary['largest_position']:.2%} exceeds limit",
                "severity": "medium"
            })
        
        # Sector concentration alerts
        for sector, allocation in risk_summary["sector_concentrations"].items():
            if allocation > self.risk_limits["max_sector_allocation_pct"]:
                alerts.append({
                    "type": "sector_concentration",
                    "message": f"{sector} allocation {allocation:.2%} exceeds limit",
                    "severity": "medium"
                })
        
        # Log risk alerts
        for alert in alerts:
            log_risk_alert(
                alert_type=alert["type"],
                severity=alert["severity"],
                description=alert["message"]
            )
        
        # Update active alerts
        self.active_alerts = alerts
        
        return {
            "type": "portfolio_risk_check",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_summary": risk_summary,
            "alerts": alerts,
            "risk_limits": self.risk_limits,
            "within_limits": len(alerts) == 0,
            "agent": self.name
        }
    
    def _validate_trade(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a trade before execution"""
        trade_data = message.get("trade", {})
        portfolio_data = message.get("portfolio", {})
        
        portfolio = self._create_portfolio_from_data(portfolio_data)
        
        symbol = trade_data.get("symbol")
        side = trade_data.get("side")
        quantity = trade_data.get("quantity", 0)
        price = trade_data.get("price", 0)
        
        validation_result = {
            "approved": True,
            "reasons": [],
            "warnings": [],
            "required_actions": []
        }
        
        # Calculate trade value
        trade_value = Decimal(str(quantity)) * Decimal(str(price))
        
        # Position size check
        if not portfolio.check_position_size_limit(symbol, trade_value):
            validation_result["approved"] = False
            validation_result["reasons"].append("Trade would exceed position size limit")
        
        # Available capital check
        if side == "buy":
            available_capital = portfolio.get_available_capital()
            if trade_value > available_capital:
                validation_result["approved"] = False
                validation_result["reasons"].append("Insufficient capital for trade")
        
        # Market hours check (simplified)
        current_hour = datetime.now().hour
        if current_hour < 9 or current_hour > 16:
            validation_result["warnings"].append("Trading outside market hours")
        
        # Liquidity check (simplified)
        min_volume = 100000  # 100k shares minimum
        if quantity > min_volume * 0.1:  # More than 10% of min volume
            validation_result["warnings"].append("Large order relative to typical volume")
        
        return {
            "type": "trade_validation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation": validation_result,
            "trade_summary": {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "value": float(trade_value)
            },
            "agent": self.name
        }
    
    def _calculate_position_size(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal position size for a signal"""
        signal_data = message.get("signal", {})
        portfolio_data = message.get("portfolio", {})
        
        portfolio = self._create_portfolio_from_data(portfolio_data)
        
        symbol = signal_data.get("symbol")
        current_price = signal_data.get("current_price", 100)
        stop_loss = signal_data.get("stop_loss")
        signal_strength = signal_data.get("signal_strength", 0.5)
        
        optimal_size = self._calculate_optimal_position_size(
            portfolio, symbol, current_price, stop_loss, signal_strength
        )
        
        # Calculate position as percentage of portfolio
        portfolio_value = portfolio.metrics.total_value
        position_value = optimal_size * Decimal(str(current_price))
        position_pct = float(position_value / portfolio_value) if portfolio_value > 0 else 0
        
        return {
            "type": "position_sizing",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "recommended_shares": optimal_size,
            "position_value": float(position_value),
            "position_percentage": position_pct,
            "rationale": self._get_sizing_rationale(signal_strength, stop_loss),
            "agent": self.name
        }
    
    def _perform_stress_test(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Perform portfolio stress test"""
        portfolio_data = message.get("portfolio", {})
        scenario = message.get("scenario", "market_crash")
        
        portfolio = self._create_portfolio_from_data(portfolio_data)
        
        # Define stress scenarios
        stress_scenarios = {
            "market_crash": {"market_move": -0.20, "volatility_spike": 2.0},
            "sector_rotation": {"sector_moves": {"tech": -0.15, "healthcare": 0.10}},
            "volatility_spike": {"volatility_multiplier": 3.0},
            "liquidity_crisis": {"liquidity_discount": 0.20}
        }
        
        scenario_params = stress_scenarios.get(scenario, stress_scenarios["market_crash"])
        
        # Simulate scenario impact
        stressed_portfolio_value = portfolio.metrics.total_value
        
        if "market_move" in scenario_params:
            market_impact = scenario_params["market_move"]
            stressed_portfolio_value *= (1 + Decimal(str(market_impact)))
        
        loss_amount = portfolio.metrics.total_value - stressed_portfolio_value
        loss_percentage = float(loss_amount / portfolio.metrics.total_value) if portfolio.metrics.total_value > 0 else 0
        
        # Risk assessment
        exceeds_limits = abs(loss_percentage) > self.risk_limits["max_drawdown_pct"]
        
        return {
            "type": "stress_test_result",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "original_value": float(portfolio.metrics.total_value),
            "stressed_value": float(stressed_portfolio_value),
            "loss_amount": float(loss_amount),
            "loss_percentage": loss_percentage,
            "exceeds_risk_limits": exceeds_limits,
            "risk_limit": self.risk_limits["max_drawdown_pct"],
            "agent": self.name
        }
    
    def _get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits"""
        return {
            "type": "risk_limits",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "limits": self.risk_limits,
            "agent": self.name
        }
    
    def _calculate_optimal_position_size(
        self,
        portfolio: Portfolio,
        symbol: str,
        price: float,
        stop_loss: Optional[float],
        signal_strength: float
    ) -> int:
        """Calculate optimal position size based on risk parameters"""
        if stop_loss and price > 0:
            # Kelly Criterion-based sizing
            risk_per_share = abs(price - stop_loss)
            risk_percentage = risk_per_share / price
            
            # Base risk per trade (1% of portfolio)
            base_risk_amount = portfolio.metrics.total_value * Decimal('0.01')
            
            # Adjust for signal strength
            strength_multiplier = Decimal(str(signal_strength))
            adjusted_risk = base_risk_amount * strength_multiplier
            
            # Calculate position size
            shares = int(adjusted_risk / Decimal(str(risk_per_share)))
            
            # Apply position size limits
            max_position_value = portfolio.metrics.total_value * Decimal(str(self.risk_limits["max_position_size_pct"]))
            max_shares = int(max_position_value / Decimal(str(price)))
            
            return min(shares, max_shares)
        else:
            # Default to small position without stop loss
            default_value = portfolio.metrics.total_value * Decimal('0.01')
            return int(default_value / Decimal(str(price)))
    
    def _get_sizing_rationale(self, signal_strength: float, stop_loss: Optional[float]) -> str:
        """Get rationale for position sizing"""
        rationale_parts = []
        
        if signal_strength > 0.8:
            rationale_parts.append("High signal confidence allows larger size")
        elif signal_strength < 0.5:
            rationale_parts.append("Low signal confidence requires smaller size")
        
        if stop_loss:
            rationale_parts.append("Position sized based on stop loss risk")
        else:
            rationale_parts.append("Conservative sizing due to no stop loss")
        
        return ". ".join(rationale_parts) if rationale_parts else "Standard risk-based position sizing"
    
    def _create_portfolio_from_data(self, data: Dict[str, Any]) -> Portfolio:
        """Create portfolio object from data (mock implementation)"""
        # In real implementation, this would deserialize actual portfolio
        portfolio = Portfolio(
            initial_cash=Decimal(str(data.get("initial_cash", 1000000))),
            current_cash=Decimal(str(data.get("current_cash", 800000)))
        )
        
        # Add mock metrics if provided
        if "total_value" in data:
            portfolio.metrics.total_value = Decimal(str(data["total_value"]))
        
        return portfolio
    
    def _general_risk_assessment(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general risk assessment requests"""
        return {
            "type": "general_response",
            "message": "Risk management services available. Use 'assess_signal_risk', 'check_portfolio_risk', or 'validate_trade' message types.",
            "available_services": [
                "assess_signal_risk",
                "check_portfolio_risk", 
                "validate_trade",
                "calculate_position_size",
                "stress_test",
                "get_risk_limits"
            ],
            "agent": self.name
        }