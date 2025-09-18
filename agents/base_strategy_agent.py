"""
Base Strategy Agent Class for AI Hedge Fund System
Foundation for all trading strategy agents with standardized proposal generation
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import json

from agents.base_agent import BaseHedgeFundAgent
from agents.technical_analyst import TechnicalAnalystAgent


class StrategyAgent(BaseHedgeFundAgent, ABC):
    """
    Base class for all trading strategy agents
    Provides standardized framework for strategy evaluation and trade proposals
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        strategy_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize strategy agent
        
        Args:
            name: Agent name
            description: Agent description
            strategy_config: Strategy-specific configuration
            **kwargs: Additional arguments for BaseHedgeFundAgent
        """
        super().__init__(name, description, **kwargs)
        
        # Strategy-specific attributes
        self.strategy_config = strategy_config or self._get_default_config()
        self.strategy_criteria = self.strategy_config.get("criteria", {})
        self.risk_parameters = self.strategy_config.get("risk_management", {})
        self.proposal_history = []
        self.active_proposals = {}
        
        # Technical analyst integration
        self.technical_analyst = None  # Will be set during initialization
        
    @abstractmethod
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default strategy configuration - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _evaluate_setup(self, symbol: str, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trading setup based on strategy criteria - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _get_strategy_reasoning(self, symbol: str, evaluation: Dict[str, Any]) -> str:
        """Generate strategy-specific reasoning - must be implemented by subclasses"""
        pass
    
    def _initialize(self) -> None:
        """Initialize strategy agent with technical analyst"""
        # Initialize technical analyst for data requests
        try:
            self.technical_analyst = TechnicalAnalystAgent(
                name=f"{self.agent_name}_tech_analyst",
                description=f"Technical analyst for {self.agent_name}"
            )
            self.logger.info(f"Strategy Agent {self.agent_name} initialized with technical analysis capability")
        except Exception as e:
            self.logger.warning(f"Could not initialize technical analyst: {e}")
            self.technical_analyst = None
    
    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process strategy-related messages and requests
        
        Args:
            message: Message from other agents or director
            context: Optional context with symbol, analysis type, etc.
            
        Returns:
            Strategy response or trade proposal
        """
        try:
            # Parse message intent
            if "analyze" in message.lower() or "evaluate" in message.lower():
                return self._analyze_symbol_for_strategy(context)
            elif "proposal" in message.lower() or "trade" in message.lower():
                return self._generate_trade_proposal(context)
            elif "defend" in message.lower() or "justify" in message.lower():
                return self._defend_proposal(message, context)
            elif "critique" in message.lower() or "review" in message.lower():
                return self._critique_proposal(message, context)
            elif "status" in message.lower() or "summary" in message.lower():
                return self._get_strategy_status()
            else:
                return self._general_strategy_response(message, context)
                
        except Exception as e:
            self.logger.error(f"Error processing strategy message: {e}")
            return {
                "type": "error",
                "message": str(e),
                "agent": self.agent_name
            }
    
    def _analyze_symbol_for_strategy(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze symbol using strategy-specific criteria
        
        Args:
            context: Context with symbol and parameters
            
        Returns:
            Strategy analysis results
        """
        symbol = self._extract_symbol(context)
        
        try:
            # Step 1: Get technical analysis
            technical_data = self._request_technical_analysis(symbol)
            
            if technical_data.get("type") == "error":
                return technical_data
            
            # Step 2: Evaluate against strategy criteria
            strategy_evaluation = self._evaluate_setup(symbol, technical_data)
            
            # Step 3: Generate strategy-specific analysis
            analysis_result = {
                "type": "strategy_analysis",
                "strategy": self.agent_name,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "technical_data": technical_data.get("formatted_for_strategies", {}),
                "strategy_evaluation": strategy_evaluation,
                "setup_quality": strategy_evaluation.get("setup_quality", "not_suitable"),
                "conviction": strategy_evaluation.get("conviction", 0),
                "criteria_met": strategy_evaluation.get("criteria_met", 0),
                "reasoning": self._get_strategy_reasoning(symbol, strategy_evaluation),
                "agent": self.agent_name
            }
            
            self.logger.info(f"Strategy analysis completed for {symbol}: {strategy_evaluation.get('setup_quality', 'unknown')}")
            return analysis_result
            
        except Exception as e:
            return self._handle_strategy_error(symbol, f"Analysis error: {str(e)}")
    
    def _generate_trade_proposal(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate complete trade proposal with entry, stops, targets, and sizing
        
        Args:
            context: Context with symbol and analysis parameters
            
        Returns:
            Complete trade proposal
        """
        symbol = self._extract_symbol(context)
        
        try:
            # Get strategy analysis
            analysis = self._analyze_symbol_for_strategy(context)
            
            if analysis.get("type") == "error":
                return analysis
            
            strategy_eval = analysis["strategy_evaluation"]
            technical_data = analysis["technical_data"]
            
            # Only generate proposals for suitable setups
            setup_quality = strategy_eval.get("setup_quality", "not_suitable")
            if setup_quality in ["not_suitable", "poor"]:
                return {
                    "type": "trade_proposal",
                    "symbol": symbol,
                    "proposal_status": "rejected",
                    "reasoning": f"Setup quality '{setup_quality}' does not meet minimum criteria",
                    "agent": self.agent_name
                }
            
            # Generate trade parameters
            trade_proposal = self._calculate_trade_parameters(symbol, strategy_eval, technical_data)
            
            # Create complete proposal
            proposal = {
                "type": "trade_proposal",
                "strategy": self.agent_name,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "proposal_id": f"{self.agent_name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                
                # Trade parameters
                "entry_signal": trade_proposal["entry_signal"],
                "entry_price": trade_proposal["entry_price"],
                "stop_loss": trade_proposal["stop_loss"],
                "profit_target": trade_proposal["profit_target"],
                "position_size": trade_proposal["position_size"],
                
                # Strategy assessment
                "conviction": strategy_eval.get("conviction", 0),
                "setup_quality": setup_quality,
                "risk_reward_ratio": trade_proposal["risk_reward_ratio"],
                "max_risk_amount": trade_proposal["max_risk_amount"],
                
                # Analysis summary
                "reasoning": self._get_strategy_reasoning(symbol, strategy_eval),
                "criteria_met": strategy_eval.get("criteria_met", 0),
                "technical_summary": self._format_technical_summary(technical_data),
                
                # Metadata
                "strategy_config": self.strategy_config.get("name", self.agent_name),
                "agent": self.agent_name
            }
            
            # Write proposal to file
            proposal_file = self._write_trade_proposal(proposal)
            proposal["proposal_file"] = proposal_file
            
            # Store proposal for defense/critique
            self.active_proposals[proposal["proposal_id"]] = proposal
            self.proposal_history.append(proposal)
            
            self.logger.info(f"Trade proposal generated for {symbol}: {setup_quality} conviction {strategy_eval.get('conviction', 0)}")
            return proposal
            
        except Exception as e:
            return self._handle_strategy_error(symbol, f"Proposal generation error: {str(e)}")
    
    def _calculate_trade_parameters(self, symbol: str, strategy_eval: Dict[str, Any], technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate specific trade parameters based on strategy and technical data
        
        Args:
            symbol: Stock symbol
            strategy_eval: Strategy evaluation results
            technical_data: Technical analysis data
            
        Returns:
            Trade parameters dictionary
        """
        current_price = technical_data.get("price_data", {}).get("current", 0)
        atr = technical_data.get("volatility_data", {}).get("atr", current_price * 0.02)
        
        # Default parameters (can be overridden by subclasses)
        entry_price = current_price
        
        # Stop loss based on ATR or strategy-specific logic
        stop_loss_distance = strategy_eval.get("stop_distance", atr * 1.5)
        stop_loss = entry_price - stop_loss_distance
        
        # Profit target based on risk/reward ratio
        risk_reward_ratio = self.risk_parameters.get("min_risk_reward", 2.0)
        profit_target = entry_price + (stop_loss_distance * risk_reward_ratio)
        
        # Position sizing based on risk management
        max_risk_amount = self.risk_parameters.get("max_risk_per_trade", 1000)
        position_size = int(max_risk_amount / stop_loss_distance) if stop_loss_distance > 0 else 0
        
        return {
            "entry_signal": strategy_eval.get("entry_signal", "market"),
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "profit_target": round(profit_target, 2),
            "position_size": position_size,
            "risk_reward_ratio": risk_reward_ratio,
            "max_risk_amount": max_risk_amount
        }
    
    def _defend_proposal(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Defend a trade proposal against criticism or questions
        
        Args:
            message: Challenge or question about the proposal
            context: Context with proposal ID or symbol
            
        Returns:
            Defense response
        """
        proposal_id = context.get("proposal_id") if context else None
        symbol = self._extract_symbol(context)
        
        try:
            # Find relevant proposal
            proposal = None
            if proposal_id and proposal_id in self.active_proposals:
                proposal = self.active_proposals[proposal_id]
            else:
                # Find most recent proposal for symbol
                symbol_proposals = [p for p in self.proposal_history if p["symbol"] == symbol]
                proposal = symbol_proposals[-1] if symbol_proposals else None
            
            if not proposal:
                return {
                    "type": "defense_response",
                    "message": f"No proposal found for {symbol or proposal_id}",
                    "agent": self.agent_name
                }
            
            # Generate defense based on proposal strength and criticism
            defense_points = self._generate_defense_points(proposal, message)
            
            return {
                "type": "defense_response",
                "proposal_id": proposal["proposal_id"],
                "symbol": proposal["symbol"],
                "defense_points": defense_points,
                "conviction_maintained": proposal["conviction"],
                "strategy_confidence": proposal["setup_quality"],
                "response": self._format_defense_response(defense_points, proposal),
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_strategy_error(symbol, f"Defense error: {str(e)}")
    
    def _critique_proposal(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Critique another strategy agent's proposal
        
        Args:
            message: Message containing proposal to critique
            context: Context with proposal data
            
        Returns:
            Critique response
        """
        try:
            # Extract proposal from context or message
            other_proposal = context.get("proposal") if context else None
            
            if not other_proposal:
                return {
                    "type": "critique_response",
                    "message": "No proposal provided for critique",
                    "agent": self.agent_name
                }
            
            # Analyze proposal from our strategy perspective
            critique_points = self._generate_critique_points(other_proposal)
            
            return {
                "type": "critique_response",
                "critiqued_proposal": other_proposal.get("proposal_id", "unknown"),
                "critiqued_symbol": other_proposal.get("symbol", "unknown"),
                "critique_points": critique_points,
                "overall_assessment": self._assess_other_proposal(other_proposal),
                "alternative_perspective": self._suggest_alternative_approach(other_proposal),
                "agent": self.agent_name
            }
            
        except Exception as e:
            return self._handle_strategy_error("unknown", f"Critique error: {str(e)}")
    
    def _request_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Request technical analysis from Technical Analyst"""
        try:
            if self.technical_analyst is None:
                # Try to initialize it again
                self.technical_analyst = TechnicalAnalystAgent(
                    name=f"{self.agent_name}_tech_analyst",
                    description=f"Technical analyst for {self.agent_name}"
                )
            
            context = {"symbol": symbol}
            message = "Please provide comprehensive technical analysis"
            return self.technical_analyst.process_message(message, context)
            
        except Exception as e:
            self.logger.error(f"Error requesting technical analysis for {symbol}: {e}")
            return {"type": "error", "message": str(e)}
    
    def _write_trade_proposal(self, proposal: Dict[str, Any]) -> str:
        """
        Write trade proposal to markdown file
        
        Args:
            proposal: Complete trade proposal
            
        Returns:
            Path to proposal file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_proposal_{proposal['symbol']}_{self.agent_name}_{timestamp}.md"
            filepath = Path("proposals") / filename
            
            # Create markdown content
            content = self._format_proposal_markdown(proposal)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Trade proposal written to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error writing proposal: {e}")
            return ""
    
    def _format_proposal_markdown(self, proposal: Dict[str, Any]) -> str:
        """Format trade proposal as markdown"""
        
        # Risk/reward calculation
        entry = proposal["entry_price"]
        stop = proposal["stop_loss"]
        target = proposal["profit_target"]
        risk_amount = (entry - stop) * proposal["position_size"]
        reward_amount = (target - entry) * proposal["position_size"]
        
        content = f"""# Trade Proposal - {proposal['symbol']}

**Strategy**: {proposal['strategy']}  
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Proposal ID**: {proposal['proposal_id']}  

## Trade Summary

**Symbol**: {proposal['symbol']}  
**Setup Quality**: {proposal['setup_quality'].title().replace('_', ' ')}  
**Conviction**: {proposal['conviction']}/10  
**Entry Signal**: {proposal['entry_signal']}  

## Trade Parameters

| Parameter | Value |
|-----------|-------|
| Entry Price | ${proposal['entry_price']:.2f} |
| Stop Loss | ${proposal['stop_loss']:.2f} |
| Profit Target | ${proposal['profit_target']:.2f} |
| Position Size | {proposal['position_size']} shares |
| Risk/Reward | {proposal['risk_reward_ratio']:.1f}:1 |

## Risk Analysis

- **Risk Amount**: ${risk_amount:.2f}
- **Reward Potential**: ${reward_amount:.2f}
- **Max Risk**: ${proposal['max_risk_amount']:.2f}
- **Stop Distance**: {((entry - stop) / entry * 100):.1f}%
- **Target Distance**: {((target - entry) / entry * 100):.1f}%

## Strategy Reasoning

{proposal['reasoning']}

## Technical Summary

{proposal['technical_summary']}

## Criteria Analysis

**Criteria Met**: {proposal['criteria_met']}/{len(self.strategy_criteria)}

"""
        
        # Add criteria details if available
        if hasattr(self, '_format_criteria_analysis'):
            content += self._format_criteria_analysis(proposal)
        
        content += f"""
---

*Proposal generated by {self.agent_name} using {self.strategy_config.get('name', 'Unknown')} strategy*  
*Generated at {proposal['timestamp']}*
"""
        
        return content
    
    # Helper methods
    
    def _extract_symbol(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Extract symbol from context"""
        if context and "symbol" in context:
            return context["symbol"]
        else:
            return "SPY"  # Default fallback
    
    def _handle_strategy_error(self, symbol: str, error_msg: str) -> Dict[str, Any]:
        """Handle strategy errors gracefully"""
        self.logger.error(f"Strategy error for {symbol}: {error_msg}")
        return {
            "type": "error",
            "symbol": symbol,
            "message": error_msg,
            "agent": self.agent_name
        }
    
    def _format_technical_summary(self, technical_data: Dict[str, Any]) -> str:
        """Format technical data into readable summary"""
        price_data = technical_data.get("price_data", {})
        momentum_data = technical_data.get("momentum_data", {})
        volume_data = technical_data.get("volume_data", {})
        
        return f"""**Technical Indicators:**
- Current Price: ${price_data.get('current', 0):.2f}
- Trend Intensity: {price_data.get('trend_intensity', 0):.3f}
- RSI: {momentum_data.get('rsi', 0):.1f}
- 22-day Performance: {momentum_data.get('gain_22d', 0):+.1f}%
- Volume Ratio: {volume_data.get('volume_ratio', 0):.2f}x
- Distance from 52w High: {momentum_data.get('distance_52w_high', 0):.1f}%"""
    
    def _generate_defense_points(self, proposal: Dict[str, Any], criticism: str) -> List[str]:
        """Generate defense points for proposal"""
        defense_points = []
        
        # Standard defense points
        if proposal["conviction"] >= 7:
            defense_points.append(f"High conviction setup ({proposal['conviction']}/10) based on strong technical alignment")
        
        if proposal["risk_reward_ratio"] >= 2.0:
            defense_points.append(f"Favorable risk/reward ratio of {proposal['risk_reward_ratio']:.1f}:1")
        
        if proposal["criteria_met"] >= len(self.strategy_criteria) * 0.8:
            defense_points.append(f"Strong criteria satisfaction: {proposal['criteria_met']}/{len(self.strategy_criteria)}")
        
        # Strategy-specific defenses can be added by subclasses
        
        return defense_points
    
    def _generate_critique_points(self, other_proposal: Dict[str, Any]) -> List[str]:
        """Generate critique points for another proposal"""
        critique_points = []
        
        # Standard critique areas
        if other_proposal.get("risk_reward_ratio", 0) < 2.0:
            critique_points.append("Risk/reward ratio below 2:1 minimum threshold")
        
        if other_proposal.get("conviction", 0) < 5:
            critique_points.append("Low conviction level suggests weak setup")
        
        # Strategy-specific critiques can be added by subclasses
        
        return critique_points
    
    def _assess_other_proposal(self, other_proposal: Dict[str, Any]) -> str:
        """Assess another proposal from this strategy's perspective"""
        conviction = other_proposal.get("conviction", 0)
        risk_reward = other_proposal.get("risk_reward_ratio", 0)
        
        if conviction >= 7 and risk_reward >= 2.0:
            return "Strong proposal with good fundamentals"
        elif conviction >= 5 and risk_reward >= 1.5:
            return "Moderate proposal with acceptable parameters"
        else:
            return "Weak proposal with concerning risk/reward characteristics"
    
    def _suggest_alternative_approach(self, other_proposal: Dict[str, Any]) -> str:
        """Suggest alternative approach from this strategy's perspective"""
        return f"From a {self.agent_name} perspective, consider waiting for stronger momentum confirmation and tighter risk parameters."
    
    def _format_defense_response(self, defense_points: List[str], proposal: Dict[str, Any]) -> str:
        """Format defense response"""
        response = f"Defending {proposal['symbol']} proposal with {proposal['conviction']}/10 conviction:\n\n"
        
        for i, point in enumerate(defense_points, 1):
            response += f"{i}. {point}\n"
        
        response += f"\nStrategy maintains confidence in this {proposal['setup_quality']} setup."
        return response
    
    def _get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        return {
            "type": "strategy_status",
            "strategy": self.agent_name,
            "active_proposals": len(self.active_proposals),
            "total_proposals": len(self.proposal_history),
            "strategy_config": self.strategy_config.get("name", self.agent_name),
            "agent": self.agent_name
        }
    
    def _general_strategy_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle general strategy inquiries"""
        return {
            "type": "general_response",
            "message": f"{self.agent_name} strategy agent ready for analysis and trade proposals.",
            "capabilities": [
                "Symbol analysis against strategy criteria",
                "Trade proposal generation with risk management",
                "Proposal defense in conversations",
                "Critique of other strategies' proposals",
                "Technical analysis integration"
            ],
            "strategy": self.strategy_config.get("name", self.agent_name),
            "agent": self.agent_name
        }