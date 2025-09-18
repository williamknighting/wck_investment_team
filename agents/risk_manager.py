"""
Risk Manager Agent for AI Hedge Fund System
Conservative portfolio risk management with proposal evaluation and veto authority
"""
import os
import json
import glob
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass

from agents.base_agent import BaseHedgeFundAgent


@dataclass
class RiskAssessment:
    """Risk assessment result for a proposal"""
    proposal_id: str
    symbol: str
    overall_risk_score: float  # 0-10 scale (0=low risk, 10=extreme risk)
    concentration_risk: float
    correlation_risk: float
    reward_risk_ratio: float
    market_regime_risk: float
    position_size_risk: float
    veto_recommended: bool
    concerns: List[str]
    suggested_adjustments: List[str]


class RiskManagerAgent(BaseHedgeFundAgent):
    """
    Conservative Risk Manager Agent that monitors and evaluates all trade proposals
    Has veto authority over trades that exceed risk parameters
    """
    
    def __init__(self, name: str = "risk_manager", description: str = "Conservative portfolio risk manager", **kwargs):
        """Initialize Risk Manager Agent"""
        super().__init__(
            name=name,
            description=description,
            system_message=self._get_system_message(),
            **kwargs
        )
    
    def _get_system_message(self) -> str:
        """Get system message for Risk Manager"""
        return """You are the Risk Manager for an AI hedge fund. Your role is to:

1. CONSERVATIVELY evaluate all trade proposals for risk
2. CHALLENGE aggressive proposals with specific concerns
3. VETO trades that exceed risk parameters
4. SUGGEST position size reductions when appropriate
5. MONITOR portfolio concentration and correlation risks

You are naturally conservative and err on the side of caution. Your primary goal is capital preservation.
You should be skeptical of high-conviction proposals and demand strong justification for concentrated positions.
Always voice specific, actionable concerns rather than general warnings."""
    
    def _initialize(self) -> None:
        """Initialize Risk Manager specific components"""
        self.risk_parameters = self._load_risk_parameters()
        self.proposals_folder = Path("proposals")
        self.current_positions = {}  # Will be loaded from portfolio system
        self.market_regime = "neutral"  # Will be determined from market analysis
        
        # Ensure proposals folder exists
        self.proposals_folder.mkdir(exist_ok=True)
        
        self.logger.info("Risk Manager Agent initialized with conservative parameters")
    
    def _load_risk_parameters(self) -> Dict[str, Any]:
        """Load risk management parameters"""
        return {
            # Portfolio concentration limits
            "max_single_position": 0.05,  # 5% max position size
            "max_sector_concentration": 0.20,  # 20% max sector exposure
            "max_correlation_exposure": 0.30,  # 30% max correlated positions
            
            # Risk/Reward requirements
            "min_risk_reward_ratio": 2.0,  # Minimum 2:1 R/R
            "max_portfolio_risk": 0.02,  # 2% max portfolio risk per trade
            
            # Position sizing limits
            "max_position_value": 100000,  # $100k max position
            "max_leverage": 1.0,  # No leverage allowed
            
            # Market regime adjustments
            "bear_market_position_reduction": 0.5,  # 50% size reduction in bear markets
            "high_volatility_threshold": 0.25,  # 25% VIX threshold
            
            # Veto thresholds
            "veto_risk_score": 8.0,  # Auto-veto if risk score > 8
            "challenge_risk_score": 6.0,  # Challenge if risk score > 6
        }
    
    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process incoming message for risk management tasks"""
        try:
            message_lower = message.lower()
            
            if "evaluate all proposals" in message_lower or "evaluate proposals" in message_lower or "risk review" in message_lower:
                return self._evaluate_all_proposals()
            elif "evaluate this" in message_lower or "evaluate specific" in message_lower:
                proposal_id = context.get("proposal_id") if context else None
                return self._evaluate_specific_proposal(proposal_id)
            elif "portfolio risk" in message_lower:
                return self._assess_portfolio_risk()
            elif "veto" in message_lower or "block" in message_lower:
                proposal_id = context.get("proposal_id") if context else None
                return self._veto_proposal(proposal_id, message)
            elif "challenge" in message_lower or "concern" in message_lower:
                proposal_id = context.get("proposal_id") if context else None
                return self._challenge_proposal(proposal_id, message)
            elif "market regime" in message_lower:
                return self._assess_market_regime()
            elif "status" in message_lower:
                return self._get_risk_status()
            else:
                return self._provide_risk_guidance(message, context)
                
        except Exception as e:
            self.logger.error(f"Error processing risk manager message: {e}")
            return {
                "type": "error",
                "message": f"Risk management error: {str(e)}",
                "agent": self.agent_name
            }
    
    def _evaluate_all_proposals(self) -> Dict[str, Any]:
        """Evaluate all proposals in the proposals folder"""
        try:
            proposals = self._scan_proposals_folder()
            
            if not proposals:
                return {
                    "type": "risk_evaluation",
                    "message": "No proposals found to evaluate",
                    "proposals_count": 0,
                    "agent": self.agent_name
                }
            
            assessments = []
            vetoed_proposals = []
            challenged_proposals = []
            
            for proposal in proposals:
                assessment = self._assess_proposal_risk(proposal)
                assessments.append(assessment)
                
                if assessment.veto_recommended:
                    vetoed_proposals.append(assessment)
                elif assessment.overall_risk_score >= self.risk_parameters["challenge_risk_score"]:
                    challenged_proposals.append(assessment)
            
            # Calculate portfolio-level risks
            portfolio_risk = self._calculate_portfolio_risk(proposals)
            
            return {
                "type": "risk_evaluation",
                "proposals_evaluated": len(proposals),
                "assessments": [self._assessment_to_dict(a) for a in assessments],
                "vetoed_count": len(vetoed_proposals),
                "challenged_count": len(challenged_proposals),
                "portfolio_risk_score": portfolio_risk,
                "vetoed_proposals": [a.proposal_id for a in vetoed_proposals],
                "challenged_proposals": [a.proposal_id for a in challenged_proposals],
                "summary": self._generate_risk_summary(assessments, portfolio_risk),
                "agent": self.agent_name
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating proposals: {e}")
            return {
                "type": "error",
                "message": f"Proposal evaluation error: {str(e)}",
                "agent": self.agent_name
            }
    
    def _scan_proposals_folder(self) -> List[Dict[str, Any]]:
        """Scan proposals folder for trade proposals"""
        proposals = []
        
        try:
            # Look for .md files in proposals folder
            proposal_files = glob.glob(str(self.proposals_folder / "*.md"))
            
            for file_path in proposal_files:
                try:
                    proposal = self._parse_proposal_file(file_path)
                    if proposal:
                        proposals.append(proposal)
                except Exception as e:
                    self.logger.warning(f"Could not parse proposal file {file_path}: {e}")
            
            self.logger.info(f"Scanned {len(proposals)} proposals from {len(proposal_files)} files")
            return proposals
            
        except Exception as e:
            self.logger.error(f"Error scanning proposals folder: {e}")
            return []
    
    def _parse_proposal_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse a proposal markdown file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract metadata from header
            proposal = {
                "file_path": file_path,
                "filename": os.path.basename(file_path)
            }
            
            # Parse header information
            lines = content.split('\n')
            for line in lines:
                if line.startswith('**Agent**:'):
                    proposal["agent"] = line.split(':', 1)[1].strip()
                elif line.startswith('**Symbol**:'):
                    proposal["symbol"] = line.split(':', 1)[1].strip()
                elif line.startswith('**Date**:'):
                    proposal["date"] = line.split(':', 1)[1].strip()
            
            # Extract proposal details from content
            proposal_data = self._extract_proposal_data(content)
            proposal.update(proposal_data)
            
            # Generate proposal ID from filename
            proposal["proposal_id"] = os.path.splitext(os.path.basename(file_path))[0]
            
            return proposal
            
        except Exception as e:
            self.logger.error(f"Error parsing proposal file {file_path}: {e}")
            return None
    
    def _extract_proposal_data(self, content: str) -> Dict[str, Any]:
        """Extract proposal data from markdown content"""
        data = {}
        
        # Look for common patterns in proposal content
        patterns = {
            "entry_price": r"(?:Entry|Enter).*?[\$]?([\d,]+\.?\d*)",
            "stop_loss": r"(?:Stop|Stop Loss).*?[\$]?([\d,]+\.?\d*)",
            "profit_target": r"(?:Target|Profit Target).*?[\$]?([\d,]+\.?\d*)",
            "position_size": r"(?:Size|Position|Shares).*?(\d+).*?shares?",
            "conviction": r"(?:Conviction).*?(\d+)(?:/10)?",
            "risk_reward": r"(?:R/R|Risk[/\\]Reward).*?(\d+\.?\d*):1"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1).replace(',', '')
                    if key in ["entry_price", "stop_loss", "profit_target"]:
                        data[key] = float(value)
                    elif key in ["position_size", "conviction"]:
                        data[key] = int(value)
                    elif key == "risk_reward":
                        data["risk_reward_ratio"] = float(value)
                except (ValueError, IndexError):
                    pass
        
        return data
    
    def _assess_proposal_risk(self, proposal: Dict[str, Any]) -> RiskAssessment:
        """Assess risk for a single proposal"""
        concerns = []
        adjustments = []
        
        # Extract proposal metrics
        symbol = proposal.get("symbol", "UNKNOWN")
        entry_price = proposal.get("entry_price", 0)
        stop_loss = proposal.get("stop_loss", 0)
        position_size = proposal.get("position_size", 0)
        conviction = proposal.get("conviction", 0)
        risk_reward_ratio = proposal.get("risk_reward_ratio", 0)
        
        # Calculate individual risk components
        concentration_risk = self._assess_concentration_risk(symbol, entry_price, position_size)
        correlation_risk = self._assess_correlation_risk(symbol)
        reward_risk_score = self._assess_reward_risk_ratio(risk_reward_ratio)
        market_regime_risk = self._assess_market_regime_risk(symbol)
        position_size_risk = self._assess_position_size_risk(entry_price, position_size)
        
        # Calculate overall risk score (weighted average)
        overall_risk = (
            concentration_risk * 0.25 +
            correlation_risk * 0.20 +
            reward_risk_score * 0.20 +
            market_regime_risk * 0.20 +
            position_size_risk * 0.15
        )
        
        # Generate concerns based on risk scores
        if concentration_risk >= 7:
            concerns.append(f"High concentration risk: {symbol} position may be too large")
            adjustments.append("Reduce position size by 30-50%")
        
        if correlation_risk >= 7:
            concerns.append(f"High correlation risk with existing positions")
            adjustments.append("Consider reducing correlated positions first")
        
        if reward_risk_score >= 7:
            concerns.append(f"Poor risk/reward ratio: {risk_reward_ratio:.1f}:1 below {self.risk_parameters['min_risk_reward_ratio']}:1 minimum")
            adjustments.append("Tighten stop loss or raise profit target")
        
        if position_size_risk >= 7:
            concerns.append(f"Position size ${entry_price * position_size:,.0f} exceeds comfort level")
            adjustments.append(f"Reduce to max ${self.risk_parameters['max_position_value']:,.0f} position")
        
        # Determine veto recommendation
        veto_recommended = (
            overall_risk >= self.risk_parameters["veto_risk_score"] or
            concentration_risk >= 9 or
            position_size_risk >= 9 or
            (risk_reward_ratio > 0 and risk_reward_ratio < 1.5)
        )
        
        return RiskAssessment(
            proposal_id=proposal.get("proposal_id", "unknown"),
            symbol=symbol,
            overall_risk_score=overall_risk,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            reward_risk_ratio=reward_risk_score,
            market_regime_risk=market_regime_risk,
            position_size_risk=position_size_risk,
            veto_recommended=veto_recommended,
            concerns=concerns,
            suggested_adjustments=adjustments
        )
    
    def _assess_concentration_risk(self, symbol: str, entry_price: float, position_size: int) -> float:
        """Assess portfolio concentration risk (0-10 scale)"""
        if entry_price <= 0 or position_size <= 0:
            return 5.0  # Default moderate risk for missing data
        
        position_value = entry_price * position_size
        
        # Assume $1M portfolio for now (would get from portfolio manager)
        portfolio_value = 1000000
        position_percentage = position_value / portfolio_value
        
        # Risk scoring based on position size
        if position_percentage > 0.10:  # > 10%
            return 10.0
        elif position_percentage > 0.07:  # > 7%
            return 8.0
        elif position_percentage > 0.05:  # > 5%
            return 6.0
        elif position_percentage > 0.03:  # > 3%
            return 4.0
        else:
            return 2.0
    
    def _assess_correlation_risk(self, symbol: str) -> float:
        """Assess correlation risk with existing positions (0-10 scale)"""
        # Simplified correlation assessment
        # In production, would analyze actual correlations
        
        # For now, assess based on sector/asset type
        if symbol in ["SPY", "QQQ", "IWM"]:  # ETFs
            return 3.0  # Lower correlation risk for diversified ETFs
        elif symbol.startswith("SPX") or symbol.startswith("NDX"):  # Index options
            return 7.0  # Higher risk for concentrated index exposure
        else:
            return 5.0  # Default moderate risk for individual stocks
    
    def _assess_reward_risk_ratio(self, risk_reward_ratio: float) -> float:
        """Assess risk/reward ratio quality (0-10 scale, higher = worse)"""
        if risk_reward_ratio <= 0:
            return 8.0  # High risk for missing R/R data
        
        min_ratio = self.risk_parameters["min_risk_reward_ratio"]
        
        if risk_reward_ratio < 1.0:
            return 10.0  # Unacceptable
        elif risk_reward_ratio < 1.5:
            return 8.0  # Poor
        elif risk_reward_ratio < min_ratio:
            return 6.0  # Below minimum
        elif risk_reward_ratio >= 3.0:
            return 1.0  # Excellent
        else:
            return 3.0  # Acceptable
    
    def _assess_market_regime_risk(self, symbol: str) -> float:
        """Assess risk based on current market regime (0-10 scale)"""
        # Simplified market regime assessment
        # In production, would analyze VIX, market trends, etc.
        
        if self.market_regime == "bear":
            return 8.0  # High risk in bear market
        elif self.market_regime == "high_volatility":
            return 7.0  # High risk in volatile conditions
        elif self.market_regime == "bull":
            return 3.0  # Lower risk in bull market
        else:
            return 5.0  # Neutral risk
    
    def _assess_position_size_risk(self, entry_price: float, position_size: int) -> float:
        """Assess position size risk (0-10 scale)"""
        if entry_price <= 0 or position_size <= 0:
            return 5.0
        
        position_value = entry_price * position_size
        max_position = self.risk_parameters["max_position_value"]
        
        if position_value > max_position * 2:
            return 10.0  # Extremely oversized
        elif position_value > max_position:
            return 8.0  # Oversized
        elif position_value > max_position * 0.7:
            return 6.0  # Large but acceptable
        else:
            return 3.0  # Reasonable size
    
    def _calculate_portfolio_risk(self, proposals: List[Dict[str, Any]]) -> float:
        """Calculate overall portfolio risk from all proposals"""
        if not proposals:
            return 0.0
        
        total_risk = 0
        total_value = 0
        
        for proposal in proposals:
            entry_price = proposal.get("entry_price", 0)
            position_size = proposal.get("position_size", 0)
            
            if entry_price > 0 and position_size > 0:
                position_value = entry_price * position_size
                total_value += position_value
                
                # Weight risk by position size
                assessment = self._assess_proposal_risk(proposal)
                total_risk += assessment.overall_risk_score * position_value
        
        if total_value > 0:
            weighted_risk = total_risk / total_value
            return min(weighted_risk, 10.0)
        
        return 5.0  # Default moderate risk
    
    def _veto_proposal(self, proposal_id: str, reason: str) -> Dict[str, Any]:
        """Veto a specific proposal"""
        return {
            "type": "veto_response",
            "proposal_id": proposal_id,
            "vetoed": True,
            "veto_reason": reason,
            "authority": "Risk Manager veto authority",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": self.agent_name
        }
    
    def _challenge_proposal(self, proposal_id: str, concern: str) -> Dict[str, Any]:
        """Challenge a proposal with specific concerns"""
        return {
            "type": "challenge_response",
            "proposal_id": proposal_id,
            "challenged": True,
            "concerns": [concern],
            "challenge_level": "formal",
            "response_required": True,
            "agent": self.agent_name
        }
    
    def _assessment_to_dict(self, assessment: RiskAssessment) -> Dict[str, Any]:
        """Convert RiskAssessment to dictionary"""
        return {
            "proposal_id": assessment.proposal_id,
            "symbol": assessment.symbol,
            "overall_risk_score": round(assessment.overall_risk_score, 1),
            "risk_breakdown": {
                "concentration": round(assessment.concentration_risk, 1),
                "correlation": round(assessment.correlation_risk, 1),
                "reward_risk": round(assessment.reward_risk_ratio, 1),
                "market_regime": round(assessment.market_regime_risk, 1),
                "position_size": round(assessment.position_size_risk, 1)
            },
            "veto_recommended": assessment.veto_recommended,
            "concerns": assessment.concerns,
            "suggested_adjustments": assessment.suggested_adjustments
        }
    
    def _generate_risk_summary(self, assessments: List[RiskAssessment], portfolio_risk: float) -> str:
        """Generate human-readable risk summary"""
        if not assessments:
            return "No proposals to evaluate"
        
        high_risk = len([a for a in assessments if a.overall_risk_score >= 8])
        moderate_risk = len([a for a in assessments if 5 <= a.overall_risk_score < 8])
        low_risk = len([a for a in assessments if a.overall_risk_score < 5])
        vetoed = len([a for a in assessments if a.veto_recommended])
        
        summary = f"Portfolio Risk Analysis:\n"
        summary += f"• {len(assessments)} proposals evaluated\n"
        summary += f"• {high_risk} high risk, {moderate_risk} moderate risk, {low_risk} low risk\n"
        summary += f"• {vetoed} proposals recommended for veto\n"
        summary += f"• Overall portfolio risk score: {portfolio_risk:.1f}/10\n"
        
        if portfolio_risk >= 7:
            summary += "\n⚠️ HIGH PORTFOLIO RISK - Recommend position size reductions"
        elif portfolio_risk >= 5:
            summary += "\n⚡ Moderate portfolio risk - Monitor concentration"
        else:
            summary += "\n✅ Acceptable portfolio risk levels"
        
        return summary
    
    def _assess_portfolio_risk(self) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        proposals = self._scan_proposals_folder()
        portfolio_risk = self._calculate_portfolio_risk(proposals)
        
        return {
            "type": "portfolio_risk_assessment",
            "portfolio_risk_score": portfolio_risk,
            "proposal_count": len(proposals),
            "risk_level": "high" if portfolio_risk >= 7 else "moderate" if portfolio_risk >= 5 else "low",
            "agent": self.agent_name
        }
    
    def _assess_market_regime(self) -> Dict[str, Any]:
        """Assess current market regime"""
        # Simplified market regime assessment
        return {
            "type": "market_regime_assessment",
            "regime": self.market_regime,
            "risk_adjustment": "conservative" if self.market_regime in ["bear", "high_volatility"] else "normal",
            "agent": self.agent_name
        }
    
    def _get_risk_status(self) -> Dict[str, Any]:
        """Get current risk manager status"""
        proposals = self._scan_proposals_folder()
        
        return {
            "type": "risk_status",
            "proposals_monitored": len(proposals),
            "risk_parameters": self.risk_parameters,
            "market_regime": self.market_regime,
            "status": "active",
            "agent": self.agent_name
        }
    
    def _provide_risk_guidance(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Provide general risk management guidance"""
        return {
            "type": "risk_guidance",
            "message": "As Risk Manager, I recommend conservative position sizing and thorough risk assessment before any trades. Please provide specific proposals for detailed evaluation.",
            "guidance": [
                "Maintain maximum 5% position sizes",
                "Require minimum 2:1 risk/reward ratios",
                "Monitor portfolio concentration carefully",
                "Adjust for current market regime"
            ],
            "agent": self.agent_name
        }
    
    def _evaluate_specific_proposal(self, proposal_id: str) -> Dict[str, Any]:
        """Evaluate a specific proposal by ID"""
        proposals = self._scan_proposals_folder()
        
        target_proposal = None
        for proposal in proposals:
            if proposal.get("proposal_id") == proposal_id:
                target_proposal = proposal
                break
        
        if not target_proposal:
            return {
                "type": "error",
                "message": f"Proposal {proposal_id} not found",
                "agent": self.agent_name
            }
        
        assessment = self._assess_proposal_risk(target_proposal)
        
        return {
            "type": "single_proposal_assessment",
            "proposal_id": proposal_id,
            "assessment": self._assessment_to_dict(assessment),
            "agent": self.agent_name
        }