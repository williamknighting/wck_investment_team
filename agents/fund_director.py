"""
Fund Director Agent for AI Hedge Fund System
Senior portfolio manager who makes final investment decisions based on agent recommendations
"""
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
from base_agent import BaseHedgeFundAgent


class FundDirectorAgent(BaseHedgeFundAgent):
    """
    Fund Director Agent - Senior portfolio manager and final decision maker
    Orchestrates investment committee meetings and makes final trading decisions
    """
    
    def _initialize(self) -> None:
        """Initialize the Fund Director"""
        self.portfolio_size = 1000000  # $1M portfolio
        self.active_positions = {}
        self.investment_committee = []
        self.decision_history = []
        self._initialized_at = datetime.now(timezone.utc)
        self.logger.info(f"Fund Director initialized - Portfolio: ${self.portfolio_size:,}")
    
    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process incoming messages and coordinate investment decisions
        
        Args:
            message: The message/request
            context: Optional context including symbol, meeting type, etc.
            
        Returns:
            Dict containing director's response or decision
        """
        try:
            if "investment committee" in message.lower() or "meeting" in message.lower():
                return self._run_investment_committee_meeting(context)
            elif "decision" in message.lower() or "approve" in message.lower():
                return self._make_investment_decision(context)
            elif "portfolio" in message.lower() or "status" in message.lower():
                return self._get_portfolio_status()
            elif "watchlist" in message.lower():
                return self._review_watchlist()
            else:
                return self._general_response(message)
                
        except Exception as e:
            self.logger.error(f"Error processing director message: {e}")
            return {
                "type": "error",
                "message": str(e),
                "agent": self.name
            }
    
    def _run_investment_committee_meeting(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run an investment committee meeting to analyze a symbol
        """
        symbol = context.get("symbol", "SPY") if context else "SPY"
        
        self.logger.info(f"Starting investment committee meeting for {symbol}")
        
        # Initialize committee members
        committee_analyses = []
        meeting_log = []
        
        # 1. Request technical analysis
        meeting_log.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": self.name,
            "message": f"Opening investment committee meeting for {symbol}. Requesting technical analysis."
        })
        
        technical_analysis = self._request_technical_analysis(symbol)
        if technical_analysis.get("type") != "error":
            committee_analyses.append({
                "agent": "Technical Analyst",
                "type": "technical_analysis",
                "data": technical_analysis
            })
            meeting_log.append({
                "timestamp": datetime.now().isoformat(), 
                "speaker": "Technical Analyst",
                "message": f"Technical analysis completed. Current price: ${technical_analysis.get('metrics', {}).get('moving_averages', {}).get('current_price', 0):.2f}"
            })
        
        # 2. Request strategy analysis from Qullamaggie agent
        qullamaggie_analysis = self._request_qullamaggie_analysis(symbol)
        if qullamaggie_analysis.get("type") != "error":
            committee_analyses.append({
                "agent": "Qullamaggie Strategy",
                "type": "strategy_analysis", 
                "data": qullamaggie_analysis
            })
            
            analysis_data = qullamaggie_analysis.get("analysis", {})
            setup_quality = analysis_data.get("setup_quality", "unknown")
            confidence = analysis_data.get("confidence", 0)
            
            meeting_log.append({
                "timestamp": datetime.now().isoformat(),
                "speaker": "Qullamaggie Agent", 
                "message": f"Strategy analysis completed. Setup quality: {setup_quality}, Confidence: {confidence:.1f}/5.0"
            })
        
        # 3. Make final investment decision based on committee input
        final_decision = self._synthesize_committee_input(symbol, committee_analyses)
        
        meeting_log.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": self.name,
            "message": f"Final decision: {final_decision['decision']} - {final_decision['reasoning']}"
        })
        
        # 4. Log the conversation and write decision
        conversation_file = self.log_conversation(meeting_log, symbol)
        decision_file = self._write_investment_decision(symbol, final_decision, committee_analyses)
        
        return {
            "type": "investment_committee_meeting",
            "symbol": symbol,
            "decision": final_decision,
            "committee_analyses": committee_analyses,
            "meeting_duration": len(meeting_log),
            "conversation_logged": conversation_file,
            "decision_logged": decision_file,
            "agent": self.name
        }
    
    def _request_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Request technical analysis from Technical Analyst agent"""
        try:
            # TODO: Replace with proper AutoGen agent-to-agent communication
            from technical_analyst import TechnicalAnalystAgent
            
            tech_agent = TechnicalAnalystAgent(
                name="technical_analyst",
                description="Technical analysis provider"
            )
            
            context = {"symbol": symbol}
            message = f"Please provide comprehensive technical analysis for {symbol}"
            
            return tech_agent.process_message(message, context)
            
        except Exception as e:
            self.logger.error(f"Error requesting technical analysis: {e}")
            return {"type": "error", "message": str(e)}
    
    def _request_qullamaggie_analysis(self, symbol: str) -> Dict[str, Any]:
        """Request Qullamaggie strategy analysis"""
        try:
            # TODO: Replace with proper AutoGen agent-to-agent communication
            from qullamaggie_agent import QullamaggieAgent
            
            strategy_agent = QullamaggieAgent(
                name="qullamaggie_agent",
                description="Qullamaggie momentum strategy specialist"
            )
            
            context = {"symbol": symbol}
            message = f"Please analyze {symbol} for Qullamaggie momentum setups"
            
            return strategy_agent.process_message(message, context)
            
        except Exception as e:
            self.logger.error(f"Error requesting Qullamaggie analysis: {e}")
            return {"type": "error", "message": str(e)}
    
    def _synthesize_committee_input(self, symbol: str, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize input from all committee members to make final decision
        """
        try:
            # Extract key information from analyses
            technical_data = None
            strategy_data = None
            
            for analysis in analyses:
                if analysis["type"] == "technical_analysis":
                    technical_data = analysis["data"]
                elif analysis["type"] == "strategy_analysis":
                    strategy_data = analysis["data"]
            
            if not technical_data or not strategy_data:
                return {
                    "decision": "HOLD",
                    "confidence": 0,
                    "reasoning": "Insufficient analysis data from committee",
                    "position_size": 0
                }
            
            # Get key metrics
            try:
                price = technical_data["metrics"]["moving_averages"]["current_price"]
                tech_summary = technical_data.get("summary", {})
                strategy_analysis = strategy_data.get("analysis", {})
                
                setup_quality = strategy_analysis.get("setup_quality", "not_suitable")
                strategy_confidence = strategy_analysis.get("confidence", 0)
                criteria_met = strategy_analysis.get("criteria_met", 0)
                position_rec = strategy_analysis.get("position_recommendation", {})
                
            except (KeyError, TypeError) as e:
                self.logger.error(f"Error extracting analysis data: {e}")
                return {
                    "decision": "HOLD",
                    "confidence": 0,
                    "reasoning": "Error parsing committee analysis data",
                    "position_size": 0
                }
            
            # Director's decision logic
            if setup_quality == "strong_buy" and strategy_confidence >= 4.0:
                decision = "BUY"
                confidence = min(strategy_confidence, 5.0)
                position_size = min(position_rec.get("position_size", 0), 1000)  # Cap at 1000 shares
                reasoning = f"Strong technical and momentum setup. Qullamaggie criteria: {criteria_met}/5 met."
                
            elif setup_quality == "watch_list" and strategy_confidence >= 3.0:
                decision = "WATCH"
                confidence = strategy_confidence
                position_size = 0
                reasoning = f"Decent setup but requires monitoring. Qullamaggie criteria: {criteria_met}/5 met."
                
            else:
                decision = "HOLD"
                confidence = strategy_confidence
                position_size = 0
                reasoning = f"Setup does not meet investment criteria. Qullamaggie criteria: {criteria_met}/5 met."
            
            # Calculate risk metrics
            risk_amount = 0
            if position_size > 0:
                risk_amount = position_size * price * 0.02  # 2% risk
            
            return {
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "position_size": position_size,
                "entry_price": price,
                "stop_loss": position_rec.get("stop_loss", price * 0.95),
                "target": position_rec.get("target", price * 1.1),
                "risk_amount": risk_amount,
                "max_portfolio_allocation": min(risk_amount / self.portfolio_size * 100, 5.0)  # Max 5% of portfolio
            }
            
        except Exception as e:
            self.logger.error(f"Error synthesizing committee input: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0,
                "reasoning": f"Error in decision synthesis: {str(e)}",
                "position_size": 0
            }
    
    def _write_investment_decision(self, symbol: str, decision: Dict[str, Any], analyses: List[Dict[str, Any]]) -> str:
        """Write final investment decision to markdown file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"investment_decision_{symbol}_{timestamp}.md"
            filepath = Path("decisions") / filename
            
            # Create decision content
            content = f"""# Investment Committee Decision - {symbol}

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Director**: {self.name}  
**Symbol**: {symbol}  
**Portfolio Size**: ${self.portfolio_size:,}  

## Final Decision

**Action**: {decision['decision']}  
**Confidence**: {decision['confidence']:.1f}/5.0  
**Position Size**: {decision['position_size']} shares  
**Entry Price**: ${decision.get('entry_price', 0):.2f}  
**Stop Loss**: ${decision.get('stop_loss', 0):.2f}  
**Target**: ${decision.get('target', 0):.2f}  
**Risk Amount**: ${decision.get('risk_amount', 0):.2f}  
**Portfolio Allocation**: {decision.get('max_portfolio_allocation', 0):.1f}%  

## Reasoning

{decision['reasoning']}

## Committee Analysis Summary

"""
            
            # Add analysis summaries
            for analysis in analyses:
                agent_name = analysis['agent']
                content += f"### {agent_name}\n\n"
                
                if analysis['type'] == 'technical_analysis':
                    tech_data = analysis['data']
                    summary = tech_data.get('summary', {})
                    content += f"- **Current Price**: ${summary.get('current_price', 0):.2f}\n"
                    content += f"- **Trend Direction**: {summary.get('trend_direction', 'unknown')}\n"
                    content += f"- **RSI Signal**: {summary.get('rsi_signal', 'unknown')}\n"
                    content += f"- **Volume Status**: {summary.get('volume_status', 'unknown')}\n\n"
                
                elif analysis['type'] == 'strategy_analysis':
                    strategy_data = analysis['data'].get('analysis', {})
                    content += f"- **Setup Quality**: {strategy_data.get('setup_quality', 'unknown')}\n"
                    content += f"- **Confidence**: {strategy_data.get('confidence', 0):.1f}/5.0\n"
                    content += f"- **Criteria Met**: {strategy_data.get('criteria_met', 0)}/5\n"
                    content += f"- **Reasoning**: {strategy_data.get('reasoning', 'N/A')}\n\n"
            
            content += f"""
## Risk Management

- Maximum risk per position: 2% of portfolio (${self.portfolio_size * 0.02:,.2f})
- Maximum portfolio allocation: 5% per position
- Risk/Reward ratio: Minimum 2:1
- Stop loss: Mandatory on all positions

---

*Decision made by Fund Director following investment committee meeting*  
*Committee members: Technical Analyst, Qullamaggie Strategy Agent*
"""
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Investment decision written to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error writing decision: {e}")
            return ""
    
    def _get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        return {
            "type": "portfolio_status",
            "portfolio_size": self.portfolio_size,
            "active_positions": len(self.active_positions),
            "cash_available": self.portfolio_size * 0.8,  # Assume 80% cash
            "positions": self.active_positions,
            "agent": self.name
        }
    
    def _review_watchlist(self) -> Dict[str, Any]:
        """Review current watchlist and prioritize symbols"""
        watchlist = self.get_watchlist()
        
        # Prioritize watchlist based on recent additions and notes
        prioritized = sorted(watchlist, key=lambda x: x['date_added'], reverse=True)
        
        return {
            "type": "watchlist_review",
            "total_symbols": len(watchlist),
            "prioritized_list": prioritized[:10],  # Top 10
            "agent": self.name
        }
    
    def _general_response(self, message: str) -> Dict[str, Any]:
        """Handle general inquiries"""
        return {
            "type": "general_response",
            "message": f"Fund Director ready. I oversee investment committee meetings and make final portfolio decisions.",
            "capabilities": [
                "Investment committee meetings",
                "Final trading decisions",
                "Portfolio management",
                "Risk oversight",
                "Agent coordination"
            ],
            "portfolio_size": self.portfolio_size,
            "agent": self.name
        }