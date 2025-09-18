"""
Director Agent for AI Hedge Fund System
Orchestrates investment committee conversations and makes final trading decisions
"""
import os
import json
import glob
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass, asdict

from agents.base_agent import BaseHedgeFundAgent


@dataclass
class ConversationRound:
    """Single round of conversation"""
    round_number: int
    agent: str
    message: str
    response: str
    timestamp: str
    question_type: str = ""  # "challenge", "clarification", "conflict_resolution"


@dataclass
class FinalDecision:
    """Final trading decision"""
    decision_id: str
    timestamp: str
    proposals_reviewed: List[str]
    approved_trades: List[Dict[str, Any]]
    rejected_trades: List[Dict[str, Any]]
    position_adjustments: List[Dict[str, Any]]
    market_context: str
    key_discussion_points: List[str]
    decision_rationale: str
    execution_instructions: str
    conversation_rounds: int


class DirectorAgent(BaseHedgeFundAgent):
    """
    Director Agent that orchestrates investment committee conversations
    Decisive personality with thorough questioning and conflict resolution
    """
    
    def __init__(self, name: str = "director", description: str = "Investment Committee Director", **kwargs):
        """Initialize Director Agent"""
        super().__init__(
            name=name,
            description=description,
            system_message=self._get_system_message(),
            **kwargs
        )
    
    def _get_system_message(self) -> str:
        """Get system message for Director"""
        return """You are the Director of an AI hedge fund's Investment Committee. Your personality is:

DECISIVE: You make final calls and don't hesitate when you have enough information
THOROUGH: You ask pointed, probing questions to expose weaknesses in proposals
DIRECT: You cut through noise and get to the core issues quickly
AUTHORITATIVE: Other agents respect your final decision
SKEPTICAL: You challenge assumptions and demand strong justification

Your role:
1. READ all proposals before starting conversations
2. ORCHESTRATE structured discussion (max 10 rounds)
3. ASK pointed questions that expose risks and assumptions
4. RESOLVE conflicts between agents with decisive authority
5. MAKE final trading decisions based on all input
6. DOCUMENT decisions with clear rationale

You have FINAL AUTHORITY over all trading decisions. Be thorough but decisive."""
    
    def _initialize(self) -> None:
        """Initialize Director specific components"""
        self.proposals_folder = Path("proposals")
        self.decisions_folder = Path("decisions")
        self.memory_file = Path("director_memory.json")
        
        # Ensure folders exist
        self.proposals_folder.mkdir(exist_ok=True)
        self.decisions_folder.mkdir(exist_ok=True)
        
        # Conversation state
        self.current_conversation = []
        self.max_rounds = 10
        self.current_round = 0
        
        # Decision memory
        self.decision_memory = self._load_decision_memory()
        
        self.logger.info("Director Agent initialized with final authority")
    
    def _load_decision_memory(self) -> List[Dict[str, Any]]:
        """Load past decisions from memory file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    memory = json.load(f)
                self.logger.info(f"Loaded {len(memory)} past decisions from memory")
                return memory
        except Exception as e:
            self.logger.warning(f"Could not load decision memory: {e}")
        
        return []
    
    def _save_decision_memory(self, decision: FinalDecision) -> None:
        """Save decision to memory file"""
        try:
            self.decision_memory.append(asdict(decision))
            
            # Keep only last 100 decisions
            if len(self.decision_memory) > 100:
                self.decision_memory = self.decision_memory[-100:]
            
            with open(self.memory_file, 'w') as f:
                json.dump(self.decision_memory, f, indent=2, default=str)
            
            self.logger.info(f"Saved decision {decision.decision_id} to memory")
            
        except Exception as e:
            self.logger.error(f"Error saving decision memory: {e}")
    
    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process incoming message for director tasks"""
        try:
            message_lower = message.lower()
            
            if "start committee" in message_lower or "start investment committee" in message_lower or "begin meeting" in message_lower:
                return self._start_investment_committee()
            elif "review all proposals" in message_lower or "review proposals" in message_lower:
                return self._review_all_proposals()
            elif "ask question" in message_lower or "challenge" in message_lower:
                agent = context.get("agent") if context else None
                return self._ask_pointed_question(agent, message)
            elif "resolve conflict" in message_lower:
                return self._resolve_agent_conflict(context)
            elif "make final decision" in message_lower or "final decision" in message_lower:
                return self._make_final_decision(context)
            elif "conversation status" in message_lower or "show conversation status" in message_lower:
                return self._get_conversation_status()
            elif "decision history" in message_lower or "show decision history" in message_lower or "past decisions" in message_lower:
                return self._get_decision_history()
            elif "market context" in message_lower or "assess market context" in message_lower:
                return self._assess_market_context()
            else:
                return self._provide_director_guidance(message, context)
                
        except Exception as e:
            self.logger.error(f"Error processing director message: {e}")
            return {
                "type": "error",
                "message": f"Director error: {str(e)}",
                "agent": self.agent_name
            }
    
    def _start_investment_committee(self) -> Dict[str, Any]:
        """Start investment committee meeting"""
        try:
            # Reset conversation state
            self.current_conversation = []
            self.current_round = 0
            
            # Read all proposals first
            proposals = self._scan_and_prepare_proposals()
            
            if not proposals:
                return {
                    "type": "committee_status",
                    "status": "no_meeting",
                    "message": "No proposals to review. Investment committee meeting not needed.",
                    "agent": self.agent_name
                }
            
            # Assess market context
            market_context = self._assess_current_market_context()
            
            # Prepare opening statement
            opening = self._generate_opening_statement(proposals, market_context)
            
            return {
                "type": "committee_started",
                "proposals_count": len(proposals),
                "proposals": [p["proposal_id"] for p in proposals],
                "market_context": market_context,
                "opening_statement": opening,
                "max_rounds": self.max_rounds,
                "current_round": self.current_round,
                "agent": self.agent_name
            }
            
        except Exception as e:
            self.logger.error(f"Error starting committee: {e}")
            return {
                "type": "error",
                "message": f"Committee startup error: {str(e)}",
                "agent": self.agent_name
            }
    
    def _scan_and_prepare_proposals(self) -> List[Dict[str, Any]]:
        """Scan and prepare all proposals for review"""
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
            
            # Sort by conviction/priority
            proposals.sort(key=lambda x: x.get("conviction", 0), reverse=True)
            
            self.logger.info(f"Prepared {len(proposals)} proposals for committee review")
            return proposals
            
        except Exception as e:
            self.logger.error(f"Error scanning proposals: {e}")
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
                elif line.startswith('**Type**:'):
                    proposal["type"] = line.split(':', 1)[1].strip()
            
            # Extract proposal details from content
            proposal_data = self._extract_proposal_details(content)
            proposal.update(proposal_data)
            
            # Generate proposal ID from filename
            proposal["proposal_id"] = os.path.splitext(os.path.basename(file_path))[0]
            
            return proposal
            
        except Exception as e:
            self.logger.error(f"Error parsing proposal file {file_path}: {e}")
            return None
    
    def _extract_proposal_details(self, content: str) -> Dict[str, Any]:
        """Extract detailed proposal data from markdown content"""
        data = {}
        
        # Look for common patterns in proposal content
        patterns = {
            "entry_price": r"(?:Entry|Enter).*?[\$]?([\d,]+\.?\d*)",
            "stop_loss": r"(?:Stop|Stop Loss).*?[\$]?([\d,]+\.?\d*)",
            "profit_target": r"(?:Target|Profit Target).*?[\$]?([\d,]+\.?\d*)",
            "position_size": r"(?:Size|Position|Shares).*?(\d+).*?shares?",
            "conviction": r"(?:Conviction).*?(\d+)(?:/10)?",
            "risk_reward": r"(?:R/R|Risk[/\\]Reward).*?(\d+\.?\d*):1",
            "setup_quality": r"(?:Setup Quality|Quality).*?:\s*(\w+)",
            "position_value": r"(?:Position Value).*?[\$]?([\d,]+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1).replace(',', '')
                    if key in ["entry_price", "stop_loss", "profit_target", "position_value"]:
                        data[key] = float(value)
                    elif key in ["position_size", "conviction"]:
                        data[key] = int(value)
                    elif key == "risk_reward":
                        data["risk_reward_ratio"] = float(value)
                    elif key == "setup_quality":
                        data[key] = value.lower()
                except (ValueError, IndexError):
                    pass
        
        # Extract reasoning/summary
        if "### Trade Parameters:" in content:
            data["has_trade_params"] = True
        
        return data
    
    def _assess_current_market_context(self) -> str:
        """Assess current market context for decision making"""
        # In production, would analyze real market data
        # For now, return simplified context
        current_time = datetime.now(timezone.utc)
        
        context = f"Market Context as of {current_time.strftime('%Y-%m-%d %H:%M')} UTC:\n"
        context += "• Market regime: Neutral with moderate volatility\n"
        context += "• Recent data freshness: All proposals reviewed with current market data\n"
        context += "• Risk environment: Standard risk parameters apply\n"
        context += "• Trading session: Active market hours"
        
        return context
    
    def _generate_opening_statement(self, proposals: List[Dict[str, Any]], market_context: str) -> str:
        """Generate opening statement for investment committee"""
        high_conviction = [p for p in proposals if p.get("conviction", 0) >= 7]
        moderate_conviction = [p for p in proposals if 4 <= p.get("conviction", 0) < 7]
        low_conviction = [p for p in proposals if p.get("conviction", 0) < 4]
        
        statement = f"**INVESTMENT COMMITTEE MEETING - {datetime.now().strftime('%Y-%m-%d %H:%M')}**\n\n"
        statement += f"I've reviewed {len(proposals)} proposals before we begin:\n\n"
        
        if high_conviction:
            statement += f"**HIGH CONVICTION ({len(high_conviction)}):**\n"
            for p in high_conviction:
                statement += f"• {p['symbol']} ({p.get('agent', 'unknown')}) - {p.get('conviction', 0)}/10\n"
            statement += "\n"
        
        if moderate_conviction:
            statement += f"**MODERATE CONVICTION ({len(moderate_conviction)}):**\n"
            for p in moderate_conviction:
                statement += f"• {p['symbol']} ({p.get('agent', 'unknown')}) - {p.get('conviction', 0)}/10\n"
            statement += "\n"
        
        if low_conviction:
            statement += f"**LOW CONVICTION ({len(low_conviction)}):**\n"
            for p in low_conviction:
                statement += f"• {p['symbol']} ({p.get('agent', 'unknown')}) - {p.get('conviction', 0)}/10\n"
            statement += "\n"
        
        statement += f"**MY INITIAL OBSERVATIONS:**\n"
        statement += f"We have {len(proposals)} proposals requiring decisions. "
        statement += f"I'll be asking pointed questions to stress-test each idea. "
        statement += f"Maximum {self.max_rounds} rounds of discussion.\n\n"
        statement += f"Let's begin with the highest conviction proposals and work our way through."
        
        return statement
    
    def _ask_pointed_question(self, agent: str, topic: str) -> Dict[str, Any]:
        """Ask pointed question to specific agent"""
        if self.current_round >= self.max_rounds:
            return {
                "type": "conversation_limit",
                "message": "Maximum conversation rounds reached. Moving to final decision.",
                "agent": self.agent_name
            }
        
        self.current_round += 1
        
        # Generate pointed questions based on topic
        questions = self._generate_pointed_questions(agent, topic)
        
        conversation_round = ConversationRound(
            round_number=self.current_round,
            agent=agent or "committee",
            message=topic,
            response="",  # Will be filled by agent response
            timestamp=datetime.now(timezone.utc).isoformat(),
            question_type="challenge"
        )
        
        self.current_conversation.append(conversation_round)
        
        return {
            "type": "pointed_question",
            "round": self.current_round,
            "target_agent": agent,
            "questions": questions,
            "tone": "direct_challenge",
            "response_required": True,
            "agent": self.agent_name
        }
    
    def _generate_pointed_questions(self, agent: str, topic: str) -> List[str]:
        """Generate pointed questions based on agent and topic"""
        questions = []
        
        if "qullamaggie" in (agent or "").lower():
            questions.extend([
                "What happens if this momentum fails and we get a sharp reversal?",
                "How do you justify the position size given the volatility?",
                "What's your exit plan if volume dries up?",
                "Are you chasing momentum or catching it early?"
            ])
        
        elif "risk" in (agent or "").lower():
            questions.extend([
                "What specific metrics trigger your veto recommendation?",
                "How does this fit within our portfolio concentration limits?",
                "What's the worst-case scenario you're protecting against?"
            ])
        
        elif "value" in (agent or "").lower():
            questions.extend([
                "What catalyst will unlock this value in our timeframe?",
                "How long are you willing to wait for this to work?",
                "What if the market disagrees with your valuation?"
            ])
        
        else:
            # Generic pointed questions
            questions.extend([
                "What's the biggest risk you're not telling us?",
                "Why should we risk capital on this right now?",
                "What would make you wrong about this trade?"
            ])
        
        # Add topic-specific questions
        if "position size" in topic.lower():
            questions.append("Defend your position sizing - why not smaller?")
        elif "risk" in topic.lower():
            questions.append("What risk are you underestimating?")
        elif "timing" in topic.lower():
            questions.append("Why now? What's the urgency?")
        
        return questions[:3]  # Limit to top 3 questions
    
    def _resolve_agent_conflict(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts between agents with authority"""
        conflict_agents = context.get("agents", []) if context else []
        conflict_topic = context.get("topic", "Unknown disagreement") if context else "Unknown"
        
        self.current_round += 1
        
        # Director's resolution approach
        resolution = self._generate_conflict_resolution(conflict_agents, conflict_topic)
        
        conversation_round = ConversationRound(
            round_number=self.current_round,
            agent="director",
            message=f"Conflict resolution: {conflict_topic}",
            response=resolution["ruling"],
            timestamp=datetime.now(timezone.utc).isoformat(),
            question_type="conflict_resolution"
        )
        
        self.current_conversation.append(conversation_round)
        
        return {
            "type": "conflict_resolution",
            "round": self.current_round,
            "conflicting_agents": conflict_agents,
            "conflict_topic": conflict_topic,
            "resolution": resolution,
            "authority": "Director final ruling",
            "agent": self.agent_name
        }
    
    def _generate_conflict_resolution(self, agents: List[str], topic: str) -> Dict[str, Any]:
        """Generate authoritative conflict resolution"""
        
        # Director's decision-making framework
        if "position size" in topic.lower():
            ruling = "Position size will be reduced to the most conservative recommendation. Risk management takes precedence over conviction."
            rationale = "Capital preservation is our primary mandate. We can always add to winning positions."
        
        elif "risk" in topic.lower() and "reward" in topic.lower():
            ruling = "Any trade with less than 2:1 risk/reward is rejected regardless of conviction."
            rationale = "Mathematical edge is non-negotiable. High conviction doesn't override poor risk/reward."
        
        elif "timing" in topic.lower():
            ruling = "We wait for clearer setup confirmation. Forced timing leads to poor entries."
            rationale = "Markets will give us another opportunity. Patience beats forcing trades."
        
        elif "correlation" in topic.lower():
            ruling = "Correlated positions are reduced to maintain portfolio diversification."
            rationale = "Portfolio-level risk management overrides individual trade merit."
        
        else:
            ruling = "Risk Manager's recommendation takes precedence in ties."
            rationale = "When in doubt, we err on the side of capital preservation."
        
        return {
            "ruling": ruling,
            "rationale": rationale,
            "final": True,
            "appeals": False
        }
    
    def _make_final_decision(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Make final trading decisions based on all input"""
        try:
            # Get all proposals and analysis
            proposals = self._scan_and_prepare_proposals()
            
            if not proposals:
                return {
                    "type": "final_decision",
                    "message": "No proposals to decide on",
                    "agent": self.agent_name
                }
            
            # Process each proposal through decision framework
            approved_trades = []
            rejected_trades = []
            position_adjustments = []
            key_discussion_points = []
            
            for proposal in proposals:
                decision = self._evaluate_proposal_for_decision(proposal)
                
                if decision["action"] == "approve":
                    approved_trades.append(decision)
                elif decision["action"] == "reject":
                    rejected_trades.append(decision)
                elif decision["action"] == "adjust":
                    position_adjustments.append(decision)
                
                key_discussion_points.extend(decision.get("discussion_points", []))
            
            # Generate final decision
            final_decision = FinalDecision(
                decision_id=f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                proposals_reviewed=[p["proposal_id"] for p in proposals],
                approved_trades=approved_trades,
                rejected_trades=rejected_trades,
                position_adjustments=position_adjustments,
                market_context=self._assess_current_market_context(),
                key_discussion_points=key_discussion_points,
                decision_rationale=self._generate_decision_rationale(approved_trades, rejected_trades),
                execution_instructions=self._generate_execution_instructions(approved_trades, position_adjustments),
                conversation_rounds=self.current_round
            )
            
            # Write decision report
            report_file = self._write_decision_report(final_decision)
            
            # Save to memory
            self._save_decision_memory(final_decision)
            
            return {
                "type": "final_decision",
                "decision_id": final_decision.decision_id,
                "approved_count": len(approved_trades),
                "rejected_count": len(rejected_trades),
                "adjusted_count": len(position_adjustments),
                "decision_file": report_file,
                "summary": self._generate_decision_summary(final_decision),
                "agent": self.agent_name
            }
            
        except Exception as e:
            self.logger.error(f"Error making final decision: {e}")
            return {
                "type": "error",
                "message": f"Decision making error: {str(e)}",
                "agent": self.agent_name
            }
    
    def _evaluate_proposal_for_decision(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate individual proposal for final decision"""
        symbol = proposal.get("symbol", "UNKNOWN")
        agent = proposal.get("agent", "unknown")
        conviction = proposal.get("conviction", 0)
        risk_reward = proposal.get("risk_reward_ratio", 0)
        position_size = proposal.get("position_size", 0)
        entry_price = proposal.get("entry_price", 0)
        setup_quality = proposal.get("setup_quality", "unknown")
        
        position_value = entry_price * position_size if entry_price and position_size else 0
        
        # Director's decision criteria
        discussion_points = []
        
        # Check minimum standards
        if risk_reward < 2.0:
            return {
                "action": "reject",
                "proposal_id": proposal["proposal_id"],
                "symbol": symbol,
                "reason": f"Risk/reward ratio {risk_reward:.1f}:1 below 2.0:1 minimum",
                "discussion_points": [f"Rejected {symbol} for insufficient risk/reward"]
            }
        
        if position_value > 100000:  # $100k limit
            # Adjust position size
            new_size = int(100000 / entry_price) if entry_price > 0 else 0
            return {
                "action": "adjust",
                "proposal_id": proposal["proposal_id"],
                "symbol": symbol,
                "original_size": position_size,
                "adjusted_size": new_size,
                "reason": f"Position size reduced from {position_size} to {new_size} shares (${position_value:,.0f} -> $100,000)",
                "discussion_points": [f"Reduced {symbol} position size for risk management"]
            }
        
        if conviction < 5:
            return {
                "action": "reject",
                "proposal_id": proposal["proposal_id"],
                "symbol": symbol,
                "reason": f"Conviction {conviction}/10 below minimum threshold",
                "discussion_points": [f"Rejected {symbol} for low conviction"]
            }
        
        # Approve if passes all checks
        return {
            "action": "approve",
            "proposal_id": proposal["proposal_id"],
            "symbol": symbol,
            "conviction": conviction,
            "risk_reward": risk_reward,
            "position_size": position_size,
            "entry_price": entry_price,
            "reason": f"Approved: {conviction}/10 conviction, {risk_reward:.1f}:1 R/R",
            "discussion_points": [f"Approved {symbol} with {conviction}/10 conviction"]
        }
    
    def _generate_decision_rationale(self, approved: List[Dict], rejected: List[Dict]) -> str:
        """Generate comprehensive decision rationale"""
        rationale = "**DIRECTOR'S DECISION RATIONALE**\n\n"
        
        if approved:
            rationale += f"**APPROVED TRADES ({len(approved)}):**\n"
            for trade in approved:
                symbol = trade['symbol']
                conviction = trade.get('conviction', 0)
                rr = trade.get('risk_reward', 0)
                rationale += f"• {symbol}: Strong fundamentals with {conviction}/10 conviction and {rr:.1f}:1 risk/reward\n"
            rationale += "\n"
        
        if rejected:
            rationale += f"**REJECTED TRADES ({len(rejected)}):**\n"
            for trade in rejected:
                symbol = trade['symbol']
                reason = trade.get('reason', 'Unknown')
                rationale += f"• {symbol}: {reason}\n"
            rationale += "\n"
        
        rationale += "**DECISION FRAMEWORK:**\n"
        rationale += "• Minimum 2:1 risk/reward ratio enforced\n"
        rationale += "• Maximum $100,000 position size limit\n"
        rationale += "• Minimum 5/10 conviction requirement\n"
        rationale += "• Capital preservation prioritized over aggressive growth\n\n"
        
        rationale += "All decisions reflect our mandate of consistent risk-adjusted returns."
        
        return rationale
    
    def _generate_execution_instructions(self, approved: List[Dict], adjustments: List[Dict]) -> str:
        """Generate execution instructions"""
        instructions = "**EXECUTION INSTRUCTIONS**\n\n"
        
        if approved:
            instructions += "**IMMEDIATE EXECUTION:**\n"
            for trade in approved:
                symbol = trade['symbol']
                size = trade.get('position_size', 0)
                entry = trade.get('entry_price', 0)
                instructions += f"• BUY {size} shares {symbol} at ${entry:.2f} (or better)\n"
            instructions += "\n"
        
        if adjustments:
            instructions += "**ADJUSTED POSITIONS:**\n"
            for adj in adjustments:
                symbol = adj['symbol']
                new_size = adj.get('adjusted_size', 0)
                instructions += f"• BUY {new_size} shares {symbol} (size reduced for risk management)\n"
            instructions += "\n"
        
        instructions += "**EXECUTION REQUIREMENTS:**\n"
        instructions += "• All orders to be placed during active market hours\n"
        instructions += "• Stop losses to be set immediately upon fill\n"
        instructions += "• Position sizes are final - no increases without committee approval\n"
        instructions += "• Report execution status within 1 hour of market close\n"
        
        return instructions
    
    def _write_decision_report(self, decision: FinalDecision) -> str:
        """Write detailed decision report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"decision_report_{timestamp}.md"
            filepath = self.decisions_folder / filename
            
            report = f"""# Investment Committee Decision Report

**Decision ID**: {decision.decision_id}  
**Date/Time**: {decision.timestamp}  
**Director**: {self.agent_name}  
**Conversation Rounds**: {decision.conversation_rounds}/{self.max_rounds}  

---

## Market Context

{decision.market_context}

---

## Proposals Reviewed

**Total Proposals**: {len(decision.proposals_reviewed)}

{', '.join(decision.proposals_reviewed)}

---

## Final Decisions

### Approved Trades ({len(decision.approved_trades)})

{self._format_approved_trades(decision.approved_trades)}

### Rejected Trades ({len(decision.rejected_trades)})

{self._format_rejected_trades(decision.rejected_trades)}

### Position Adjustments ({len(decision.position_adjustments)})

{self._format_position_adjustments(decision.position_adjustments)}

---

## Key Discussion Points

{self._format_discussion_points(decision.key_discussion_points)}

---

## Decision Rationale

{decision.decision_rationale}

---

## Execution Instructions

{decision.execution_instructions}

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Authority**: Investment Committee Director  
**Status**: FINAL DECISION  
"""
            
            with open(filepath, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Decision report written to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error writing decision report: {e}")
            return ""
    
    def _format_approved_trades(self, trades: List[Dict]) -> str:
        """Format approved trades for report"""
        if not trades:
            return "None"
        
        formatted = ""
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            size = trade.get('position_size', 0)
            entry = trade.get('entry_price', 0)
            conviction = trade.get('conviction', 0)
            rr = trade.get('risk_reward', 0)
            
            formatted += f"**{symbol}**\n"
            formatted += f"- Size: {size} shares\n"
            formatted += f"- Entry: ${entry:.2f}\n"
            formatted += f"- Conviction: {conviction}/10\n"
            formatted += f"- Risk/Reward: {rr:.1f}:1\n\n"
        
        return formatted
    
    def _format_rejected_trades(self, trades: List[Dict]) -> str:
        """Format rejected trades for report"""
        if not trades:
            return "None"
        
        formatted = ""
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            reason = trade.get('reason', 'Unknown')
            
            formatted += f"**{symbol}**: {reason}\n\n"
        
        return formatted
    
    def _format_position_adjustments(self, adjustments: List[Dict]) -> str:
        """Format position adjustments for report"""
        if not adjustments:
            return "None"
        
        formatted = ""
        for adj in adjustments:
            symbol = adj.get('symbol', 'UNKNOWN')
            original = adj.get('original_size', 0)
            adjusted = adj.get('adjusted_size', 0)
            reason = adj.get('reason', 'Unknown')
            
            formatted += f"**{symbol}**: {original} → {adjusted} shares\n"
            formatted += f"Reason: {reason}\n\n"
        
        return formatted
    
    def _format_discussion_points(self, points: List[str]) -> str:
        """Format discussion points for report"""
        if not points:
            return "No major discussion points recorded"
        
        formatted = ""
        for i, point in enumerate(points, 1):
            formatted += f"{i}. {point}\n"
        
        return formatted
    
    def _generate_decision_summary(self, decision: FinalDecision) -> str:
        """Generate concise decision summary"""
        approved = len(decision.approved_trades)
        rejected = len(decision.rejected_trades)
        adjusted = len(decision.position_adjustments)
        total = len(decision.proposals_reviewed)
        
        summary = f"Investment Committee Decision: {approved} approved, {rejected} rejected, {adjusted} adjusted out of {total} proposals reviewed. "
        
        if approved > 0:
            symbols = [t.get('symbol', 'UNKNOWN') for t in decision.approved_trades]
            summary += f"Approved trades: {', '.join(symbols)}. "
        
        summary += f"Decision made after {decision.conversation_rounds} rounds of discussion."
        
        return summary
    
    def _review_all_proposals(self) -> Dict[str, Any]:
        """Review all proposals before committee meeting"""
        proposals = self._scan_and_prepare_proposals()
        
        review_summary = {
            "type": "proposal_review",
            "proposals_count": len(proposals),
            "high_conviction": len([p for p in proposals if p.get("conviction", 0) >= 7]),
            "moderate_conviction": len([p for p in proposals if 4 <= p.get("conviction", 0) < 7]),
            "low_conviction": len([p for p in proposals if p.get("conviction", 0) < 4]),
            "proposals": []
        }
        
        for proposal in proposals:
            review_summary["proposals"].append({
                "proposal_id": proposal["proposal_id"],
                "symbol": proposal.get("symbol", "UNKNOWN"),
                "agent": proposal.get("agent", "unknown"),
                "conviction": proposal.get("conviction", 0),
                "risk_reward": proposal.get("risk_reward_ratio", 0),
                "position_value": proposal.get("position_value", 0),
                "setup_quality": proposal.get("setup_quality", "unknown")
            })
        
        review_summary["agent"] = self.agent_name
        return review_summary
    
    def _get_conversation_status(self) -> Dict[str, Any]:
        """Get current conversation status"""
        return {
            "type": "conversation_status",
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "rounds_remaining": self.max_rounds - self.current_round,
            "conversation_length": len(self.current_conversation),
            "can_continue": self.current_round < self.max_rounds,
            "agent": self.agent_name
        }
    
    def _get_decision_history(self) -> Dict[str, Any]:
        """Get decision history from memory"""
        recent_decisions = self.decision_memory[-10:] if len(self.decision_memory) > 10 else self.decision_memory
        
        return {
            "type": "decision_history",
            "total_decisions": len(self.decision_memory),
            "recent_decisions": recent_decisions,
            "agent": self.agent_name
        }
    
    def _assess_market_context(self) -> Dict[str, Any]:
        """Assess market context for decision making"""
        context = self._assess_current_market_context()
        
        return {
            "type": "market_context",
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": self.agent_name
        }
    
    def _provide_director_guidance(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Provide general director guidance"""
        return {
            "type": "director_guidance",
            "message": "As Director, I orchestrate investment committee meetings and make final trading decisions. Use 'start committee' to begin formal review.",
            "commands": [
                "start committee - Begin investment committee meeting",
                "review proposals - Review all current proposals",
                "make decision - Make final trading decisions",
                "conversation status - Check meeting progress",
                "decision history - Review past decisions"
            ],
            "agent": self.agent_name
        }