#!/usr/bin/env python3
"""
Investment Committee Orchestrator
Main script that manages the full investment committee workflow with all agents
"""
import os
import sys
import json
import time
import asyncio
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add project root to path
sys.path.append('.')

# Set up environment
os.environ.setdefault("OPENAI_API_KEY", "test_key")

# Import agents
from agents.director import DirectorAgent
from agents.risk_manager import RiskManagerAgent
from agents.qullamaggie_agent import QullamaggieAgent
from agents.technical_analyst import TechnicalAnalystAgent
from src.utils.logging_config import get_logger


@dataclass
class ConversationTurn:
    """Single turn in the investment committee conversation"""
    turn_number: int
    speaker: str
    message: str
    response: str
    timestamp: str
    turn_type: str = "normal"  # "normal", "challenge", "response", "interruption"
    target_agent: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class CommitteeSession:
    """Complete investment committee session"""
    session_id: str
    start_time: str
    end_time: str
    total_duration_seconds: float
    participants: List[str]
    proposals_count: int
    conversation_turns: int
    final_decision_id: str
    conversation_log: List[ConversationTurn]
    errors_encountered: List[str]
    session_summary: str


class InvestmentCommitteeOrchestrator:
    """
    Main orchestrator for investment committee meetings
    Manages agent initialization, conversation flow, and decision making
    """
    
    def __init__(self, config_path: str = "config/committee_config.json"):
        """Initialize orchestrator with configuration"""
        self.logger = get_logger("committee_orchestrator")
        self.config_path = Path(config_path)
        self.config = self._load_configuration()
        
        # Initialize folders
        self.conversations_folder = Path("conversations")
        self.conversations_folder.mkdir(exist_ok=True)
        
        # Session state
        self.session_id = f"committee_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = None
        self.agents = {}
        self.conversation_log = []
        self.errors = []
        self.interrupted = False
        
        self.logger.info(f"Investment Committee Orchestrator initialized: {self.session_id}")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load committee configuration"""
        default_config = {
            "agents": {
                "director": {
                    "enabled": True,
                    "max_conversation_rounds": 10,
                    "early_termination_threshold": 8
                },
                "risk_manager": {
                    "enabled": True,
                    "auto_veto_threshold": 8.0,
                    "challenge_threshold": 6.0
                },
                "qullamaggie_agent": {
                    "enabled": True,
                    "conviction_threshold": 6
                },
                "technical_analyst": {
                    "enabled": True,
                    "analysis_symbols": ["SPY", "QQQ", "TSLA", "AAPL"]
                }
            },
            "conversation": {
                "max_turns": 50,
                "turn_timeout_seconds": 30,
                "allow_interruptions": True,
                "challenge_cooldown_turns": 2
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay_seconds": 1,
                "continue_on_agent_failure": True
            }
        }
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
                self.logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            self.logger.warning(f"Could not load config file: {e}, using defaults")
        
        return default_config
    
    def initialize_agents(self) -> bool:
        """Initialize all enabled agents"""
        self.logger.info("Initializing investment committee agents...")
        
        try:
            # Initialize agents based on configuration
            agent_classes = {
                "director": DirectorAgent,
                "risk_manager": RiskManagerAgent,
                "qullamaggie_agent": QullamaggieAgent,
                "technical_analyst": TechnicalAnalystAgent
            }
            
            for agent_name, agent_class in agent_classes.items():
                if self.config["agents"].get(agent_name, {}).get("enabled", False):
                    try:
                        self.logger.info(f"Initializing {agent_name}...")
                        agent = agent_class()
                        self.agents[agent_name] = agent
                        self.logger.info(f"✅ {agent_name} initialized successfully")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to initialize {agent_name}: {e}")
                        self.errors.append(f"Agent initialization failed: {agent_name} - {str(e)}")
                        
                        if not self.config["error_handling"]["continue_on_agent_failure"]:
                            return False
            
            if not self.agents:
                self.logger.error("No agents were successfully initialized")
                return False
            
            self.logger.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error during agent initialization: {e}")
            self.errors.append(f"Critical initialization error: {str(e)}")
            return False
    
    def run_investment_committee(self) -> CommitteeSession:
        """Run complete investment committee session"""
        self.start_time = datetime.now(timezone.utc)
        self.logger.info(f"🏛️ Starting Investment Committee Session: {self.session_id}")
        
        try:
            # Phase 1: Agent Initialization
            if not self.initialize_agents():
                return self._create_failed_session("Agent initialization failed")
            
            # Phase 2: Data Research and Preparation
            self._run_research_phase()
            
            # Phase 3: Technical Analysis Preparation
            self._run_technical_analysis_phase()
            
            # Phase 4: Strategy Agent Proposal Generation (Parallel)
            self._run_proposal_generation_phase()
            
            # Phase 5: Director Preparation
            self._run_director_preparation_phase()
            
            # Phase 6: Investment Committee Conversation
            self._run_committee_conversation()
            
            # Phase 7: Final Decision
            final_decision = self._run_final_decision_phase()
            
            # Create session summary
            session = self._create_session_summary(final_decision)
            
            # Log conversation
            self._log_conversation_to_file(session)
            
            self.logger.info(f"🎉 Investment Committee Session completed: {self.session_id}")
            return session
            
        except Exception as e:
            self.logger.error(f"Critical error in investment committee: {e}")
            traceback.print_exc()
            self.errors.append(f"Critical session error: {str(e)}")
            return self._create_failed_session(f"Critical error: {str(e)}")
    
    def _run_research_phase(self) -> None:
        """Phase 1: Research agent updates data and reports status"""
        self.logger.info("📊 Phase 1: Research and Data Update")
        
        # In production, would run research agent to update market data
        # For now, simulate research phase
        research_turn = ConversationTurn(
            turn_number=len(self.conversation_log) + 1,
            speaker="research_agent",
            message="Update market data and prepare research report",
            response="Market data updated. Key markets: SPY +0.5%, QQQ +0.8%, VIX 18.5. Sector rotation continuing.",
            timestamp=datetime.now(timezone.utc).isoformat(),
            turn_type="research_update",
            duration_seconds=2.0
        )
        
        self.conversation_log.append(research_turn)
        self.logger.info(f"✅ Research phase completed")
    
    def _run_technical_analysis_phase(self) -> None:
        """Phase 2: Technical analyst prepares metrics"""
        self.logger.info("📈 Phase 2: Technical Analysis Preparation")
        
        if "technical_analyst" in self.agents:
            try:
                start_time = time.time()
                
                # Get technical analysis for key symbols
                symbols = self.config["agents"]["technical_analyst"].get("analysis_symbols", ["SPY"])
                
                for symbol in symbols:
                    try:
                        message = f"Prepare comprehensive technical analysis for {symbol}"
                        result = self._call_agent_with_retry("technical_analyst", message, {"symbol": symbol})
                        
                        tech_turn = ConversationTurn(
                            turn_number=len(self.conversation_log) + 1,
                            speaker="technical_analyst",
                            message=message,
                            response=f"Technical analysis completed for {symbol}",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            turn_type="technical_preparation",
                            duration_seconds=time.time() - start_time
                        )
                        
                        self.conversation_log.append(tech_turn)
                        
                    except Exception as e:
                        self.logger.warning(f"Technical analysis failed for {symbol}: {e}")
                        self.errors.append(f"Technical analysis error for {symbol}: {str(e)}")
                
                self.logger.info(f"✅ Technical analysis phase completed")
                
            except Exception as e:
                self.logger.error(f"Technical analysis phase failed: {e}")
                self.errors.append(f"Technical analysis phase error: {str(e)}")
        else:
            self.logger.warning("Technical analyst not available")
    
    def _run_proposal_generation_phase(self) -> None:
        """Phase 3: Strategy agents create proposals in parallel"""
        self.logger.info("💼 Phase 3: Strategy Agent Proposal Generation (Parallel)")
        
        strategy_agents = [name for name in self.agents.keys() if "agent" in name and name != "technical_analyst"]
        
        if not strategy_agents:
            self.logger.warning("No strategy agents available for proposal generation")
            return
        
        # Run proposal generation in parallel
        with ThreadPoolExecutor(max_workers=len(strategy_agents)) as executor:
            future_to_agent = {}
            
            for agent_name in strategy_agents:
                future = executor.submit(self._generate_agent_proposals, agent_name)
                future_to_agent[future] = agent_name
            
            # Collect results
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    proposals = future.result()
                    self.logger.info(f"✅ {agent_name} generated {len(proposals)} proposals")
                except Exception as e:
                    self.logger.error(f"❌ {agent_name} proposal generation failed: {e}")
                    self.errors.append(f"Proposal generation error ({agent_name}): {str(e)}")
        
        self.logger.info(f"✅ Proposal generation phase completed")
    
    def _generate_agent_proposals(self, agent_name: str) -> List[Dict[str, Any]]:
        """Generate proposals for a specific strategy agent"""
        proposals = []
        
        try:
            # Test symbols for proposal generation
            test_symbols = ["SPY", "TSLA", "AAPL"]
            
            for symbol in test_symbols:
                try:
                    start_time = time.time()
                    message = f"Generate trade proposal for {symbol}"
                    result = self._call_agent_with_retry(agent_name, message, {"symbol": symbol})
                    
                    if result and result.get("type") == "trade_proposal":
                        proposals.append(result)
                        
                        # Log proposal generation
                        proposal_turn = ConversationTurn(
                            turn_number=len(self.conversation_log) + 1,
                            speaker=agent_name,
                            message=message,
                            response=f"Trade proposal generated for {symbol}",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            turn_type="proposal_generation",
                            duration_seconds=time.time() - start_time
                        )
                        
                        self.conversation_log.append(proposal_turn)
                        
                except Exception as e:
                    self.logger.warning(f"{agent_name} failed to generate proposal for {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in proposal generation for {agent_name}: {e}")
            raise
        
        return proposals
    
    def _run_director_preparation_phase(self) -> None:
        """Phase 4: Director reads and prepares for committee meeting"""
        self.logger.info("👔 Phase 4: Director Preparation")
        
        if "director" not in self.agents:
            self.logger.error("Director agent not available")
            return
        
        try:
            start_time = time.time()
            
            # Director reviews all proposals
            result = self._call_agent_with_retry("director", "Review all proposals")
            
            prep_turn = ConversationTurn(
                turn_number=len(self.conversation_log) + 1,
                speaker="director",
                message="Review all proposals and prepare for committee meeting",
                response=f"Reviewed {result.get('proposals_count', 0)} proposals",
                timestamp=datetime.now(timezone.utc).isoformat(),
                turn_type="director_preparation",
                duration_seconds=time.time() - start_time
            )
            
            self.conversation_log.append(prep_turn)
            self.logger.info(f"✅ Director preparation completed")
            
        except Exception as e:
            self.logger.error(f"Director preparation failed: {e}")
            self.errors.append(f"Director preparation error: {str(e)}")
    
    def _run_committee_conversation(self) -> None:
        """Phase 5: Main investment committee conversation"""
        self.logger.info("🏛️ Phase 5: Investment Committee Conversation")
        
        if "director" not in self.agents:
            self.logger.error("Cannot run committee without Director")
            return
        
        try:
            # Director starts the meeting
            meeting_result = self._call_agent_with_retry("director", "Start investment committee meeting")
            
            if meeting_result.get("type") == "committee_started":
                self.logger.info(f"📋 Committee meeting started: {meeting_result.get('proposals_count', 0)} proposals")
                
                # Log opening
                opening_turn = ConversationTurn(
                    turn_number=len(self.conversation_log) + 1,
                    speaker="director",
                    message="Start investment committee meeting",
                    response=meeting_result.get("opening_statement", "Meeting started"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    turn_type="meeting_start",
                    duration_seconds=1.0
                )
                
                self.conversation_log.append(opening_turn)
                
                # Run conversation rounds
                self._run_conversation_rounds()
                
            else:
                self.logger.warning("Failed to start committee meeting")
                
        except Exception as e:
            self.logger.error(f"Committee conversation failed: {e}")
            self.errors.append(f"Committee conversation error: {str(e)}")
    
    def _run_conversation_rounds(self) -> None:
        """Run turn-based conversation rounds"""
        max_turns = self.config["conversation"]["max_turns"]
        current_turn = 0
        
        # Define conversation flow
        conversation_topics = [
            ("director", "qullamaggie_agent", "Challenge the TSLA momentum proposal - justify the position size"),
            ("qullamaggie_agent", None, "Defend TSLA proposal"),
            ("director", "risk_manager", "What's your assessment of the current proposals?"),
            ("risk_manager", None, "Risk assessment"),
            ("director", "qullamaggie_agent", "Address the risk manager's concerns"),
            ("qullamaggie_agent", "risk_manager", "Challenge the risk assessment"),
            ("risk_manager", "qullamaggie_agent", "Defend risk position"),
            ("director", None, "Resolve any conflicts and summarize positions")
        ]
        
        for speaker, target, topic in conversation_topics:
            if current_turn >= max_turns or self.interrupted:
                break
                
            if speaker not in self.agents:
                continue
                
            try:
                current_turn += 1
                start_time = time.time()
                
                # Prepare message context
                context = {"target_agent": target} if target else {}
                
                # Get agent response
                result = self._call_agent_with_retry(speaker, topic, context)
                
                # Determine turn type
                turn_type = "challenge" if target else "response"
                if "resolve" in topic.lower():
                    turn_type = "conflict_resolution"
                
                # Log conversation turn
                conv_turn = ConversationTurn(
                    turn_number=current_turn,
                    speaker=speaker,
                    message=topic,
                    response=self._extract_response_summary(result),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    turn_type=turn_type,
                    target_agent=target,
                    duration_seconds=time.time() - start_time
                )
                
                self.conversation_log.append(conv_turn)
                
                self.logger.info(f"Turn {current_turn}: {speaker} -> {target or 'committee'}")
                
                # Check for early termination
                if self._should_end_conversation(current_turn):
                    self.logger.info("Director ending conversation early")
                    break
                    
                # Brief pause between turns
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error in conversation turn {current_turn}: {e}")
                self.errors.append(f"Conversation turn error: {str(e)}")
        
        self.logger.info(f"✅ Committee conversation completed ({current_turn} turns)")
    
    def _should_end_conversation(self, current_turn: int) -> bool:
        """Check if Director wants to end conversation early"""
        if current_turn < 4:  # Minimum turns
            return False
            
        # Simple heuristic: end if we've had good coverage
        if current_turn >= 6:
            return True
            
        return False
    
    def _run_final_decision_phase(self) -> Dict[str, Any]:
        """Phase 6: Director makes final decision"""
        self.logger.info("⚖️ Phase 6: Final Decision")
        
        if "director" not in self.agents:
            self.logger.error("Cannot make final decision without Director")
            return {}
        
        try:
            start_time = time.time()
            
            # Director makes final decision
            decision_result = self._call_agent_with_retry("director", "Make final decision on all proposals")
            
            # Log final decision
            decision_turn = ConversationTurn(
                turn_number=len(self.conversation_log) + 1,
                speaker="director",
                message="Make final decision on all proposals",
                response=decision_result.get("summary", "Final decision made"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                turn_type="final_decision",
                duration_seconds=time.time() - start_time
            )
            
            self.conversation_log.append(decision_turn)
            
            self.logger.info(f"✅ Final decision completed: {decision_result.get('decision_id', 'unknown')}")
            return decision_result
            
        except Exception as e:
            self.logger.error(f"Final decision failed: {e}")
            self.errors.append(f"Final decision error: {str(e)}")
            return {}
    
    def _call_agent_with_retry(self, agent_name: str, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call agent with retry logic"""
        max_retries = self.config["error_handling"]["max_retries"]
        retry_delay = self.config["error_handling"]["retry_delay_seconds"]
        
        for attempt in range(max_retries):
            try:
                if agent_name not in self.agents:
                    raise ValueError(f"Agent {agent_name} not available")
                
                agent = self.agents[agent_name]
                result = agent.process_message(message, context)
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Agent {agent_name} call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise e
    
    def _extract_response_summary(self, result: Dict[str, Any]) -> str:
        """Extract summary from agent response"""
        if not result:
            return "No response"
        
        # Try various response fields
        for field in ["summary", "message", "response", "ruling", "assessment"]:
            if field in result and result[field]:
                response = str(result[field])
                # Truncate if too long
                if len(response) > 200:
                    response = response[:197] + "..."
                return response
        
        # Fallback to result type
        return f"{result.get('type', 'unknown')} response"
    
    def _create_session_summary(self, final_decision: Dict[str, Any]) -> CommitteeSession:
        """Create complete session summary"""
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - datetime.fromisoformat(self.start_time.isoformat())).total_seconds()
        
        # Count proposals
        proposal_turns = [t for t in self.conversation_log if t.turn_type == "proposal_generation"]
        proposals_count = len(proposal_turns)
        
        # Generate session summary
        approved = final_decision.get("approved_count", 0)
        rejected = final_decision.get("rejected_count", 0)
        adjusted = final_decision.get("adjusted_count", 0)
        
        summary = f"Investment Committee Session {self.session_id}: "
        summary += f"{proposals_count} proposals reviewed, "
        summary += f"{approved} approved, {rejected} rejected, {adjusted} adjusted. "
        summary += f"Duration: {total_duration:.1f}s, {len(self.conversation_log)} conversation turns."
        
        if self.errors:
            summary += f" {len(self.errors)} errors encountered."
        
        return CommitteeSession(
            session_id=self.session_id,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=total_duration,
            participants=list(self.agents.keys()),
            proposals_count=proposals_count,
            conversation_turns=len(self.conversation_log),
            final_decision_id=final_decision.get("decision_id", ""),
            conversation_log=self.conversation_log,
            errors_encountered=self.errors,
            session_summary=summary
        )
    
    def _create_failed_session(self, error_message: str) -> CommitteeSession:
        """Create session summary for failed session"""
        end_time = datetime.now(timezone.utc)
        
        if self.start_time:
            total_duration = (end_time - self.start_time).total_seconds()
            start_time_iso = self.start_time.isoformat()
        else:
            total_duration = 0.0
            start_time_iso = end_time.isoformat()
        
        return CommitteeSession(
            session_id=self.session_id,
            start_time=start_time_iso,
            end_time=end_time.isoformat(),
            total_duration_seconds=total_duration,
            participants=list(self.agents.keys()),
            proposals_count=0,
            conversation_turns=len(self.conversation_log),
            final_decision_id="",
            conversation_log=self.conversation_log,
            errors_encountered=self.errors + [error_message],
            session_summary=f"FAILED SESSION: {error_message}"
        )
    
    def _log_conversation_to_file(self, session: CommitteeSession) -> None:
        """Log complete conversation to file"""
        try:
            filename = f"committee_session_{self.session_id}.json"
            filepath = self.conversations_folder / filename
            
            # Convert to serializable format
            session_dict = asdict(session)
            
            with open(filepath, 'w') as f:
                json.dump(session_dict, f, indent=2, default=str)
            
            self.logger.info(f"📄 Conversation logged to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to log conversation: {e}")


def main():
    """Main entry point for investment committee orchestrator"""
    print("🏛️ Investment Committee Orchestrator")
    print("=" * 50)
    
    try:
        # Initialize orchestrator
        orchestrator = InvestmentCommitteeOrchestrator()
        
        # Run investment committee session
        session = orchestrator.run_investment_committee()
        
        # Print summary report
        print(f"\n📊 SESSION SUMMARY")
        print("-" * 30)
        print(f"Session ID: {session.session_id}")
        print(f"Duration: {session.total_duration_seconds:.1f} seconds")
        print(f"Participants: {', '.join(session.participants)}")
        print(f"Proposals: {session.proposals_count}")
        print(f"Conversation Turns: {session.conversation_turns}")
        print(f"Errors: {len(session.errors_encountered)}")
        
        if session.final_decision_id:
            print(f"Final Decision: {session.final_decision_id}")
        
        print(f"\nSummary: {session.session_summary}")
        
        if session.errors_encountered:
            print(f"\n⚠️ Errors Encountered:")
            for error in session.errors_encountered:
                print(f"  • {error}")
        
        print(f"\n💡 Next Steps:")
        print(f"  • Review conversation log in conversations/ folder")
        print(f"  • Check decision report in decisions/ folder")
        print(f"  • Execute approved trades")
        
        return session
        
    except Exception as e:
        print(f"❌ Critical orchestrator error: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    session = main()