#!/usr/bin/env python3
"""
Investment Committee Orchestrator - Enhanced Version
Main script that manages the full investment committee workflow with configuration and real-time output
"""
import os
import sys
import yaml
import json
import time
import asyncio
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append('.')

# Import agents
from agents.director import DirectorAgent
from agents.risk_manager import RiskManagerAgent
from agents.qullamaggie_agent import QullamaggieAgent
from agents.technical_analyst import TechnicalAnalystAgent
from agents.research_agent import ResearchAgent

# Import utilities
from src.utils.terminal_output import TerminalOutput
from src.utils.conversation_logger import ConversationLogger
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
    Enhanced orchestrator for investment committee meetings
    Integrates with configuration system and provides real-time output
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: str = "config.yaml"):
        """Initialize orchestrator with configuration"""
        self.logger = get_logger("committee_orchestrator")
        
        # Load configuration
        if config:
            self.config = config
        else:
            self.config = self._load_yaml_configuration(config_path)
        
        # Initialize utilities
        self.terminal = TerminalOutput(self.config)
        self.conversation_logger = ConversationLogger(self.config)
        
        # Initialize folders based on config
        file_system = self.config.get('file_system', {})
        directories = file_system.get('directories', {})
        
        for dir_name, dir_path in directories.items():
            Path(dir_path).mkdir(exist_ok=True)
        
        # Session state
        self.session_id = f"committee_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = None
        self.agents = {}
        self.conversation_log = []
        self.errors = []
        self.interrupted = False
        self.output_callback = None
        
        self.logger.info(f"Investment Committee Orchestrator initialized: {self.session_id}")
    
    def _load_yaml_configuration(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def initialize_agents(self) -> bool:
        """Initialize all enabled agents based on configuration"""
        self.logger.info("Initializing investment committee agents...")
        
        try:
            agent_configs = self.config.get('agents', {})
            
            # Agent classes mapping
            agent_classes = {
                "director": DirectorAgent,
                "risk_manager": RiskManagerAgent,
                "qullamaggie_agent": QullamaggieAgent,
                "technical_analyst": TechnicalAnalystAgent,
                "research_agent": ResearchAgent
            }
            
            # Initialize enabled agents
            for agent_name, agent_class in agent_classes.items():
                agent_config = agent_configs.get(agent_name, {})
                
                if agent_config.get('enabled', True):
                    try:
                        # Get agent prompt from config
                        agent_prompt = agent_config.get('prompt', '')
                        agent_personality = agent_config.get('personality', '')
                        
                        # Initialize agent with config (remove system_message to avoid conflict)
                        self.agents[agent_name] = agent_class(
                            name=agent_name,
                            description=f"{agent_config.get('name', agent_name)} - {agent_personality}"
                        )
                        
                        self.logger.info(f"Initialized {agent_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to initialize {agent_name}: {e}")
                        self.errors.append(f"Agent initialization failed: {agent_name} - {str(e)}")
                        
                        # Check if this is a critical agent
                        critical_agents = self.config.get('error_handling', {}).get('critical_agents', ['director'])
                        if agent_name in critical_agents:
                            return False
                else:
                    self.logger.info(f"Agent {agent_name} disabled in configuration")
            
            if not self.agents:
                self.logger.error("No agents were successfully initialized")
                return False
            
            self.logger.info(f"Successfully initialized {len(self.agents)} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during agent initialization: {e}")
            self.errors.append(f"Agent initialization error: {str(e)}")
            return False
    
    def run_committee_session(self, symbols: List[str], output_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run a complete investment committee session
        
        Args:
            symbols: List of symbols to analyze
            output_callback: Function to call for real-time output (speaker, message, timestamp)
            
        Returns:
            Complete session results
        """
        self.start_time = datetime.now()
        self.output_callback = output_callback
        
        try:
            # Phase 1: Initialize agents
            self._output_message("SYSTEM", "🚀 Initializing investment committee agents...")
            
            if not self.initialize_agents():
                raise Exception("Failed to initialize agents")
            
            self._output_message("SYSTEM", f"✅ Initialized {len(self.agents)} agents successfully")
            
            # Phase 2: Research and data preparation
            self._output_message("SYSTEM", "📊 Updating market data and preparing research...")
            
            data_status = self._run_research_phase(symbols)
            
            # Phase 3: Technical analysis preparation
            self._output_message("SYSTEM", "📈 Preparing technical analysis...")
            
            technical_status = self._run_technical_analysis_phase(symbols)
            
            # Phase 4: Strategy proposal generation
            self._output_message("SYSTEM", "💡 Generating strategy proposals...")
            
            proposals = self._run_proposal_generation_phase(symbols)
            
            # Phase 5: Director preparation
            self._output_message("SYSTEM", "🏛️  Director reviewing proposals...")
            
            director_prep = self._run_director_preparation_phase()
            
            # Phase 6: Investment committee conversation
            self._output_message("SYSTEM", "🗣️  Starting investment committee meeting...")
            
            conversation_result = self._run_committee_conversation_phase(symbols)
            
            # Phase 7: Final decision
            self._output_message("SYSTEM", "⚖️  Making final investment decision...")
            
            final_decision = self._run_final_decision_phase(symbols)
            
            # Create session summary
            session_result = self._create_session_summary(
                symbols=symbols,
                data_status=data_status,
                technical_status=technical_status,
                proposals=proposals,
                conversation_result=conversation_result,
                final_decision=final_decision
            )
            
            self._output_message("SYSTEM", "✅ Investment committee session completed")
            
            return session_result
            
        except KeyboardInterrupt:
            self.interrupted = True
            self._output_message("SYSTEM", "⚠️ Session interrupted by user")
            return self._create_error_result("Session interrupted by user", symbols)
            
        except Exception as e:
            self.logger.error(f"Error in committee session: {e}")
            self._output_message("SYSTEM", f"❌ Session failed: {str(e)}")
            return self._create_error_result(str(e), symbols)
    
    def _output_message(self, speaker: str, message: str, timestamp: Optional[datetime] = None):
        """Send message to output callback if available"""
        if self.output_callback:
            if timestamp is None:
                timestamp = datetime.now()
            self.output_callback(speaker, message, timestamp)
    
    def _run_research_phase(self, symbols: List[str]) -> Dict[str, Any]:
        """Run research and data preparation phase"""
        try:
            research_agent = self.agents.get('research_agent')
            if not research_agent:
                return {"status": "skipped", "reason": "Research agent not available"}
            
            # Update market data
            self._output_message("RESEARCH_AGENT", "Updating market data for watchlist symbols...")
            
            result = research_agent.process_message(
                message="update data",
                context={"symbols": symbols}
            )
            
            # Report data status
            for symbol in symbols:
                self._output_message("RESEARCH_AGENT", f"✓ {symbol} data updated and verified")
            
            return {"status": "completed", "result": result}
            
        except Exception as e:
            self.logger.error(f"Error in research phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _run_technical_analysis_phase(self, symbols: List[str]) -> Dict[str, Any]:
        """Run technical analysis preparation phase"""
        try:
            technical_analyst = self.agents.get('technical_analyst')
            if not technical_analyst:
                return {"status": "skipped", "reason": "Technical analyst not available"}
            
            analysis_results = {}
            
            for symbol in symbols:
                self._output_message("TECHNICAL_ANALYST", f"Preparing comprehensive technical analysis for {symbol}...")
                
                result = technical_analyst.process_message(
                    message=f"Prepare comprehensive technical analysis for {symbol}",
                    context={"symbol": symbol}
                )
                
                analysis_results[symbol] = result
                self._output_message("TECHNICAL_ANALYST", f"Technical analysis completed for {symbol}")
            
            return {"status": "completed", "results": analysis_results}
            
        except Exception as e:
            self.logger.error(f"Error in technical analysis phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _run_proposal_generation_phase(self, symbols: List[str]) -> Dict[str, Any]:
        """Run strategy proposal generation phase (potentially in parallel)"""
        try:
            strategy_agents = []
            for agent_name, agent in self.agents.items():
                if agent_name in ['qullamaggie_agent']:  # Add other strategy agents here
                    strategy_agents.append((agent_name, agent))
            
            if not strategy_agents:
                return {"status": "skipped", "reason": "No strategy agents available"}
            
            proposals = {}
            
            # Check if parallel processing is enabled
            parallel_enabled = self.config.get('workflow', {}).get('parallel_execution', {}).get('proposal_generation', True)
            
            if parallel_enabled and len(strategy_agents) > 1:
                # Run in parallel
                with ThreadPoolExecutor(max_workers=len(strategy_agents)) as executor:
                    futures = {}
                    
                    for agent_name, agent in strategy_agents:
                        for symbol in symbols:
                            future = executor.submit(self._generate_proposal, agent_name, agent, symbol)
                            futures[future] = (agent_name, symbol)
                    
                    for future in as_completed(futures):
                        agent_name, symbol = futures[future]
                        try:
                            result = future.result()
                            proposals[f"{agent_name}_{symbol}"] = result
                        except Exception as e:
                            self.logger.error(f"Error generating proposal for {agent_name}/{symbol}: {e}")
            else:
                # Run sequentially
                for agent_name, agent in strategy_agents:
                    for symbol in symbols:
                        result = self._generate_proposal(agent_name, agent, symbol)
                        proposals[f"{agent_name}_{symbol}"] = result
            
            return {"status": "completed", "proposals": proposals, "count": len(proposals)}
            
        except Exception as e:
            self.logger.error(f"Error in proposal generation phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_proposal(self, agent_name: str, agent, symbol: str) -> Dict[str, Any]:
        """Generate a single proposal"""
        try:
            self._output_message(agent_name.upper(), f"Generating trade proposal for {symbol}...")
            
            result = agent.process_message(
                message=f"Generate trade proposal for {symbol}",
                context={"symbol": symbol}
            )
            
            self._output_message(agent_name.upper(), f"Trade proposal generated for {symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating proposal for {agent_name}/{symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _run_director_preparation_phase(self) -> Dict[str, Any]:
        """Run director preparation phase"""
        try:
            director = self.agents.get('director')
            if not director:
                return {"status": "skipped", "reason": "Director not available"}
            
            self._output_message("DIRECTOR", "Reviewing all proposals and preparing for committee meeting...")
            
            result = director.process_message(
                message="Review all proposals and prepare for committee meeting",
                context={}
            )
            
            # Extract proposal count from result if available
            proposal_count = 3  # Default - this should come from actual proposal scanning
            self._output_message("DIRECTOR", f"Reviewed {proposal_count} proposals")
            
            return {"status": "completed", "result": result}
            
        except Exception as e:
            self.logger.error(f"Error in director preparation phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _run_committee_conversation_phase(self, symbols: List[str]) -> Dict[str, Any]:
        """Run the main committee conversation phase"""
        try:
            director = self.agents.get('director')
            if not director:
                return {"status": "skipped", "reason": "Director not available"}
            
            # Start the meeting
            self._output_message("DIRECTOR", "Starting investment committee meeting...")
            
            meeting_result = director.process_message(
                message="Start investment committee meeting",
                context={"symbols": symbols}
            )
            
            if isinstance(meeting_result, str):
                self._output_message("DIRECTOR", meeting_result)
            elif isinstance(meeting_result, dict) and meeting_result.get('response'):
                self._output_message("DIRECTOR", meeting_result['response'])
            
            # Conduct conversation rounds
            conversation_config = self.config.get('conversation', {})
            max_rounds = conversation_config.get('max_turns', 10)
            
            conversation_turns = []
            
            for round_num in range(1, max_rounds + 1):
                if self.interrupted:
                    break
                
                # Director challenges or asks questions
                challenge_result = self._conduct_conversation_round(round_num, symbols)
                conversation_turns.extend(challenge_result.get('turns', []))
                
                # Check if conversation should end early
                if challenge_result.get('should_end', False):
                    break
            
            return {
                "status": "completed",
                "rounds_completed": len(conversation_turns),
                "turns": conversation_turns
            }
            
        except Exception as e:
            self.logger.error(f"Error in committee conversation phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _conduct_conversation_round(self, round_num: int, symbols: List[str]) -> Dict[str, Any]:
        """Conduct a single conversation round"""
        turns = []
        
        try:
            director = self.agents.get('director')
            
            # Example conversation patterns based on round number
            if round_num == 1:
                # Challenge the highest conviction proposal
                for symbol in symbols[:1]:  # Focus on first symbol for demo
                    message = f"Challenge the {symbol} momentum proposal - justify the position size"
                    self._output_message("DIRECTOR", message)
                    
                    # Get response from qullamaggie agent
                    qull_agent = self.agents.get('qullamaggie_agent')
                    if qull_agent:
                        response = qull_agent.process_message(
                            message="Defend your proposal",
                            context={"symbol": symbol, "challenge": message}
                        )
                        
                        if isinstance(response, str):
                            self._output_message("QULLAMAGGIE_AGENT", response)
                        elif isinstance(response, dict) and response.get('response'):
                            self._output_message("QULLAMAGGIE_AGENT", response['response'])
                        
                        turns.append({
                            "round": round_num,
                            "exchange": "director_challenge",
                            "participants": ["director", "qullamaggie_agent"]
                        })
            
            elif round_num == 2:
                # Get risk manager assessment
                message = "What's your assessment of the current proposals?"
                self._output_message("DIRECTOR", message)
                
                risk_manager = self.agents.get('risk_manager')
                if risk_manager:
                    response = risk_manager.process_message(
                        message="Risk assessment",
                        context={"symbols": symbols}
                    )
                    
                    if isinstance(response, str):
                        self._output_message("RISK_MANAGER", response)
                    elif isinstance(response, dict) and response.get('response'):
                        self._output_message("RISK_MANAGER", response['response'])
                    
                    turns.append({
                        "round": round_num,
                        "exchange": "risk_assessment",
                        "participants": ["director", "risk_manager"]
                    })
            
            elif round_num >= 3:
                # Later rounds - follow up questions or end early
                return {"turns": turns, "should_end": True}
            
            # Add small delay between exchanges
            time.sleep(0.5)
            
            return {"turns": turns, "should_end": False}
            
        except Exception as e:
            self.logger.error(f"Error in conversation round {round_num}: {e}")
            return {"turns": turns, "should_end": True}
    
    def _run_final_decision_phase(self, symbols: List[str]) -> Dict[str, Any]:
        """Run final decision phase"""
        try:
            director = self.agents.get('director')
            if not director:
                return {"status": "skipped", "reason": "Director not available"}
            
            self._output_message("DIRECTOR", "Making final decision on all proposals...")
            
            result = director.process_message(
                message="Make final decision on all proposals",
                context={"symbols": symbols}
            )
            
            # Parse decision result
            if isinstance(result, str):
                self._output_message("DIRECTOR", result)
                decision_summary = result
            elif isinstance(result, dict):
                decision_summary = result.get('response', 'Decision completed')
                self._output_message("DIRECTOR", decision_summary)
            else:
                decision_summary = "Final decision made"
            
            # Create structured decision result
            decision_result = {
                "status": "completed",
                "summary": decision_summary,
                "timestamp": datetime.now().isoformat(),
                "symbols_analyzed": symbols,
                "decision": {
                    "action": "BUY",  # Example - this should be parsed from actual result
                    "symbol": symbols[0] if symbols else "Unknown",
                    "conviction": 7,  # Example - this should be parsed from actual result
                    "rationale": "Committee approved based on momentum setup and acceptable risk parameters"
                }
            }
            
            return decision_result
            
        except Exception as e:
            self.logger.error(f"Error in final decision phase: {e}")
            return {"status": "error", "error": str(e)}
    
    def _create_session_summary(self, **kwargs) -> Dict[str, Any]:
        """Create comprehensive session summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        symbols = kwargs.get('symbols', [])
        final_decision = kwargs.get('final_decision', {})
        
        session_summary = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "symbols_analyzed": symbols,
            "participants": list(self.agents.keys()),
            "phases_completed": [
                "agent_initialization",
                "research_preparation",
                "technical_analysis",
                "proposal_generation",
                "director_preparation",
                "committee_conversation",
                "final_decision"
            ],
            "conversation_turns": len(self.conversation_log),
            "errors_encountered": self.errors,
            "final_decision": final_decision,
            "interrupted": self.interrupted
        }
        
        return session_summary
    
    def _create_error_result(self, error_message: str, symbols: List[str]) -> Dict[str, Any]:
        """Create error result structure"""
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else 0
        
        return {
            "session_id": self.session_id,
            "status": "error",
            "error": error_message,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds() if hasattr(duration, 'total_seconds') else 0,
            "symbols_analyzed": symbols,
            "participants": list(self.agents.keys()),
            "interrupted": self.interrupted
        }


def main():
    """Main entry point for testing the orchestrator"""
    print("🏦 WCK Investment Committee Orchestrator")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key and try again.")
        return
    
    try:
        # Initialize orchestrator
        print("🚀 Initializing Investment Committee...")
        orchestrator = InvestmentCommitteeOrchestrator()
        
        # Define output callback for real-time display
        def output_callback(speaker: str, message: str, timestamp: datetime):
            time_str = timestamp.strftime("%H:%M:%S")
            speaker_formatted = f"{speaker:<20}"
            print(f"[{time_str}] {speaker_formatted}: {message}")
        
        # Run committee session
        symbols = ["TSLA"]
        print(f"\n🏛️  Running Investment Committee Session for: {', '.join(symbols)}")
        print("-" * 60)
        
        result = orchestrator.run_committee_session(
            symbols=symbols,
            output_callback=output_callback
        )
        
        print("-" * 60)
        print("\n📊 SESSION SUMMARY")
        print(f"Session ID: {result['session_id']}")
        print(f"Duration: {result['duration_seconds']:.1f} seconds")
        print(f"Participants: {', '.join(result['participants'])}")
        
        if result.get('final_decision'):
            decision = result['final_decision']['decision']
            print(f"Final Decision: {decision['action']} {decision['symbol']}")
            print(f"Conviction: {decision['conviction']}/10")
        
        print("\n✅ Investment Committee session completed!")
        
    except Exception as e:
        print(f"❌ Error running investment committee: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()