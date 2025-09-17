"""
AutoGen orchestrator for AI Hedge Fund System
Main coordination hub for multi-agent conversations and workflows
"""
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
import uuid
import logging

try:
    from autogen import GroupChat, GroupChatManager, UserProxyAgent
    from autogen.agentchat.agent import Agent
except ImportError:
    # Fallback for development without autogen installed
    class GroupChat:
        def __init__(self, **kwargs):
            pass
    class GroupChatManager:
        def __init__(self, **kwargs):
            pass
    class UserProxyAgent:
        def __init__(self, **kwargs):
            pass
    class Agent:
        pass

from ..agents.base_agent import BaseHedgeFundAgent, AgentCapability
from ..models.market_state import MarketState
from ..models.portfolio import Portfolio
from ..models.trade import TradeSignal, Order
from ..utils.logging_config import get_logger


class ConversationType(str, Enum):
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_REVIEW = "portfolio_review"
    STRATEGY_EXECUTION = "strategy_execution"
    EMERGENCY_RESPONSE = "emergency_response"


class ConversationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ConversationFlow:
    """Defines a conversation workflow between agents"""
    conversation_id: str
    conversation_type: ConversationType
    participants: List[str]  # Agent names
    initiator: str
    sequence: List[Dict[str, Any]]  # Conversation steps
    timeout_minutes: int = 30
    max_rounds: int = 10
    require_consensus: bool = False
    
    
@dataclass
class ConversationResult:
    """Result of a completed conversation"""
    conversation_id: str
    status: ConversationStatus
    outcome: Dict[str, Any]
    participants: List[str]
    messages: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


class HedgeFundOrchestrator:
    """
    Main orchestrator for coordinating multi-agent conversations
    Manages conversation flows, consensus building, and result aggregation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the orchestrator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger("orchestrator")
        
        # Agent management
        self.agents: Dict[str, BaseHedgeFundAgent] = {}
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationFlow] = {}
        self.conversation_history: List[ConversationResult] = []
        
        # AutoGen components
        self.group_chats: Dict[str, GroupChat] = {}
        self.chat_managers: Dict[str, GroupChatManager] = {}
        
        # System proxy agent for coordination
        self.system_proxy = UserProxyAgent(
            name="system_coordinator",
            human_input_mode="NEVER",
            code_execution_config={"use_docker": False},
            system_message="System coordinator for hedge fund operations."
        )
        
        # Conversation patterns
        self._conversation_patterns = self._initialize_conversation_patterns()
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        self.logger.info("HedgeFundOrchestrator initialized")
    
    def register_agent(self, agent: BaseHedgeFundAgent) -> None:
        """
        Register an agent with the orchestrator
        
        Args:
            agent: Agent to register
        """
        self.agents[agent.name] = agent
        self.agent_capabilities[agent.name] = agent.capabilities
        
        # Add event handlers
        agent.add_event_handler("signal_generated", self._handle_signal_event)
        agent.add_event_handler("risk_alert", self._handle_risk_alert)
        
        self.logger.info(f"Registered agent: {agent.name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Remove an agent from the orchestrator"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            del self.agent_capabilities[agent_name]
            self.logger.info(f"Unregistered agent: {agent_name}")
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[BaseHedgeFundAgent]:
        """Get agents with specific capability"""
        return [
            agent for agent_name, agent in self.agents.items()
            if capability in self.agent_capabilities.get(agent_name, [])
        ]
    
    async def start_conversation(
        self,
        conversation_type: ConversationType,
        initiator: str,
        context: Dict[str, Any] = None,
        participants: List[str] = None
    ) -> str:
        """
        Start a new conversation workflow
        
        Args:
            conversation_type: Type of conversation
            initiator: Agent or system initiating the conversation
            context: Conversation context data
            participants: Specific agents to include (optional)
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        # Determine participants if not specified
        if not participants:
            participants = self._select_participants(conversation_type, context)
        
        # Create conversation flow
        flow = ConversationFlow(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            participants=participants,
            initiator=initiator,
            sequence=self._get_conversation_sequence(conversation_type),
            **self._get_conversation_config(conversation_type)
        )
        
        self.active_conversations[conversation_id] = flow
        
        try:
            # Execute conversation
            result = await self._execute_conversation(flow, context or {})
            
            # Store result
            self.conversation_history.append(result)
            
            # Remove from active conversations
            del self.active_conversations[conversation_id]
            
            self.logger.info(f"Completed conversation {conversation_id}: {result.status}")
            
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"Conversation {conversation_id} failed: {e}")
            
            # Create failed result
            result = ConversationResult(
                conversation_id=conversation_id,
                status=ConversationStatus.FAILED,
                outcome={"error": str(e)},
                participants=participants,
                messages=[],
                start_time=datetime.now(timezone.utc)
            )
            
            self.conversation_history.append(result)
            
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]
            
            raise
    
    async def _execute_conversation(self, flow: ConversationFlow, context: Dict[str, Any]) -> ConversationResult:
        """Execute a conversation flow"""
        start_time = datetime.now(timezone.utc)
        messages = []
        
        try:
            # Get participating agents
            participating_agents = [
                self.agents[name] for name in flow.participants
                if name in self.agents
            ]
            
            if not participating_agents:
                raise ValueError("No valid participating agents found")
            
            # Create group chat
            group_chat = GroupChat(
                agents=participating_agents + [self.system_proxy],
                messages=[],
                max_round=flow.max_rounds,
                speaker_selection_method="auto"
            )
            
            # Create chat manager
            chat_manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=self._get_manager_llm_config()
            )
            
            # Store for tracking
            self.group_chats[flow.conversation_id] = group_chat
            self.chat_managers[flow.conversation_id] = chat_manager
            
            # Execute conversation sequence
            for step in flow.sequence:
                step_result = await self._execute_conversation_step(
                    step, participating_agents, chat_manager, context
                )
                messages.extend(step_result.get("messages", []))
            
            # Build outcome
            outcome = self._build_conversation_outcome(flow, messages, context)
            
            # Check for consensus if required
            if flow.require_consensus:
                consensus_result = await self._check_consensus(participating_agents, outcome)
                outcome["consensus"] = consensus_result
            
            end_time = datetime.now(timezone.utc)
            
            return ConversationResult(
                conversation_id=flow.conversation_id,
                status=ConversationStatus.COMPLETED,
                outcome=outcome,
                participants=flow.participants,
                messages=messages,
                start_time=start_time,
                end_time=end_time
            )
            
        except Exception as e:
            end_time = datetime.now(timezone.utc)
            
            return ConversationResult(
                conversation_id=flow.conversation_id,
                status=ConversationStatus.FAILED,
                outcome={"error": str(e)},
                participants=flow.participants,
                messages=messages,
                start_time=start_time,
                end_time=end_time
            )
    
    async def _execute_conversation_step(
        self,
        step: Dict[str, Any],
        agents: List[BaseHedgeFundAgent],
        chat_manager: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single conversation step"""
        step_type = step.get("type")
        
        if step_type == "broadcast":
            return await self._execute_broadcast_step(step, agents, context)
        elif step_type == "sequential":
            return await self._execute_sequential_step(step, agents, context)
        elif step_type == "consensus":
            return await self._execute_consensus_step(step, agents, context)
        elif step_type == "analysis":
            return await self._execute_analysis_step(step, agents, context)
        else:
            self.logger.warning(f"Unknown step type: {step_type}")
            return {"messages": []}
    
    async def _execute_broadcast_step(
        self,
        step: Dict[str, Any],
        agents: List[BaseHedgeFundAgent],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute broadcast step - send message to all agents"""
        message = step.get("message", "")
        responses = []
        
        for agent in agents:
            try:
                response = agent.process_message({
                    "content": message,
                    "context": context,
                    "step": step
                })
                responses.append({
                    "agent": agent.name,
                    "response": response,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                self.logger.error(f"Agent {agent.name} failed to process broadcast: {e}")
                responses.append({
                    "agent": agent.name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        return {"messages": responses}
    
    async def _execute_sequential_step(
        self,
        step: Dict[str, Any],
        agents: List[BaseHedgeFundAgent],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute sequential step - agents respond in order"""
        sequence = step.get("sequence", [agent.name for agent in agents])
        message = step.get("message", "")
        responses = []
        accumulated_context = context.copy()
        
        for agent_name in sequence:
            agent = next((a for a in agents if a.name == agent_name), None)
            if not agent:
                continue
            
            try:
                response = agent.process_message({
                    "content": message,
                    "context": accumulated_context,
                    "step": step,
                    "previous_responses": responses
                })
                
                response_entry = {
                    "agent": agent.name,
                    "response": response,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                responses.append(response_entry)
                
                # Update context with response
                accumulated_context[f"{agent.name}_response"] = response
                
            except Exception as e:
                self.logger.error(f"Agent {agent.name} failed in sequential step: {e}")
                responses.append({
                    "agent": agent.name,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        return {"messages": responses}
    
    async def _execute_consensus_step(
        self,
        step: Dict[str, Any],
        agents: List[BaseHedgeFundAgent],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute consensus step - build agreement among agents"""
        topic = step.get("topic", "")
        max_rounds = step.get("max_rounds", 3)
        
        responses = []
        consensus_reached = False
        
        for round_num in range(max_rounds):
            round_responses = []
            
            # Get responses from all agents
            for agent in agents:
                try:
                    response = agent.process_message({
                        "content": f"Round {round_num + 1} consensus on: {topic}",
                        "context": context,
                        "step": step,
                        "round": round_num,
                        "previous_round_responses": responses
                    })
                    
                    round_responses.append({
                        "agent": agent.name,
                        "response": response,
                        "round": round_num,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(f"Agent {agent.name} failed in consensus round {round_num}: {e}")
            
            responses.extend(round_responses)
            
            # Check for consensus
            consensus_reached = self._check_round_consensus(round_responses)
            if consensus_reached:
                break
        
        return {
            "messages": responses,
            "consensus_reached": consensus_reached,
            "rounds_completed": round_num + 1
        }
    
    async def _execute_analysis_step(
        self,
        step: Dict[str, Any],
        agents: List[BaseHedgeFundAgent],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analysis step - focused analysis by relevant agents"""
        analysis_type = step.get("analysis_type", "general")
        
        # Filter agents by capability
        relevant_agents = []
        if analysis_type == "market":
            relevant_agents = self.get_agents_by_capability(AgentCapability.MARKET_ANALYSIS)
        elif analysis_type == "risk":
            relevant_agents = self.get_agents_by_capability(AgentCapability.RISK_ASSESSMENT)
        elif analysis_type == "portfolio":
            relevant_agents = self.get_agents_by_capability(AgentCapability.PORTFOLIO_MANAGEMENT)
        else:
            relevant_agents = agents
        
        # Execute analysis
        return await self._execute_broadcast_step(step, relevant_agents, context)
    
    def _select_participants(self, conversation_type: ConversationType, context: Dict[str, Any]) -> List[str]:
        """Select appropriate agents for conversation type"""
        participants = []
        
        if conversation_type == ConversationType.MARKET_ANALYSIS:
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.MARKET_ANALYSIS)
            ])
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.RESEARCH)
            ])
            
        elif conversation_type == ConversationType.SIGNAL_GENERATION:
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.SIGNAL_GENERATION)
            ])
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.MARKET_ANALYSIS)
            ])
            
        elif conversation_type == ConversationType.RISK_ASSESSMENT:
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.RISK_ASSESSMENT)
            ])
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.PORTFOLIO_MANAGEMENT)
            ])
            
        elif conversation_type == ConversationType.PORTFOLIO_REVIEW:
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.PORTFOLIO_MANAGEMENT)
            ])
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.RISK_ASSESSMENT)
            ])
            
        elif conversation_type == ConversationType.STRATEGY_EXECUTION:
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.SIGNAL_GENERATION)
            ])
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.RISK_ASSESSMENT)
            ])
            participants.extend([
                agent.name for agent in self.get_agents_by_capability(AgentCapability.TRADE_EXECUTION)
            ])
        
        # Remove duplicates and ensure we have participants
        participants = list(set(participants))
        
        if not participants:
            # Fall back to all available agents
            participants = list(self.agents.keys())
        
        return participants
    
    def _initialize_conversation_patterns(self) -> Dict[ConversationType, Dict[str, Any]]:
        """Initialize conversation patterns for different types"""
        return {
            ConversationType.MARKET_ANALYSIS: {
                "sequence": [
                    {"type": "broadcast", "message": "Analyze current market conditions"},
                    {"type": "sequential", "message": "Provide detailed market assessment"},
                    {"type": "consensus", "topic": "market regime classification", "max_rounds": 2}
                ],
                "timeout_minutes": 20,
                "max_rounds": 8,
                "require_consensus": True
            },
            
            ConversationType.SIGNAL_GENERATION: {
                "sequence": [
                    {"type": "broadcast", "message": "Screen for trading opportunities"},
                    {"type": "analysis", "analysis_type": "market", "message": "Validate market context"},
                    {"type": "sequential", "message": "Generate and rank signals"},
                    {"type": "analysis", "analysis_type": "risk", "message": "Assess signal risks"}
                ],
                "timeout_minutes": 30,
                "max_rounds": 10,
                "require_consensus": False
            },
            
            ConversationType.RISK_ASSESSMENT: {
                "sequence": [
                    {"type": "broadcast", "message": "Assess current portfolio risks"},
                    {"type": "analysis", "analysis_type": "risk", "message": "Detailed risk analysis"},
                    {"type": "consensus", "topic": "risk mitigation actions", "max_rounds": 3}
                ],
                "timeout_minutes": 15,
                "max_rounds": 6,
                "require_consensus": True
            }
        }
    
    def _get_conversation_sequence(self, conversation_type: ConversationType) -> List[Dict[str, Any]]:
        """Get conversation sequence for type"""
        pattern = self._conversation_patterns.get(conversation_type, {})
        return pattern.get("sequence", [])
    
    def _get_conversation_config(self, conversation_type: ConversationType) -> Dict[str, Any]:
        """Get conversation configuration for type"""
        pattern = self._conversation_patterns.get(conversation_type, {})
        return {
            "timeout_minutes": pattern.get("timeout_minutes", 30),
            "max_rounds": pattern.get("max_rounds", 10),
            "require_consensus": pattern.get("require_consensus", False)
        }
    
    def _get_manager_llm_config(self) -> Dict[str, Any]:
        """Get LLM config for chat manager"""
        from ..config.settings import settings
        
        return {
            "model": settings.autogen.model_name,
            "temperature": 0.1,
            "max_tokens": 1000,
            "timeout": settings.autogen.timeout,
            "api_key": settings.api_keys.get("openai_api_key")
        }
    
    def _build_conversation_outcome(
        self,
        flow: ConversationFlow,
        messages: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build conversation outcome from messages"""
        outcome = {
            "conversation_type": flow.conversation_type.value,
            "total_messages": len(messages),
            "participants": flow.participants,
            "summary": self._summarize_conversation(messages),
            "decisions": self._extract_decisions(messages),
            "action_items": self._extract_action_items(messages),
            "context": context
        }
        
        return outcome
    
    def _summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Create conversation summary"""
        if not messages:
            return "No messages in conversation"
        
        # Simple summarization - in production, could use LLM summarization
        key_points = []
        for msg in messages[-5:]:  # Last 5 messages
            if "response" in msg and isinstance(msg["response"], dict):
                if "summary" in msg["response"]:
                    key_points.append(msg["response"]["summary"])
        
        return "; ".join(key_points) if key_points else "Conversation completed"
    
    def _extract_decisions(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract decisions from conversation"""
        decisions = []
        
        for msg in messages:
            if "response" in msg and isinstance(msg["response"], dict):
                if "decision" in msg["response"]:
                    decisions.append({
                        "agent": msg.get("agent"),
                        "decision": msg["response"]["decision"],
                        "timestamp": msg.get("timestamp")
                    })
        
        return decisions
    
    def _extract_action_items(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract action items from conversation"""
        actions = []
        
        for msg in messages:
            if "response" in msg and isinstance(msg["response"], dict):
                if "actions" in msg["response"]:
                    agent_actions = msg["response"]["actions"]
                    if isinstance(agent_actions, list):
                        for action in agent_actions:
                            actions.append({
                                "agent": msg.get("agent"),
                                "action": action,
                                "timestamp": msg.get("timestamp")
                            })
        
        return actions
    
    async def _check_consensus(self, agents: List[BaseHedgeFundAgent], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Check for consensus among agents"""
        # Simple consensus checking - could be enhanced
        consensus_responses = []
        
        for agent in agents:
            try:
                response = agent.process_message({
                    "content": "Do you agree with the conversation outcome?",
                    "outcome": outcome,
                    "type": "consensus_check"
                })
                
                consensus_responses.append({
                    "agent": agent.name,
                    "response": response
                })
                
            except Exception as e:
                self.logger.error(f"Consensus check failed for {agent.name}: {e}")
        
        # Analyze consensus
        agreements = sum(1 for r in consensus_responses 
                        if isinstance(r.get("response"), dict) and 
                        r["response"].get("agreement", False))
        
        return {
            "total_agents": len(agents),
            "agreements": agreements,
            "consensus_ratio": agreements / len(agents) if agents else 0,
            "consensus_reached": agreements / len(agents) > 0.66 if agents else False,
            "responses": consensus_responses
        }
    
    def _check_round_consensus(self, responses: List[Dict[str, Any]]) -> bool:
        """Check if consensus reached in a round"""
        if len(responses) < 2:
            return False
        
        # Simple consensus check - could be enhanced with NLP
        agreements = 0
        for response in responses:
            if isinstance(response.get("response"), dict):
                if response["response"].get("consensus", False):
                    agreements += 1
        
        return agreements / len(responses) > 0.66
    
    def _handle_signal_event(self, agent: BaseHedgeFundAgent, event: str, data: Any) -> None:
        """Handle signal generated event"""
        self.logger.info(f"Signal generated by {agent.name}: {data}")
        
        # Could trigger risk assessment conversation
        # asyncio.create_task(self.start_conversation(
        #     ConversationType.RISK_ASSESSMENT,
        #     agent.name,
        #     {"signal": data}
        # ))
    
    def _handle_risk_alert(self, agent: BaseHedgeFundAgent, event: str, data: Any) -> None:
        """Handle risk alert event"""
        self.logger.warning(f"Risk alert from {agent.name}: {data}")
        
        # Could trigger emergency response conversation
    
    def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get status of a conversation"""
        if conversation_id in self.active_conversations:
            flow = self.active_conversations[conversation_id]
            return {
                "conversation_id": conversation_id,
                "status": "in_progress",
                "type": flow.conversation_type.value,
                "participants": flow.participants,
                "start_time": datetime.now(timezone.utc).isoformat()  # Approximate
            }
        
        # Check history
        for result in self.conversation_history:
            if result.conversation_id == conversation_id:
                return {
                    "conversation_id": conversation_id,
                    "status": result.status.value,
                    "outcome": result.outcome,
                    "duration_seconds": result.duration_seconds
                }
        
        return {"error": "Conversation not found"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "registered_agents": len(self.agents),
            "active_conversations": len(self.active_conversations),
            "total_conversations": len(self.conversation_history),
            "agent_status": {
                name: agent.get_status() for name, agent in self.agents.items()
            }
        }