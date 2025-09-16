"""
Base agent class for AI Hedge Fund System
Abstract base class extending AutoGen's AssistantAgent with hedge fund specific functionality
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json
import uuid

try:
    from autogen import AssistantAgent, ConversableAgent
    from autogen.agentchat.agent import Agent
except ImportError:
    # Fallback for development without autogen installed
    class AssistantAgent:
        pass
    class ConversableAgent:
        pass
    class Agent:
        pass

try:
    # Try relative imports first (works in package context)
    from ..models.trade import TradeSignal, Order
    from ..models.market_state import MarketState
    from ..models.portfolio import Portfolio
    from ..utils.logging_config import get_logger
except ImportError:
    # Fall back to absolute imports (works in script context)
    from models.trade import TradeSignal, Order
    from models.market_state import MarketState
    from models.portfolio import Portfolio
    from utils.logging_config import get_logger


class AgentStatus(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    DISABLED = "disabled"


class AgentCapability(str, Enum):
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    TRADE_EXECUTION = "trade_execution"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    RESEARCH = "research"
    REPORTING = "reporting"


@dataclass
class AgentMetrics:
    """Performance metrics for an agent"""
    total_messages: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_response_time_ms: float = 0.0
    last_activity: Optional[datetime] = None
    uptime_start: datetime = None
    
    def __post_init__(self):
        if self.uptime_start is None:
            self.uptime_start = datetime.now(timezone.utc)
    
    @property
    def success_rate(self) -> float:
        total_ops = self.successful_operations + self.failed_operations
        return self.successful_operations / total_ops if total_ops > 0 else 0.0
    
    @property
    def uptime_hours(self) -> float:
        if self.uptime_start:
            return (datetime.now(timezone.utc) - self.uptime_start).total_seconds() / 3600
        return 0.0


class BaseHedgeFundAgent(AssistantAgent, ABC):
    """
    Abstract base class for all hedge fund agents
    Extends AutoGen's AssistantAgent with hedge fund specific functionality
    """
    
    def __init__(
        self,
        name: str,
        system_message: str,
        capabilities: List[AgentCapability],
        llm_config: Dict[str, Any] = None,
        code_execution_config: Dict[str, Any] = None,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 3,
        **kwargs
    ):
        """
        Initialize base hedge fund agent
        
        Args:
            name: Agent name
            system_message: System prompt for the agent
            capabilities: List of agent capabilities
            llm_config: AutoGen LLM configuration
            code_execution_config: Code execution configuration
            human_input_mode: Human input mode for AutoGen
            max_consecutive_auto_reply: Max auto replies
            **kwargs: Additional arguments
        """
        # Initialize AutoGen AssistantAgent
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config or self._get_default_llm_config(),
            code_execution_config=code_execution_config or {"use_docker": False},
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            **kwargs
        )
        
        # Agent metadata
        self.agent_id = str(uuid.uuid4())
        self.capabilities = capabilities
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        
        # Logging
        self.logger = get_logger(f"agent.{name}")
        
        # State management
        self._state: Dict[str, Any] = {}
        self._context_data: Dict[str, Any] = {}
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize agent-specific components
        try:
            self._initialize()
            self.status = AgentStatus.READY
            self.logger.info(f"Agent {self.name} initialized successfully")
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Failed to initialize agent {self.name}: {e}")
            raise
    
    def _get_default_llm_config(self) -> Dict[str, Any]:
        """Get default LLM configuration"""
        try:
            from ..config.settings import settings
        except ImportError:
            try:
                from config.settings import settings
            except ImportError:
                # Fallback config if settings not available
                settings = type('Settings', (), {
                    'OPENAI_API_KEY': 'test-key',
                    'MODEL_NAME': 'gpt-4'
                })()
        
        return {
            "model": getattr(settings, 'MODEL_NAME', 'gpt-4'),
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 30,
            "api_key": getattr(settings, 'OPENAI_API_KEY', 'test-key')
        }
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize agent-specific components"""
        pass
    
    @abstractmethod
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """
        Process incoming message and return response
        
        Args:
            message: Message data
            sender: Sender agent name
            
        Returns:
            Response message
        """
        pass
    
    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Union[str, Dict, None]:
        """
        AutoGen generate_reply override with hedge fund specific logic
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            self.status = AgentStatus.PROCESSING
            self.metrics.total_messages += 1
            
            # Extract message content
            if messages:
                last_message = messages[-1] if messages else {}
                content = last_message.get("content", "")
                
                # Parse structured message if JSON
                try:
                    message_data = json.loads(content) if content.startswith("{") else {"content": content}
                except json.JSONDecodeError:
                    message_data = {"content": content}
                
                # Process with agent-specific logic
                response = self.process_message(message_data, sender.name if sender else None)
                
                # Format response for AutoGen
                if isinstance(response, dict):
                    reply = json.dumps(response, default=str, indent=2)
                else:
                    reply = str(response)
                
                self.metrics.successful_operations += 1
                self.status = AgentStatus.READY
                
                return reply
            
            # Fall back to parent implementation
            return super().generate_reply(messages, sender, config)
            
        except Exception as e:
            self.metrics.failed_operations += 1
            self.status = AgentStatus.ERROR
            self.logger.error(f"Error generating reply: {e}")
            
            return f"Error processing message: {str(e)}"
        
        finally:
            # Update metrics
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000
            
            # Update rolling average
            total_ops = self.metrics.total_messages
            if total_ops > 1:
                self.metrics.avg_response_time_ms = (
                    (self.metrics.avg_response_time_ms * (total_ops - 1) + response_time) / total_ops
                )
            else:
                self.metrics.avg_response_time_ms = response_time
            
            self.metrics.last_activity = end_time
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            return self._validate_input(data)
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False
    
    @abstractmethod
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Agent-specific input validation"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status and metrics
        
        Returns:
            Status dictionary
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "metrics": {
                "total_messages": self.metrics.total_messages,
                "success_rate": self.metrics.success_rate,
                "avg_response_time_ms": self.metrics.avg_response_time_ms,
                "uptime_hours": self.metrics.uptime_hours,
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            },
            "state_keys": list(self._state.keys()),
            "context_keys": list(self._context_data.keys())
        }
    
    def update_context(self, key: str, value: Any) -> None:
        """Update agent context data"""
        self._context_data[key] = value
        self.logger.debug(f"Updated context: {key}")
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context data"""
        return self._context_data.get(key, default)
    
    def clear_context(self) -> None:
        """Clear all context data"""
        self._context_data.clear()
        self.logger.debug("Context cleared")
    
    def set_state(self, key: str, value: Any) -> None:
        """Set agent state"""
        self._state[key] = value
        self.logger.debug(f"State updated: {key}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get agent state"""
        return self._state.get(key, default)
    
    def add_event_handler(self, event: str, handler: Callable) -> None:
        """Add event handler"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def emit_event(self, event: str, data: Any = None) -> None:
        """Emit event to handlers"""
        if event in self._event_handlers:
            for handler in self._event_handlers[event]:
                try:
                    handler(self, event, data)
                except Exception as e:
                    self.logger.error(f"Event handler error for {event}: {e}")
    
    def reset(self) -> None:
        """Reset agent to initial state"""
        self._state.clear()
        self._context_data.clear()
        self.status = AgentStatus.READY
        self.logger.info(f"Agent {self.name} reset")
    
    def shutdown(self) -> None:
        """Shutdown agent"""
        self.status = AgentStatus.DISABLED
        self.emit_event("shutdown")
        self.logger.info(f"Agent {self.name} shutdown")
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability"""
        return capability in self.capabilities
    
    def create_signal(
        self,
        symbol: str,
        side: str,
        signal_strength: float,
        reasoning: str,
        **kwargs
    ) -> TradeSignal:
        """
        Create a trade signal
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            signal_strength: Signal confidence (0-1)
            reasoning: Explanation for the signal
            **kwargs: Additional signal parameters
            
        Returns:
            TradeSignal object
        """
        from ..models.trade import OrderSide
        
        return TradeSignal(
            symbol=symbol,
            side=OrderSide(side.lower()),
            signal_strength=signal_strength,
            strategy=kwargs.get("strategy", "unknown"),
            agent_name=self.name,
            reasoning=reasoning,
            **{k: v for k, v in kwargs.items() if k != "strategy"}
        )
    
    def log_activity(self, activity: str, level: str = "info", **kwargs) -> None:
        """Log agent activity"""
        log_data = {
            "agent": self.name,
            "activity": activity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        
        if level == "debug":
            self.logger.debug(json.dumps(log_data))
        elif level == "warning":
            self.logger.warning(json.dumps(log_data))
        elif level == "error":
            self.logger.error(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status='{self.status.value}')"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"capabilities={[c.value for c in self.capabilities]}, "
                f"status='{self.status.value}')")