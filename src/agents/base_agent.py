"""
Base agent class for AI Hedge Fund System
Abstract base class extending AutoGen's AssistantAgent with hedge fund specific functionality
"""
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json
import uuid

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError:
    # Fallback for development without autogen installed
    class AssistantAgent:
        pass
    class OpenAIChatCompletionClient:
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


class BaseHedgeFundAgent(AssistantAgent):
    """
    Base class for all hedge fund agents
    Extends AutoGen's AssistantAgent with hedge fund specific functionality
    """
    
    def __init__(
        self,
        name: str,
        capabilities: List[AgentCapability],
        system_message: str = None,
        model_name: str = "gpt-4o-mini",
        description: str = None,
        **kwargs
    ):
        """
        Initialize base hedge fund agent
        
        Args:
            name: Agent name
            capabilities: List of agent capabilities
            system_message: System prompt for the agent
            model_name: OpenAI model to use
            description: Agent description
            **kwargs: Additional arguments
        """
        
        # Default system message if not provided
        if system_message is None:
            system_message = f"You are a {name} for an AI hedge fund system. Follow instructions precisely."
        
        # Default description if not provided
        if description is None:
            description = f"AI hedge fund agent: {name}"
        
        # Create model client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        model_client = OpenAIChatCompletionClient(
            model=model_name,
            api_key=api_key
        )
        
        # Initialize AutoGen AssistantAgent with new API
        super().__init__(
            name=name,
            model_client=model_client,
            description=description,
            system_message=system_message,
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

    # For compatibility with our existing code that expects certain methods
    def analyze_symbol(self, symbol: str, technical_metrics: Any) -> Dict[str, Any]:
        """
        Compatibility method for symbol analysis
        Delegates to process_message with structured data
        """
        message = {
            "type": "analyze_symbol",
            "symbol": symbol,
            "technical_metrics": technical_metrics
        }
        return self.process_message(message)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information for compatibility
        """
        return {
            "name": self.name,
            "persona": getattr(self, 'persona', 'AI Trading Agent'),
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities]
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status='{self.status.value}')"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"capabilities={[c.value for c in self.capabilities]}, "
                f"status='{self.status.value}')")