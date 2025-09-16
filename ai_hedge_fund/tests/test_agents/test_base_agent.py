"""
Tests for base agent functionality
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.agents.base_agent import BaseHedgeFundAgent, AgentCapability, AgentStatus
from src.models.trade import TradeSignal, OrderSide


class TestAgent(BaseHedgeFundAgent):
    """Test implementation of base agent"""
    
    def _initialize(self):
        self.initialized = True
    
    def process_message(self, message, sender=None):
        return {
            "type": "test_response",
            "message": "Test response",
            "sender": sender,
            "agent": self.name
        }
    
    def _validate_input(self, data):
        return isinstance(data, dict)


class TestBaseHedgeFundAgent:
    """Test suite for BaseHedgeFundAgent"""
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent instance"""
        agent = TestAgent(
            name="test_agent",
            system_message="Test agent for unit testing",
            capabilities=[AgentCapability.MARKET_ANALYSIS, AgentCapability.RESEARCH]
        )
        return agent
    
    def test_agent_initialization(self, test_agent):
        """Test agent initialization"""
        assert test_agent.name == "test_agent"
        assert test_agent.status == AgentStatus.READY
        assert AgentCapability.MARKET_ANALYSIS in test_agent.capabilities
        assert AgentCapability.RESEARCH in test_agent.capabilities
        assert hasattr(test_agent, 'initialized')
        assert test_agent.initialized == True
    
    def test_agent_has_capability(self, test_agent):
        """Test capability checking"""
        assert test_agent.has_capability(AgentCapability.MARKET_ANALYSIS)
        assert test_agent.has_capability(AgentCapability.RESEARCH)
        assert not test_agent.has_capability(AgentCapability.TRADE_EXECUTION)
    
    def test_process_message(self, test_agent):
        """Test message processing"""
        message = {"type": "test", "content": "test message"}
        response = test_agent.process_message(message, "test_sender")
        
        assert response["type"] == "test_response"
        assert response["sender"] == "test_sender"
        assert response["agent"] == "test_agent"
    
    def test_validate_input(self, test_agent):
        """Test input validation"""
        # Valid input
        assert test_agent.validate_input({"valid": "data"})
        
        # Invalid input
        assert not test_agent.validate_input("invalid")
        assert not test_agent.validate_input(None)
    
    def test_context_management(self, test_agent):
        """Test context data management"""
        # Set context
        test_agent.update_context("test_key", "test_value")
        assert test_agent.get_context("test_key") == "test_value"
        
        # Get with default
        assert test_agent.get_context("missing_key", "default") == "default"
        
        # Clear context
        test_agent.clear_context()
        assert test_agent.get_context("test_key") is None
    
    def test_state_management(self, test_agent):
        """Test state management"""
        # Set state
        test_agent.set_state("active_trades", [])
        assert test_agent.get_state("active_trades") == []
        
        # Get with default
        assert test_agent.get_state("missing_state", {}) == {}
    
    def test_event_handling(self, test_agent):
        """Test event handling system"""
        events_received = []
        
        def test_handler(agent, event, data):
            events_received.append((agent.name, event, data))
        
        # Add handler
        test_agent.add_event_handler("test_event", test_handler)
        
        # Emit event
        test_agent.emit_event("test_event", {"test": "data"})
        
        assert len(events_received) == 1
        assert events_received[0][0] == "test_agent"
        assert events_received[0][1] == "test_event"
        assert events_received[0][2] == {"test": "data"}
    
    def test_signal_creation(self, test_agent):
        """Test trade signal creation"""
        signal = test_agent.create_signal(
            symbol="AAPL",
            side="buy",
            signal_strength=0.8,
            reasoning="Test signal",
            strategy="test_strategy"
        )
        
        assert isinstance(signal, TradeSignal)
        assert signal.symbol == "AAPL"
        assert signal.side == OrderSide.BUY
        assert signal.signal_strength == 0.8
        assert signal.reasoning == "Test signal"
        assert signal.strategy == "test_strategy"
        assert signal.agent_name == "test_agent"
    
    def test_activity_logging(self, test_agent):
        """Test activity logging"""
        with patch.object(test_agent.logger, 'info') as mock_log:
            test_agent.log_activity("test_activity", symbol="AAPL", price=150.0)
            
            # Verify logging was called
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "test_activity" in call_args
    
    def test_get_status(self, test_agent):
        """Test status reporting"""
        status = test_agent.get_status()
        
        assert "agent_id" in status
        assert status["name"] == "test_agent"
        assert status["status"] == AgentStatus.READY.value
        assert AgentCapability.MARKET_ANALYSIS.value in status["capabilities"]
        assert "metrics" in status
    
    def test_metrics_tracking(self, test_agent):
        """Test metrics tracking"""
        initial_messages = test_agent.metrics.total_messages
        
        # Process message to increment metrics
        test_agent.process_message({"test": "message"})
        
        assert test_agent.metrics.total_messages == initial_messages + 1
        assert test_agent.metrics.last_activity is not None
    
    def test_reset_functionality(self, test_agent):
        """Test agent reset"""
        # Set some state and context
        test_agent.set_state("test", "value")
        test_agent.update_context("test", "value")
        
        # Reset
        test_agent.reset()
        
        assert test_agent.get_state("test") is None
        assert test_agent.get_context("test") is None
        assert test_agent.status == AgentStatus.READY
    
    def test_shutdown(self, test_agent):
        """Test agent shutdown"""
        # Mock event handler to verify shutdown event
        shutdown_called = []
        
        def shutdown_handler(agent, event, data):
            shutdown_called.append(True)
        
        test_agent.add_event_handler("shutdown", shutdown_handler)
        
        # Shutdown
        test_agent.shutdown()
        
        assert test_agent.status == AgentStatus.DISABLED
        assert len(shutdown_called) == 1
    
    @pytest.mark.asyncio
    async def test_generate_reply_integration(self, test_agent):
        """Test AutoGen generate_reply integration"""
        # Mock AutoGen message format
        messages = [
            {"content": '{"type": "test", "data": "test_data"}'}
        ]
        
        # Mock sender
        sender = Mock()
        sender.name = "test_sender"
        
        # Generate reply
        reply = test_agent.generate_reply(messages, sender)
        
        assert reply is not None
        assert isinstance(reply, str)
        
        # Should be JSON format
        import json
        response_data = json.loads(reply)
        assert response_data["type"] == "test_response"
    
    def test_error_handling(self, test_agent):
        """Test error handling in message processing"""
        # Create agent that raises exception
        class ErrorAgent(TestAgent):
            def process_message(self, message, sender=None):
                raise ValueError("Test error")
        
        error_agent = ErrorAgent(
            name="error_agent",
            system_message="Error agent",
            capabilities=[AgentCapability.MARKET_ANALYSIS]
        )
        
        # Mock messages for AutoGen
        messages = [{"content": "test"}]
        sender = Mock()
        sender.name = "test_sender"
        
        reply = error_agent.generate_reply(messages, sender)
        
        # Should return error message
        assert "Error processing message" in reply
        assert error_agent.metrics.failed_operations > 0


class TestAgentMetrics:
    """Test agent metrics functionality"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        from src.agents.base_agent import AgentMetrics
        
        metrics = AgentMetrics()
        assert metrics.total_messages == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0
        assert metrics.success_rate == 0.0
        assert metrics.uptime_start is not None
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        from src.agents.base_agent import AgentMetrics
        
        metrics = AgentMetrics()
        metrics.successful_operations = 8
        metrics.failed_operations = 2
        
        assert metrics.success_rate == 0.8
    
    def test_uptime_calculation(self):
        """Test uptime calculation"""
        from src.agents.base_agent import AgentMetrics
        import time
        
        metrics = AgentMetrics()
        time.sleep(0.1)  # Wait a bit
        
        assert metrics.uptime_hours > 0