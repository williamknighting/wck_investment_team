"""
Base Agent Class for AI Hedge Fund System
All agents inherit from this class for consistent AutoGen integration and DuckDB access
"""
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import logging
import json
import uuid

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError:
    # Fallback for development without autogen installed
    class AssistantAgent:
        def __init__(self, *args, **kwargs):
            pass
    
    class OpenAIChatCompletionClient:
        def __init__(self, *args, **kwargs):
            pass

# Import DuckDB manager
import sys
sys.path.append('..')
try:
    from src.data.duckdb_manager import get_duckdb_manager
    from src.utils.logging_config import get_logger
except ImportError:
    # Fallback imports
    def get_duckdb_manager():
        from src.data.duckdb_manager import DuckDBDataManager
        return DuckDBDataManager()
    
    def get_logger(name):
        return logging.getLogger(name)


class BaseHedgeFundAgent(AssistantAgent, ABC):
    """
    Base class for all AI Hedge Fund agents
    Provides AutoGen integration, DuckDB access, and common functionality
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        system_message: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        **kwargs
    ):
        """
        Initialize base hedge fund agent
        
        Args:
            name: Agent name (unique identifier)
            description: Agent description/role
            system_message: System message for the agent
            model_name: OpenAI model to use
        """
        self.agent_name = name  # Store our own name to avoid AutoGen property conflict
        self.agent_description = description  # Store our own description to avoid AutoGen property conflict
        self.agent_id = str(uuid.uuid4())
        
        # Initialize logger
        self.logger = get_logger(f"agent.{name}")
        
        # Initialize DuckDB connection
        self.db = get_duckdb_manager()
        
        # Setup OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize AutoGen with OpenAI client
        model_client = OpenAIChatCompletionClient(
            model=model_name,
            api_key=api_key
        )
        
        # Default system message if none provided
        if not system_message:
            system_message = f"""You are {name}, {description}.
You have access to a DuckDB database with market data and can analyze stocks and make trading decisions.
You should communicate with other agents to gather information and provide your specialized analysis.
Always provide clear, actionable insights based on your expertise."""
        
        # Initialize AutoGen AssistantAgent
        super().__init__(
            name=name,
            model_client=model_client,
            description=description,
            system_message=system_message,
            **kwargs
        )
        
        # Agent-specific initialization
        self._initialize()
        
        self.logger.info(f"Agent {name} initialized successfully")
    
    @abstractmethod
    def _initialize(self) -> None:
        """Agent-specific initialization - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process incoming message and return response - must be implemented by subclasses"""
        pass
    
    def get_market_data(self, symbol: str, days: int = 252, interval: str = "1Day") -> Optional[Any]:
        """
        Get market data from DuckDB
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            days: Number of days of data to retrieve
            interval: Data interval ('1Day', '1Hour', etc.)
            
        Returns:
            DataFrame with market data or None if not found
        """
        try:
            from datetime import timedelta
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            data = self.db.get_market_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"Retrieved {len(data)} records for {symbol}")
                return data
            else:
                self.logger.warning(f"No market data found for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving market data for {symbol}: {e}")
            return None
    
    def add_to_watchlist(self, ticker: str, notes: str = "") -> bool:
        """
        Add a ticker to the watchlist
        
        Args:
            ticker: Stock ticker symbol
            notes: Optional notes about why it was added
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            self.db.conn.execute("""
                INSERT OR REPLACE INTO watchlist (ticker, date_added, active, added_by, notes)
                VALUES (?, NOW(), TRUE, ?, ?)
            """, [ticker, self.agent_name, notes])
            
            self.logger.info(f"Added {ticker} to watchlist")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding {ticker} to watchlist: {e}")
            return False
    
    def get_watchlist(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get current watchlist
        
        Args:
            active_only: Only return active tickers
            
        Returns:
            List of watchlist entries
        """
        try:
            query = "SELECT * FROM watchlist"
            if active_only:
                query += " WHERE active = TRUE"
            query += " ORDER BY date_added DESC"
            
            result = self.db.conn.execute(query).fetchall()
            
            watchlist = []
            if result:
                columns = [desc[0] for desc in self.db.conn.description]
                watchlist = [dict(zip(columns, row)) for row in result]
            
            return watchlist
            
        except Exception as e:
            self.logger.error(f"Error retrieving watchlist: {e}")
            return []
    
    def write_proposal(self, content: str, symbol: str, proposal_type: str = "trade") -> str:
        """
        Write a trade proposal to markdown file
        
        Args:
            content: Proposal content in markdown
            symbol: Stock symbol this proposal is for
            proposal_type: Type of proposal ('trade', 'analysis', etc.)
            
        Returns:
            Path to created file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.agent_name}_{symbol}_{proposal_type}_{timestamp}.md"
            filepath = Path("proposals") / filename
            
            # Add header to proposal
            header = f"""# {proposal_type.title()} Proposal - {symbol}

**Agent**: {self.agent_name}  
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Symbol**: {symbol}  
**Type**: {proposal_type}  

---

"""
            
            with open(filepath, 'w') as f:
                f.write(header + content)
            
            self.logger.info(f"Proposal written to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error writing proposal: {e}")
            return ""
    
    def log_conversation(self, conversation: List[Dict[str, Any]], topic: str = "general") -> str:
        """
        Log agent conversation to file
        
        Args:
            conversation: List of message dictionaries
            topic: Topic/symbol of conversation
            
        Returns:
            Path to log file
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{topic}_{timestamp}.json"
            filepath = Path("conversations") / filename
            
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "participants": list(set([msg.get("agent", "unknown") for msg in conversation])),
                "message_count": len(conversation),
                "conversation": conversation
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            self.logger.info(f"Conversation logged to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error logging conversation: {e}")
            return ""
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "name": self.agent_name,
            "description": self.agent_description,
            "agent_id": self.agent_id,
            "status": "active",
            "db_connected": self.db is not None,
            "initialized_at": getattr(self, '_initialized_at', None)
        }