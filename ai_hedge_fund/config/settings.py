"""
Global configuration management for AI Hedge Fund System
"""
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import yaml
from pathlib import Path


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "ai_hedge_fund"
    user: str = "postgres"
    password: Optional[str] = None
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class AutoGenConfig:
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 60
    max_consecutive_auto_reply: int = 3
    human_input_mode: str = "NEVER"
    code_execution_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.code_execution_config is None:
            self.code_execution_config = {"use_docker": False}


@dataclass
class TradingConfig:
    default_position_size: float = 10000.0  # USD
    max_position_size: float = 50000.0      # USD
    max_portfolio_risk: float = 0.02        # 2% portfolio risk
    max_single_position_risk: float = 0.005 # 0.5% per position
    commission_rate: float = 0.001          # 0.1%
    slippage_rate: float = 0.0005           # 0.05%
    

@dataclass
class DataConfig:
    market_data_provider: str = "yahoo"
    fundamental_data_provider: str = "alpha_vantage"
    cache_duration_minutes: int = 5
    max_cache_size_mb: int = 100


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/ai_hedge_fund.log"
    max_file_size_mb: int = 10
    backup_count: int = 5


class Settings:
    def __init__(self):
        self.env = Environment(os.getenv("ENVIRONMENT", "development"))
        self.config_dir = Path(__file__).parent
        
        # Load configuration files
        self._load_configs()
        
        # Initialize configuration objects
        self.database = DatabaseConfig(**self._get_database_config())
        self.autogen = AutoGenConfig(**self._get_autogen_config())
        self.trading = TradingConfig(**self._get_trading_config())
        self.data = DataConfig(**self._get_data_config())
        self.logging = LoggingConfig(**self._get_logging_config())
        
        # API Keys (loaded from environment)
        self.api_keys = self._load_api_keys()
    
    def _load_configs(self):
        """Load YAML configuration files"""
        self.market_config = self._load_yaml("market_config.yaml")
        self.strategy_config = self._load_yaml("strategy_config.yaml")
        self.risk_limits = self._load_yaml("risk_limits.yaml")
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file"""
        file_path = self.config_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _get_database_config(self) -> Dict[str, Any]:
        """Get database configuration based on environment"""
        base_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "name": os.getenv("DB_NAME", "ai_hedge_fund"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD")
        }
        
        if self.env == Environment.PRODUCTION:
            base_config.update({
                "host": os.getenv("PROD_DB_HOST", base_config["host"]),
                "name": os.getenv("PROD_DB_NAME", "ai_hedge_fund_prod")
            })
        
        return base_config
    
    def _get_autogen_config(self) -> Dict[str, Any]:
        """Get AutoGen configuration"""
        return {
            "model_name": os.getenv("AUTOGEN_MODEL", "gpt-4-turbo-preview"),
            "temperature": float(os.getenv("AUTOGEN_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("AUTOGEN_MAX_TOKENS", "2000")),
            "timeout": int(os.getenv("AUTOGEN_TIMEOUT", "60")),
            "max_consecutive_auto_reply": int(os.getenv("AUTOGEN_MAX_AUTO_REPLY", "3"))
        }
    
    def _get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        config = {
            "default_position_size": float(os.getenv("DEFAULT_POSITION_SIZE", "10000")),
            "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "50000")),
            "max_portfolio_risk": float(os.getenv("MAX_PORTFOLIO_RISK", "0.02")),
            "max_single_position_risk": float(os.getenv("MAX_SINGLE_POSITION_RISK", "0.005")),
            "commission_rate": float(os.getenv("COMMISSION_RATE", "0.001")),
            "slippage_rate": float(os.getenv("SLIPPAGE_RATE", "0.0005"))
        }
        
        # Override with risk limits from YAML if available
        if self.risk_limits:
            config.update(self.risk_limits.get("trading", {}))
        
        return config
    
    def _get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return {
            "market_data_provider": os.getenv("MARKET_DATA_PROVIDER", "yahoo"),
            "fundamental_data_provider": os.getenv("FUNDAMENTAL_DATA_PROVIDER", "alpha_vantage"),
            "cache_duration_minutes": int(os.getenv("CACHE_DURATION_MINUTES", "5")),
            "max_cache_size_mb": int(os.getenv("MAX_CACHE_SIZE_MB", "100"))
        }
    
    def _get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        level = "DEBUG" if self.env == Environment.DEVELOPMENT else "INFO"
        return {
            "level": os.getenv("LOG_LEVEL", level),
            "file_path": os.getenv("LOG_FILE", "logs/ai_hedge_fund.log"),
            "max_file_size_mb": int(os.getenv("LOG_MAX_SIZE_MB", "10")),
            "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5"))
        }
    
    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        """Load API keys from environment variables"""
        return {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "polygon_api_key": os.getenv("POLYGON_API_KEY"),
            "yahoo_finance_api_key": os.getenv("YAHOO_FINANCE_API_KEY"),
            "broker_api_key": os.getenv("BROKER_API_KEY"),
            "broker_secret_key": os.getenv("BROKER_SECRET_KEY")
        }
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy"""
        return self.strategy_config.get(strategy_name, {})
    
    def get_market_config(self, market: str = "us_equities") -> Dict[str, Any]:
        """Get configuration for a specific market"""
        return self.market_config.get(market, {})
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.env == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.env == Environment.PRODUCTION


# Global settings instance
settings = Settings()