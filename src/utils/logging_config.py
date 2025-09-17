"""
Logging configuration for AI Hedge Fund System
Structured logging with JSON format and multiple outputs
"""
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import json
import os


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from LoggerAdapter or explicit extra parameters
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        # Add process/thread info for debugging
        if record.process:
            log_entry["process_id"] = record.process
        if record.thread:
            log_entry["thread_id"] = record.thread
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class HedgeFundLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding hedge fund specific context"""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: Any, kwargs: Dict[str, Any]) -> tuple:
        """Process log message with extra context"""
        if self.extra:
            # Merge extra fields
            if "extra" not in kwargs:
                kwargs["extra"] = {}
            kwargs["extra"].update(self.extra)
        
        # Add extra fields to the record
        if "extra" in kwargs:
            record_extra = kwargs.pop("extra")
            kwargs["extra"] = {"extra_fields": record_extra}
        
        return msg, kwargs


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    include_console: bool = True,
    json_format: bool = True
) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        max_file_size_mb: Maximum file size in MB before rotation
        backup_count: Number of backup files to keep
        include_console: Include console output
        json_format: Use JSON formatting
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "ai_hedge_fund.log",
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Separate file for errors
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "errors.log",
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Trading activity log
    trading_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "trading.log",
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count * 2,  # Keep more trading history
        encoding="utf-8"
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(formatter)
    
    # Create trading logger
    trading_logger = logging.getLogger("trading")
    trading_logger.addHandler(trading_handler)
    trading_logger.propagate = False  # Don't propagate to root
    
    # Performance metrics log
    metrics_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / "metrics.log",
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8"
    )
    metrics_handler.setLevel(logging.INFO)
    metrics_handler.setFormatter(formatter)
    
    # Create metrics logger
    metrics_logger = logging.getLogger("metrics")
    metrics_logger.addHandler(metrics_handler)
    metrics_logger.propagate = False
    
    # Console handler
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Use simpler format for console
        if json_format and os.getenv("ENVIRONMENT") == "development":
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            console_formatter = formatter
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger("autogen").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Initialize from settings if available
    try:
        from ..config.settings import settings
        
        # Override with settings if available
        if hasattr(settings, 'logging'):
            root_logger.setLevel(getattr(logging, settings.logging.level.upper()))
            for handler in root_logger.handlers:
                handler.setLevel(getattr(logging, settings.logging.level.upper()))
    
    except ImportError:
        pass  # Settings not available during initial setup


def get_logger(name: str, extra_context: Dict[str, Any] = None) -> HedgeFundLoggerAdapter:
    """
    Get a logger with hedge fund specific context
    
    Args:
        name: Logger name
        extra_context: Additional context to include in logs
        
    Returns:
        Configured logger adapter
    """
    logger = logging.getLogger(name)
    return HedgeFundLoggerAdapter(logger, extra_context)


def get_trading_logger(strategy: str = None, symbol: str = None) -> HedgeFundLoggerAdapter:
    """Get specialized trading logger"""
    context = {}
    if strategy:
        context["strategy"] = strategy
    if symbol:
        context["symbol"] = symbol
    
    return get_logger("trading", context)


def get_metrics_logger() -> HedgeFundLoggerAdapter:
    """Get specialized metrics logger"""
    return get_logger("metrics")


def log_trade_execution(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    strategy: str,
    agent: str,
    **kwargs
) -> None:
    """
    Log trade execution with structured data
    
    Args:
        symbol: Trading symbol
        side: Buy/sell
        quantity: Trade quantity
        price: Execution price
        strategy: Trading strategy
        agent: Executing agent
        **kwargs: Additional trade details
    """
    logger = get_trading_logger(strategy, symbol)
    
    trade_data = {
        "event_type": "trade_execution",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "strategy": strategy,
        "agent": agent,
        **kwargs
    }
    
    logger.info("Trade executed", extra=trade_data)


def log_signal_generation(
    symbol: str,
    signal_strength: float,
    strategy: str,
    agent: str,
    reasoning: str,
    **kwargs
) -> None:
    """
    Log signal generation with structured data
    
    Args:
        symbol: Trading symbol
        signal_strength: Signal confidence (0-1)
        strategy: Trading strategy
        agent: Generating agent
        reasoning: Signal reasoning
        **kwargs: Additional signal details
    """
    logger = get_trading_logger(strategy, symbol)
    
    signal_data = {
        "event_type": "signal_generation",
        "symbol": symbol,
        "signal_strength": signal_strength,
        "strategy": strategy,
        "agent": agent,
        "reasoning": reasoning,
        **kwargs
    }
    
    logger.info("Signal generated", extra=signal_data)


def log_risk_alert(
    alert_type: str,
    severity: str,
    description: str,
    affected_positions: list = None,
    metrics: dict = None,
    **kwargs
) -> None:
    """
    Log risk alert with structured data
    
    Args:
        alert_type: Type of risk alert
        severity: Alert severity (low/medium/high/critical)
        description: Alert description
        affected_positions: List of affected positions
        metrics: Risk metrics
        **kwargs: Additional alert details
    """
    logger = get_logger("risk")
    
    alert_data = {
        "event_type": "risk_alert",
        "alert_type": alert_type,
        "severity": severity,
        "description": description,
        "affected_positions": affected_positions or [],
        "metrics": metrics or {},
        **kwargs
    }
    
    # Use appropriate log level based on severity
    if severity == "critical":
        logger.critical("Critical risk alert", extra=alert_data)
    elif severity == "high":
        logger.error("High risk alert", extra=alert_data)
    elif severity == "medium":
        logger.warning("Medium risk alert", extra=alert_data)
    else:
        logger.info("Low risk alert", extra=alert_data)


def log_performance_metrics(
    period: str,
    total_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    win_rate: float,
    **kwargs
) -> None:
    """
    Log performance metrics
    
    Args:
        period: Performance period
        total_return: Total return percentage
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown percentage
        win_rate: Win rate percentage
        **kwargs: Additional metrics
    """
    logger = get_metrics_logger()
    
    metrics_data = {
        "event_type": "performance_metrics",
        "period": period,
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        **kwargs
    }
    
    logger.info("Performance metrics", extra=metrics_data)


def log_system_health(
    component: str,
    status: str,
    metrics: dict = None,
    **kwargs
) -> None:
    """
    Log system health metrics
    
    Args:
        component: System component name
        status: Component status
        metrics: Health metrics
        **kwargs: Additional health data
    """
    logger = get_logger("system")
    
    health_data = {
        "event_type": "system_health",
        "component": component,
        "status": status,
        "metrics": metrics or {},
        **kwargs
    }
    
    if status == "error":
        logger.error("System health issue", extra=health_data)
    elif status == "warning":
        logger.warning("System health warning", extra=health_data)
    else:
        logger.info("System health check", extra=health_data)


# Initialize logging on module import
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs"),
    include_console=os.getenv("ENVIRONMENT", "development") == "development"
)