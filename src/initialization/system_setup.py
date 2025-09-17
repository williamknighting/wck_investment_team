"""
System Setup - Initialize AI Hedge Fund System with Alpaca + DuckDB
"""
import os
from pathlib import Path

from ..services.market_data_service import initialize_market_data_service
from ..data.alpaca_market_provider import create_alpaca_provider
from ..data.duckdb_manager import get_duckdb_manager
from ..utils.logging_config import get_logger


def setup_ai_hedge_fund_system(
    alpaca_api_key: str = None,
    alpaca_secret: str = None,
    paper_trading: bool = True,
    db_path: str = "data_store/market_data.duckdb"
) -> dict:
    """
    Initialize complete AI hedge fund system with Alpaca data and DuckDB storage
    
    Args:
        alpaca_api_key: Alpaca API key (defaults to environment variable)
        alpaca_secret: Alpaca secret key (defaults to environment variable)  
        paper_trading: Use paper trading environment
        db_path: Path to DuckDB database
        
    Returns:
        Dictionary with initialized system components
    """
    logger = get_logger("system_setup")
    logger.info("Initializing AI Hedge Fund System...")
    
    # Get API credentials
    api_key = alpaca_api_key or os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
    secret_key = alpaca_secret or os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")
    
    try:
        # 1. Initialize DuckDB Manager
        logger.info("Initializing DuckDB data store...")
        duckdb_manager = get_duckdb_manager()
        logger.info(f"✅ DuckDB initialized: {db_path}")
        
        # 2. Create Alpaca Provider
        logger.info("Setting up Alpaca data provider...")
        alpaca_provider = create_alpaca_provider(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper_trading
        )
        logger.info(f"✅ Alpaca provider ready (paper: {paper_trading})")
        
        # 3. Initialize Market Data Service
        logger.info("Initializing market data service layer...")
        market_service = initialize_market_data_service(alpaca_provider)
        logger.info("✅ Market data service initialized")
        
        # 4. Test the system
        logger.info("Testing system integration...")
        test_result = test_system_integration()
        
        if test_result["success"]:
            logger.info("✅ System integration test passed")
        else:
            logger.warning(f"⚠️ System test issues: {test_result['message']}")
        
        # System summary
        summary = {
            "status": "ready",
            "data_provider": "alpaca",
            "data_store": "duckdb", 
            "database_path": db_path,
            "paper_trading": paper_trading,
            "supported_intervals": market_service.get_supported_intervals(),
            "test_result": test_result,
            "components": {
                "duckdb_manager": duckdb_manager,
                "alpaca_provider": alpaca_provider,
                "market_service": market_service
            }
        }
        
        logger.info("🚀 AI Hedge Fund System ready!")
        return summary
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


def test_system_integration() -> dict:
    """Test system integration with a simple data fetch"""
    logger = get_logger("system_test")
    
    try:
        from ..services.market_data_service import get_market_data_service
        
        # Test fetching some data
        market_service = get_market_data_service()
        
        # Try to fetch SPY data
        logger.info("Testing data fetch with SPY...")
        data = market_service.get_stock_data("SPY", period="5d", interval="1d")
        
        if data.empty:
            return {
                "success": False,
                "message": "No data returned for SPY test"
            }
        
        # Check DuckDB storage
        duckdb_manager = get_duckdb_manager()
        summary = duckdb_manager.get_data_summary()
        
        return {
            "success": True,
            "message": f"Successfully fetched {len(data)} records for SPY",
            "data_summary": {
                "total_records": summary.get("total_records", 0),
                "unique_symbols": summary.get("unique_symbols", 0)
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Integration test failed: {e}"
        }


def get_system_status() -> dict:
    """Get current system status and statistics"""
    try:
        from ..services.market_data_service import get_market_data_service
        
        duckdb_manager = get_duckdb_manager()
        market_service = get_market_data_service()
        
        # Get database summary
        db_summary = duckdb_manager.get_data_summary()
        
        # Get provider info
        provider_info = {
            "type": "alpaca",
            "supported_intervals": market_service.get_supported_intervals(),
            "supported_periods": market_service.get_supported_periods(),
            "market_open": market_service.is_market_open()
        }
        
        return {
            "status": "operational",
            "database": db_summary,
            "provider": provider_info,
            "timestamp": get_logger("status").info.__self__.name if hasattr(get_logger("status"), "info") else "unknown"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Convenience initialization function
def quick_setup() -> dict:
    """Quick setup with default parameters"""
    return setup_ai_hedge_fund_system(
        paper_trading=True,
        db_path="data_store/market_data.duckdb"
    )