#!/usr/bin/env python3
"""
Production System Initialization - REAL ALPACA DATA ONLY
NO MOCK DATA - DuckDB will only contain real market data
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.services.market_data_service import initialize_market_data_service
from src.data.alpaca_market_provider import create_alpaca_provider
from src.data.duckdb_manager import get_duckdb_manager
from src.utils.logging_config import get_logger


def initialize_production_system():
    """Initialize production system with REAL Alpaca API credentials"""
    print("🚀 Initializing Production AI Hedge Fund System")
    print("=" * 60)
    
    logger = get_logger("production_init")
    
    # Get API credentials from environment
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper_trading = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    
    print(f"\n🔑 API Configuration:")
    print(f"   API Key: {'✅ Set' if api_key and api_key != 'YOUR_ALPACA_API_KEY' else '❌ Missing/Default'}")
    print(f"   Secret Key: {'✅ Set' if secret_key and secret_key != 'YOUR_ALPACA_SECRET_KEY' else '❌ Missing/Default'}")
    print(f"   Paper Trading: {paper_trading}")
    
    if not api_key or api_key == "YOUR_ALPACA_API_KEY":
        print("\n❌ ERROR: ALPACA_API_KEY not set!")
        print("   Please set your real Alpaca API key:")
        print("   export ALPACA_API_KEY='your_real_api_key'")
        return False
    
    if not secret_key or secret_key == "YOUR_ALPACA_SECRET_KEY":  
        print("\n❌ ERROR: ALPACA_SECRET_KEY not set!")
        print("   Please set your real Alpaca secret key:")
        print("   export ALPACA_SECRET_KEY='your_real_secret_key'")
        return False
    
    try:
        # Initialize DuckDB (real data only)
        print(f"\n🗄️ Initializing DuckDB (real data only)...")
        duckdb_manager = get_duckdb_manager()
        print("   ✅ DuckDB ready for real market data")
        
        # Create Alpaca provider with REAL credentials
        print(f"\n📡 Creating Alpaca provider...")
        alpaca_provider = create_alpaca_provider(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper_trading
        )
        print(f"   ✅ Alpaca provider ready (paper: {paper_trading})")
        
        # Initialize market data service
        print(f"\n🔧 Initializing market data service...")
        market_service = initialize_market_data_service(alpaca_provider)
        print("   ✅ Market data service ready")
        
        # Test with a small data fetch
        print(f"\n🧪 Testing with real API call...")
        print("   Fetching SPY daily data (last 5 days)...")
        
        try:
            data = market_service.get_stock_data("SPY", period="5d", interval="1d")
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                print(f"   ✅ SUCCESS: Got {len(data)} records")
                print(f"   💰 SPY Latest Price: ${latest_price:.2f}")
                
                # Verify it's stored in DuckDB
                summary = duckdb_manager.get_data_summary()
                total_records = summary.get("total_records", 0)
                data_sources = summary.get("data_sources", [])
                
                print(f"   📊 DuckDB: {total_records} records from {data_sources}")
                
                if 'alpaca' in data_sources and 'mock' not in data_sources:
                    print("   ✅ Real Alpaca data stored in DuckDB!")
                else:
                    print("   ⚠️ Unexpected data sources in DuckDB")
                
            else:
                print("   ❌ No data returned - check API credentials")
                return False
                
        except Exception as e:
            print(f"   ❌ API test failed: {e}")
            print("   Check your Alpaca API credentials and connectivity")
            return False
        
        # Final system status
        print(f"\n" + "=" * 60)
        print("🎉 Production System Ready!")
        print(f"   • Data Provider: Alpaca Markets ({'Paper' if paper_trading else 'Live'})")
        print(f"   • Data Store: DuckDB (real data only)")
        print(f"   • API Safety: 1 req/sec, daily data only")
        print(f"   • Mock Data: ❌ Blocked from DuckDB")
        print(f"   • Real Data: ✅ Stored in DuckDB")
        
        print(f"\n🔧 Usage:")
        print(f"   # Get market data service")
        print(f"   from src.services.market_data_service import get_market_data_service")
        print(f"   service = get_market_data_service()")
        print(f"   data = service.get_stock_data('AAPL', period='1mo')")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        logger.error(f"Production initialization failed: {e}")
        return False


def main():
    """Main entry point"""
    success = initialize_production_system()
    
    if success:
        print(f"\n🚀 System ready for production trading!")
        exit(0)
    else:
        print(f"\n💥 System initialization failed!")
        print(f"   Fix the issues above and try again.")
        exit(1)


if __name__ == "__main__":
    main()