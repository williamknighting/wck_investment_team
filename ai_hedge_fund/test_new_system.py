#!/usr/bin/env python3
"""
Test script for new Alpaca + DuckDB architecture
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.initialization.system_setup import setup_ai_hedge_fund_system, get_system_status
from src.utils.logging_config import get_logger


def main():
    """Test the complete system"""
    print("🚀 Testing AI Hedge Fund System with Alpaca + DuckDB")
    print("=" * 60)
    
    # Initialize system
    print("\n📋 Step 1: System Initialization")
    system = setup_ai_hedge_fund_system(
        paper_trading=True,
        db_path="data_store/market_data.duckdb"
    )
    
    print(f"Status: {system['status']}")
    if system['status'] == 'failed':
        print(f"Error: {system['error']}")
        return
    
    print(f"✅ Data Provider: {system['data_provider']}")
    print(f"✅ Data Store: {system['data_store']}")
    print(f"✅ Paper Trading: {system['paper_trading']}")
    print(f"✅ Supported Intervals: {system['supported_intervals']}")
    
    # Test data fetching
    print("\n📊 Step 2: Test Data Fetching")
    try:
        market_service = system['components']['market_service']
        
        # Test multiple symbols
        test_symbols = ['SPY', 'AAPL', 'NVDA']
        
        for symbol in test_symbols:
            print(f"\n  Testing {symbol}...")
            data = market_service.get_stock_data(symbol, period='5d', interval='1d')
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                print(f"    ✅ {symbol}: {len(data)} records, latest price: ${latest_price:.2f}")
            else:
                print(f"    ❌ {symbol}: No data received")
        
    except Exception as e:
        print(f"    ❌ Data fetch error: {e}")
    
    # Test DuckDB storage
    print("\n🗄️ Step 3: Test DuckDB Storage")
    try:
        duckdb_manager = system['components']['duckdb_manager']
        summary = duckdb_manager.get_data_summary()
        
        print(f"  Total Records: {summary.get('total_records', 0)}")
        print(f"  Unique Symbols: {summary.get('unique_symbols', 0)}")
        print(f"  Data Sources: {summary.get('data_sources', [])}")
        
        if 'summary_by_symbol' in summary:
            print("  \n  Symbol Details:")
            for record in summary['summary_by_symbol'][:5]:  # Show first 5
                print(f"    {record['symbol']}: {record['record_count']} records ({record['data_source']})")
        
    except Exception as e:
        print(f"  ❌ DuckDB error: {e}")
    
    # Test Research Agent integration
    print("\n🔬 Step 4: Test Research Agent Integration")
    try:
        from src.agents.core.research_agent import ResearchAgent
        
        research_agent = ResearchAgent()
        
        # Test data fetch through agent
        test_data = research_agent.fetch_stock_data_for_analysis("AAPL", period="5d")
        
        if test_data:
            print(f"  ✅ Research Agent: Fetched data for AAPL")
            print(f"    Current Price: ${test_data.get('current_price', 'N/A')}")
            print(f"    Trading Days: {test_data.get('total_trading_days', 'N/A')}")
        else:
            print("  ❌ Research Agent: No data received")
            
    except Exception as e:
        print(f"  ❌ Research Agent error: {e}")
    
    # Final system status
    print("\n📈 Step 5: Final System Status")
    status = get_system_status()
    print(f"  System Status: {status['status']}")
    
    if status['status'] == 'operational':
        print(f"  Market Open: {status['provider']['market_open']}")
        print(f"  Database Records: {status['database'].get('total_records', 0)}")
    
    print("\n" + "=" * 60)
    print("🎉 System Test Complete!")


if __name__ == "__main__":
    main()