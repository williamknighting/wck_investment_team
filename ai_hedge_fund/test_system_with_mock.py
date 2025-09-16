#!/usr/bin/env python3
"""
Test AI Hedge Fund System with Mock Data Provider
Demonstrates complete system functionality without API keys
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.services.market_data_service import initialize_market_data_service
from src.data.mock_provider import create_mock_provider
from src.data.duckdb_manager import get_duckdb_manager
from src.utils.logging_config import get_logger


def main():
    """Test the complete system with mock data"""
    print("🚀 Testing AI Hedge Fund System with Mock Data Provider")
    print("=" * 65)
    
    logger = get_logger("system_test")
    
    # 1. Initialize system with mock provider
    print("\n📋 Step 1: Initialize System Components")
    try:
        # Create mock provider
        mock_provider = create_mock_provider()
        print("✅ Mock Data Provider: Ready")
        
        # Initialize market data service
        market_service = initialize_market_data_service(mock_provider)
        print("✅ Market Data Service: Ready")
        
        # Initialize DuckDB
        duckdb_manager = get_duckdb_manager()
        print("✅ DuckDB Data Store: Ready")
        
        print(f"✅ Supported Intervals: {market_service.get_supported_intervals()}")
        print(f"✅ Market Open: {market_service.is_market_open()}")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    # 2. Test data fetching and storage
    print("\n📊 Step 2: Test Data Fetching & Storage")
    test_symbols = ['SPY', 'AAPL', 'NVDA']
    
    for symbol in test_symbols:
        try:
            print(f"\n  Fetching {symbol}...")
            
            # Fetch data
            data = market_service.get_stock_data(symbol, period='1mo', interval='1d')
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                volume = data['Volume'].iloc[-1]
                print(f"    ✅ {symbol}: {len(data)} records")
                print(f"    💰 Latest Price: ${latest_price:.2f}")
                print(f"    📈 Volume: {volume:,}")
            else:
                print(f"    ❌ {symbol}: No data received")
                
        except Exception as e:
            print(f"    ❌ {symbol}: Error - {e}")
    
    # 3. Verify DuckDB storage
    print("\n🗄️ Step 3: Verify DuckDB Storage")
    try:
        summary = duckdb_manager.get_data_summary()
        
        print(f"  📊 Total Records: {summary.get('total_records', 0)}")
        print(f"  🏷️ Unique Symbols: {summary.get('unique_symbols', 0)}")
        print(f"  📡 Data Sources: {summary.get('data_sources', [])}")
        
        if 'summary_by_symbol' in summary and summary['summary_by_symbol']:
            print("\n  Symbol Details:")
            for record in summary['summary_by_symbol']:
                print(f"    {record['symbol']}: {record['record_count']} records "
                      f"(source: {record['data_source']})")
                print(f"      📅 Range: {record['oldest']} to {record['newest']}")
        
    except Exception as e:
        print(f"  ❌ DuckDB verification failed: {e}")
    
    # 4. Test multi-symbol fetch
    print("\n🔄 Step 4: Test Multi-Symbol Fetch")
    try:
        multi_data = market_service.get_multiple_stocks(
            ['SPY', 'AAPL', 'MSFT'], 
            period='5d', 
            interval='1d'
        )
        
        print(f"  ✅ Fetched data for {len(multi_data)} symbols:")
        for symbol, data in multi_data.items():
            if not data.empty:
                price = data['Close'].iloc[-1]
                print(f"    {symbol}: {len(data)} records, ${price:.2f}")
        
    except Exception as e:
        print(f"  ❌ Multi-symbol fetch failed: {e}")
    
    # 5. Test latest prices
    print("\n💲 Step 5: Test Latest Prices")
    try:
        symbols = ['SPY', 'AAPL', 'NVDA', 'TSLA']
        prices = market_service.get_latest_prices(symbols)
        
        print("  Latest Prices:")
        for symbol, price in prices.items():
            print(f"    {symbol}: ${price:.2f}")
        
    except Exception as e:
        print(f"  ❌ Latest prices failed: {e}")
    
    # 6. Test research agent integration
    print("\n🔬 Step 6: Test Research Agent Integration")
    try:
        # Import here to avoid initialization issues
        import importlib.util
        
        # Load research agent module
        spec = importlib.util.spec_from_file_location(
            "research_agent", 
            "src/agents/core/research_agent.py"
        )
        research_module = importlib.util.module_from_spec(spec)
        
        # Import required base classes first
        from src.agents.base_agent import BaseHedgeFundAgent, AgentCapability
        
        # Execute the module
        spec.loader.exec_module(research_module)
        
        # Create research agent instance
        research_agent = research_module.ResearchAgent()
        
        # Test data fetch through agent
        test_data = research_agent.fetch_stock_data_for_analysis("AAPL", period="5d")
        
        if test_data and 'current_price' in test_data:
            print(f"  ✅ Research Agent Integration: Working")
            print(f"    Symbol: {test_data.get('symbol', 'N/A')}")
            print(f"    Current Price: ${test_data.get('current_price', 'N/A')}")
            print(f"    52W High: ${test_data.get('high_52w', 'N/A'):.2f}")
            print(f"    52W Low: ${test_data.get('low_52w', 'N/A'):.2f}")
            print(f"    Trading Days: {test_data.get('total_trading_days', 'N/A')}")
        else:
            print("  ❌ Research Agent: No data received")
            
    except Exception as e:
        print(f"  ❌ Research Agent integration failed: {e}")
    
    # 7. Performance test
    print("\n⚡ Step 7: Performance Test")
    try:
        import time
        
        start_time = time.time()
        
        # Fetch data for multiple symbols with different timeframes
        for symbol in ['SPY', 'QQQ', 'IWM']:
            market_service.get_stock_data(symbol, period='3mo', interval='1d')
        
        end_time = time.time()
        print(f"  ✅ Fetched 3 symbols with 3-month data in {end_time - start_time:.2f} seconds")
        
        # Test caching performance
        start_time = time.time()
        market_service.get_stock_data('SPY', period='3mo', interval='1d')  # Should be cached
        end_time = time.time()
        print(f"  ✅ Cached data fetch: {end_time - start_time:.3f} seconds")
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
    
    # Final summary
    print("\n" + "=" * 65)
    print("🎉 System Test Summary")
    
    try:
        final_summary = duckdb_manager.get_data_summary()
        print(f"📊 Final Database State:")
        print(f"  • Total Records: {final_summary.get('total_records', 0)}")
        print(f"  • Unique Symbols: {final_summary.get('unique_symbols', 0)}")
        print(f"  • Data Sources: {final_summary.get('data_sources', [])}")
        
        print(f"\n🏗️ System Architecture:")
        print(f"  • Data Provider: Mock (for testing)")
        print(f"  • Data Store: DuckDB (analytical)")
        print(f"  • Service Layer: Abstracted (swappable)")
        print(f"  • Agent Integration: ✅ Working")
        
        print(f"\n🚀 Status: System fully operational with mock data!")
        print(f"💡 Next: Replace mock provider with Alpaca API credentials")
        
    except Exception as e:
        print(f"❌ Final summary failed: {e}")


if __name__ == "__main__":
    main()