#!/usr/bin/env python3
"""
Test that NO MOCK DATA is written to DuckDB
DuckDB should ONLY contain real Alpaca data
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.services.market_data_service import initialize_market_data_service
from src.data.mock_provider import create_mock_provider
from src.data.duckdb_manager import get_duckdb_manager


def test_no_mock_data_in_duckdb():
    """Verify mock data is NEVER written to DuckDB"""
    print("🚫 Testing: NO Mock Data in DuckDB")
    print("=" * 50)
    
    # Clean start
    duckdb_manager = get_duckdb_manager()
    
    # Check database is empty
    print("\n📊 Step 1: Verify DuckDB is clean")
    summary = duckdb_manager.get_data_summary()
    initial_records = summary.get("total_records", 0)
    print(f"  Initial records in DuckDB: {initial_records}")
    
    # Create mock provider and service
    print("\n🎭 Step 2: Create Mock Provider")
    mock_provider = create_mock_provider()
    mock_service = initialize_market_data_service(mock_provider)
    print("  ✅ Mock provider and service created")
    
    # Try to fetch data with mock provider
    print("\n📈 Step 3: Fetch Mock Data")
    test_symbols = ['SPY', 'AAPL', 'NVDA']
    
    for symbol in test_symbols:
        print(f"  Fetching mock data for {symbol}...")
        data = mock_service.get_stock_data(symbol, period='1mo', interval='1d')
        
        if not data.empty:
            print(f"    ✅ Got {len(data)} records (in memory only)")
        else:
            print(f"    ❌ No data received")
    
    # CHECK: Verify DuckDB is still empty
    print("\n🗄️ Step 4: Verify DuckDB Remains Empty")
    final_summary = duckdb_manager.get_data_summary()
    final_records = final_summary.get("total_records", 0)
    data_sources = final_summary.get("data_sources", [])
    
    print(f"  Final records in DuckDB: {final_records}")
    print(f"  Data sources: {data_sources}")
    
    # Test results
    if final_records == 0:
        print("  ✅ SUCCESS: DuckDB contains NO mock data!")
    else:
        print(f"  ❌ FAILURE: DuckDB contains {final_records} records!")
        
        # Show what's in there
        if 'summary_by_symbol' in final_summary:
            print("  Contents:")
            for record in final_summary['summary_by_symbol']:
                source = record.get('data_source', 'unknown')
                print(f"    {record['symbol']}: {record['record_count']} records from {source}")
    
    if 'mock' in data_sources:
        print("  ❌ CRITICAL: Mock data source found in DuckDB!")
        return False
    
    # Test latest prices (should work without caching)
    print("\n💰 Step 5: Test Latest Prices (No Caching)")
    try:
        prices = mock_service.get_latest_prices(['SPY', 'AAPL'])
        print(f"  Latest prices: {prices}")
        print("  ✅ Mock data works without DuckDB caching")
    except Exception as e:
        print(f"  ❌ Latest prices failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Mock Data Isolation Test Results:")
    print(f"  DuckDB Records: {final_records} (should be 0)")
    print(f"  Mock Data Sources: {'mock' in data_sources} (should be False)")
    
    if final_records == 0 and 'mock' not in data_sources:
        print("  🎉 SUCCESS: Mock data properly isolated from DuckDB!")
        return True
    else:
        print("  💥 FAILURE: Mock data contaminated DuckDB!")
        return False


if __name__ == "__main__":
    success = test_no_mock_data_in_duckdb()
    if not success:
        exit(1)