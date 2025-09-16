#!/usr/bin/env python3
"""
Test that DuckDB only accepts real Alpaca data
Simulates what happens with real vs mock data sources
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.duckdb_manager import get_duckdb_manager


def test_duckdb_data_source_filtering():
    """Test DuckDB only accepts specific data sources"""
    print("🛡️ Testing DuckDB Data Source Filtering")
    print("=" * 50)
    
    duckdb_manager = get_duckdb_manager()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'timestamp': [datetime.now(timezone.utc)],
        'open': [100.0], 'high': [101.0], 'low': [99.0], 'close': [100.5],
        'volume': [1000000], 'adj_close': [100.5], 'vwap': [100.2], 'trade_count': [500]
    })
    
    print("\n📊 Testing different data sources:")
    
    # Test 1: Mock data (should be REJECTED)
    print("\n1. Testing MOCK data source:")
    success = duckdb_manager.store_market_data(
        symbol="TEST_MOCK",
        data=sample_data,
        interval="1d", 
        data_source="mock"
    )
    print(f"   Result: {'✅ Accepted' if success else '❌ REJECTED (correct!)'}")
    
    # Test 2: Alpaca data (should be ACCEPTED)
    print("\n2. Testing ALPACA data source:")
    success = duckdb_manager.store_market_data(
        symbol="TEST_ALPACA", 
        data=sample_data,
        interval="1d",
        data_source="alpaca"
    )
    print(f"   Result: {'✅ ACCEPTED (correct!)' if success else '❌ Rejected'}")
    
    # Test 3: Other data source (should be ACCEPTED)
    print("\n3. Testing OTHER data source:")
    success = duckdb_manager.store_market_data(
        symbol="TEST_OTHER",
        data=sample_data, 
        interval="1d",
        data_source="yahoo"
    )
    print(f"   Result: {'✅ ACCEPTED' if success else '❌ Rejected'}")
    
    # Check what's in the database
    print("\n🗄️ Final DuckDB Contents:")
    summary = duckdb_manager.get_data_summary()
    total_records = summary.get("total_records", 0)
    data_sources = summary.get("data_sources", [])
    
    print(f"   Total Records: {total_records}")
    print(f"   Data Sources: {data_sources}")
    
    if 'summary_by_symbol' in summary:
        for record in summary['summary_by_symbol']:
            source = record.get('data_source', 'unknown')
            symbol = record.get('symbol', 'unknown')
            count = record.get('record_count', 0)
            print(f"   {symbol}: {count} records from {source}")
    
    # Verify mock is blocked
    has_mock = 'mock' in data_sources
    print(f"\n🎯 Results:")
    print(f"   Mock data blocked: {not has_mock} {'✅' if not has_mock else '❌'}")
    print(f"   Real data accepted: {'alpaca' in data_sources} {'✅' if 'alpaca' in data_sources else '❌'}")
    
    if not has_mock and 'alpaca' in data_sources:
        print("   🎉 SUCCESS: DuckDB properly filters data sources!")
        return True
    else:
        print("   💥 FAILURE: DuckDB filtering not working correctly!")
        return False


if __name__ == "__main__":
    success = test_duckdb_data_source_filtering()
    if not success:
        exit(1)