#!/usr/bin/env python3
"""Debug column count issue"""
import sys
sys.path.append('src')

from src.data.mock_provider import create_mock_provider
from src.data.duckdb_manager import get_duckdb_manager
import pandas as pd
from datetime import datetime, timezone

def debug_data_flow():
    """Debug the exact data flow to find the extra column"""
    
    print("🔍 Debugging DuckDB Column Count Issue")
    print("=" * 50)
    
    # 1. Generate mock data
    provider = create_mock_provider()
    
    # Generate date range like mock provider does
    from datetime import timedelta
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=3)
    date_range = pd.bdate_range(start=start_date, end=end_date)
    
    print(f"\n1. Generate raw price data:")
    df_raw = provider._generate_price_data('SPY', date_range)
    print(f"   Columns: {df_raw.columns.tolist()}")
    print(f"   Index: {df_raw.index.name}")
    print(f"   Shape: {df_raw.shape}")
    
    # 2. Reset index (like cache_data does)
    print(f"\n2. After reset_index():")
    df_reset = df_raw.reset_index()
    print(f"   Columns: {df_reset.columns.tolist()}")
    print(f"   Shape: {df_reset.shape}")
    
    # 3. Rename columns (like cache_data does)
    print(f"\n3. After renaming to lowercase:")
    df_renamed = df_reset.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
        'Volume': 'volume', 'Adj Close': 'adj_close', 'VWAP': 'vwap',
        'Trade Count': 'trade_count'
    })
    print(f"   Columns: {df_renamed.columns.tolist()}")
    print(f"   Shape: {df_renamed.shape}")
    
    # 4. What DuckDBManager store_market_data will do
    print(f"\n4. What DuckDBManager.store_market_data() expects:")
    
    # Simulate store_market_data processing
    df = df_renamed.copy()
    symbol = 'SPY'
    interval = '1d'
    data_source = 'mock'
    
    # Ensure timestamp index (but we already have timestamp column!)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            print("   ERROR: No timestamp column or index!")
            return
    
    # Add required columns (this is where extra columns might come from)
    df = df.reset_index()  # This might be the issue - double reset?
    print(f"   After second reset_index: {df.columns.tolist()}")
    print(f"   Shape after second reset: {df.shape}")
    
    df['symbol'] = symbol
    df['interval'] = interval
    df['data_source'] = data_source
    df['created_at'] = datetime.now(timezone.utc)
    df['updated_at'] = datetime.now(timezone.utc)
    
    print(f"   After adding metadata: {df.columns.tolist()}")
    print(f"   Total columns: {len(df.columns)}")
    
    # 5. Column selection for insert
    print(f"\n5. Column selection for DuckDB insert:")
    required_cols = [
        'symbol', 'timestamp', 'interval', 'open', 'high', 'low', 
        'close', 'volume', 'adj_close', 'vwap', 'trade_count',
        'data_source', 'created_at', 'updated_at'
    ]
    print(f"   Required: {required_cols}")
    print(f"   Required count: {len(required_cols)}")
    
    available_cols = df.columns.tolist()
    print(f"   Available: {available_cols}")
    print(f"   Available count: {len(available_cols)}")
    
    # Check for missing/extra columns
    missing = [col for col in required_cols if col not in available_cols]
    extra = [col for col in available_cols if col not in required_cols]
    
    if missing:
        print(f"   ❌ Missing columns: {missing}")
    if extra:
        print(f"   ⚠️ Extra columns: {extra}")
    
    try:
        df_clean = df[required_cols]
        print(f"   ✅ Clean DataFrame shape: {df_clean.shape}")
        print(f"   ✅ Ready for DuckDB insert!")
        
        # Try the actual insert
        manager = get_duckdb_manager()
        success = manager.store_market_data(
            symbol='TEST_DEBUG',
            data=df_clean,
            interval='1d',
            data_source='debug'
        )
        print(f"   Insert result: {'✅ Success' if success else '❌ Failed'}")
        
    except Exception as e:
        print(f"   ❌ Column selection failed: {e}")


if __name__ == "__main__":
    debug_data_flow()