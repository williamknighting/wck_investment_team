#!/usr/bin/env python3
"""
Backfill Historical Data - Conservative approach
Fetch 2 years of daily data for SPY and QQQ with careful API usage
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.services.market_data_service import initialize_market_data_service
from src.data.alpaca_market_provider import create_alpaca_provider
from src.data.duckdb_manager import get_duckdb_manager
from src.utils.logging_config import get_logger


def backfill_historical_data():
    """Conservative backfill of 2 years daily data for SPY and QQQ"""
    print("📈 Backfilling Historical Data - Conservative Approach")
    print("=" * 60)
    
    logger = get_logger("backfill")
    
    # Get API credentials from .env
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper_trading = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    
    print(f"\n🔑 Configuration:")
    print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"   Secret Key: {'✅ Set' if secret_key else '❌ Missing'}")
    print(f"   Paper Trading: {paper_trading}")
    
    if not api_key or not secret_key:
        print("❌ ERROR: Alpaca API credentials not found in .env file")
        return False
    
    try:
        # Initialize system
        print(f"\n🗄️ Initializing DuckDB...")
        duckdb_manager = get_duckdb_manager()
        
        print(f"📡 Initializing Alpaca provider...")
        alpaca_provider = create_alpaca_provider(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper_trading
        )
        
        market_service = initialize_market_data_service(alpaca_provider)
        print("✅ System ready for backfill")
        
        # Check current data
        print(f"\n📊 Current DuckDB Status:")
        summary = duckdb_manager.get_data_summary()
        print(f"   Total Records: {summary.get('total_records', 0)}")
        print(f"   Data Sources: {summary.get('data_sources', [])}")
        
        # Symbols to backfill
        symbols = ['SPY', 'QQQ']
        
        print(f"\n🎯 Backfill Plan:")
        print(f"   Symbols: {symbols}")
        print(f"   Period: 2 years daily data")
        print(f"   Conservative: 1 request per second")
        print(f"   Estimated time: ~{len(symbols)} seconds")
        
        # Backfill each symbol
        successful_backfills = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n📈 [{i}/{len(symbols)}] Backfilling {symbol}...")
            
            try:
                # Check if we already have data
                existing_data = market_service.get_stock_data(symbol, period="5d", interval="1d")
                if not existing_data.empty:
                    print(f"   ⚠️ {symbol} already has {len(existing_data)} recent records")
                    print(f"   Proceeding with 2-year backfill anyway...")
                
                # Fetch 2 years of daily data
                start_time = time.time()
                data = market_service.get_stock_data(symbol, period="2y", interval="1d")
                end_time = time.time()
                
                if not data.empty:
                    latest_price = data['Close'].iloc[-1]
                    oldest_date = data.index[0].strftime('%Y-%m-%d')
                    newest_date = data.index[-1].strftime('%Y-%m-%d')
                    
                    print(f"   ✅ SUCCESS: {len(data)} records")
                    print(f"   📅 Date Range: {oldest_date} to {newest_date}")
                    print(f"   💰 Latest Price: ${latest_price:.2f}")
                    print(f"   ⏱️ Fetch Time: {end_time - start_time:.2f}s")
                    
                    successful_backfills += 1
                    
                else:
                    print(f"   ❌ FAILED: No data returned for {symbol}")
                
                # Conservative delay between symbols (API protection)
                if i < len(symbols):
                    print(f"   ⏸️ Waiting 2 seconds before next symbol...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                logger.error(f"Backfill failed for {symbol}: {e}")
        
        # Final status
        print(f"\n" + "=" * 60)
        print(f"📊 Backfill Complete!")
        
        # Check final database status
        final_summary = duckdb_manager.get_data_summary()
        total_records = final_summary.get("total_records", 0)
        data_sources = final_summary.get("data_sources", [])
        
        print(f"   ✅ Successfully backfilled: {successful_backfills}/{len(symbols)} symbols")
        print(f"   📊 Total DuckDB Records: {total_records}")
        print(f"   📡 Data Sources: {data_sources}")
        
        if 'summary_by_symbol' in final_summary:
            print(f"   📈 Symbol Breakdown:")
            for record in final_summary['summary_by_symbol']:
                symbol = record['symbol']
                count = record['record_count']
                source = record['data_source']
                oldest = record.get('oldest', 'N/A')
                newest = record.get('newest', 'N/A')
                print(f"     {symbol}: {count} records from {source}")
                print(f"       Range: {oldest} to {newest}")
        
        if successful_backfills == len(symbols):
            print(f"   🎉 All symbols successfully backfilled!")
            return True
        else:
            print(f"   ⚠️ Some symbols failed - check logs")
            return False
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        logger.error(f"Backfill process failed: {e}")
        return False


if __name__ == "__main__":
    success = backfill_historical_data()
    if success:
        print(f"\n🚀 Historical data backfill complete!")
    else:
        print(f"\n💥 Backfill failed - check errors above")
        exit(1)