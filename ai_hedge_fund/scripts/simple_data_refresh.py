#!/usr/bin/env python3
"""
Simple Data Refresh - No Agent Framework
Direct implementation of intelligent delta updates for DuckDB
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
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


class SimpleDataRefresh:
    """Simple data refresh functionality without agent framework"""
    
    def __init__(self):
        self.duckdb_manager = get_duckdb_manager()
        self.logger = get_logger("simple_data_refresh")
        
        # Initialize market service
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials not found in .env")
        
        alpaca_provider = create_alpaca_provider(api_key=api_key, secret_key=secret_key, paper=True)
        self.market_service = initialize_market_data_service(alpaca_provider)
        
        self.logger.info("Simple Data Refresh initialized")
    
    def analyze_freshness(self) -> Dict[str, Any]:
        """Analyze data freshness for all symbols"""
        summary = self.duckdb_manager.get_data_summary()
        current_time = datetime.now(timezone.utc)
        current_date = current_time.date()
        
        freshness_report = {
            "analysis_time": current_time.isoformat(),
            "total_symbols": summary.get("unique_symbols", 0),
            "total_records": summary.get("total_records", 0),
            "symbol_analysis": [],
            "refresh_needed": []
        }
        
        if 'summary_by_symbol' not in summary:
            return freshness_report
        
        for record in summary['summary_by_symbol']:
            symbol = record['symbol']
            count = record['record_count']
            newest_str = record.get('newest')
            
            if not newest_str:
                continue
                
            newest_date = pd.to_datetime(newest_str).date()
            days_behind = (current_date - newest_date).days
            
            symbol_analysis = {
                "symbol": symbol,
                "record_count": count,
                "newest_date": newest_date.isoformat(),
                "days_behind": days_behind,
                "needs_refresh": days_behind > 1
            }
            
            freshness_report["symbol_analysis"].append(symbol_analysis)
            
            if symbol_analysis["needs_refresh"]:
                freshness_report["refresh_needed"].append(symbol)
        
        return freshness_report
    
    def refresh_symbols(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Refresh data for specified symbols or all stale symbols"""
        if symbols is None:
            freshness = self.analyze_freshness()
            symbols = freshness.get("refresh_needed", [])
        
        if not symbols:
            return {"message": "No symbols need refreshing", "results": []}
        
        results = {
            "processed_symbols": [],
            "successful_updates": [],
            "failed_updates": [],
            "total_records_added": 0
        }
        
        for symbol in symbols:
            try:
                self.logger.info(f"Refreshing {symbol}...")
                
                # Get latest data for this symbol
                latest_data = self.duckdb_manager.get_market_data(symbol, interval="1d", limit=1)
                
                if latest_data.empty:
                    self.logger.warning(f"{symbol}: No existing data found")
                    continue
                
                latest_date = latest_data.index[0]
                days_to_fetch = (datetime.now(timezone.utc).date() - latest_date.date()).days
                
                if days_to_fetch <= 0:
                    self.logger.info(f"{symbol}: Already up to date")
                    continue
                
                # Fetch fresh data (with buffer for weekends)
                fresh_data = self.market_service.get_stock_data(
                    symbol, 
                    period=f"{days_to_fetch + 5}d",
                    interval="1d"
                )
                
                if not fresh_data.empty:
                    # Filter to only truly new records
                    new_records = fresh_data[fresh_data.index > latest_date]
                    
                    if not new_records.empty:
                        results["successful_updates"].append({
                            "symbol": symbol,
                            "records_added": len(new_records),
                            "date_range": f"{new_records.index[0].date()} to {new_records.index[-1].date()}"
                        })
                        results["total_records_added"] += len(new_records)
                        self.logger.info(f"{symbol}: Added {len(new_records)} new records")
                    else:
                        self.logger.info(f"{symbol}: No new records available")
                else:
                    results["failed_updates"].append({
                        "symbol": symbol,
                        "error": "No data returned from API"
                    })
                
                results["processed_symbols"].append(symbol)
                
                # Conservative delay
                if symbol != symbols[-1]:
                    time.sleep(1.5)
                    
            except Exception as e:
                error_msg = f"Failed to refresh {symbol}: {e}"
                self.logger.error(error_msg)
                results["failed_updates"].append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        return results


def main():
    """Test the simple data refresh system"""
    print("🔄 Simple Data Refresh Test")
    print("=" * 50)
    
    try:
        # Initialize refresh system
        print("\n🔧 Initializing...")
        refresh_system = SimpleDataRefresh()
        print("✅ System ready")
        
        # Analyze current data freshness
        print("\n📊 Analyzing Data Freshness...")
        freshness = refresh_system.analyze_freshness()
        
        total_symbols = freshness.get("total_symbols", 0)
        refresh_needed = freshness.get("refresh_needed", [])
        
        print(f"   Total Symbols: {total_symbols}")
        print(f"   Symbols Needing Refresh: {len(refresh_needed)}")
        
        # Show symbol status
        for analysis in freshness.get("symbol_analysis", []):
            symbol = analysis["symbol"]
            days_behind = analysis["days_behind"]
            status = "✅ Current" if days_behind <= 1 else f"🔄 {days_behind} days behind"
            print(f"   {symbol}: {status}")
        
        # Refresh stale data if needed
        if refresh_needed:
            print(f"\n🔄 Refreshing {len(refresh_needed)} symbols...")
            results = refresh_system.refresh_symbols(refresh_needed[:3])  # Limit to 3 for testing
            
            successful = len(results.get("successful_updates", []))
            failed = len(results.get("failed_updates", []))
            total_added = results.get("total_records_added", 0)
            
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            print(f"   📊 Records Added: {total_added}")
            
            for update in results.get("successful_updates", []):
                symbol = update["symbol"]
                records = update["records_added"]
                date_range = update["date_range"]
                print(f"     {symbol}: +{records} records ({date_range})")
                
        else:
            print("\n✅ All data is current - no refresh needed!")
        
        # Final status
        print(f"\n📈 Final Database Status:")
        final_summary = refresh_system.duckdb_manager.get_data_summary()
        print(f"   Total Records: {final_summary.get('total_records', 0)}")
        print(f"   Unique Symbols: {final_summary.get('unique_symbols', 0)}")
        
        print(f"\n🎉 Data refresh test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)