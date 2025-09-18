"""
Research Agent for AI Hedge Fund System
Monitors data freshness, updates market data, and coordinates research activities
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json

from agents.base_agent import BaseHedgeFundAgent

# Import data providers
import sys
sys.path.append('..')
from src.data.alpaca_provider import get_alpaca_provider
from src.data.duckdb_manager import get_duckdb_manager


class ResearchAgent(BaseHedgeFundAgent):
    """
    Research Agent with AutoGen integration
    Primary responsibilities:
    1. Monitor data freshness for watchlist symbols
    2. Coordinate market data updates via Alpaca
    3. Generate data status reports
    4. Placeholder for news/fundamental research integration
    """
    
    def _initialize(self) -> None:
        """Initialize the research agent"""
        self.alpaca_provider = get_alpaca_provider()
        self.data_freshness_threshold = timedelta(hours=24)  # 1 business day
        self.weekend_adjustment = True  # Account for weekends/holidays
        self.last_data_check = None
        self.data_status_cache = {}
        self._initialized_at = datetime.now(timezone.utc)
        self.logger.info("Research Agent initialized - Data monitoring active")
    
    def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process incoming messages and coordinate research activities
        
        Args:
            message: The message/request from another agent
            context: Optional context including symbols, report type, etc.
            
        Returns:
            Dict containing research results or data status
        """
        try:
            # Parse message intent
            if "data freshness" in message.lower() or "check data" in message.lower():
                return self._check_data_freshness(context)
            elif "update data" in message.lower() or "refresh data" in message.lower():
                return self._update_market_data(context)
            elif "data status" in message.lower() or "status report" in message.lower():
                return self._generate_data_status_report(context)
            elif "news" in message.lower() or "research" in message.lower():
                return self._research_news_and_fundamentals(context)
            elif "watchlist" in message.lower():
                return self._analyze_watchlist_data_status()
            else:
                return self._general_research_response(message, context)
                
        except Exception as e:
            self.logger.error(f"Error processing research message: {e}")
            return {
                "type": "error",
                "message": str(e),
                "agent": self.name
            }
    
    def _check_data_freshness(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check data freshness for all active tickers in watchlist
        
        Returns:
            Dict with freshness status for each ticker
        """
        try:
            # Get active watchlist
            watchlist = self.get_watchlist(active_only=True)
            
            if not watchlist:
                return {
                    "type": "data_freshness_check",
                    "message": "No active symbols in watchlist",
                    "fresh_data": [],
                    "stale_data": [],
                    "agent": self.name
                }
            
            fresh_data = []
            stale_data = []
            
            # Get current market data summary
            data_summary = self.db.get_data_summary()
            
            # Check each watchlist symbol
            for item in watchlist:
                symbol = item["ticker"]
                freshness_status = self._analyze_symbol_freshness(symbol, data_summary)
                
                if freshness_status["is_fresh"]:
                    fresh_data.append({
                        "symbol": symbol,
                        "last_updated": freshness_status["last_updated"],
                        "age_hours": freshness_status["age_hours"],
                        "record_count": freshness_status["record_count"]
                    })
                else:
                    stale_data.append({
                        "symbol": symbol,
                        "last_updated": freshness_status["last_updated"],
                        "age_hours": freshness_status["age_hours"],
                        "needs_update": True,
                        "reason": freshness_status["reason"]
                    })
            
            self.last_data_check = datetime.now(timezone.utc)
            
            result = {
                "type": "data_freshness_check",
                "timestamp": self.last_data_check.isoformat(),
                "total_symbols": len(watchlist),
                "fresh_count": len(fresh_data),
                "stale_count": len(stale_data),
                "fresh_data": fresh_data,
                "stale_data": stale_data,
                "agent": self.name
            }
            
            # Cache results
            self.data_status_cache = result
            
            self.logger.info(f"Data freshness check completed: {len(fresh_data)} fresh, {len(stale_data)} stale")
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {e}")
            return {
                "type": "error",
                "message": f"Data freshness check failed: {str(e)}",
                "agent": self.name
            }
    
    def _analyze_symbol_freshness(self, symbol: str, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze freshness for a single symbol accounting for business days
        
        Args:
            symbol: Stock symbol to check
            data_summary: DuckDB data summary
            
        Returns:
            Dict with freshness analysis
        """
        try:
            # Find symbol in data summary
            symbol_data = None
            for record in data_summary.get("summary_by_symbol", []):
                if record["symbol"] == symbol and record["interval"] == "1d":
                    symbol_data = record
                    break
            
            if not symbol_data:
                return {
                    "is_fresh": False,
                    "last_updated": None,
                    "age_hours": float('inf'),
                    "record_count": 0,
                    "reason": "No data found in database"
                }
            
            # Parse timestamps
            last_updated = pd.to_datetime(symbol_data["newest"])
            now = pd.Timestamp.now(tz='UTC')
            age = now - last_updated
            age_hours = age.total_seconds() / 3600
            
            # Business day logic - account for weekends and trading hours
            is_fresh = self._is_data_fresh_for_business_day(last_updated, now)
            
            reason = ""
            if not is_fresh:
                if self._is_weekend(now):
                    reason = "Weekend - data acceptable if from Friday"
                elif age_hours > 24:
                    reason = f"Data {age_hours:.1f} hours old, exceeds threshold"
                else:
                    reason = "Market hours consideration"
            
            return {
                "is_fresh": is_fresh,
                "last_updated": last_updated.isoformat(),
                "age_hours": age_hours,
                "record_count": symbol_data["record_count"],
                "reason": reason
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing freshness for {symbol}: {e}")
            return {
                "is_fresh": False,
                "last_updated": None,
                "age_hours": float('inf'),
                "record_count": 0,
                "reason": f"Analysis error: {str(e)}"
            }
    
    def _is_data_fresh_for_business_day(self, last_updated: pd.Timestamp, now: pd.Timestamp) -> bool:
        """
        Determine if data is fresh considering business days and market hours
        
        Args:
            last_updated: When data was last updated
            now: Current timestamp
            
        Returns:
            True if data is considered fresh
        """
        try:
            # Convert to Eastern Time (market timezone)
            et_now = now.tz_convert('US/Eastern')
            et_last = last_updated.tz_convert('US/Eastern')
            
            # If it's weekend, data from Friday is acceptable
            if et_now.weekday() >= 5:  # Saturday=5, Sunday=6
                # Find last Friday
                days_since_friday = (et_now.weekday() - 4) % 7
                if days_since_friday == 0:
                    days_since_friday = 7
                last_friday = et_now.date() - timedelta(days=days_since_friday)
                
                # Data is fresh if it's from Friday or later
                return et_last.date() >= last_friday
            
            # Weekday logic
            else:
                # Before market open (9:30 AM ET), yesterday's data is acceptable
                market_open = et_now.replace(hour=9, minute=30, second=0, microsecond=0)
                
                if et_now < market_open:
                    # Accept data from yesterday
                    yesterday = et_now.date() - timedelta(days=1)
                    return et_last.date() >= yesterday
                else:
                    # After market open, need today's data (or very recent)
                    age = et_now - et_last
                    return age < self.data_freshness_threshold
            
        except Exception as e:
            self.logger.error(f"Error in business day freshness check: {e}")
            # Default to simple age check
            age = now - last_updated
            return age < self.data_freshness_threshold
    
    def _is_weekend(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is on weekend"""
        return timestamp.weekday() >= 5
    
    def _update_market_data(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Update market data for stale symbols using Alpaca provider
        
        Args:
            context: Optional context with specific symbols to update
            
        Returns:
            Dict with update results
        """
        try:
            # Determine which symbols to update
            if context and "symbols" in context:
                symbols_to_update = context["symbols"]
            else:
                # Check freshness and update stale data
                freshness_check = self._check_data_freshness()
                symbols_to_update = [item["symbol"] for item in freshness_check["stale_data"]]
            
            if not symbols_to_update:
                return {
                    "type": "data_update",
                    "message": "No symbols require data updates",
                    "updated_symbols": [],
                    "failed_symbols": [],
                    "agent": self.name
                }
            
            self.logger.info(f"Updating data for {len(symbols_to_update)} symbols: {symbols_to_update}")
            
            updated_symbols = []
            failed_symbols = []
            
            # Update each symbol using Alpaca provider
            for symbol in symbols_to_update:
                try:
                    # Calculate date range for delta update
                    start_date = self._get_last_update_date(symbol)
                    end_date = datetime.now(timezone.utc)
                    
                    # Fetch data from Alpaca (this caches automatically)
                    data = self.alpaca_provider.get_stock_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval="1Day"
                    )
                    
                    if not data.empty:
                        updated_symbols.append({
                            "symbol": symbol,
                            "records_updated": len(data),
                            "date_range": f"{start_date.date()} to {end_date.date()}"
                        })
                        self.logger.info(f"Updated {symbol}: {len(data)} records")
                    else:
                        failed_symbols.append({
                            "symbol": symbol,
                            "reason": "No data returned from Alpaca"
                        })
                        
                except Exception as e:
                    failed_symbols.append({
                        "symbol": symbol,
                        "reason": str(e)
                    })
                    self.logger.error(f"Failed to update {symbol}: {e}")
            
            return {
                "type": "data_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requested_symbols": len(symbols_to_update),
                "updated_count": len(updated_symbols),
                "failed_count": len(failed_symbols),
                "updated_symbols": updated_symbols,
                "failed_symbols": failed_symbols,
                "agent": self.name
            }
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return {
                "type": "error",
                "message": f"Data update failed: {str(e)}",
                "agent": self.name
            }
    
    def _get_last_update_date(self, symbol: str) -> datetime:
        """
        Get the last update date for a symbol to perform delta updates
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Date to start delta update from
        """
        try:
            # Get latest data for symbol
            latest_data = self.db.get_market_data(symbol, interval="1d", limit=1)
            
            if not latest_data.empty:
                # Start from day after last record
                last_date = latest_data.index[-1]
                return last_date + timedelta(days=1)
            else:
                # No existing data, get 1 year of history
                return datetime.now(timezone.utc) - timedelta(days=365)
                
        except Exception as e:
            self.logger.error(f"Error getting last update date for {symbol}: {e}")
            # Default to 30 days back
            return datetime.now(timezone.utc) - timedelta(days=30)
    
    def _analyze_watchlist_data_status(self) -> Dict[str, Any]:
        """
        Analyze data status for all watchlist symbols
        
        Returns:
            Comprehensive watchlist data analysis
        """
        try:
            freshness_check = self._check_data_freshness()
            
            # Additional analysis
            total_symbols = freshness_check["total_symbols"]
            fresh_percentage = (freshness_check["fresh_count"] / total_symbols * 100) if total_symbols > 0 else 0
            
            # Identify symbols that need immediate attention
            critical_symbols = [
                item for item in freshness_check["stale_data"] 
                if item["age_hours"] > 48  # More than 2 days old
            ]
            
            return {
                "type": "watchlist_data_analysis",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_symbols": total_symbols,
                    "fresh_count": freshness_check["fresh_count"],
                    "stale_count": freshness_check["stale_count"],
                    "fresh_percentage": round(fresh_percentage, 1),
                    "critical_count": len(critical_symbols)
                },
                "freshness_details": freshness_check,
                "critical_symbols": critical_symbols,
                "recommendations": self._generate_data_recommendations(freshness_check),
                "agent": self.name
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing watchlist data status: {e}")
            return {
                "type": "error",
                "message": f"Watchlist analysis failed: {str(e)}",
                "agent": self.name
            }
    
    def _generate_data_status_report(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive data status report
        
        Returns:
            Formatted data status report
        """
        try:
            # Get current data status
            watchlist_analysis = self._analyze_watchlist_data_status()
            
            # Generate report content
            report_content = self._format_data_status_report(watchlist_analysis)
            
            # Write report to file
            report_file = self._write_data_status_report(report_content, watchlist_analysis)
            
            return {
                "type": "data_status_report",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": watchlist_analysis["summary"],
                "report_file": report_file,
                "recommendations": watchlist_analysis["recommendations"],
                "agent": self.name
            }
            
        except Exception as e:
            self.logger.error(f"Error generating data status report: {e}")
            return {
                "type": "error",
                "message": f"Report generation failed: {str(e)}",
                "agent": self.name
            }
    
    def _format_data_status_report(self, analysis: Dict[str, Any]) -> str:
        """Format data status report as markdown"""
        summary = analysis["summary"]
        
        content = f"""# Market Data Status Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Agent**: {self.name}  

## Executive Summary

- **Total Watchlist Symbols**: {summary['total_symbols']}
- **Fresh Data**: {summary['fresh_count']} ({summary['fresh_percentage']}%)
- **Stale Data**: {summary['stale_count']} symbols
- **Critical Issues**: {summary['critical_count']} symbols

## Data Freshness Analysis

### Fresh Data Symbols
"""
        
        for item in analysis["freshness_details"]["fresh_data"]:
            content += f"- **{item['symbol']}**: Last updated {item['age_hours']:.1f} hours ago ({item['record_count']} records)\n"
        
        content += "\n### Stale Data Symbols\n"
        
        for item in analysis["freshness_details"]["stale_data"]:
            content += f"- **{item['symbol']}**: {item['age_hours']:.1f} hours old - {item['reason']}\n"
        
        if analysis["critical_symbols"]:
            content += "\n### Critical Issues (>48 hours old)\n"
            for item in analysis["critical_symbols"]:
                content += f"- **{item['symbol']}**: {item['age_hours']:.1f} hours old - REQUIRES IMMEDIATE ATTENTION\n"
        
        content += "\n## Recommendations\n\n"
        for rec in analysis["recommendations"]:
            content += f"- {rec}\n"
        
        content += f"""
---

*Data status report generated by Research Agent*  
*Next check recommended in 4 hours*
"""
        
        return content
    
    def _write_data_status_report(self, content: str, analysis: Dict[str, Any]) -> str:
        """Write data status report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_status_report_{timestamp}.md"
            filepath = Path("proposals") / filename
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            self.logger.info(f"Data status report written to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error writing report: {e}")
            return ""
    
    def _generate_data_recommendations(self, freshness_check: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on data status"""
        recommendations = []
        
        stale_count = freshness_check["stale_count"]
        total_count = freshness_check["total_symbols"]
        
        if stale_count == 0:
            recommendations.append("All watchlist data is current - no immediate action required")
        elif stale_count < total_count * 0.2:
            recommendations.append(f"Update {stale_count} stale symbols using delta refresh")
        else:
            recommendations.append(f"High number of stale symbols ({stale_count}) - consider bulk data refresh")
        
        # Check for critical symbols
        critical_symbols = [item for item in freshness_check["stale_data"] if item["age_hours"] > 48]
        if critical_symbols:
            recommendations.append(f"URGENT: {len(critical_symbols)} symbols have critically old data (>48h)")
        
        # General recommendations
        recommendations.append("Schedule automated data updates during off-market hours")
        recommendations.append("Consider implementing real-time data feeds for active trading symbols")
        
        return recommendations
    
    def _research_news_and_fundamentals(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Placeholder for news and fundamental research integration
        
        Args:
            context: Optional context with symbols or research parameters
            
        Returns:
            Placeholder response for news research
        """
        symbol = context.get("symbol", "general") if context else "general"
        
        # PLACEHOLDER: This would integrate with news APIs like Alpha Vantage, NewsAPI, etc.
        return {
            "type": "news_research",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "placeholder",
            "message": "News and fundamental research API integration not yet implemented",
            "planned_features": [
                "Real-time news sentiment analysis",
                "Earnings announcements and estimates",
                "Insider trading activity monitoring",
                "SEC filing analysis",
                "Analyst recommendations tracking",
                "Economic calendar integration"
            ],
            "next_steps": [
                "Integrate with NewsAPI or Alpha Vantage News",
                "Set up SEC EDGAR API for filings",
                "Add fundamental data provider (Financial Modeling Prep, etc.)",
                "Implement sentiment analysis pipeline"
            ],
            "agent": self.name
        }
    
    def _general_research_response(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle general research inquiries"""
        return {
            "type": "general_response",
            "message": "Research Agent ready. I monitor data freshness and coordinate market data updates.",
            "capabilities": [
                "Data freshness monitoring for watchlist symbols",
                "Automated market data updates via Alpaca",
                "Business day aware freshness checking",
                "Delta updates to minimize API usage",
                "Comprehensive data status reporting",
                "Placeholder for news and fundamental research"
            ],
            "commands": [
                "check data freshness",
                "update market data", 
                "generate status report",
                "analyze watchlist data",
                "research news [symbol]"
            ],
            "agent": self.name
        }