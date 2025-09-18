#!/usr/bin/env python3
"""
AI Hedge Fund System - Main Orchestrator
Coordinates AutoGen agents for investment analysis and decision making
"""
import os
import sys
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Import agents
from agents.fund_director import FundDirectorAgent
from agents.technical_analyst import TechnicalAnalystAgent
from agents.qullamaggie_agent import QullamaggieAgent
from agents.research_agent import ResearchAgent

# Import utilities
from src.utils.logging_config import get_logger
from src.data.duckdb_manager import get_duckdb_manager


class AIHedgeFundOrchestrator:
    """
    Main orchestrator for the AI hedge fund system
    Coordinates agents and manages investment workflow
    """
    
    def __init__(self):
        """Initialize the hedge fund orchestrator"""
        self.logger = get_logger("hedge_fund_orchestrator")
        self.db = get_duckdb_manager()
        
        # Initialize agents
        self.fund_director = FundDirectorAgent(
            name="fund_director",
            description="Senior portfolio manager and investment committee chair"
        )
        
        self.technical_analyst = TechnicalAnalystAgent(
            name="technical_analyst", 
            description="Technical analysis specialist"
        )
        
        self.qullamaggie_agent = QullamaggieAgent(
            name="qullamaggie_agent",
            description="Qullamaggie momentum strategy specialist"
        )
        
        self.research_agent = ResearchAgent(
            name="research_agent",
            description="Data monitoring and research coordination specialist"
        )
        
        # Create required directories
        self._ensure_directories()
        
        self.logger.info("AI Hedge Fund Orchestrator initialized")
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = ['proposals', 'decisions', 'conversations', 'config']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def run_investment_committee_meeting(self, symbol: str) -> Dict[str, Any]:
        """
        Run a complete investment committee meeting for a symbol
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Complete meeting results with final decision
        """
        self.logger.info(f"Starting investment committee meeting for {symbol}")
        
        try:
            # Run the meeting through the Fund Director
            context = {"symbol": symbol}
            message = "investment committee meeting"
            
            result = self.fund_director.process_message(message, context)
            
            if result.get("type") == "investment_committee_meeting":
                self.logger.info(f"Meeting completed for {symbol}: {result['decision']['decision']}")
                return result
            else:
                self.logger.error(f"Meeting failed for {symbol}: {result}")
                return result
                
        except Exception as e:
            self.logger.error(f"Error in investment committee meeting for {symbol}: {e}")
            return {
                "type": "error",
                "message": str(e),
                "symbol": symbol
            }
    
    def analyze_watchlist(self) -> Dict[str, Any]:
        """
        Analyze all symbols in the watchlist
        
        Returns:
            Summary of watchlist analysis
        """
        self.logger.info("Starting watchlist analysis")
        
        try:
            # Get current watchlist
            watchlist = self.fund_director.get_watchlist()
            
            if not watchlist:
                self.logger.warning("No symbols in watchlist")
                return {
                    "type": "watchlist_analysis",
                    "message": "No symbols in watchlist",
                    "results": []
                }
            
            results = []
            
            # Analyze each symbol
            for item in watchlist:
                symbol = item["ticker"]
                self.logger.info(f"Analyzing {symbol}")
                
                meeting_result = self.run_investment_committee_meeting(symbol)
                
                if meeting_result.get("type") == "investment_committee_meeting":
                    results.append({
                        "symbol": symbol,
                        "decision": meeting_result["decision"]["decision"],
                        "confidence": meeting_result["decision"]["confidence"],
                        "reasoning": meeting_result["decision"]["reasoning"],
                        "position_size": meeting_result["decision"].get("position_size", 0)
                    })
                else:
                    results.append({
                        "symbol": symbol,
                        "decision": "ERROR",
                        "error": meeting_result.get("message", "Unknown error")
                    })
            
            return {
                "type": "watchlist_analysis",
                "total_symbols": len(watchlist),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in watchlist analysis: {e}")
            return {
                "type": "error",
                "message": str(e)
            }
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        return self.fund_director._get_portfolio_status()
    
    def add_to_watchlist(self, symbol: str, notes: str = "") -> bool:
        """Add symbol to watchlist"""
        return self.fund_director.add_to_watchlist(symbol, notes)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available market data"""
        return self.db.get_data_summary()
    
    def check_data_freshness(self) -> Dict[str, Any]:
        """Check data freshness for all watchlist symbols"""
        message = "check data freshness"
        return self.research_agent.process_message(message)
    
    def update_stale_data(self) -> Dict[str, Any]:
        """Update stale market data for watchlist symbols"""
        message = "update data"
        return self.research_agent.process_message(message)
    
    def generate_data_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive data status report"""
        message = "data status report"
        return self.research_agent.process_message(message)
    
    def run_full_data_workflow(self) -> Dict[str, Any]:
        """
        Run complete data monitoring and update workflow
        
        Returns:
            Complete workflow results
        """
        self.logger.info("Starting full data monitoring workflow")
        
        try:
            workflow_results = {}
            
            # Step 1: Check data freshness
            print("🔍 Checking data freshness...")
            freshness_check = self.check_data_freshness()
            workflow_results["freshness_check"] = freshness_check
            
            if freshness_check.get("type") == "data_freshness_check":
                stale_count = freshness_check.get("stale_count", 0)
                print(f"   Found {stale_count} symbols with stale data")
                
                # Step 2: Update stale data if needed
                if stale_count > 0:
                    print("📈 Updating stale market data...")
                    update_result = self.update_stale_data()
                    workflow_results["data_update"] = update_result
                    
                    if update_result.get("type") == "data_update":
                        updated_count = update_result.get("updated_count", 0)
                        print(f"   Successfully updated {updated_count} symbols")
                
                # Step 3: Generate status report
                print("📊 Generating data status report...")
                report_result = self.generate_data_status_report()
                workflow_results["status_report"] = report_result
                
                if report_result.get("report_file"):
                    print(f"   Report saved: {report_result['report_file']}")
            
            workflow_results["workflow_status"] = "completed"
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Error in data workflow: {e}")
            return {
                "workflow_status": "error",
                "error": str(e)
            }
    
    def run_single_symbol_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Run analysis for a single symbol without full committee meeting
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Analysis results from all agents
        """
        self.logger.info(f"Running single symbol analysis for {symbol}")
        
        try:
            results = {}
            
            # Technical analysis
            tech_context = {"symbol": symbol}
            tech_message = f"Please provide technical analysis for {symbol}"
            tech_result = self.technical_analyst.process_message(tech_message, tech_context)
            results["technical_analysis"] = tech_result
            
            # Qullamaggie analysis
            qull_context = {"symbol": symbol}
            qull_message = f"Please analyze {symbol} for momentum setups"
            qull_result = self.qullamaggie_agent.process_message(qull_message, qull_context)
            results["qullamaggie_analysis"] = qull_result
            
            return {
                "type": "single_symbol_analysis",
                "symbol": symbol,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in single symbol analysis for {symbol}: {e}")
            return {
                "type": "error",
                "message": str(e),
                "symbol": symbol
            }


def main():
    """Main entry point for the AI hedge fund system"""
    print("🏦 AI Hedge Fund System")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key and try again.")
        return
    
    try:
        # Initialize orchestrator
        print("🚀 Initializing AI Hedge Fund System...")
        orchestrator = AIHedgeFundOrchestrator()
        
        # Get data summary
        print("\n📊 Market Data Summary:")
        data_summary = orchestrator.get_data_summary()
        if "error" not in data_summary:
            print(f"   Total records: {data_summary['total_records']:,}")
            print(f"   Unique symbols: {data_summary['unique_symbols']}")
            print(f"   Data sources: {', '.join(data_summary['data_sources'])}")
        else:
            print(f"   Error: {data_summary['error']}")
        
        # Example usage
        test_symbol = "SPY"
        
        # Add to watchlist if not already there
        print(f"\n📝 Adding {test_symbol} to watchlist...")
        orchestrator.add_to_watchlist(test_symbol, "S&P 500 ETF for testing")
        
        # Run data monitoring workflow
        print(f"\n🔍 Running Data Monitoring Workflow...")
        data_workflow = orchestrator.run_full_data_workflow()
        
        if data_workflow.get("workflow_status") == "completed":
            print("   ✅ Data monitoring workflow completed successfully")
        else:
            print(f"   ❌ Data workflow error: {data_workflow.get('error', 'Unknown error')}")
        
        # Run investment committee meeting
        print(f"\n🏛️  Running Investment Committee Meeting for {test_symbol}...")
        meeting_result = orchestrator.run_investment_committee_meeting(test_symbol)
        
        if meeting_result.get("type") == "investment_committee_meeting":
            decision = meeting_result["decision"]
            print(f"   Decision: {decision['decision']}")
            print(f"   Confidence: {decision['confidence']:.1f}/5.0")
            print(f"   Position Size: {decision.get('position_size', 0)} shares")
            print(f"   Reasoning: {decision['reasoning']}")
            
            if meeting_result.get("conversation_logged"):
                print(f"   Meeting logged: {meeting_result['conversation_logged']}")
            if meeting_result.get("decision_logged"):
                print(f"   Decision logged: {meeting_result['decision_logged']}")
        else:
            print(f"   Error: {meeting_result.get('message', 'Unknown error')}")
        
        # Portfolio status
        print(f"\n💼 Portfolio Status:")
        portfolio = orchestrator.get_portfolio_status()
        print(f"   Portfolio Size: ${portfolio['portfolio_size']:,}")
        print(f"   Active Positions: {portfolio['active_positions']}")
        print(f"   Cash Available: ${portfolio['cash_available']:,}")
        
        print(f"\n✅ AI Hedge Fund System demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running AI hedge fund system: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()