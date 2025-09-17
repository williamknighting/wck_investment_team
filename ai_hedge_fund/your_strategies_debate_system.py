#!/usr/bin/env python3
"""
Your Custom Trading Strategies - Multi-Agent Debate System
Built for YOUR specific trading strategies with agent debates and portfolio manager decisions
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
import time
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import all system components
from src.services.market_data_service import initialize_market_data_service, get_market_data_service
from src.data.alpaca_market_provider import create_alpaca_provider
from src.agents.technical_analyst import SimpleTechnicalAnalyst
from src.agents.strategies.qullamaggie_strategy_agent import QullamaggieStrategyAgent
from src.agents.fund_head_agent import FundHeadAgent
from src.data.duckdb_manager import get_duckdb_manager
from scripts.simple_data_refresh import SimpleDataRefresh
from src.utils.logging_config import get_logger


class YourStrategiesDebateSystem:
    """
    Your Custom Trading Strategies System
    
    Features YOUR specific trading strategies:
    - Qullamaggie: Momentum breakouts, episodic pivots, parabolic shorts
    - [Add your next strategy here]
    - [Add your next strategy here]
    
    With debate system for strategy disagreements and portfolio manager decisions
    """
    
    def __init__(self):
        self.logger = get_logger("your_strategies_debate_system")
        
        # Initialize data layer first
        self.duckdb_manager = get_duckdb_manager()
        
        # Initialize market data service with Alpaca provider
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials not found in .env file")
        
        alpaca_provider = create_alpaca_provider(api_key=api_key, secret_key=secret_key, paper=True)
        self.market_service = initialize_market_data_service(alpaca_provider)
        
        # Initialize refresh service
        self.data_refresh = SimpleDataRefresh()
        
        # Initialize technical analyst
        self.technical_analyst = SimpleTechnicalAnalyst()
        
        # Initialize YOUR specific strategy agents
        self.strategy_agents = {
            "qullamaggie": QullamaggieStrategyAgent(account_size=100000),
            # Add your other strategies here:
            # "your_second_strategy": YourSecondStrategyAgent(),
            # "your_third_strategy": YourThirdStrategyAgent(),
        }
        
        # Initialize fund head agent
        self.fund_head = FundHeadAgent()
        
        # System state
        self.tracked_symbols = ['SPY', 'QQQ']  # Start with our backfilled data
        self.investment_decisions = {}
        
        self.logger.info("Your Strategies Debate System initialized")
    
    def startup_checks(self) -> bool:
        """Perform system startup checks and data refresh"""
        print("🚀 Your Custom Trading Strategies System Starting Up")
        print("=" * 70)
        
        try:
            # 1. Check system components
            print("\n🔧 Step 1: System Component Check")
            if not self._check_system_components():
                return False
            
            # 2. Check data freshness and refresh if needed
            print("\n📊 Step 2: Data Freshness Check")
            if not self._check_and_refresh_data():
                return False
            
            # 3. Test technical analysis
            print("\n🔬 Step 3: Technical Analysis Test")
            if not self._test_technical_analysis():
                return False
            
            # 4. Test YOUR strategy agents
            print("\n🎯 Step 4: Your Strategy Agents Test")
            if not self._test_strategy_agents():
                return False
            
            print("\n✅ All startup checks passed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Startup failed: {e}")
            self.logger.error(f"Startup checks failed: {e}")
            return False
    
    def _check_system_components(self) -> bool:
        """Check all system components are ready"""
        try:
            # Check database
            summary = self.duckdb_manager.get_data_summary()
            total_records = summary.get("total_records", 0)
            unique_symbols = summary.get("unique_symbols", 0)
            
            print(f"   📊 Database: {total_records} records, {unique_symbols} symbols")
            
            if total_records == 0:
                print("   ❌ No data in database - run backfill_historical_data.py first")
                return False
            
            # Check market service
            print(f"   🔌 Market Service: {'✅ Ready' if self.market_service else '❌ Failed'}")
            
            # Check technical analyst
            print(f"   🤖 Technical Analyst: {'✅ Ready' if self.technical_analyst else '❌ Failed'}")
            
            # Check YOUR strategy agents
            print(f"   📈 Your Strategy Agents: {len(self.strategy_agents)} strategies ready")
            for name, agent in self.strategy_agents.items():
                if hasattr(agent, 'get_agent_info'):
                    info = agent.get_agent_info()
                    print(f"     • {info.get('name', name)} ({info.get('persona', 'Your Strategy')})")
                else:
                    # For Qullamaggie agent (doesn't have get_agent_info method)
                    print(f"     • {name.title()} Strategy (Your Trading System)")
            
            # Check fund head
            print(f"   👔 Fund Head: {'✅ Ready' if self.fund_head else '❌ Failed'}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Component check failed: {e}")
            return False
    
    def _check_and_refresh_data(self) -> bool:
        """Check data freshness and refresh if needed"""
        try:
            # Analyze data freshness
            freshness = self.data_refresh.analyze_freshness()
            
            total_symbols = freshness.get("total_symbols", 0)
            refresh_needed = freshness.get("refresh_needed", [])
            
            print(f"   📊 Tracked Symbols: {total_symbols}")
            print(f"   🔄 Need Refresh: {len(refresh_needed)}")
            
            # Show symbol status
            for analysis in freshness.get("symbol_analysis", []):
                symbol = analysis["symbol"]
                days_behind = analysis["days_behind"]
                newest_date = analysis["newest_date"]
                status = "✅ Current" if days_behind <= 1 else f"🔄 {days_behind} days behind"
                print(f"     {symbol}: {status} (latest: {newest_date})")
            
            # Refresh stale data if needed
            if refresh_needed:
                print(f"\n   🔄 Refreshing {len(refresh_needed)} stale symbols...")
                refresh_results = self.data_refresh.refresh_symbols(refresh_needed)
                
                successful = len(refresh_results.get("successful_updates", []))
                failed = len(refresh_results.get("failed_updates", []))
                total_added = refresh_results.get("total_records_added", 0)
                
                print(f"     ✅ Successful: {successful}")
                print(f"     ❌ Failed: {failed}")
                print(f"     📊 Records Added: {total_added}")
            else:
                print("   ✅ All data is current!")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Data refresh failed: {e}")
            return False
    
    def _test_technical_analysis(self) -> bool:
        """Test technical analysis on our data"""
        try:
            print("   🧪 Testing technical analysis with SPY...")
            
            metrics = self.technical_analyst.calculate_all_metrics("SPY", period="1y", interval="1d")
            
            if metrics:
                ma = metrics.moving_averages
                momentum = metrics.momentum_indicators
                
                print(f"     💰 Current Price: ${ma['current_price']:.2f}")
                print(f"     📈 SMA 20: ${ma['sma_20']:.2f}")
                print(f"     📊 RSI: {momentum['rsi_14']:.1f}")
                print(f"     🎯 5-day gain: {momentum['gain_5d']:.1f}%")
                
                print("   ✅ Technical analysis working!")
                return True
            else:
                print("   ❌ No metrics returned")
                return False
                
        except Exception as e:
            print(f"   ❌ Technical analysis test failed: {e}")
            return False
    
    def _test_strategy_agents(self) -> bool:
        """Test YOUR strategy agents"""
        try:
            print("   🧪 Testing your strategy agents...")
            
            for name, agent in self.strategy_agents.items():
                try:
                    if name == "qullamaggie":
                        # Test Qullamaggie agent with its specific interface
                        result = agent.process_message({
                            "type": "analyze_setup",
                            "symbol": "SPY"
                        })
                        
                        if result.get("type") == "trade_setup":
                            setup = result["setup"]
                            print(f"     • {name}: {setup.setup_type.value} setup ({setup.confidence:.1f} stars)")
                        elif result.get("type") == "no_setup":
                            print(f"     • {name}: No setup found (working correctly)")
                        else:
                            print(f"     • {name}: {result.get('type', 'working')}")
                    else:
                        # Test other strategy agents with standard interface
                        # Get SPY technical metrics for testing
                        metrics = self.technical_analyst.calculate_all_metrics("SPY", period="1y", interval="1d")
                        analysis = agent.analyze_symbol("SPY", metrics)
                        assessment = analysis.get("assessment", {})
                        print(f"     • {name}: {assessment.get('recommendation', 'Working')}")
                    
                except Exception as e:
                    print(f"     • {name}: ❌ Failed ({e})")
                    return False
            
            print("   ✅ All your strategy agents working!")
            return True
            
        except Exception as e:
            print(f"   ❌ Strategy agent test failed: {e}")
            return False
    
    def run_investment_committee_meeting(self, symbol: str) -> Dict[str, Any]:
        """
        Run investment committee meeting for YOUR trading strategies
        
        YOUR strategies will debate and the portfolio manager will decide
        """
        
        print(f"\n🏛️ INVESTMENT COMMITTEE MEETING - YOUR STRATEGIES")
        print(f"Symbol: {symbol} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)
        
        try:
            # Step 1: Prepare technical analysis
            print(f"\n📋 PREPARATION: Technical Analysis")
            technical_metrics = self.technical_analyst.calculate_all_metrics(symbol, period="1y", interval="1d")
            
            ma = technical_metrics.moving_averages
            momentum = technical_metrics.momentum_indicators
            
            print(f"   Current Price: ${ma['current_price']:.2f}")
            print(f"   20-day SMA: ${ma['sma_20']:.2f} ({((ma['current_price'] - ma['sma_20']) / ma['sma_20'] * 100):+.1f}%)")
            print(f"   RSI(14): {momentum['rsi_14']:.1f}")
            print(f"   5-day Performance: {momentum['gain_5d']:+.1f}%")
            
            # Step 2: Collect YOUR strategy analyses
            print(f"\n📊 YOUR STRATEGY ANALYSES")
            agent_analyses = []
            
            for name, agent in self.strategy_agents.items():
                try:
                    if name == "qullamaggie":
                        # Handle Qullamaggie agent
                        result = agent.process_message({
                            "type": "analyze_setup",
                            "symbol": symbol
                        })
                        
                        if result.get("type") == "trade_setup":
                            setup = result["setup"]
                            
                            # Convert to standard analysis format for debate
                            analysis = {
                                "agent": f"Qullamaggie Strategy",
                                "persona": "Momentum Breakout Specialist",
                                "symbol": symbol,
                                "assessment": {
                                    "setup_score": setup.confidence,
                                    "recommendation": self._convert_qulla_to_recommendation(setup),
                                    "key_factors": [
                                        f"Setup Type: {setup.setup_type.value}",
                                        f"Confidence: {setup.confidence:.1f}/5 stars",
                                        f"Position Size: {setup.position_size:.1f}%",
                                        f"Risk: ${setup.risk_amount:.0f}"
                                    ]
                                },
                                "signal": {
                                    "action": "buy",
                                    "strength": setup.confidence / 5.0,
                                    "entry_price": setup.entry_price,
                                    "stop_loss": setup.stop_loss,
                                    "target": setup.target_1
                                },
                                "reasoning": f"""🎯 QULLAMAGGIE {setup.setup_type.value} SETUP:
                                
{setup.notes}

Entry: ${setup.entry_price:.2f}
Stop: ${setup.stop_loss:.2f}  
Target: ${setup.target_1:.2f}
Risk: ${setup.risk_amount:.0f} ({setup.position_size:.1f}% position)

Market Regime: {setup.market_regime.value}
Confidence: {setup.confidence:.1f}/5 stars

This meets our mechanical Qullamaggie criteria for a {setup.confidence:.0f}-star {setup.setup_type.value.lower()} setup.""",
                                "debate_points": {
                                    "bullish_points": [
                                        f"Meets {setup.setup_type.value} mechanical criteria",
                                        f"High confidence: {setup.confidence:.1f}/5 stars",
                                        f"Defined risk: ${setup.risk_amount:.0f}"
                                    ],
                                    "key_metric": f"Qullamaggie Score: {setup.confidence:.1f}/5"
                                }
                            }
                            
                        elif result.get("type") == "no_setup":
                            # No setup found
                            analysis = {
                                "agent": "Qullamaggie Strategy",
                                "persona": "Momentum Breakout Specialist", 
                                "symbol": symbol,
                                "assessment": {
                                    "setup_score": 0,
                                    "recommendation": "Pass",
                                    "key_factors": ["No valid Qullamaggie setups found"]
                                },
                                "signal": None,
                                "reasoning": f"""❌ NO QULLAMAGGIE SETUP: 
                                
{symbol} does not meet our mechanical criteria for any Qullamaggie setups:
- Breakout: No consolidation breakout with volume
- Episodic Pivot: No gap with catalyst  
- Parabolic Short: No parabolic extension

We pass on this opportunity and wait for setups that meet our strict criteria.""",
                                "debate_points": {
                                    "bearish_points": ["No mechanical setup criteria met"],
                                    "key_metric": "Qullamaggie Score: 0/5"
                                }
                            }
                        else:
                            continue  # Skip if error
                            
                    else:
                        # Handle other strategy agents with standard interface
                        analysis = agent.analyze_symbol(symbol, technical_metrics)
                    
                    agent_analyses.append(analysis)
                    
                    # Brief summary
                    assessment = analysis.get("assessment", {})
                    recommendation = assessment.get("recommendation", "None")
                    key_metric = analysis.get("debate_points", {}).get("key_metric", "")
                    
                    print(f"   {analysis.get('agent', name)}: {recommendation} ({key_metric})")
                    
                except Exception as e:
                    self.logger.error(f"Error with {name} strategy: {e}")
                    print(f"   {name}: ❌ Analysis failed")
            
            if not agent_analyses:
                print("   ❌ No successful strategy analyses")
                return {"error": "No strategy analyses available"}
            
            # Step 3: Facilitate debate and final decision
            market_context = {
                "trading_session": "regular",
                "market_regime": "normal",  # Could be enhanced
                "volatility_environment": "normal"
            }
            
            final_decision = self.fund_head.make_investment_decision(
                symbol, agent_analyses, market_context
            )
            
            # Store decision
            self.investment_decisions[symbol] = final_decision
            
            return final_decision
            
        except Exception as e:
            error_msg = f"Investment committee meeting failed for {symbol}: {e}"
            print(f"\n❌ {error_msg}")
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def _convert_qulla_to_recommendation(self, setup) -> str:
        """Convert Qullamaggie setup to standard recommendation"""
        if setup.confidence >= 4.5:
            return "Strong Buy"
        elif setup.confidence >= 3.5:
            return "Buy"
        elif setup.confidence >= 2.5:
            return "Weak Buy"
        else:
            return "Pass"
    
    def run_full_portfolio_review(self) -> Dict[str, Any]:
        """Run investment committee meetings for all tracked symbols"""
        
        print(f"\n🏛️ FULL PORTFOLIO REVIEW - YOUR STRATEGIES")
        print(f"Reviewing {len(self.tracked_symbols)} symbols with your trading strategies")
        print("=" * 70)
        
        portfolio_decisions = {}
        meeting_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols_reviewed": [],
            "decisions_made": [],
            "errors": []
        }
        
        for symbol in self.tracked_symbols:
            try:
                print(f"\n🔍 Analyzing {symbol} with YOUR strategies...")
                
                decision = self.run_investment_committee_meeting(symbol)
                
                if "error" not in decision:
                    portfolio_decisions[symbol] = decision
                    meeting_results["symbols_reviewed"].append(symbol)
                    meeting_results["decisions_made"].append({
                        "symbol": symbol,
                        "decision": decision["decision"],
                        "position_size": decision["position_size"],
                        "consensus_level": decision["consensus_level"]
                    })
                else:
                    meeting_results["errors"].append(f"{symbol}: {decision['error']}")
                
                # Brief pause between meetings
                if symbol != self.tracked_symbols[-1]:
                    time.sleep(1)
                    
            except Exception as e:
                error_msg = f"{symbol}: Portfolio review failed - {str(e)}"
                print(f"   ❌ {error_msg}")
                meeting_results["errors"].append(error_msg)
        
        # Portfolio summary
        print(f"\n📈 YOUR PORTFOLIO REVIEW COMPLETE")
        print("-" * 50)
        print(f"   Symbols Reviewed: {len(meeting_results['symbols_reviewed'])}")
        print(f"   Decisions Made: {len(meeting_results['decisions_made'])}")
        print(f"   Errors: {len(meeting_results['errors'])}")
        
        # Show key decisions
        if meeting_results['decisions_made']:
            print(f"\n🎯 YOUR STRATEGY DECISIONS:")
            for decision in meeting_results['decisions_made']:
                symbol = decision['symbol']
                action = decision['decision']
                size = decision['position_size']
                consensus = decision['consensus_level']
                
                action_emoji = {
                    "Strong Buy": "🟢", "Buy": "🟢",
                    "Strong Sell": "🔴", "Sell": "🔴", 
                    "Hold": "🟡", "Pass": "⚪"
                }.get(action, "⚪")
                
                print(f"   {action_emoji} {symbol}: {action} ({size} position, {consensus})")
        
        return meeting_results
    
    def run_system(self):
        """Run YOUR complete multi-strategy trading system"""
        try:
            # Startup checks
            if not self.startup_checks():
                print("\n❌ System startup failed!")
                return False
            
            print(f"\n🎉 YOUR Multi-Strategy System Ready!")
            print(f"   Your Strategies: {len(self.strategy_agents)}")
            print(f"   Tracked Symbols: {len(self.tracked_symbols)}")
            print(f"   Fund Head: {self.fund_head.get_agent_info()['name']}")
            
            # Run full portfolio review with YOUR strategies
            portfolio_results = self.run_full_portfolio_review()
            
            print(f"\n🚀 YOUR Multi-Strategy System operational!")
            print(f"   Conducted {len(portfolio_results['decisions_made'])} investment meetings")
            print(f"   Decisions made using YOUR trading strategies")
            
            print(f"\n💡 YOUR SYSTEM FEATURES:")
            print(f"   ✅ Your specific trading strategies")
            print(f"   ✅ Strategy debate and discussion")
            print(f"   ✅ Portfolio manager consensus building")
            print(f"   ✅ Real hedge fund simulation")
            
            print(f"\n📈 TO ADD MORE OF YOUR STRATEGIES:")
            print(f"   1. Create your strategy agent in src/agents/strategies/your_strategy/")
            print(f"   2. Add to self.strategy_agents in __init__")
            print(f"   3. Your strategies will automatically participate in debates!")
            
            return True
            
        except Exception as e:
            print(f"\n💥 System error: {e}")
            self.logger.error(f"System run failed: {e}")
            return False


def main():
    """Main entry point"""
    print("🚀 YOUR Trading Strategies - Multi-Agent Debate System")
    print("Built for YOUR specific trading strategies with debate and decisions")
    print("=" * 80)
    
    try:
        # Create and run system
        system = YourStrategiesDebateSystem()
        success = system.run_system()
        
        if success:
            print(f"\n✅ YOUR multi-strategy system completed successfully!")
            print(f"\n🎯 YOUR system features:")
            print(f"   • YOUR specific trading strategies")
            print(f"   • Multi-strategy debates and discussions") 
            print(f"   • Portfolio manager moderation")
            print(f"   • Real hedge fund investment committee simulation")
            exit(0)
        else:
            print(f"\n❌ YOUR system run failed!")
            exit(1)
            
    except KeyboardInterrupt:
        print(f"\n🛑 System stopped by user")
        exit(0)
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        exit(1)


if __name__ == "__main__":
    main()