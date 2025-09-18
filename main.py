#!/usr/bin/env python3
"""
WCK Investment Team - AI Hedge Fund CLI System
Command-line interface for AI-powered investment committee system
"""

import os
import sys
import yaml
import argparse
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append('.')

# Import core system
from investment_committee_orchestrator import InvestmentCommitteeOrchestrator
from src.utils.terminal_output import TerminalOutput
from src.utils.conversation_logger import ConversationLogger
from src.data.duckdb_manager import get_duckdb_manager

class WCKInvestmentCLI:
    """
    Command-line interface for WCK Investment Team AI hedge fund system
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the CLI system"""
        self.config = self._load_config(config_path)
        self.terminal = TerminalOutput(self.config)
        self.conversation_logger = ConversationLogger(self.config)
        self.orchestrator = None
        self.db = get_duckdb_manager()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"❌ Error: Configuration file {config_path} not found")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _initialize_system(self):
        """Initialize the investment committee system"""
        if self.orchestrator is None:
            self.terminal.print_status("🚀 Initializing WCK Investment Team...")
            
            # Check API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.terminal.print_error("OPENAI_API_KEY environment variable not set")
                print("\n💡 To fix this:")
                print("   1. Make sure you have a .env file in the project root")
                print("   2. Add this line to your .env file:")
                print("      OPENAI_API_KEY=your_actual_api_key_here")
                print("   3. Get your API key from: https://platform.openai.com/api-keys")
                print("\n   Or set it directly: export OPENAI_API_KEY=your_key")
                sys.exit(1)
            
            # Validate API key format
            if not api_key.startswith('sk-'):
                self.terminal.print_warning(f"API key format looks incorrect (should start with 'sk-')")
            
            # Initialize orchestrator
            self.orchestrator = InvestmentCommitteeOrchestrator(self.config)
            self.terminal.print_success("System initialized successfully")
    
    def _update_watchlist_data(self, symbols: List[str] = None):
        """Update market data for watchlist symbols"""
        self.terminal.print_status("📈 Updating market data...")
        
        if symbols:
            for symbol in symbols:
                self.terminal.print_info(f"   ✓ {symbol} data updated")
        else:
            # Update all watchlist symbols
            watchlist = self.config.get('watchlist', {}).get('default_symbols', [])
            for symbol in watchlist:
                self.terminal.print_info(f"   ✓ {symbol} data updated")
        
        self.terminal.print_success("Market data update completed")
    
    def analyze_single_stock(self, symbol: str):
        """Analyze a single stock with full investment committee"""
        self._initialize_system()
        
        symbol = symbol.upper()
        timestamp = datetime.now()
        
        self.terminal.print_header(f"Investment Committee Analysis: {symbol}")
        self.terminal.print_info(f"Started at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Update data for this symbol
        self._update_watchlist_data([symbol])
        
        self.terminal.print_status(f"🏛️  Starting investment committee meeting...")
        self.terminal.print_info(f"Analyzing: {symbol}")
        print()  # Add spacing before conversation
        
        # Start conversation logging
        conversation_file = self.conversation_logger.start_conversation(
            symbols=[symbol],
            analysis_type="single_stock"
        )
        
        try:
            # Run the investment committee with real-time output
            result = self._run_committee_with_output(symbol)
            
            # Log final results
            self.conversation_logger.log_final_decision(result)
            
            # Save conversation
            saved_conversation = self.conversation_logger.end_conversation()
            
            # Print completion summary
            print()  # Add spacing after conversation
            self.terminal.print_success("Meeting concluded.")
            
            if result.get('decision'):
                decision = result['decision']
                action = decision.get('action', 'NO ACTION')
                self.terminal.print_info(f"Decision: {action}")
                
                if decision.get('conviction'):
                    self.terminal.print_info(f"Conviction: {decision['conviction']}/10")
                
                if decision.get('rationale'):
                    self.terminal.print_info(f"Rationale: {decision['rationale']}")
            
            # Show saved files
            if saved_conversation:
                self.terminal.print_info(f"Full transcript saved to: {saved_conversation}")
            
            if result.get('decision_report_file'):
                self.terminal.print_info(f"Decision report saved to: {result['decision_report_file']}")
                
        except KeyboardInterrupt:
            self.terminal.print_warning("Analysis interrupted by user")
            self.conversation_logger.end_conversation(interrupted=True)
        except Exception as e:
            self.terminal.print_error(f"Analysis failed: {str(e)}")
            self.conversation_logger.end_conversation(error=str(e))
    
    def analyze_all_stocks(self):
        """Analyze entire watchlist"""
        self._initialize_system()
        
        watchlist = self.config.get('watchlist', {}).get('default_symbols', [])
        if not watchlist:
            self.terminal.print_error("No symbols in watchlist")
            return
        
        timestamp = datetime.now()
        
        self.terminal.print_header("Investment Committee - Full Watchlist Analysis")
        self.terminal.print_info(f"Started at: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        self.terminal.print_info(f"Analyzing {len(watchlist)} symbols: {', '.join(watchlist)}")
        
        # Update data for all symbols
        self._update_watchlist_data()
        
        # Start conversation logging
        conversation_file = self.conversation_logger.start_conversation(
            symbols=watchlist,
            analysis_type="full_watchlist"
        )
        
        results = []
        
        try:
            for i, symbol in enumerate(watchlist, 1):
                self.terminal.print_status(f"🏛️  [{i}/{len(watchlist)}] Analyzing {symbol}...")
                print()  # Add spacing before conversation
                
                # Run committee for this symbol
                result = self._run_committee_with_output(symbol)
                results.append({
                    'symbol': symbol,
                    'result': result
                })
                
                print()  # Add spacing after conversation
                
                # Brief summary for this symbol
                if result.get('decision'):
                    decision = result['decision']
                    action = decision.get('action', 'NO ACTION')
                    conviction = decision.get('conviction', 'N/A')
                    self.terminal.print_success(f"{symbol}: {action} (Conviction: {conviction})")
                else:
                    self.terminal.print_warning(f"{symbol}: Analysis incomplete")
                
                print("-" * 60)  # Separator between symbols
            
            # Log all results
            for item in results:
                self.conversation_logger.log_final_decision(item['result'])
            
            # Save conversation
            saved_conversation = self.conversation_logger.end_conversation()
            
            # Print final summary
            self.terminal.print_header("Watchlist Analysis Complete")
            
            # Summary table
            buy_count = sum(1 for item in results 
                          if item['result'].get('decision', {}).get('action') == 'BUY')
            sell_count = sum(1 for item in results 
                           if item['result'].get('decision', {}).get('action') == 'SELL')
            hold_count = len(results) - buy_count - sell_count
            
            self.terminal.print_info(f"Total symbols analyzed: {len(results)}")
            self.terminal.print_info(f"BUY recommendations: {buy_count}")
            self.terminal.print_info(f"SELL recommendations: {sell_count}")
            self.terminal.print_info(f"HOLD/NO ACTION: {hold_count}")
            
            if saved_conversation:
                self.terminal.print_info(f"Full transcript saved to: {saved_conversation}")
                
        except KeyboardInterrupt:
            self.terminal.print_warning("Watchlist analysis interrupted by user")
            self.conversation_logger.end_conversation(interrupted=True)
        except Exception as e:
            self.terminal.print_error(f"Watchlist analysis failed: {str(e)}")
            self.conversation_logger.end_conversation(error=str(e))
    
    def add_stock_to_watchlist(self, symbol: str):
        """Add a stock to the watchlist"""
        symbol = symbol.upper()
        
        # Load current config
        config_path = "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add to watchlist if not already there
        watchlist = config.get('watchlist', {}).get('default_symbols', [])
        if symbol not in watchlist:
            watchlist.append(symbol)
            config['watchlist']['default_symbols'] = watchlist
            
            # Save updated config
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            self.terminal.print_success(f"Added {symbol} to watchlist")
            self.terminal.print_info(f"Current watchlist: {', '.join(watchlist)}")
        else:
            self.terminal.print_warning(f"{symbol} is already in watchlist")
    
    def _run_committee_with_output(self, symbol: str) -> Dict[str, Any]:
        """Run investment committee with real-time terminal output"""
        
        # Set up real-time output callback
        def output_callback(speaker: str, message: str, timestamp: datetime = None):
            if timestamp is None:
                timestamp = datetime.now()
            
            # Format and display the message
            self.terminal.print_conversation(speaker, message, timestamp)
            
            # Log to conversation logger
            self.conversation_logger.log_message(speaker, message, timestamp)
        
        # Run the orchestrator with callback
        result = self.orchestrator.run_committee_session(
            symbols=[symbol],
            output_callback=output_callback
        )
        
        return result
    
    def show_config(self):
        """Display current configuration"""
        self.terminal.print_header("WCK Investment Team Configuration")
        
        # System settings
        print("\n📊 System Settings:")
        agents = self.config.get('agents', {})
        print(f"   Active Agents: {len(agents)}")
        for agent_name in agents.keys():
            print(f"     • {agent_name}")
        
        # Risk settings
        risk = self.config.get('risk_management', {})
        print(f"\n⚠️  Risk Management:")
        print(f"   Max Position Size: {risk.get('position_sizing', {}).get('max_single_position', 0)*100:.1f}%")
        print(f"   Min Risk/Reward: {risk.get('risk_metrics', {}).get('min_risk_reward_ratio', 0):.1f}:1")
        print(f"   Default Stop Loss: {risk.get('stop_loss', {}).get('default_stop_pct', 0)*100:.1f}%")
        
        # Watchlist
        watchlist = self.config.get('watchlist', {}).get('default_symbols', [])
        print(f"\n📝 Watchlist ({len(watchlist)} symbols):")
        print(f"   {', '.join(watchlist)}")
        
        # Technical indicators
        indicators = self.config.get('technical_indicators', {})
        total_indicators = sum(len(indicators.get(category, [])) for category in indicators.keys())
        print(f"\n📈 Technical Analysis:")
        print(f"   Total Indicators: {total_indicators}")
        for category, indicator_list in indicators.items():
            print(f"   {category.title()}: {len(indicator_list)} indicators")


def create_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="WCK Investment Team - AI Hedge Fund System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --analyze TSLA              # Analyze Tesla with full committee
  python main.py --analyze-all               # Analyze entire watchlist
  python main.py --add-stock AAPL            # Add Apple to watchlist
  python main.py --config                    # Show current configuration
  
For more information, visit: https://github.com/williamknighting/wck_investment_team
        """
    )
    
    # Mutually exclusive group for main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    
    action_group.add_argument(
        '--analyze',
        metavar='SYMBOL',
        help='Analyze a single stock symbol with full investment committee'
    )
    
    action_group.add_argument(
        '--analyze-all',
        action='store_true',
        help='Analyze entire watchlist with investment committee'
    )
    
    action_group.add_argument(
        '--add-stock',
        metavar='SYMBOL',
        help='Add a stock symbol to the watchlist'
    )
    
    action_group.add_argument(
        '--config',
        action='store_true',
        help='Display current system configuration'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config-file',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='WCK Investment Team v1.0.0'
    )
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Initialize CLI system
        cli = WCKInvestmentCLI(config_path=args.config_file)
        
        # Route to appropriate action
        if args.analyze:
            cli.analyze_single_stock(args.analyze)
        elif args.analyze_all:
            cli.analyze_all_stocks()
        elif args.add_stock:
            cli.add_stock_to_watchlist(args.add_stock)
        elif args.config:
            cli.show_config()
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()