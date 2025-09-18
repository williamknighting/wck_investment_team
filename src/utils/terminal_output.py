"""
Terminal Output Utility
Provides formatted terminal output for the investment committee system
"""

import sys
from datetime import datetime
from typing import Dict, Any, Optional
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform color support
init(autoreset=True)

class TerminalOutput:
    """
    Handles formatted terminal output for the investment committee system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize terminal output with configuration"""
        self.config = config
        self.conversation_config = config.get('conversation', {}).get('settings', {})
        self.timestamp_format = self.conversation_config.get('timestamp_format', '%H:%M:%S')
        self.agent_name_width = self.conversation_config.get('agent_name_width', 20)
        self.real_time_output = self.conversation_config.get('real_time_output', True)
        
        # Color scheme
        self.colors = {
            'header': Fore.CYAN + Style.BRIGHT,
            'success': Fore.GREEN + Style.BRIGHT,
            'warning': Fore.YELLOW + Style.BRIGHT,
            'error': Fore.RED + Style.BRIGHT,
            'info': Fore.BLUE,
            'status': Fore.MAGENTA,
            'director': Fore.RED + Style.BRIGHT,
            'risk_manager': Fore.YELLOW + Style.BRIGHT,
            'qullamaggie_agent': Fore.GREEN + Style.BRIGHT,
            'technical_analyst': Fore.CYAN + Style.BRIGHT,
            'default': Fore.WHITE
        }
    
    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{self.colors['header']}{'=' * 60}")
        print(f"{text.center(60)}")
        print(f"{'=' * 60}{Style.RESET_ALL}\n")
    
    def print_success(self, text: str):
        """Print a success message"""
        print(f"{self.colors['success']}✅ {text}{Style.RESET_ALL}")
    
    def print_warning(self, text: str):
        """Print a warning message"""
        print(f"{self.colors['warning']}⚠️  {text}{Style.RESET_ALL}")
    
    def print_error(self, text: str):
        """Print an error message"""
        print(f"{self.colors['error']}❌ {text}{Style.RESET_ALL}")
    
    def print_info(self, text: str):
        """Print an info message"""
        print(f"{self.colors['info']}ℹ️  {text}{Style.RESET_ALL}")
    
    def print_status(self, text: str):
        """Print a status message"""
        print(f"{self.colors['status']}🔄 {text}{Style.RESET_ALL}")
    
    def print_conversation(self, speaker: str, message: str, timestamp: Optional[datetime] = None):
        """
        Print a conversation message with formatted speaker and timestamp
        
        Args:
            speaker: Name of the agent speaking
            message: The message content
            timestamp: When the message was sent
        """
        if not self.real_time_output:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format timestamp
        time_str = timestamp.strftime(self.timestamp_format)
        
        # Get speaker color
        speaker_color = self.colors.get(speaker.lower(), self.colors['default'])
        
        # Format speaker name with consistent width
        speaker_display = speaker.upper().replace('_', ' ')
        speaker_formatted = f"{speaker_display:<{self.agent_name_width}}"
        
        # Format and print the message
        print(f"[{time_str}] {speaker_color}{speaker_formatted}{Style.RESET_ALL}: {message}")
        
        # Flush output to ensure real-time display
        sys.stdout.flush()
    
    def print_separator(self, char: str = "-", length: int = 60):
        """Print a separator line"""
        print(char * length)
    
    def print_phase_header(self, phase_name: str, phase_number: int = None):
        """Print a phase header for workflow steps"""
        if phase_number:
            header = f"Phase {phase_number}: {phase_name}"
        else:
            header = phase_name
        
        print(f"\n{self.colors['status']}{'=' * 40}")
        print(f"{header}")
        print(f"{'=' * 40}{Style.RESET_ALL}")
    
    def print_agent_status(self, agent_name: str, status: str, details: str = ""):
        """Print agent status update"""
        agent_color = self.colors.get(agent_name.lower(), self.colors['default'])
        
        if details:
            print(f"{agent_color}{agent_name.upper()}{Style.RESET_ALL}: {status} - {details}")
        else:
            print(f"{agent_color}{agent_name.upper()}{Style.RESET_ALL}: {status}")
    
    def print_decision_summary(self, decision: Dict[str, Any]):
        """Print a formatted decision summary"""
        print(f"\n{self.colors['header']}📊 DECISION SUMMARY{Style.RESET_ALL}")
        print(f"{self.colors['status']}{'─' * 40}{Style.RESET_ALL}")
        
        action = decision.get('action', 'NO ACTION')
        if action == 'BUY':
            action_color = self.colors['success']
        elif action == 'SELL':
            action_color = self.colors['error']
        else:
            action_color = self.colors['warning']
        
        print(f"Action: {action_color}{action}{Style.RESET_ALL}")
        
        if decision.get('symbol'):
            print(f"Symbol: {decision['symbol']}")
        
        if decision.get('conviction'):
            conviction = decision['conviction']
            if conviction >= 8:
                conviction_color = self.colors['success']
            elif conviction >= 6:
                conviction_color = self.colors['warning']
            else:
                conviction_color = self.colors['error']
            
            print(f"Conviction: {conviction_color}{conviction}/10{Style.RESET_ALL}")
        
        if decision.get('position_size'):
            print(f"Position Size: {decision['position_size']} shares")
        
        if decision.get('entry_price'):
            print(f"Entry Price: ${decision['entry_price']:.2f}")
        
        if decision.get('stop_loss'):
            print(f"Stop Loss: ${decision['stop_loss']:.2f}")
        
        if decision.get('profit_target'):
            print(f"Profit Target: ${decision['profit_target']:.2f}")
        
        if decision.get('rationale'):
            print(f"\nRationale: {decision['rationale']}")
        
        print(f"{self.colors['status']}{'─' * 40}{Style.RESET_ALL}")
    
    def print_portfolio_summary(self, portfolio: Dict[str, Any]):
        """Print portfolio status summary"""
        print(f"\n{self.colors['header']}💼 PORTFOLIO STATUS{Style.RESET_ALL}")
        print(f"{self.colors['status']}{'─' * 40}{Style.RESET_ALL}")
        
        if portfolio.get('total_value'):
            print(f"Total Value: ${portfolio['total_value']:,.2f}")
        
        if portfolio.get('cash_available'):
            print(f"Cash Available: ${portfolio['cash_available']:,.2f}")
        
        if portfolio.get('active_positions'):
            print(f"Active Positions: {portfolio['active_positions']}")
        
        if portfolio.get('daily_pnl'):
            pnl = portfolio['daily_pnl']
            pnl_color = self.colors['success'] if pnl >= 0 else self.colors['error']
            print(f"Daily P&L: {pnl_color}${pnl:,.2f}{Style.RESET_ALL}")
        
        print(f"{self.colors['status']}{'─' * 40}{Style.RESET_ALL}")
    
    def print_watchlist_summary(self, symbols: list, analysis_type: str = "single"):
        """Print watchlist analysis summary"""
        symbol_list = ", ".join(symbols)
        
        if analysis_type == "single":
            print(f"\n{self.colors['info']}📈 Analyzing: {symbol_list}{Style.RESET_ALL}")
        else:
            print(f"\n{self.colors['info']}📈 Watchlist Analysis: {len(symbols)} symbols{Style.RESET_ALL}")
            print(f"{self.colors['info']}   Symbols: {symbol_list}{Style.RESET_ALL}")
    
    def print_progress_bar(self, current: int, total: int, prefix: str = "", suffix: str = ""):
        """Print a simple progress bar"""
        if total == 0:
            return
        
        progress = current / total
        bar_length = 30
        filled_length = int(bar_length * progress)
        
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        percentage = f"{progress * 100:.1f}%"
        
        print(f"\r{prefix} [{bar}] {percentage} {suffix}", end='', flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def clear_line(self):
        """Clear the current terminal line"""
        print('\r' + ' ' * 80 + '\r', end='', flush=True)
    
    def print_data_freshness_status(self, freshness_data: Dict[str, Any]):
        """Print data freshness status"""
        print(f"\n{self.colors['header']}📊 DATA FRESHNESS STATUS{Style.RESET_ALL}")
        print(f"{self.colors['status']}{'─' * 50}{Style.RESET_ALL}")
        
        if freshness_data.get('fresh_symbols'):
            fresh_count = len(freshness_data['fresh_symbols'])
            print(f"{self.colors['success']}✅ Fresh data: {fresh_count} symbols{Style.RESET_ALL}")
        
        if freshness_data.get('stale_symbols'):
            stale_count = len(freshness_data['stale_symbols'])
            print(f"{self.colors['warning']}⚠️  Stale data: {stale_count} symbols{Style.RESET_ALL}")
            
            for symbol_info in freshness_data['stale_symbols']:
                symbol = symbol_info.get('symbol', 'Unknown')
                age = symbol_info.get('age_hours', 0)
                print(f"   {symbol}: {age:.1f} hours old")
        
        if freshness_data.get('missing_symbols'):
            missing_count = len(freshness_data['missing_symbols'])
            print(f"{self.colors['error']}❌ Missing data: {missing_count} symbols{Style.RESET_ALL}")
            
            for symbol in freshness_data['missing_symbols']:
                print(f"   {symbol}: No data available")
        
        print(f"{self.colors['status']}{'─' * 50}{Style.RESET_ALL}")
    
    def print_error_details(self, error: Exception, context: str = ""):
        """Print detailed error information"""
        print(f"\n{self.colors['error']}❌ ERROR{Style.RESET_ALL}")
        
        if context:
            print(f"{self.colors['error']}Context: {context}{Style.RESET_ALL}")
        
        print(f"{self.colors['error']}Type: {type(error).__name__}{Style.RESET_ALL}")
        print(f"{self.colors['error']}Message: {str(error)}{Style.RESET_ALL}")
    
    def print_system_status(self, status: Dict[str, Any]):
        """Print overall system status"""
        print(f"\n{self.colors['header']}🖥️  SYSTEM STATUS{Style.RESET_ALL}")
        print(f"{self.colors['status']}{'─' * 40}{Style.RESET_ALL}")
        
        if status.get('agents_initialized'):
            agent_count = len(status['agents_initialized'])
            print(f"{self.colors['success']}✅ Agents: {agent_count} initialized{Style.RESET_ALL}")
        
        if status.get('database_connected'):
            print(f"{self.colors['success']}✅ Database: Connected{Style.RESET_ALL}")
        else:
            print(f"{self.colors['error']}❌ Database: Not connected{Style.RESET_ALL}")
        
        if status.get('api_key_valid'):
            print(f"{self.colors['success']}✅ API Key: Valid{Style.RESET_ALL}")
        else:
            print(f"{self.colors['error']}❌ API Key: Invalid or missing{Style.RESET_ALL}")
        
        if status.get('config_loaded'):
            print(f"{self.colors['success']}✅ Configuration: Loaded{Style.RESET_ALL}")
        
        print(f"{self.colors['status']}{'─' * 40}{Style.RESET_ALL}")