"""
Conversation Logger
Manages logging of investment committee conversations to markdown files
"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

class ConversationLogger:
    """
    Handles logging of investment committee conversations to markdown files
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize conversation logger with configuration"""
        self.config = config
        self.file_system = config.get('file_system', {})
        self.conversation_dir = self.file_system.get('directories', {}).get('conversations', 'conversations')
        self.decisions_dir = self.file_system.get('directories', {}).get('decisions', 'decisions')
        
        # Ensure directories exist
        Path(self.conversation_dir).mkdir(exist_ok=True)
        Path(self.decisions_dir).mkdir(exist_ok=True)
        
        # Current session state
        self.current_session = None
        self.conversation_log = []
        self.session_start_time = None
        self.session_symbols = []
        self.analysis_type = None
        
    def start_conversation(self, symbols: List[str], analysis_type: str = "single_stock") -> str:
        """
        Start a new conversation session
        
        Args:
            symbols: List of symbols being analyzed
            analysis_type: Type of analysis (single_stock, full_watchlist)
            
        Returns:
            Filename that will be used for the conversation log
        """
        self.session_start_time = datetime.now()
        self.session_symbols = symbols
        self.analysis_type = analysis_type
        self.conversation_log = []
        
        # Generate filename
        timestamp = self.session_start_time.strftime("%Y-%m-%d-%H-%M-%S")
        
        if analysis_type == "single_stock" and len(symbols) == 1:
            filename = f"{timestamp}-{symbols[0]}.md"
        else:
            filename = f"{timestamp}-ALL.md"
        
        self.current_session = os.path.join(self.conversation_dir, filename)
        
        return self.current_session
    
    def log_message(self, speaker: str, message: str, timestamp: Optional[datetime] = None):
        """
        Log a conversation message
        
        Args:
            speaker: Name of the agent speaking
            message: The message content
            timestamp: When the message was sent (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.conversation_log.append({
            'timestamp': timestamp,
            'speaker': speaker,
            'message': message
        })
    
    def log_final_decision(self, decision_result: Dict[str, Any]):
        """
        Log the final investment decision
        
        Args:
            decision_result: The complete decision result from the committee
        """
        if not hasattr(self, 'final_decisions'):
            self.final_decisions = []
        
        self.final_decisions.append(decision_result)
    
    def end_conversation(self, interrupted: bool = False, error: str = None) -> Optional[str]:
        """
        End the conversation session and save to markdown file
        
        Args:
            interrupted: Whether the conversation was interrupted
            error: Error message if conversation failed
            
        Returns:
            Path to the saved conversation file, or None if save failed
        """
        if not self.current_session or not self.session_start_time:
            return None
        
        try:
            # Generate markdown content
            markdown_content = self._generate_markdown_content(interrupted, error)
            
            # Write to file
            with open(self.current_session, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # Reset session state
            saved_file = self.current_session
            self._reset_session()
            
            return saved_file
            
        except Exception as e:
            print(f"Error saving conversation log: {e}")
            return None
    
    def _generate_markdown_content(self, interrupted: bool = False, error: str = None) -> str:
        """Generate markdown content for the conversation log"""
        
        # Header information
        session_end_time = datetime.now()
        duration = session_end_time - self.session_start_time
        
        # Start building markdown
        lines = []
        lines.append(f"# Investment Committee Meeting - {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Meeting information
        if len(self.session_symbols) == 1:
            lines.append(f"## Stock(s) Analyzed: {self.session_symbols[0]}")
        else:
            lines.append(f"## Stock(s) Analyzed: {', '.join(self.session_symbols)}")
        
        # Participants (extract from conversation log)
        participants = set()
        for entry in self.conversation_log:
            participants.add(entry['speaker'])
        
        lines.append(f"## Participants: {', '.join(sorted(participants))}")
        lines.append("")
        
        # Session metadata
        lines.append("### Session Information")
        lines.append(f"- **Start Time**: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **End Time**: {session_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Duration**: {self._format_duration(duration)}")
        lines.append(f"- **Analysis Type**: {self.analysis_type}")
        lines.append(f"- **Total Messages**: {len(self.conversation_log)}")
        
        if interrupted:
            lines.append(f"- **Status**: ⚠️ INTERRUPTED BY USER")
        elif error:
            lines.append(f"- **Status**: ❌ ERROR - {error}")
        else:
            lines.append(f"- **Status**: ✅ COMPLETED")
        
        lines.append("")
        
        # Data status section (if available)
        lines.append("### Data Status")
        for symbol in self.session_symbols:
            lines.append(f"- {symbol}: Data current as of session start")
        lines.append("")
        
        # Conversation transcript
        lines.append("### Conversation")
        lines.append("")
        
        if not self.conversation_log:
            lines.append("*No conversation messages recorded*")
        else:
            for entry in self.conversation_log:
                timestamp_str = entry['timestamp'].strftime('%H:%M:%S')
                speaker = entry['speaker'].upper().replace('_', ' ')
                message = entry['message']
                
                lines.append(f"**[{timestamp_str}] {speaker}:** {message}")
                lines.append("")
        
        # Proposals reviewed section
        lines.append("### Proposals Reviewed")
        lines.append("")
        
        # Look for proposal files that might have been generated
        proposal_files = self._find_related_proposals()
        if proposal_files:
            for proposal_file in proposal_files:
                relative_path = os.path.relpath(proposal_file, self.conversation_dir)
                lines.append(f"- [{os.path.basename(proposal_file)}](../{relative_path})")
        else:
            lines.append("*No proposal files found*")
        
        lines.append("")
        
        # Final decision section
        lines.append("### Final Decision")
        lines.append("")
        
        if hasattr(self, 'final_decisions') and self.final_decisions:
            for decision_result in self.final_decisions:
                if decision_result.get('decision'):
                    decision = decision_result['decision']
                    
                    # Decision summary
                    action = decision.get('action', 'NO ACTION')
                    symbol = decision.get('symbol', 'Unknown')
                    conviction = decision.get('conviction', 'N/A')
                    
                    lines.append(f"**{symbol}: {action}**")
                    if conviction != 'N/A':
                        lines.append(f"- Conviction: {conviction}/10")
                    
                    if decision.get('position_size'):
                        lines.append(f"- Position Size: {decision['position_size']} shares")
                    
                    if decision.get('entry_price'):
                        lines.append(f"- Entry Price: ${decision['entry_price']:.2f}")
                    
                    if decision.get('stop_loss'):
                        lines.append(f"- Stop Loss: ${decision['stop_loss']:.2f}")
                    
                    if decision.get('profit_target'):
                        lines.append(f"- Target Price: ${decision['profit_target']:.2f}")
                    
                    if decision.get('rationale'):
                        lines.append(f"- Rationale: {decision['rationale']}")
                    
                    lines.append("")
                else:
                    lines.append("*Decision details not available*")
                    lines.append("")
        else:
            if interrupted:
                lines.append("*Meeting was interrupted before a final decision could be reached*")
            elif error:
                lines.append(f"*Meeting ended with error: {error}*")
            else:
                lines.append("*No final decision recorded*")
        
        lines.append("")
        
        # Execution instructions (if decision was made)
        if hasattr(self, 'final_decisions') and self.final_decisions:
            lines.append("### Execution Instructions")
            lines.append("")
            
            for decision_result in self.final_decisions:
                if decision_result.get('decision'):
                    decision = decision_result['decision']
                    action = decision.get('action', 'NO ACTION')
                    
                    if action in ['BUY', 'SELL']:
                        lines.append(f"**{action} ORDER INSTRUCTIONS:**")
                        
                        if decision.get('symbol'):
                            lines.append(f"- Symbol: {decision['symbol']}")
                        
                        if decision.get('position_size'):
                            lines.append(f"- Quantity: {decision['position_size']} shares")
                        
                        if decision.get('entry_price'):
                            lines.append(f"- Target Entry: ${decision['entry_price']:.2f}")
                        
                        if decision.get('stop_loss'):
                            lines.append(f"- Stop Loss: ${decision['stop_loss']:.2f}")
                        
                        lines.append("- Execution: Market hours only")
                        lines.append("- Set stop loss immediately upon fill")
                        lines.append("- Report execution status within 1 hour")
                        lines.append("")
                    else:
                        lines.append("**NO IMMEDIATE ACTION REQUIRED**")
                        lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"**Report Generated**: {session_end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("**Authority**: WCK Investment Team AI Committee")
        
        if interrupted:
            lines.append("**Status**: MEETING INTERRUPTED")
        elif error:
            lines.append("**Status**: MEETING ERROR")
        else:
            lines.append("**Status**: FINAL DECISION")
        
        return '\n'.join(lines)
    
    def _format_duration(self, duration) -> str:
        """Format duration as human-readable string"""
        total_seconds = int(duration.total_seconds())
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _find_related_proposals(self) -> List[str]:
        """Find proposal files that might be related to this session"""
        proposal_files = []
        
        # Look for proposal files generated around the session time
        proposals_dir = self.file_system.get('directories', {}).get('proposals', 'proposals')
        
        if os.path.exists(proposals_dir):
            # Get session date for filtering
            session_date = self.session_start_time.strftime("%Y-%m-%d")
            
            for filename in os.listdir(proposals_dir):
                if filename.endswith('.md') and session_date in filename:
                    # Check if any of our symbols are in the filename
                    for symbol in self.session_symbols:
                        if symbol.upper() in filename.upper():
                            proposal_files.append(os.path.join(proposals_dir, filename))
                            break
        
        return proposal_files
    
    def _reset_session(self):
        """Reset session state"""
        self.current_session = None
        self.conversation_log = []
        self.session_start_time = None
        self.session_symbols = []
        self.analysis_type = None
        
        if hasattr(self, 'final_decisions'):
            delattr(self, 'final_decisions')
    
    def get_conversation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.conversation_log[-limit:] if limit else self.conversation_log
    
    def save_decision_report(self, decision_result: Dict[str, Any], symbol: str) -> Optional[str]:
        """
        Save a separate decision report file
        
        Args:
            decision_result: The decision result to save
            symbol: The symbol this decision relates to
            
        Returns:
            Path to saved decision report file
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = f"decision-report-{timestamp}-{symbol}.md"
            filepath = os.path.join(self.decisions_dir, filename)
            
            # Generate decision report content
            content = self._generate_decision_report(decision_result, symbol)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return filepath
            
        except Exception as e:
            print(f"Error saving decision report: {e}")
            return None
    
    def _generate_decision_report(self, decision_result: Dict[str, Any], symbol: str) -> str:
        """Generate decision report markdown content"""
        timestamp = datetime.now()
        
        lines = []
        lines.append(f"# Investment Decision Report")
        lines.append("")
        lines.append(f"**Symbol**: {symbol}")
        lines.append(f"**Date/Time**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append(f"**Decision ID**: {timestamp.strftime('%Y%m%d_%H%M%S')}_{symbol}")
        lines.append("")
        
        if decision_result.get('decision'):
            decision = decision_result['decision']
            
            lines.append("## Executive Summary")
            lines.append("")
            
            action = decision.get('action', 'NO ACTION')
            lines.append(f"**Recommendation**: {action}")
            
            if decision.get('conviction'):
                lines.append(f"**Conviction Level**: {decision['conviction']}/10")
            
            if decision.get('rationale'):
                lines.append(f"**Rationale**: {decision['rationale']}")
            
            lines.append("")
            
            # Trade details
            if action in ['BUY', 'SELL']:
                lines.append("## Trade Details")
                lines.append("")
                
                if decision.get('position_size'):
                    lines.append(f"- **Position Size**: {decision['position_size']} shares")
                
                if decision.get('entry_price'):
                    lines.append(f"- **Entry Price**: ${decision['entry_price']:.2f}")
                
                if decision.get('stop_loss'):
                    lines.append(f"- **Stop Loss**: ${decision['stop_loss']:.2f}")
                
                if decision.get('profit_target'):
                    lines.append(f"- **Profit Target**: ${decision['profit_target']:.2f}")
                
                # Calculate risk/reward if possible
                if decision.get('entry_price') and decision.get('stop_loss') and decision.get('profit_target'):
                    entry = decision['entry_price']
                    stop = decision['stop_loss']
                    target = decision['profit_target']
                    
                    if action == 'BUY':
                        risk = entry - stop
                        reward = target - entry
                    else:  # SELL
                        risk = stop - entry
                        reward = entry - target
                    
                    if risk > 0:
                        risk_reward_ratio = reward / risk
                        lines.append(f"- **Risk/Reward Ratio**: {risk_reward_ratio:.2f}:1")
                
                lines.append("")
        
        # Committee analysis section
        lines.append("## Committee Analysis")
        lines.append("")
        
        if hasattr(self, 'conversation_log') and self.conversation_log:
            # Summarize key points from conversation
            lines.append("### Key Discussion Points")
            lines.append("")
            
            # Extract key messages (this is simplified - could be enhanced)
            key_speakers = ['director', 'risk_manager']
            for entry in self.conversation_log:
                if entry['speaker'].lower() in key_speakers:
                    speaker = entry['speaker'].upper().replace('_', ' ')
                    message = entry['message'][:200]  # Truncate long messages
                    if len(entry['message']) > 200:
                        message += "..."
                    lines.append(f"- **{speaker}**: {message}")
            
            lines.append("")
        
        # Risk assessment
        lines.append("## Risk Assessment")
        lines.append("")
        lines.append("*Risk assessment completed by committee risk manager*")
        lines.append("")
        
        # Execution instructions
        if decision_result.get('decision', {}).get('action') in ['BUY', 'SELL']:
            lines.append("## Execution Instructions")
            lines.append("")
            lines.append("1. Execute during regular market hours only")
            lines.append("2. Set stop loss order immediately upon fill")
            lines.append("3. Monitor position and report any significant moves")
            lines.append("4. Review position at next committee meeting")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"**Report Generated**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("**Authority**: WCK Investment Team AI Committee")
        lines.append("**Status**: FINAL DECISION")
        
        return '\n'.join(lines)