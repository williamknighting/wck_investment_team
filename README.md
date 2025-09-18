# WCK Investment Team - AI Hedge Fund System

A sophisticated AI-powered investment committee system that orchestrates multiple specialized agents to make collaborative trading decisions through structured debate and analysis.

## 🏛️ System Overview

The WCK Investment Team simulates a real hedge fund investment committee with AI agents that:
- Analyze market data and generate technical metrics
- Create trading proposals based on different strategies  
- Engage in structured debate about trade ideas
- Assess portfolio risk and challenge aggressive proposals
- Make final trading decisions with comprehensive documentation

## 🤖 AI Agents

### Director Agent (`director`)
- **Role**: Investment Committee Chair with final decision authority
- **Personality**: Decisive, thorough, direct, and skeptical
- **Responsibilities**: Orchestrates meetings, asks pointed questions, resolves conflicts, makes final trading decisions

### Risk Manager (`risk_manager`) 
- **Role**: Conservative portfolio risk oversight
- **Personality**: Conservative, cautious, detail-oriented
- **Responsibilities**: Evaluates concentration risk, vetoes dangerous trades, suggests position adjustments

### Strategy Agents
- **QullamaggieAgent (`qullamaggie_agent`)**: Momentum-based trading strategy
- **Future agents**: Value, technical breakout, etc.

### Technical Analyst (`technical_analyst`)
- **Role**: Quantitative market analysis specialist
- **Responsibilities**: Calculates technical indicators, momentum metrics, volume analysis

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd wck_investment_team

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY required)
```

### 2. Command Line Interface

**WCK Investment Team provides a comprehensive CLI for all operations:**

```bash
# Analyze a single stock with full investment committee
python main.py --analyze TSLA

# Analyze entire watchlist
python main.py --analyze-all

# Add a stock to the watchlist
python main.py --add-stock AAPL

# View current configuration
python main.py --config

# Get help
python main.py --help
```

**Real-Time Terminal Output:**
Watch the investment committee conversation unfold in real-time:
```
[14:30:15] DIRECTOR           : Good afternoon team. We're reviewing proposals for TSLA.
[14:30:18] TECHNICAL_ANALYST  : Current TSLA metrics: RSI: 67.5, Volume: 15% above average
[14:30:22] QULLAMAGGIE_AGENT  : I'm seeing an episodic pivot setup here! Textbook momentum...
[14:30:25] RISK_MANAGER       : Let's discuss the risk/reward. Position size seems aggressive...
```

**Alternative - Direct Orchestrator:**
```bash
python investment_committee_orchestrator.py
```

### 3. Review Results

After running, check these folders for outputs:
- **`conversations/`**: Complete committee meeting transcripts (markdown format)
- **`decisions/`**: Final trading decisions and rationale
- **`proposals/`**: Individual strategy agent proposals

**Sample Output Files:**
```
conversations/2024-01-15-14-30-00-TSLA.md    # Full meeting transcript
decisions/decision-report-2024-01-15-TSLA.md  # Decision summary
proposals/qullamaggie-TSLA-proposal.md        # Strategy proposals
```

## 📁 Directory Structure

```
wck_investment_team/
├── main.py                              # 🎯 CLI ENTRY POINT
├── config.yaml                          # 📋 Main configuration file  
├── investment_committee_orchestrator.py # Core orchestrator
├── requirements.txt                     # Python dependencies
├── agents/                              # AI Agent implementations
│   ├── director.py                      # Investment committee director
│   ├── risk_manager.py                  # Portfolio risk management
│   ├── qullamaggie_agent.py            # Momentum strategy agent
│   ├── technical_analyst.py            # Technical analysis specialist
│   └── base_*.py                       # Base classes and utilities
├── src/utils/                          # System utilities
│   ├── terminal_output.py              # Real-time terminal display
│   ├── conversation_logger.py          # Markdown conversation logging
│   └── logging_config.py               # Logging configuration
├── conversations/                      # Meeting transcripts (output)
├── decisions/                         # Final trading decisions (output)
├── proposals/                         # Strategy proposals (output)
├── src/                              # Core system components
├── tests/                            # Test files (development)
└── data_store/                       # Market data storage
```

## ⚙️ Configuration

### Main Configuration (`config.yaml`)

The system uses a comprehensive YAML configuration file:

```yaml
# Agent personalities and prompts
agents:
  director:
    name: "Investment Committee Director"
    personality: "decisive, thorough, skeptical"
    prompt: "You are the Investment Committee Director..."
  
  risk_manager:
    personality: "conservative, detail-oriented"
    # ... full prompts and settings

# Risk management parameters
risk_management:
  position_sizing:
    max_single_position: 0.05  # 5% max per position
    min_risk_reward_ratio: 2.0  # 2:1 minimum
  
# Technical indicators
technical_indicators:
  momentum: [RSI_14, MACD, ADX_14]
  trend: [EMA_8, EMA_20, SMA_200]
  # ... complete indicator list

# Watchlist management  
watchlist:
  default_symbols: [SPY, QQQ, TSLA, AAPL, NVDA]
  auto_update: true
```

### Legacy Configuration (`config/committee_config.json`)

```json
{
  "agents": {
    "director": {
      "enabled": true,
      "max_conversation_rounds": 10,
      "early_termination_threshold": 8
    },
    "risk_manager": {
      "enabled": true,
      "auto_veto_threshold": 8.0,
      "challenge_threshold": 6.0
    }
  },
  "conversation": {
    "max_turns": 20,
    "allow_interruptions": true
  }
}
```

### Agent Behavior Tuning

Edit `config/committee_config.json` to adjust:
- **Risk thresholds**: How conservative the risk manager should be
- **Conversation limits**: Maximum debate rounds
- **Agent personalities**: Decision-making parameters

## 📊 Understanding the Output

### Decision Reports (`decisions/decision_report_YYYYMMDD_HHMMSS.md`)

Contains:
- **Market Context**: Current market conditions
- **Proposals Reviewed**: All strategy proposals evaluated
- **Final Decisions**: Approved/rejected/adjusted trades with rationale
- **Execution Instructions**: Clear trading instructions

### Conversation Logs (`conversations/committee_session_YYYYMMDD_HHMMSS.json`)

Contains:
- **Complete Transcript**: Every conversation turn with timestamps
- **Agent Interactions**: Who said what and when
- **Decision Process**: How final decisions were reached
- **Performance Metrics**: Session duration and statistics

### Sample Output

```
📊 SESSION SUMMARY
Session ID: committee_20250918_125949
Duration: 2.6 seconds
Participants: director, risk_manager, qullamaggie_agent, technical_analyst
Proposals: 2 generated + 3 reviewed
Final Decision: 0 approved, 2 rejected, 1 adjusted
```

## 🔧 Advanced Usage

### Individual Agent Testing

Test specific agents in isolation:
```bash
python tests/test_director.py           # Test director functionality
python tests/test_risk_manager.py       # Test risk management
python tests/test_strategy_agent.py     # Test strategy agents
python tests/test_technical_analyst.py  # Test technical analysis
```

### Custom Strategies

Add new strategy agents by:
1. Inheriting from `StrategyAgent` base class
2. Implementing required methods (`_evaluate_setup`, `_get_strategy_reasoning`)
3. Adding to orchestrator configuration

### Market Data

The system uses DuckDB for market data storage:
- **Location**: `data_store/market_data.duckdb`
- **Format**: OHLCV data with technical indicators
- **Update**: Use scripts in `scripts/` folder

## 🎯 Key Features

### Real Investment Committee Experience
- **Turn-based debate**: Structured conversation rounds
- **Pointed questioning**: Director challenges weak proposals  
- **Conflict resolution**: Authoritative decision-making
- **Conservative oversight**: Risk manager vetoes dangerous trades

### Comprehensive Risk Management
- **Position size limits**: Maximum exposure controls
- **Risk/reward requirements**: Minimum 2:1 ratios enforced
- **Portfolio concentration**: Diversification requirements
- **Market regime awareness**: Risk adjustments based on conditions

### Production-Ready Architecture
- **Error handling**: Retry logic and graceful degradation
- **Parallel processing**: Concurrent proposal generation
- **Audit trails**: Complete decision documentation
- **Configuration-driven**: Easy behavior modification

## 🔍 Troubleshooting

### Common Issues

**"No proposals found"**
- Run individual strategy agents to generate proposals first
- Check `proposals/` folder exists and has .md files

**"Agent initialization failed"**  
- Verify OPENAI_API_KEY is set in .env file
- Check all dependencies are installed correctly

**"Technical analysis errors"**
- Ensure market data is available in `data_store/`
- Run data update scripts if needed

### Debug Mode

Enable detailed logging by setting environment variable:
```bash
export DEBUG=1
python investment_committee_orchestrator.py
```

## 📈 Example Session Flow

1. **Research Phase**: Market data updates and analysis
2. **Technical Analysis**: Comprehensive metrics for key symbols
3. **Proposal Generation**: Strategy agents create trade ideas (parallel)
4. **Director Preparation**: Review all proposals before meeting
5. **Committee Debate**: Structured conversation with challenges
6. **Final Decision**: Director makes binding trading decisions
7. **Documentation**: Complete audit trail and execution instructions

## 🛠️ Development

### Adding New Agents
1. Create new agent class inheriting from appropriate base class
2. Implement required abstract methods
3. Add to orchestrator initialization
4. Update configuration file

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_<agent_name>.py
```

## 📊 Current System Status

**🤖 Multi-Agent System**: ✅ Operational with AutoGen framework  
**📊 Technical Analysis**: ✅ DuckDB-powered with comprehensive metrics  
**🎯 Strategy Implementation**: ✅ Qullamaggie momentum strategy  
**🏛️ Investment Committee**: ✅ Realistic hedge fund decision process  
**📈 Data Infrastructure**: ✅ High-performance DuckDB storage  
**🔗 Risk Management**: ✅ Conservative portfolio oversight  

## 📝 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and always consult with financial professionals before making investment decisions.

---

**🎯 To run the system: `python investment_committee_orchestrator.py`**

This single command orchestrates the complete AI hedge fund investment committee process from market analysis to final trading decisions.