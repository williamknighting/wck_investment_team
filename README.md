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

### 2. Run Investment Committee Meeting

**Main Command - Run Full Committee Session:**
```bash
python investment_committee_orchestrator.py
```

This single command runs the complete investment committee workflow:
- Initializes all agents
- Runs market research and technical analysis  
- Generates strategy proposals in parallel
- Conducts structured committee debate
- Makes final trading decisions
- Generates comprehensive reports

### 3. Review Results

After running, check these folders for outputs:
- **`decisions/`**: Final trading decisions and rationale
- **`conversations/`**: Complete committee meeting transcripts  
- **`proposals/`**: Individual strategy agent proposals

## 📁 Directory Structure

```
wck_investment_team/
├── investment_committee_orchestrator.py  # 🎯 MAIN SCRIPT
├── agents/                               # AI Agent implementations
│   ├── director.py                       # Investment committee director
│   ├── risk_manager.py                   # Portfolio risk management
│   ├── qullamaggie_agent.py             # Momentum strategy agent
│   ├── technical_analyst.py             # Technical analysis specialist
│   └── base_*.py                        # Base classes and utilities
├── config/                              # Configuration files
│   └── committee_config.json            # Agent behavior settings
├── decisions/                           # Final trading decisions (output)
├── conversations/                       # Meeting transcripts (output)  
├── proposals/                           # Strategy proposals (output)
├── src/                                # Core system components
├── scripts/                            # Utility scripts
├── tests/                              # Test files (development)
└── data_store/                         # Market data storage
```

## ⚙️ Configuration

### Committee Settings (`config/committee_config.json`)

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