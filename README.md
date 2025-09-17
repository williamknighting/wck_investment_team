# AI Hedge Fund Multi-Agent System

A sophisticated multi-agent trading system that simulates a real hedge fund investment committee. The system uses AutoGen framework to orchestrate AI agents specialized in different trading strategies, with the Fund Head Agent making final investment decisions based on collaborative analysis.

## 🎯 Overview

This system implements a realistic hedge fund structure where specialized strategy agents analyze markets and debate trading decisions, moderated by a Fund Head Agent who makes final investment choices. The system focuses on momentum strategies, particularly the Qullamaggie approach, with comprehensive technical analysis and risk management.

**Current Status**: 
- ✅ Complete multi-agent system with AutoGen 0.7.4 integration
- ✅ Fund Head Agent as investment committee leader and decision maker
- ✅ Qullamaggie strategy agent with 3 momentum setups
- ✅ Comprehensive technical analyst using DuckDB (no API abuse)
- ✅ 2-year SPY/QQQ historical data in high-performance DuckDB
- ✅ Alpaca Markets integration for paper trading
- ✅ Clean hedge fund organizational structure

## 🏗️ Multi-Agent System Architecture

### Investment Committee Structure

The system mirrors a real hedge fund investment committee with specialized roles:

**🏛️ Fund Head Agent** (`src/agents/fund_head_agent.py`)
- Senior portfolio manager and final decision maker
- Orchestrates investment committee meetings
- Analyzes market regime (Bull/Bear/Choppy)
- Reviews all strategy analyses and makes final trading decisions

**📊 Technical Analyst** (`src/agents/technical_analyst.py`)
- Provides comprehensive technical analysis using DuckDB data ONLY
- Calculates 30+ technical indicators (RSI, moving averages, momentum, volatility)
- No API calls for maximum performance - uses local DuckDB exclusively

**🎯 Strategy Agents** (`src/agents/strategies/`)
- **Qullamaggie Strategy Agent**: Your momentum trading strategy with 3 setups
- **Extensible Framework**: Add your specific trading strategies here
- Each agent analyzes opportunities through their specialized strategy lens

**🔍 Research Agent** (`src/agents/research_agent.py`)
- Handles data refresh operations and fundamental research
- Manages API backfill calls and incremental data updates

**⚖️ Risk Manager** (`src/agents/risk_manager_agent.py`)
- Portfolio risk assessment and position sizing recommendations
- ATR-based risk calculations and portfolio risk management

### System Workflow

```
1. Fund Head Agent → Starts investment committee meeting
2. Strategy Agents → Analyze symbol using Technical Analyst data  
3. Risk Manager → Provides risk assessment
4. Research Agent → Supplies additional context
5. Fund Head Agent → Makes final investment decision with position sizing
```

### Strategy Implementation

**Qullamaggie Momentum Strategy** with three setups:

1. **Breakouts**: 20-day high breakouts with volume confirmation
2. **Episodic Pivots**: Low-risk entries on pullbacks to key support levels  
3. **Parabolic Shorts**: Counter-trend plays on parabolic extensions

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key for AutoGen LLM integration
- Alpaca Markets API key for market data and paper trading

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd wck_investment_team
```

2. **Set up Python environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

3. **Configure API keys**:
```bash
# Set environment variables
export OPENAI_API_KEY="your_openai_api_key"
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"

# Or create .env file in project root
```

### Running the Multi-Agent System

#### Run Full Investment Committee Meeting
```bash
# Run complete multi-agent analysis for a symbol
python your_strategies_debate_system.py

# Example output:
# Fund Head Agent analyzing SPY...
# Strategy agents providing recommendations...
# Final investment decision: BUY 100 shares at $425.50
```

#### Individual Agent Testing
```bash
# Test technical analyst with DuckDB data
python -c "
from src.agents.technical_analyst import SimpleTechnicalAnalyst
analyst = SimpleTechnicalAnalyst()
metrics = analyst.calculate_all_metrics('SPY')
print(f'RSI: {metrics.momentum_indicators[\"rsi_14\"]:.2f}')
"

# Test Qullamaggie strategy agent
python -c "
from src.agents.strategies.qullamaggie_strategy_agent import QullamaggieStrategyAgent
agent = QullamaggieStrategyAgent()
print('✅ Qullamaggie Strategy Agent: Ready')
"
```

## 📁 Project Structure

```
wck_investment_team/
├── your_strategies_debate_system.py    # Main multi-agent system entry point
│
├── src/                                # Core source code
│   ├── agents/                        # Multi-agent system components
│   │   ├── base_agent.py             # AutoGen base agent class
│   │   ├── fund_head_agent.py        # Fund head decision maker
│   │   ├── technical_analyst.py      # Technical analysis (DuckDB only)
│   │   ├── research_agent.py         # Data refresh and research
│   │   ├── risk_manager_agent.py     # Portfolio risk management
│   │   └── strategies/               # Strategy agent implementations
│   │       └── qullamaggie_strategy_agent.py  # Your momentum strategy
│   │
│   ├── data/                         # Data management layer
│   │   ├── duckdb_manager.py        # High-performance DuckDB operations
│   │   └── market_data.py           # Market data integration
│   │
│   ├── services/                     # Service layer
│   │   ├── market_data_service.py   # Abstracted market data service
│   │   └── providers/               # Data provider implementations
│   │       └── alpaca_provider.py   # Alpaca Markets integration
│   │
│   ├── models/                      # Data models
│   │   ├── trade.py                # Trade and order models
│   │   └── position.py             # Position tracking
│   │
│   └── utils/                       # Utilities
│       └── logging_config.py        # Structured logging system
│
├── scripts/                         # Utility scripts
│   ├── backfill_historical_data.py # Historical data population
│   └── simple_data_refresh.py      # Incremental data updates
│
├── strategies/                      # Strategy configuration
│   └── qullamaggie/                # Qullamaggie strategy rules
│       └── rules.json              # Strategy parameters and setup criteria
│
├── data_cache/                     # Local data storage
│   └── market_data.duckdb         # DuckDB database with SPY/QQQ data
│
└── requirements.txt                # Python dependencies (AutoGen 0.7.4)
```

## 🔧 Configuration

### Qullamaggie Strategy Rules

Complete strategy configuration in `strategies/qullamaggie/rules.json`:

```json
{
  "scanning_criteria": {
    "filters": {
      "min_price": 5.0,
      "min_market_cap": 1000000000,
      "min_dollar_volume": 10000000,
      "min_adr_percent": 4.0
    }
  },
  "setups": {
    "breakout": {
      "lookback_period": 20,
      "volume_threshold": 1.5,
      "gap_threshold": 2.0
    },
    "episodic_pivot": {
      "pullback_percentage": 8.0,
      "support_test_days": 3
    },
    "parabolic_short": {
      "extension_threshold": 100.0,
      "volume_exhaustion_ratio": 0.5
    }
  },
  "position_sizing": {
    "account_tiers": {
      "nano": {"max_size": 1000, "max_risk": 100},
      "micro": {"max_size": 5000, "max_risk": 500},
      "small": {"max_size": 25000, "max_risk": 2000},
      "medium": {"max_size": 100000, "max_risk": 5000}
    }
  }
}
```

## 📊 Technical Analysis Capabilities

### 30+ Technical Indicators

**Trend Indicators**:
- Simple Moving Averages (10, 20, 50, 200-day)
- Exponential Moving Averages (10, 20, 50-day)
- Moving Average alignment and angles

**Volatility Indicators**:
- Average True Range (ATR)
- Average Daily Range (ADR)
- Bollinger Bands
- Standard deviation bands

**Momentum Indicators**:
- RSI (14-day)
- MACD with histogram
- Rate of Change (22, 67-day)
- Momentum oscillators

**Volume Indicators**:
- Volume ratios and surges
- Dollar volume calculations
- Volume-weighted metrics
- On-Balance Volume (OBV)

**Support/Resistance**:
- Pivot points
- Key level identification
- Breakout detection
- Extension measurements

### Market Regime Classification

Automatic detection of market conditions:
- **Bull Market**: Rising trends, strong momentum
- **Choppy Market**: Sideways action, mixed signals  
- **Bear Market**: Declining trends, weak momentum

## 🎯 Trading Strategy Features

### Qullamaggie Implementation

**Setup Detection**:
- Mechanical rule-based identification
- Confidence scoring (0-5 stars)
- Entry/exit price calculations
- Risk/reward analysis

**Position Sizing**:
- Account tier-based sizing
- ATR-based risk calculation
- Maximum risk per trade limits
- Portfolio risk management

**Risk Management**:
- ATR-based stop losses
- 2:1 minimum risk/reward ratios
- Position size limits
- Sector concentration limits

### Multi-Agent System Usage

```python
from your_strategies_debate_system import StrategiesDebateSystem

# Initialize the complete multi-agent system
debate_system = StrategiesDebateSystem()

# Run investment committee meeting for a symbol
symbol = "SPY"
decision = debate_system.run_investment_committee_meeting(symbol)

if decision["action"] != "HOLD":
    print(f"Decision: {decision['action']} {decision['quantity']} shares")
    print(f"Entry Price: ${decision['entry_price']:.2f}")
    print(f"Stop Loss: ${decision['stop_loss']:.2f}")
    print(f"Confidence: {decision['confidence']}/5.0")
    print(f"Risk: ${decision['total_risk']:.2f}")
```

### Individual Agent Usage

```python
# Technical Analysis (DuckDB only - no API calls)
from src.agents.technical_analyst import SimpleTechnicalAnalyst
analyst = SimpleTechnicalAnalyst()
metrics = analyst.calculate_all_metrics("SPY", period="1y")
print(f"RSI: {metrics.momentum_indicators['rsi_14']:.2f}")

# Qullamaggie Strategy Analysis
from src.agents.strategies.qullamaggie_strategy_agent import QullamaggieStrategyAgent
qullamaggie = QullamaggieStrategyAgent()
# Agent uses technical analyst internally with DuckDB data

# Fund Head Decision Making
from src.agents.fund_head_agent import FundHeadAgent
fund_head = FundHeadAgent()
final_decision = fund_head.make_investment_decision(symbol, agent_analyses, market_context)
```

## 🛡️ Risk Management

### Risk Controls

- **Position Sizing**: Maximum position sizes per account tier
- **Stop Losses**: ATR-based automatic stop loss calculation
- **Risk/Reward**: Minimum 2:1 risk/reward ratios required
- **Portfolio Risk**: Maximum risk percentage per trade
- **Sector Limits**: Concentration risk management

### Risk Metrics

- Average True Range (ATR) for volatility
- Position size as % of account
- Risk per trade calculations
- Portfolio heat monitoring
- Drawdown tracking

## 📈 Data Integration & Performance

### High-Performance DuckDB Architecture

**Local Database Strategy** - No API abuse, maximum performance:

```python
from src.data.duckdb_manager import get_duckdb_manager
from src.agents.technical_analyst import SimpleTechnicalAnalyst

# DuckDB manager for high-performance analytics
duckdb_manager = get_duckdb_manager()
data = duckdb_manager.get_market_data("SPY", "1Day", start_date, end_date)

# Technical analyst uses ONLY DuckDB data (no API calls)
analyst = SimpleTechnicalAnalyst()
metrics = analyst.calculate_all_metrics("SPY")  # Pure DuckDB query
```

### Current Data Setup: Alpaca Markets + DuckDB

**Production-Ready Architecture**:
- **DuckDB Storage**: 2 years of SPY/QQQ daily data for instant analysis
- **Alpaca Integration**: Professional-grade market data and paper trading
- **Zero API Abuse**: Technical analysis uses cached DuckDB data exclusively
- **Incremental Updates**: Scripts for daily data refresh without hitting API limits
- **Service Layer**: Abstracted providers for easy data source swapping

### Database Performance

**Current DuckDB Contents**:
- **SPY**: 500+ daily OHLCV records (2022-2024)
- **QQQ**: 500+ daily OHLCV records (2022-2024) 
- **Query Speed**: Sub-millisecond technical analysis queries
- **Storage**: Compressed analytical columnar format
- **Reliability**: No network dependencies for core analysis

## 🧪 Testing the Multi-Agent System

### Complete System Testing

```bash
# Test full multi-agent investment committee
python your_strategies_debate_system.py

# Expected output:
# Fund Head Agent: Starting investment committee meeting for SPY
# Technical Analyst: Calculating metrics from DuckDB (no API calls)
# Qullamaggie Agent: Analyzing momentum setup...
# Risk Manager: Evaluating position sizing...
# Fund Head Agent: Final decision - BUY 100 shares
```

### Individual Agent Testing

```bash
# Test technical analyst with DuckDB performance
python -c "
from src.agents.technical_analyst import SimpleTechnicalAnalyst
import time

analyst = SimpleTechnicalAnalyst()
start = time.time()
metrics = analyst.calculate_all_metrics('SPY')
end = time.time()

print(f'✅ Technical Analysis: {(end-start)*1000:.1f}ms')
print(f'📊 RSI: {metrics.momentum_indicators[\"rsi_14\"]:.2f}')
print(f'📈 Current Price: ${metrics.moving_averages[\"current_price\"]:.2f}')
"

# Test AutoGen agent initialization
python -c "
from src.agents.fund_head_agent import FundHeadAgent
from src.agents.strategies.qullamaggie_strategy_agent import QullamaggieStrategyAgent

fund_head = FundHeadAgent()
qullamaggie = QullamaggieStrategyAgent()

print('✅ Fund Head Agent: Ready (AutoGen 0.7.4)')
print('✅ Qullamaggie Strategy Agent: Ready')
print('✅ Multi-Agent System: Operational')
"

# Test DuckDB data availability
python -c "
from src.data.duckdb_manager import get_duckdb_manager

manager = get_duckdb_manager()
spy_count = manager.get_record_count('SPY', '1Day')
qqq_count = manager.get_record_count('QQQ', '1Day')

print(f'📊 SPY records in DuckDB: {spy_count}')
print(f'📊 QQQ records in DuckDB: {qqq_count}')
print('✅ Database: Ready for analysis')
"
```

## 📋 Logging and Monitoring

### Structured Logging

Comprehensive logging system with JSON formatting:

```python
from src.utils.logging_config import get_trading_logger, log_trade_execution

# Specialized trading logs
logger = get_trading_logger("qullamaggie", "AAPL")
logger.info("Signal generated", extra={
    "signal_strength": 0.85,
    "setup_type": "breakout"
})

# Trade execution logging
log_trade_execution(
    symbol="AAPL",
    side="buy",
    quantity=100, 
    price=150.0,
    strategy="qullamaggie",
    agent="breakout_detector"
)
```

### Log Categories

- `wck_investment_team.log`: General system logs
- `trading.log`: Trade executions and signals
- `metrics.log`: Performance metrics
- `errors.log`: Error tracking

## 🔧 Development & Extension

### Adding New Technical Indicators

```python
# In src/agents/technical_analyst.py
def _calculate_new_indicator(self, df: pd.DataFrame) -> Dict[str, float]:
    """Add your custom indicator to the technical metrics"""
    # Your custom calculation using pandas/numpy
    custom_value = your_calculation(df)
    return {'custom_indicator': float(custom_value)}

# Update calculate_all_metrics() to include your indicator
custom_metrics = self._calculate_new_indicator(df)
```

### Creating New Strategy Agents

```python
# 1. Create new agent in src/agents/strategies/
from ..base_agent import BaseAgent
from ..technical_analyst import SimpleTechnicalAnalyst

class YourStrategyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="your_strategy_agent",
            description="Your trading strategy description"
        )
        self.technical_analyst = SimpleTechnicalAnalyst()
    
    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        # Get technical metrics from DuckDB (no API calls)
        metrics = self.technical_analyst.calculate_all_metrics(symbol)
        # Your strategy logic here
        return analysis_result

# 2. Add strategy rules in strategies/your_strategy/rules.json
# 3. Register with Fund Head Agent in your_strategies_debate_system.py
```

### Adding New Data Providers

```python
# Create new provider in src/services/providers/
class YourDataProvider:
    def get_daily_data(self, symbol: str, period: str) -> pd.DataFrame:
        # Your data source implementation
        return standardized_ohlcv_dataframe
    
    def get_supported_intervals(self) -> List[str]:
        return ["1Day", "1Hour"]  # Your supported intervals

# Register in market_data_service.py
```

## ⚠️ Current Limitations

- **Strategy Expansion**: Currently only has Qullamaggie strategy; add more of your specific strategies
- **Live Trading**: Paper trading only; no live broker execution yet  
- **Backtesting Engine**: Limited historical testing capabilities
- **Agent Conversations**: AutoGen conversations could be more sophisticated
- **Portfolio Management**: Single-position focus; needs multi-position portfolio tracking

## 🚀 Future Enhancements

### High Priority
- [ ] **More Strategy Agents**: Add your other trading strategies to the investment committee
- [ ] **Enhanced Agent Debates**: More sophisticated AutoGen conversations between agents
- [ ] **Live Trading Integration**: Connect Alpaca paper trading to actual trade execution
- [ ] **Portfolio Management**: Multi-position tracking and portfolio-level risk management

### Agent System Improvements
- [ ] **Agent Memory**: Persistent memory across investment committee meetings
- [ ] **Market Regime Adaptation**: Dynamic strategy weighting based on market conditions
- [ ] **Performance Tracking**: Track individual agent performance over time
- [ ] **Agent Learning**: Improve agent decision-making based on historical performance

### Data & Performance  
- [ ] **Real-time Data**: Live market data feeds for intraday strategies
- [ ] **More Symbols**: Expand DuckDB beyond SPY/QQQ to full universe
- [ ] **Fundamental Data**: Earnings, financial metrics, news sentiment integration
- [ ] **Options Data**: Greeks, volatility surface, options strategies

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and always consult with financial professionals before making investment decisions.

---

## 🏆 System Status

**🤖 Multi-Agent System**: ✅ Operational with AutoGen 0.7.4  
**📊 Technical Analysis**: ✅ DuckDB-powered (no API abuse)  
**🎯 Strategy Implementation**: ✅ Qullamaggie momentum strategy  
**🏛️ Fund Structure**: ✅ Realistic hedge fund investment committee  
**📈 Data Infrastructure**: ✅ 2-year SPY/QQQ historical data  
**🔗 Trading Integration**: ✅ Alpaca Markets paper trading  

**Last Updated**: September 2025  
**Framework**: AutoGen 0.7.4 with OpenAI LLM integration  
**Database**: DuckDB high-performance analytics  
**Data Provider**: Alpaca Markets with local caching  

## 📊 Current DuckDB Contents

**High-Performance Database** (`data_cache/market_data.duckdb`):

| Symbol | Records | Date Range | Timeframe | Performance |
|--------|---------|------------|-----------|-------------|
| SPY    | 500+    | 2022-2024 | Daily (1Day) | <1ms queries |
| QQQ    | 500+    | 2022-2024 | Daily (1Day) | <1ms queries |

**Total**: 1000+ daily OHLCV records optimized for analytical workloads, enabling instant technical analysis without API dependencies.

## 🚀 Ready to Use

The system is production-ready for:
- ✅ **Investment Committee Simulations**: Run realistic hedge fund meetings
- ✅ **Strategy Development**: Add your trading strategies to the committee  
- ✅ **Technical Analysis**: Lightning-fast analysis using local DuckDB
- ✅ **Risk Management**: Professional-grade position sizing and risk controls
- ✅ **Paper Trading**: Test strategies with Alpaca Markets integration