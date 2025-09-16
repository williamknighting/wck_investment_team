# AI Hedge Fund System

A quantitative trading system implementing the Qullamaggie momentum strategy with comprehensive technical analysis, market data integration, and risk management capabilities.

## 🎯 Overview

This system implements a systematic trading platform focused on momentum strategies, particularly the Qullamaggie approach. The current implementation is primarily algorithmic/quantitative with optional AI agent integration using the AutoGen framework.

**Current Status**: 
- ✅ Complete Qullamaggie strategy implementation with 3 setups
- ✅ 30+ technical indicators and market regime detection
- ✅ Real market data integration (Alpha Vantage API)
- ✅ SQLite data caching and management
- ✅ Comprehensive risk management and position sizing
- ✅ Paper trading simulation capabilities
- 🔄 AutoGen AI agents (infrastructure ready, not actively used)

## 🏗️ System Architecture

### Core Components

- **Market Data Manager**: Fetches and caches real market data using Alpha Vantage API
- **Technical Analysis Engine**: 30+ indicators including SMA, EMA, ATR, ADR, RSI, MACD, Bollinger Bands
- **Qullamaggie Strategy Engine**: Complete implementation of 3 momentum setups
- **Market Regime Classifier**: Bull/Choppy/Bear market detection
- **Position Sizing System**: Account-tier based position sizing (Nano/Micro/Small/Medium accounts)
- **Risk Management**: ATR-based stop losses and comprehensive risk controls
- **Paper Trading System**: Realistic trade execution simulation

### Strategy Implementation

**Qullamaggie Momentum Strategy** with three setups:

1. **Breakouts**: 20-day high breakouts with volume confirmation
2. **Episodic Pivots**: Low-risk entries on pullbacks to key support levels  
3. **Parabolic Shorts**: Counter-trend plays on parabolic extensions

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Alpha Vantage API key (free at alphavantage.co)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd ai_hedge_fund
```

2. **Set up Python environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

3. **Configure API key**:
```bash
# Edit utils/data_integration.py and replace with your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "your_api_key_here"
```

### Running the System

#### Test Complete System
```bash
# Run full system test with real market data
python utils/data_integration.py
```

#### Individual Components
```bash
# Test technical analysis
python src/agents/core/technical_analyst_agent.py

# Test Qullamaggie strategy
python src/agents/strategies/qullamaggie/qullamaggie_strategy_agent.py

# Test market data fetching
python src/data/market_data.py
```

## 📁 Project Structure

```
ai_hedge_fund/
├── config/                     # Configuration files
│   ├── settings.py            # Main configuration
│   ├── market_config.yaml     # Market parameters  
│   ├── strategy_config.yaml   # Strategy settings
│   └── risk_limits.yaml       # Risk management rules
│
├── src/                       # Core source code
│   ├── agents/                # AI agents and strategy engines
│   │   ├── base_agent.py      # Base agent class
│   │   ├── core/              # Core system agents
│   │   │   ├── technical_analyst_agent.py  # Technical analysis engine
│   │   │   ├── market_regime_agent.py      # Market regime classifier
│   │   │   └── risk_manager_agent.py       # Risk management
│   │   └── strategies/        # Strategy implementations
│   │       └── qullamaggie/   # Qullamaggie strategy agent
│   ├── coordination/          # AutoGen orchestration (optional)
│   ├── data/                  # Data management
│   │   └── market_data.py     # Market data fetcher and cache
│   ├── execution/             # Trade execution engine
│   ├── models/                # Data models (Trade, Position, Signal)
│   │   ├── trade.py           # Trade and order models
│   │   └── position.py        # Position tracking model
│   └── utils/                 # Utilities
│       └── logging_config.py  # Structured logging system
│
├── strategies/                # Strategy configuration
│   └── qullamaggie/          # Qullamaggie strategy rules
│       ├── rules.json        # Complete strategy parameters
│       └── prompts/          # AI agent prompts (optional)
│
├── utils/                     # Utilities and integrations
│   └── data_integration.py    # Working Alpha Vantage integration
│
└── requirements.txt           # Python dependencies
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

### Example Usage

```python
from src.agents.strategies.qullamaggie.qullamaggie_strategy_agent import QullamaggieStrategyAgent
from src.data.market_data import MarketDataManager

# Initialize components
strategy = QullamaggieStrategyAgent()
data_manager = MarketDataManager()

# Analyze a stock
symbol = "NVDA"
data = data_manager.fetch_stock_data(symbol, period="6mo")
analysis = strategy.analyze_stock(symbol, data)

if analysis["valid"]:
    print(f"Setup found: {analysis['setup_type']}")
    print(f"Confidence: {analysis['confidence']}/5.0")
    print(f"Entry: ${analysis['entry_price']:.2f}")
    print(f"Stop: ${analysis['stop_loss']:.2f}")
    print(f"Target: ${analysis['target']:.2f}")
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

## 📈 Data Integration

### Abstracted Market Data Service

**Service Layer Architecture** - Easy to swap data providers without changing trading logic:

```python
from src.services.market_data_service import get_market_data_service

# Get market data (provider-agnostic)
service = get_market_data_service()
data = service.get_stock_data("AAPL", period="1y", interval="1d")

# Multiple timeframes
daily_data = service.get_daily_data("AAPL", "6mo")
intraday_data = service.get_intraday_data("AAPL", "5m", days=1)

# Multiple stocks
multi_data = service.get_multiple_stocks(["AAPL", "NVDA", "TSLA"])
```

### Current Data Provider: Alpaca Markets

**Production-Ready Setup** with conservative API usage:
- **Daily Data Only**: OHLCV data at 1-day intervals (API-friendly)
- **Rate Limiting**: 1 second minimum between requests (very conservative)
- **Aggressive Caching**: DuckDB stores all data to minimize API calls
- **Fallback**: Returns cached data when API limits hit
- **Paper Trading**: Uses Alpaca paper trading environment by default

### Data Management

- **DuckDB Storage**: High-performance analytical database for market data
- **Conservative Rate Limiting**: 1-second delays with intelligent caching
- **Data Persistence**: All data cached permanently with metadata tracking
- **Daily Timeframes**: Focus on daily OHLCV data to respect API limits  
- **Standardized Format**: Consistent OHLCV columns across all providers

## 🧪 Testing

### System Testing

```bash
# Test market data service layer
python -c "
from src.services.market_data_service import initialize_market_data_service
from src.services.providers.yfinance_provider import create_yfinance_provider

provider = create_yfinance_provider()
service = initialize_market_data_service(provider)
data = service.get_daily_data('AAPL', '5d')
print(f'✅ Got {len(data)} days, latest: ${data[\"Close\"].iloc[-1]:.2f}')
"

# Expected output:
# ✅ Got 100 days, latest: $234.07
```

### Individual Component Testing

```bash
# Test technical analysis with market service
python -c "
from src.services.market_data_service import initialize_market_data_service
from src.services.providers.yfinance_provider import create_yfinance_provider

# Initialize service
provider = create_yfinance_provider()
service = initialize_market_data_service(provider)
print('✅ Market Data Service: Ready')
print(f'📊 Supports: {len(service.get_supported_intervals())} intervals')
"

# Test strategy with real data
python -c "
from src.agents.strategies.qullamaggie.qullamaggie_strategy_agent import QullamaggieStrategyAgent
agent = QullamaggieStrategyAgent()
print('✅ Qullamaggie Strategy Agent: Ready')
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

- `ai_hedge_fund.log`: General system logs
- `trading.log`: Trade executions and signals
- `metrics.log`: Performance metrics
- `errors.log`: Error tracking

## 🔧 Development

### Adding New Technical Indicators

```python
# In technical_analyst_agent.py
def _calculate_new_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
    """Add your custom indicator"""
    data['custom_indicator'] = your_calculation(data)
    return data
```

### Creating New Strategies

1. Create strategy directory in `strategies/`
2. Define rules in `rules.json`
3. Implement strategy agent class
4. Add to system configuration

### API Integration

The system includes working Alpha Vantage integration. To add other data sources:

1. Create new fetcher class in `src/data/`
2. Implement rate limiting and caching
3. Update market data manager
4. Test with real data

## ⚠️ Current Limitations

- **LLM Integration**: System is currently algorithmic; AI agents available but not actively used
- **Live Trading**: Paper trading only; no broker integration yet
- **Backtesting**: Limited historical testing capabilities
- **Rate Limits**: yfinance can be heavily rate limited; cached data provides fallback

## 🚀 Future Enhancements

### High Priority
- [ ] **Interactive Brokers Integration**: Create `IBKRProvider` implementing `MarketDataProvider` interface for live trading and real-time data feeds
- [ ] **Advanced Backtesting**: Walk-forward analysis engine with realistic slippage and commission modeling
- [ ] **Real-time Alerts**: Price/volume breakout notifications and strategy signal alerts

### Medium Priority  
- [ ] Machine learning signal enhancement and pattern recognition
- [ ] Portfolio optimization algorithms with risk constraints
- [ ] Options trading strategies and Greeks analysis
- [ ] Fundamental data integration (earnings, financials, estimates)

### Data Provider Roadmap
- [x] **yfinance**: Current provider with caching and rate limiting
- [ ] **Interactive Brokers**: Professional-grade data and execution
- [ ] **Alpha Vantage**: Fundamental data and news sentiment
- [ ] **Polygon.io**: High-frequency intraday data

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk and always consult with financial professionals before making investment decisions.

---

**System Status**: ✅ Core functionality complete and tested with real market data
**Last Updated**: September 2025  
**Data Integration**: yfinance with abstracted service layer (swappable to IBKR)
**Strategy Implementation**: Qullamaggie (complete)

## 📊 Current Database Contents

**SQLite Database** (`data_cache/market_data.db`):

| Symbol | Records | Date Range | Timeframe | Last Updated |
|--------|---------|------------|-----------|---------------|
| AAPL   | 100     | 2025-04-22 to 2025-09-12 | Daily (1d) | 2025-09-13 20:32:54 |
| MSFT   | 100     | 2025-04-22 to 2025-09-12 | Daily (1d) | 2025-09-13 20:33:06 |  
| NVDA   | 100     | 2025-04-22 to 2025-09-12 | Daily (1d) | 2025-09-13 20:32:42 |

**Total**: 300 daily OHLCV records across 3 major tech stocks, cached locally for fast analysis without API limits.