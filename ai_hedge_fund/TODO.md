# AI Hedge Fund System - TODO List

## 📊 Data Requirements for Full Technical Analysis

### ✅ **Available with OHLCV Data** (Can implement now)
- All Moving Averages (SMA, EMA)
- All Volatility Metrics (ADR, ATR)
- All Performance/Momentum Metrics (gains, returns)
- Most Price Pattern Metrics
- Most Volume Metrics
- Gap & Extension Metrics
- Most Intraday Metrics
- Most Setup Detection Metrics
- Most Risk Management Metrics

### ❌ **Missing Data - Need Additional Sources**

#### **Market/Index Data**
- [ ] **Nasdaq Index Data** - Need QQQ or ^IXIC data for market regime
  - `nasdaq_10ma`, `nasdaq_20ma`, `nasdaq_ma_slope`
  - `market_regime` classification
- [ ] **Sector ETF Data** - Need sector ETFs for relative strength
  - `sector_strength_rank`
  - `sector_breakout_count`

#### **Fundamental Data** 
- [ ] **Earnings Data API** - Need earnings calendar/estimates
  - `eps_growth_yoy`, `revenue_growth_yoy`
  - `earnings_beat`, `days_to_earnings`, `days_since_earnings`
- [ ] **Financial Statements** - Need quarterly/annual data
  - Could use `yfinance` `.info` and `.financials` but limited

#### **Market Microstructure Data**
- [ ] **Real-time Intraday** - Some metrics need live data
  - `opening_volume_ratio` (first 20-min volume)
  - `first_red_candle`, `first_green_candle` timestamps
- [ ] **Market Breadth Data** - For percentile rankings
  - `rank_1m`, `rank_3m`, `rank_6m` vs entire market

#### **Alternative Data Sources**
- [ ] **Options Data** - For sentiment/volatility
- [ ] **News Sentiment** - For fundamental catalyst detection
- [ ] **Insider Trading** - For EP qualification rules

## 🔧 Technical Implementation TODOs

### **Phase 1: Core Technical Indicators** ✅ (Ready to implement)
- [ ] Moving Averages (SMA 10/20/50, EMA 10/20/65)
- [ ] Volatility (ADR, ATR, range contraction)
- [ ] Momentum (all gain periods, consecutive days)
- [ ] Basic Volume metrics
- [ ] VWAP calculation
- [ ] Gap detection
- [ ] MA alignment and angles

### **Phase 2: Pattern Recognition** 
- [ ] Higher lows/lower highs detection algorithm
- [ ] Consolidation period detection
- [ ] Breakout quality scoring (1-5 stars)
- [ ] Parabolic extension detection

### **Phase 3: Setup Detection**
- [ ] Qullamaggie Breakout scoring
- [ ] Episodic Pivot qualification (needs gap + volume rules)
- [ ] Parabolic Short exhaustion signals
- [ ] Risk/reward calculations

### **Phase 4: Market Context** (Requires additional data)
- [ ] Market regime classification
- [ ] Sector rotation analysis
- [ ] Relative strength rankings

## 📈 Data Source Integration

### **Immediate Solutions** (Free/Easy)
- [ ] **yfinance extended**: Use `.info` for basic fundamentals
- [ ] **Index ETFs**: Pull QQQ, SPY data for market regime
- [ ] **Sector ETFs**: XLK, XLV, XLF etc. for sector analysis
- [ ] **Multiple timeframes**: 1min, 5min, 1hour, 1day data

### **Future Enhancements** (Paid APIs)
- [ ] **Alpha Vantage**: Fundamental data, earnings calendar
- [ ] **Polygon**: Real-time market data, better intraday
- [ ] **Quandl/IEX**: Market breadth, sector data
- [ ] **Fred Economic Data**: Macro indicators

## 🎯 Strategy-Specific TODOs

### **Qullamaggie Strategy Requirements**
- [ ] **Breakout Detection**: Daily consolidation + volume surge
- [ ] **Episodic Pivot Rules**: Gap >10%, volume in first 20min, no recent rally
- [ ] **Parabolic Short**: Extension >20%, exhaustion signals
- [ ] **Position Sizing**: Kelly Criterion + risk-adjusted sizing

### **Risk Management**
- [ ] **ATR-based stops**: Dynamic stop loss calculation
- [ ] **Position sizing**: Multiple methods (Kelly, fixed risk, volatility-adjusted)
- [ ] **Portfolio heat**: Track total risk across all positions

## 🏗️ System Architecture TODOs

### **Agent Coordination**
- [ ] **Technical Analysis Agent**: Calculate all metrics
- [ ] **Market Regime Agent**: Enhanced with real index data  
- [ ] **Strategy Agents**: Request specific metrics they need
- [ ] **Caching Strategy**: Avoid recalculating same metrics

### **Performance Optimization**
- [ ] **Batch Calculations**: Calculate multiple stocks efficiently
- [ ] **Incremental Updates**: Only calculate new data points
- [ ] **Memory Management**: Handle large datasets efficiently

## 📋 Testing & Validation

### **Data Quality**
- [ ] **Historical Accuracy**: Validate calculations against known sources
- [ ] **Missing Data Handling**: Graceful degradation when data unavailable
- [ ] **Edge Cases**: Handle stock splits, dividends, halts

### **Performance Testing**
- [ ] **Speed Benchmarks**: Ensure calculations complete quickly
- [ ] **Memory Usage**: Monitor resource consumption
- [ ] **Concurrent Access**: Test multiple agents accessing data

---

**Next Steps**: Start with Phase 1 technical indicators using OHLCV data, then progressively add more complex features as data sources are integrated.