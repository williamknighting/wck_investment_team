"""
Research Agent for AI Hedge Fund System
Conducts fundamental and technical research on securities
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from decimal import Decimal
import json
import pandas as pd

from ..base_agent import BaseHedgeFundAgent, AgentCapability
from ...models.trade import TradeSignal, OrderSide
from ...utils.logging_config import get_logger
from ...services.market_data_service import get_market_data_service


class ResearchAgent(BaseHedgeFundAgent):
    """
    Agent responsible for conducting research on securities
    Provides fundamental analysis, technical analysis, and market research
    """
    
    def __init__(self, **kwargs):
        system_message = """You are a Research Agent for an AI hedge fund.

Your responsibilities:
1. Conduct fundamental analysis on securities (earnings, revenue, ratios)
2. Perform technical analysis using various indicators
3. Research market trends and sector dynamics
4. Analyze news sentiment and market events
5. Screen for investment opportunities
6. Provide comprehensive research reports

Research capabilities:
- Financial statement analysis
- Valuation metrics (P/E, P/B, PEG, DCF)
- Technical indicators (RSI, MACD, moving averages)
- Relative strength analysis
- Sector and industry comparison
- News sentiment analysis
- Earnings estimates and revisions
- Insider trading activity
- Institutional ownership changes

Research methodology:
- Multi-timeframe analysis (daily, weekly, monthly)
- Quantitative screening with qualitative overlay
- Risk-adjusted opportunity assessment
- Catalyst identification and timing
- Competitive landscape analysis

Provide actionable research insights with clear investment thesis."""
        
        super().__init__(
            name="research_agent",
            system_message=system_message,
            capabilities=[
                AgentCapability.RESEARCH,
                AgentCapability.MARKET_ANALYSIS
            ],
            **kwargs
        )
    
    def _initialize(self) -> None:
        """Initialize research agent"""
        self.research_cache = {}
        self.screening_criteria = {
            "min_market_cap": 1_000_000_000,  # $1B
            "min_avg_volume": 1_000_000,      # 1M shares
            "max_pe_ratio": 25,
            "min_revenue_growth": 0.10,       # 10%
            "min_rsi": 30,
            "max_rsi": 70
        }
        
        self.technical_indicators = [
            "sma_20", "sma_50", "sma_200",
            "rsi_14", "macd", "bollinger_bands",
            "volume_sma", "relative_strength"
        ]
        
        self.fundamental_metrics = [
            "market_cap", "pe_ratio", "pb_ratio", "peg_ratio",
            "debt_to_equity", "roe", "roa", "profit_margin",
            "revenue_growth", "earnings_growth", "free_cash_flow"
        ]
        
        self.logger.info("Research Agent initialized")
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data for research requests"""
        if not isinstance(data, dict):
            return False
        
        message_type = data.get("type", "general")
        
        if message_type in ["analyze_security", "fundamental_analysis", "technical_analysis"]:
            return "symbol" in data
        elif message_type == "screen_securities":
            return True  # No specific requirements
        elif message_type == "sector_analysis":
            return "sector" in data or "symbols" in data
        
        return True
    
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """Process research requests"""
        try:
            message_type = message.get("type", "general")
            
            if message_type == "analyze_security":
                return self._analyze_security(message)
            elif message_type == "fundamental_analysis":
                return self._fundamental_analysis(message)
            elif message_type == "technical_analysis":
                return self._technical_analysis(message)
            elif message_type == "screen_securities":
                return self._screen_securities(message)
            elif message_type == "sector_analysis":
                return self._sector_analysis(message)
            elif message_type == "research_report":
                return self._generate_research_report(message)
            else:
                return self._general_research_response(message)
        
        except Exception as e:
            self.logger.error(f"Error processing research message: {e}")
            return {
                "type": "error",
                "error": str(e),
                "agent": self.name
            }
    
    def _analyze_security(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive security analysis"""
        symbol = message.get("symbol")
        analysis_depth = message.get("depth", "standard")  # quick, standard, deep
        
        # Fetch real stock data from yfinance
        stock_data = self.fetch_stock_data_for_analysis(symbol, period="1y", interval="1d")
        
        if not stock_data:
            return {
                "type": "error",
                "error": f"No data available for {symbol}",
                "agent": self.name
            }
        
        # Perform fundamental analysis
        fundamental = self._perform_fundamental_analysis(stock_data)
        
        # Perform technical analysis
        technical = self._perform_technical_analysis(stock_data)
        
        # Generate overall assessment
        assessment = self._generate_security_assessment(symbol, fundamental, technical)
        
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fundamental_analysis": fundamental,
            "technical_analysis": technical,
            "overall_assessment": assessment,
            "investment_thesis": self._generate_investment_thesis(symbol, fundamental, technical),
            "risks": self._identify_risks(symbol, fundamental, technical),
            "catalysts": self._identify_catalysts(symbol, fundamental, technical),
            "target_price": self._calculate_target_price(symbol, fundamental, technical),
            "recommendation": assessment["recommendation"]
        }
        
        # Cache results
        self.research_cache[symbol] = analysis
        
        self.log_activity("security_analyzed", level="info",
                         symbol=symbol, recommendation=assessment["recommendation"],
                         confidence=assessment["confidence"])
        
        return {
            "type": "security_analysis",
            "analysis": analysis,
            "agent": self.name
        }
    
    def _fundamental_analysis(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Focused fundamental analysis"""
        symbol = message.get("symbol")
        stock_data = self.fetch_stock_data_for_analysis(symbol, period="1y", interval="1d")
        
        if not stock_data:
            return {
                "type": "error",
                "error": f"No data available for {symbol}",
                "agent": self.name
            }
        
        fundamental = self._perform_fundamental_analysis(stock_data)
        
        return {
            "type": "fundamental_analysis",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis": fundamental,
            "agent": self.name
        }
    
    def _technical_analysis(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Focused technical analysis"""
        symbol = message.get("symbol")
        stock_data = self.fetch_stock_data_for_analysis(symbol, period="1y", interval="1d")
        
        if not stock_data:
            return {
                "type": "error", 
                "error": f"No data available for {symbol}",
                "agent": self.name
            }
        
        technical = self._perform_technical_analysis(stock_data)
        
        return {
            "type": "technical_analysis",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis": technical,
            "agent": self.name
        }
    
    def _screen_securities(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Screen securities based on criteria"""
        criteria = message.get("criteria", {})
        screen_type = message.get("screen_type", "momentum")
        max_results = message.get("max_results", 20)
        
        # Merge with default criteria
        screening_criteria = {**self.screening_criteria, **criteria}
        
        # Mock screening results (in real implementation, would screen universe)
        if screen_type == "momentum":
            candidates = self._momentum_screen(screening_criteria, max_results)
        elif screen_type == "value":
            candidates = self._value_screen(screening_criteria, max_results)
        elif screen_type == "growth":
            candidates = self._growth_screen(screening_criteria, max_results)
        elif screen_type == "breakout":
            candidates = self._breakout_screen(screening_criteria, max_results)
        else:
            candidates = self._general_screen(screening_criteria, max_results)
        
        return {
            "type": "screening_results",
            "screen_type": screen_type,
            "criteria": screening_criteria,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidates": candidates,
            "total_found": len(candidates),
            "agent": self.name
        }
    
    def _sector_analysis(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sector performance and trends"""
        sector = message.get("sector")
        symbols = message.get("symbols", [])
        
        # Mock sector analysis
        sector_metrics = {
            "sector": sector,
            "performance_1w": 0.02,   # 2% weekly
            "performance_1m": 0.08,   # 8% monthly
            "performance_3m": 0.15,   # 15% quarterly
            "relative_strength": 1.2, # vs market
            "avg_pe_ratio": 18.5,
            "avg_revenue_growth": 0.12,
            "rotation_score": 0.7,    # 0-1 scale
            "momentum_score": 0.8
        }
        
        top_performers = [
            {"symbol": "EXAMPLE1", "performance": 0.25, "score": 0.9},
            {"symbol": "EXAMPLE2", "performance": 0.18, "score": 0.8},
            {"symbol": "EXAMPLE3", "performance": 0.15, "score": 0.7}
        ]
        
        return {
            "type": "sector_analysis",
            "sector": sector,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sector_metrics": sector_metrics,
            "top_performers": top_performers,
            "sector_outlook": self._generate_sector_outlook(sector, sector_metrics),
            "agent": self.name
        }
    
    def _generate_research_report(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive research report"""
        symbol = message.get("symbol")
        report_type = message.get("report_type", "full")
        
        # Get cached analysis or perform new analysis
        if symbol in self.research_cache:
            analysis = self.research_cache[symbol]
        else:
            analysis_result = self._analyze_security({"symbol": symbol})
            analysis = analysis_result["analysis"]
        
        # Generate formatted report
        report = {
            "symbol": symbol,
            "report_type": report_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "executive_summary": self._generate_executive_summary(symbol, analysis),
            "investment_recommendation": analysis["recommendation"],
            "target_price": analysis["target_price"],
            "key_metrics": self._extract_key_metrics(analysis),
            "risk_assessment": analysis["risks"],
            "catalysts": analysis["catalysts"],
            "full_analysis": analysis if report_type == "full" else None
        }
        
        return {
            "type": "research_report",
            "report": report,
            "agent": self.name
        }
    
    def fetch_stock_data_for_analysis(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        """
        Fetch real stock data using market data service and calculate analysis metrics
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            period: Time period for data
            interval: Data interval
            
        Returns:
            Dictionary with stock data and calculated metrics
        """
        try:
            # Get market data service
            market_service = get_market_data_service()
            
            # Fetch OHLCV data using the service
            df = market_service.get_stock_data(symbol, period, interval)
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return {}
            
            # Get current price (latest close) - note: service returns title case columns
            current_price = float(df['Close'].iloc[-1])
            
            # Basic stock information
            avg_volume = int(df['Volume'].mean())
            current_volume = int(df['Volume'].iloc[-1])
            
            # Get latest values - just raw price data, no technical calculations
            latest_data = {
                "symbol": symbol,
                "current_price": current_price,
                "volume": current_volume, 
                "avg_volume": avg_volume,
                "high_52w": float(df['High'].max()),
                "low_52w": float(df['Low'].min()),
                "total_trading_days": len(df)
            }
            
            # Add raw dataframe for additional analysis
            latest_data['price_data'] = df
            
            self.logger.info(f"Successfully fetched and analyzed data for {symbol}")
            return latest_data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return {}
    
    
    def _perform_fundamental_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform basic fundamental analysis with available data
        Note: Limited to basic metrics until fundamental data source is added
        """
        # Basic analysis with price data only
        current_price = data.get("current_price", 0)
        high_52w = data.get("high_52w", current_price)
        low_52w = data.get("low_52w", current_price)
        
        # Price position analysis
        if high_52w > low_52w:
            price_position = (current_price - low_52w) / (high_52w - low_52w)
        else:
            price_position = 0.5
        
        # Volume analysis
        current_volume = data.get("volume", 0)
        avg_volume = data.get("avg_volume", current_volume)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Basic scoring
        valuation_score = 0.5  # Neutral without fundamental data
        growth_score = 0.6 if price_position > 0.7 else 0.4  # High price = growth potential
        quality_score = 0.6 if volume_ratio > 1.2 else 0.4  # Volume = quality signal
        
        return {
            "valuation_score": min(1.0, max(0.0, valuation_score)),
            "growth_score": min(1.0, max(0.0, growth_score)),
            "quality_score": min(1.0, max(0.0, quality_score)),
            "key_metrics": {
                "price_position_52w": price_position,
                "volume_ratio": volume_ratio,
                "high_52w": data.get("high_52w"),
                "low_52w": data.get("low_52w"),
                "current_price": current_price,
                "note": "Limited fundamental data - using price-based analysis"
            },
            "fundamental_rating": self._calculate_fundamental_rating(valuation_score, growth_score, quality_score)
        }
    
    def _perform_technical_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform basic technical analysis with available data
        Note: For full technical analysis, use Technical Analysis Agent
        """
        current_price = data.get("current_price", 0)
        high_52w = data.get("high_52w", current_price)
        low_52w = data.get("low_52w", current_price)
        
        # Basic price position analysis
        if high_52w > low_52w:
            price_position = (current_price - low_52w) / (high_52w - low_52w)
        else:
            price_position = 0.5
        
        # Simple trend analysis based on 52-week position
        if price_position > 0.8:
            trend_score = 0.8  # Near highs
        elif price_position > 0.6:
            trend_score = 0.6  # Upper range
        elif price_position < 0.2:
            trend_score = 0.2  # Near lows
        else:
            trend_score = 0.5  # Middle range
        
        # Volume-based momentum
        volume_ratio = data.get("volume", 0) / data.get("avg_volume", 1) 
        momentum_score = min(0.9, 0.3 + (volume_ratio * 0.2))  # Higher volume = momentum
        
        return {
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "price_position_52w": price_position,
            "volume_ratio": volume_ratio,
            "note": "Basic technical analysis only. Use Technical Analysis Agent for full indicators.",
            "technical_rating": self._calculate_technical_rating(trend_score, momentum_score)
        }
    
    def _generate_security_assessment(self, symbol: str, fundamental: Dict[str, Any], technical: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall security assessment"""
        # Combine fundamental and technical scores
        fundamental_weight = 0.6
        technical_weight = 0.4
        
        overall_score = (
            fundamental["fundamental_rating"] * fundamental_weight +
            technical["technical_rating"] * technical_weight
        )
        
        # Generate recommendation
        if overall_score >= 0.7:
            recommendation = "BUY"
            confidence = 0.8
        elif overall_score >= 0.6:
            recommendation = "WEAK_BUY"
            confidence = 0.6
        elif overall_score <= 0.3:
            recommendation = "SELL"
            confidence = 0.7
        elif overall_score <= 0.4:
            recommendation = "WEAK_SELL"
            confidence = 0.5
        else:
            recommendation = "HOLD"
            confidence = 0.4
        
        return {
            "overall_score": overall_score,
            "recommendation": recommendation,
            "confidence": confidence,
            "fundamental_contribution": fundamental["fundamental_rating"] * fundamental_weight,
            "technical_contribution": technical["technical_rating"] * technical_weight
        }
    
    def _generate_investment_thesis(self, symbol: str, fundamental: Dict[str, Any], technical: Dict[str, Any]) -> str:
        """Generate investment thesis"""
        thesis_parts = []
        
        # Fundamental thesis
        if fundamental["growth_score"] > 0.7:
            thesis_parts.append("Strong growth metrics support expansion story")
        if fundamental["quality_score"] > 0.7:
            thesis_parts.append("High-quality business with strong fundamentals")
        if fundamental["valuation_score"] > 0.6:
            thesis_parts.append("Attractive valuation relative to growth")
        
        # Technical thesis
        if technical["trend_score"] > 0.7:
            thesis_parts.append("Strong technical momentum supports upward move")
        if technical["momentum_score"] > 0.6:
            thesis_parts.append("Healthy momentum indicators suggest continued strength")
        
        return ". ".join(thesis_parts) + "." if thesis_parts else "Mixed signals require careful monitoring."
    
    def _identify_risks(self, symbol: str, fundamental: Dict[str, Any], technical: Dict[str, Any]) -> List[str]:
        """Identify key risks"""
        risks = []
        
        if fundamental["valuation_score"] < 0.3:
            risks.append("High valuation relative to fundamentals")
        if technical["momentum_score"] < 0.3:
            risks.append("Weak technical momentum")
        if technical.get("rsi_14", 50) > 80:
            risks.append("Overbought conditions may lead to pullback")
        
        # Add generic risks
        risks.extend([
            "Market volatility and systematic risk",
            "Sector-specific headwinds",
            "Execution risk on growth initiatives"
        ])
        
        return risks
    
    def _identify_catalysts(self, symbol: str, fundamental: Dict[str, Any], technical: Dict[str, Any]) -> List[str]:
        """Identify potential catalysts"""
        catalysts = []
        
        if fundamental["growth_score"] > 0.6:
            catalysts.append("Earnings beat and guidance raise")
        if technical["trend_score"] > 0.6:
            catalysts.append("Technical breakout above resistance")
        
        # Add generic catalysts
        catalysts.extend([
            "Positive industry developments",
            "New product launches or partnerships",
            "Market share gains"
        ])
        
        return catalysts
    
    def _calculate_target_price(self, symbol: str, fundamental: Dict[str, Any], technical: Dict[str, Any]) -> Dict[str, float]:
        """Calculate target price based on analysis"""
        # Get current price from fundamental metrics
        current_price = fundamental["key_metrics"]["current_price"]
        
        # Fundamental target (simple growth model)
        fundamental_target = current_price * (1 + fundamental["growth_score"] * 0.3)
        
        # Technical target (52-week high with adjustment)
        high_52w = fundamental["key_metrics"]["high_52w"]
        technical_target = high_52w * 1.1  # 10% above 52-week high
        
        # Weighted average
        target_price = (fundamental_target * 0.7 + technical_target * 0.3)
        
        return {
            "target_price": round(target_price, 2),
            "upside_potential": round((target_price - current_price) / current_price, 3),
            "fundamental_target": round(fundamental_target, 2),
            "technical_target": round(technical_target, 2)
        }
    
    def _calculate_fundamental_rating(self, valuation: float, growth: float, quality: float) -> float:
        """Calculate overall fundamental rating"""
        return (valuation * 0.3 + growth * 0.4 + quality * 0.3)
    
    def _calculate_technical_rating(self, trend: float, momentum: float) -> float:
        """Calculate overall technical rating"""
        return (trend * 0.6 + momentum * 0.4)
    
    def _momentum_screen(self, criteria: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Mock momentum screening"""
        return [
            {"symbol": f"MOM{i}", "score": 0.8 - i*0.05, "rsi": 65 - i*2, "momentum": 0.9 - i*0.1}
            for i in range(min(max_results, 10))
        ]
    
    def _value_screen(self, criteria: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Mock value screening"""
        return [
            {"symbol": f"VAL{i}", "score": 0.7 - i*0.04, "pe_ratio": 12 + i, "pb_ratio": 1.2 + i*0.2}
            for i in range(min(max_results, 10))
        ]
    
    def _growth_screen(self, criteria: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Mock growth screening"""
        return [
            {"symbol": f"GRW{i}", "score": 0.8 - i*0.03, "revenue_growth": 0.25 - i*0.02, "earnings_growth": 0.30 - i*0.03}
            for i in range(min(max_results, 10))
        ]
    
    def _breakout_screen(self, criteria: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Mock breakout screening"""
        return [
            {"symbol": f"BRK{i}", "score": 0.85 - i*0.04, "price_vs_resistance": 1.02 + i*0.01, "volume_spike": 2.5 - i*0.2}
            for i in range(min(max_results, 10))
        ]
    
    def _general_screen(self, criteria: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Mock general screening"""
        return [
            {"symbol": f"GEN{i}", "score": 0.6 - i*0.03, "overall_rating": 0.65 - i*0.04}
            for i in range(min(max_results, 10))
        ]
    
    def _generate_sector_outlook(self, sector: str, metrics: Dict[str, Any]) -> str:
        """Generate sector outlook commentary"""
        if metrics["relative_strength"] > 1.1:
            return f"{sector} sector showing strong outperformance with positive momentum indicators."
        elif metrics["relative_strength"] < 0.9:
            return f"{sector} sector underperforming with potential rotation concerns."
        else:
            return f"{sector} sector performance in line with market averages."
    
    def _generate_executive_summary(self, symbol: str, analysis: Dict[str, Any]) -> str:
        """Generate executive summary for research report"""
        recommendation = analysis["recommendation"]
        confidence = analysis["overall_assessment"]["confidence"]
        
        summary = f"{symbol} receives a {recommendation} recommendation with {confidence:.0%} confidence. "
        summary += analysis["investment_thesis"]
        
        return summary
    
    def _extract_key_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for report"""
        return {
            "overall_score": analysis["overall_assessment"]["overall_score"],
            "fundamental_rating": analysis["fundamental_analysis"]["fundamental_rating"],
            "technical_rating": analysis["technical_analysis"]["technical_rating"],
            "target_price": analysis["target_price"]["target_price"],
            "upside_potential": analysis["target_price"]["upside_potential"]
        }
    
    def _general_research_response(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general research requests"""
        content = message.get("content", "")
        
        if "screen" in content.lower():
            return {
                "type": "general_response",
                "message": "Research screening available. Use 'screen_securities' with criteria.",
                "available_screens": ["momentum", "value", "growth", "breakout"],
                "agent": self.name
            }
        else:
            return {
                "type": "general_response",
                "message": "Research services available. Use 'analyze_security', 'screen_securities', or 'sector_analysis'.",
                "available_services": [
                    "analyze_security",
                    "fundamental_analysis",
                    "technical_analysis", 
                    "screen_securities",
                    "sector_analysis",
                    "research_report"
                ],
                "agent": self.name
            }