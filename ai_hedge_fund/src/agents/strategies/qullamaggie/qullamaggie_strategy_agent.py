"""
Qullamaggie Strategy Agent for AI Hedge Fund System
Implements momentum swing trading strategy with breakouts, episodic pivots, and parabolic shorts
"""
import json
from datetime import datetime, timezone, time
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    # Try relative imports first (works in package context)
    from ...base_agent import BaseHedgeFundAgent, AgentCapability
    from ....models.trade import TradeSignal, OrderSide
    from ....utils.logging_config import get_logger
except ImportError:
    # Fall back to absolute imports (works in script context)
    from agents.base_agent import BaseHedgeFundAgent, AgentCapability
    from models.trade import TradeSignal, OrderSide
    from utils.logging_config import get_logger


class SetupType(str, Enum):
    BREAKOUT = "BREAKOUT"
    EPISODIC_PIVOT = "EP" 
    PARABOLIC_SHORT = "PARABOLIC_SHORT"


class MarketRegime(str, Enum):
    BULL = "BULL"
    CHOPPY = "CHOPPY" 
    BEAR = "BEAR"


class SectorStrength(str, Enum):
    HOT = "HOT"
    NORMAL = "NORMAL"
    WEAK = "WEAK"


@dataclass
class TradeSetup:
    """Container for trade setup data"""
    setup_type: SetupType
    symbol: str
    confidence: float
    entry_price: float
    stop_loss: float
    position_size: float
    risk_amount: float
    target_1: Optional[float]
    trailing_ma: int
    sector_strength: SectorStrength
    market_regime: MarketRegime
    notes: str
    
    # Technical data
    technical_metrics: Dict[str, Any]
    opening_ranges: Dict[str, Dict[str, float]]
    
    # Setup-specific data
    setup_data: Dict[str, Any]


class QullamaggieStrategyAgent(BaseHedgeFundAgent):
    """
    Qullamaggie momentum trading strategy agent
    Implements mechanical rule-based trading for breakouts, episodic pivots, and parabolic shorts
    """
    
    def __init__(self, account_size: float = 100000, **kwargs):
        system_message = """You are a Qullamaggie Strategy Agent for an AI hedge fund.

Your responsibilities:
1. Detect three types of momentum setups: Breakouts, Episodic Pivots, Parabolic Shorts
2. Apply strict market regime filtering (Bull/Choppy/Bear market rules)
3. Calculate precise position sizing based on account size and risk rules
4. Score setup confidence from 1-5 stars using mechanical criteria
5. Generate detailed trade signals with entry, stop, targets, and risk data
6. Enforce all Qullamaggie rules mechanically with no subjective interpretation

Setup Types:
- BREAKOUT: Consolidation breaks with volume on rising stocks
- EPISODIC_PIVOT: 10%+ gap up on catalyst with no recent rally
- PARABOLIC_SHORT: Overextended stocks showing exhaustion (large accounts only)

You MUST follow market regime rules:
- BULL: Trade all setups with full size
- CHOPPY: Only 5-star setups with 50% size  
- BEAR: Cash only, no new positions

Focus on mechanical rule execution with comprehensive risk management."""
        
        super().__init__(
            name="qullamaggie_strategy_agent",
            system_message=system_message,
            capabilities=[
                AgentCapability.SIGNAL_GENERATION,
                AgentCapability.MARKET_ANALYSIS
            ],
            **kwargs
        )
        
        self.account_size = account_size
    
    def _initialize(self) -> None:
        """Initialize Qullamaggie strategy agent"""
        # Load strategy configuration
        self.config = self._load_strategy_config()
        
        # Initialize account tier
        self.account_tier = self._determine_account_tier()
        
        # Setup tracking
        self.active_setups: Dict[str, TradeSetup] = {}
        self.rejected_setups: List[Dict[str, Any]] = []
        self.scan_history: List[Dict[str, Any]] = []
        
        # Market state tracking
        self.current_market_regime = MarketRegime.BULL
        self.last_regime_check = None
        
        self.logger.info(f"Qullamaggie Strategy Agent initialized - Account: ${self.account_size:,.0f}, Tier: {self.account_tier}")
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data"""
        message_type = data.get("type", "general")
        
        if message_type in ["scan_for_setups", "analyze_setup"]:
            return "symbol" in data or "symbols" in data
        elif message_type == "check_market_regime":
            return True
        
        return True
    
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """Process Qullamaggie strategy requests"""
        try:
            message_type = message.get("type", "general")
            
            if message_type == "scan_for_setups":
                return self._scan_for_setups(message)
            elif message_type == "analyze_setup":
                return self._analyze_setup(message)
            elif message_type == "check_market_regime":
                return self._check_market_regime(message)
            elif message_type == "get_position_sizing":
                return self._get_position_sizing(message)
            elif message_type == "validate_setup":
                return self._validate_setup(message)
            else:
                return self._general_response(message)
        
        except Exception as e:
            self.logger.error(f"Error processing Qullamaggie message: {e}")
            return {
                "type": "error",
                "error": str(e),
                "agent": self.name
            }
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load Qullamaggie strategy configuration"""
        config_path = Path(__file__).parent.parent.parent.parent.parent / "strategies" / "qullamaggie" / "rules.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info("Qullamaggie configuration loaded successfully")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load Qullamaggie config: {e}")
            # Return minimal default config
            return {
                "setups": {
                    "breakout": {"enabled": True, "min_confidence": 3.0},
                    "episodic_pivot": {"enabled": True, "min_confidence": 4.0},
                    "parabolic_short": {"enabled": False, "min_account_size": 1000000}
                }
            }
    
    def _determine_account_tier(self) -> str:
        """Determine account tier based on size"""
        if self.account_size < 100000:
            return "small"
        elif self.account_size < 1000000:
            return "medium"
        else:
            return "large"
    
    def _scan_for_setups(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for Qullamaggie setups across multiple symbols"""
        symbols = message.get("symbols", [])
        if not symbols:
            symbols = self._get_scan_universe()
        
        # Check market regime first
        regime_result = self._check_market_regime({})
        current_regime = MarketRegime(regime_result["market_regime"])
        
        # If bear market, return no setups
        if current_regime == MarketRegime.BEAR:
            self.logger.info("Bear market detected - no new setups")
            return {
                "type": "scan_results",
                "setups": [],
                "market_regime": current_regime.value,
                "message": "Bear market - cash only",
                "agent": self.name
            }
        
        # Scan each symbol
        found_setups = []
        rejected_count = 0
        
        for symbol in symbols:
            try:
                setup_result = self._analyze_setup({"symbol": symbol, "scan_mode": True})
                
                if setup_result.get("type") == "trade_setup":
                    setup = setup_result["setup"]
                    
                    # Apply regime filtering
                    if current_regime == MarketRegime.CHOPPY and setup.confidence < 5.0:
                        rejected_count += 1
                        self._log_rejected_setup(symbol, "choppy_market_not_5_star")
                        continue
                    
                    found_setups.append(setup)
                    self.logger.info(f"Found {setup.setup_type} setup for {symbol} - {setup.confidence:.1f} stars")
                else:
                    rejected_count += 1
            
            except Exception as e:
                self.logger.error(f"Error scanning {symbol}: {e}")
                rejected_count += 1
        
        # Sort setups by confidence (highest first)
        found_setups.sort(key=lambda x: x.confidence, reverse=True)
        
        # Log scan results
        scan_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols_scanned": len(symbols),
            "setups_found": len(found_setups),
            "setups_rejected": rejected_count,
            "market_regime": current_regime.value
        }
        
        self.scan_history.append(scan_summary)
        
        self.log_activity("setups_scanned", level="info", 
                         found=len(found_setups), rejected=rejected_count,
                         regime=current_regime.value)
        
        return {
            "type": "scan_results",
            "setups": [self._setup_to_dict(setup) for setup in found_setups],
            "scan_summary": scan_summary,
            "market_regime": current_regime.value,
            "agent": self.name
        }
    
    def _analyze_setup(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single symbol for Qullamaggie setups"""
        symbol = message.get("symbol")
        scan_mode = message.get("scan_mode", False)
        
        # Get technical metrics from Technical Analysis Agent
        technical_metrics = self._get_technical_metrics(symbol)
        if not technical_metrics:
            return {
                "type": "error",
                "error": f"Could not get technical data for {symbol}",
                "agent": self.name
            }
        
        # Apply basic filters first
        if not self._passes_basic_filters(symbol, technical_metrics):
            if not scan_mode:
                return {
                    "type": "rejected_setup",
                    "symbol": symbol,
                    "reason": "Failed basic filters",
                    "agent": self.name
                }
            return {"type": "no_setup"}
        
        # Check each setup type
        best_setup = None
        best_confidence = 0.0
        
        # 1. Check Breakout Setup
        if self.config["setups"]["breakout"]["enabled"]:
            breakout_setup = self._detect_breakout_setup(symbol, technical_metrics)
            if breakout_setup and breakout_setup.confidence > best_confidence:
                best_setup = breakout_setup
                best_confidence = breakout_setup.confidence
        
        # 2. Check Episodic Pivot Setup
        if self.config["setups"]["episodic_pivot"]["enabled"]:
            ep_setup = self._detect_episodic_pivot_setup(symbol, technical_metrics)
            if ep_setup and ep_setup.confidence > best_confidence:
                best_setup = ep_setup
                best_confidence = ep_setup.confidence
        
        # 3. Check Parabolic Short Setup (if account allows)
        if (self.config["setups"]["parabolic_short"]["enabled"] and 
            self.account_size >= self.config["setups"]["parabolic_short"]["min_account_size"]):
            short_setup = self._detect_parabolic_short_setup(symbol, technical_metrics)
            if short_setup and short_setup.confidence > best_confidence:
                best_setup = short_setup
                best_confidence = short_setup.confidence
        
        if best_setup:
            # Store active setup
            self.active_setups[symbol] = best_setup
            
            return {
                "type": "trade_setup",
                "symbol": symbol,
                "setup": best_setup,
                "agent": self.name
            }
        else:
            if not scan_mode:
                return {
                    "type": "no_setup",
                    "symbol": symbol, 
                    "message": "No valid Qullamaggie setups found",
                    "agent": self.name
                }
            return {"type": "no_setup"}
    
    def _get_technical_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get technical metrics from Technical Analysis Agent"""
        try:
            # Import here to avoid circular imports
            try:
                from ...core.technical_analyst_agent import TechnicalAnalystAgent
            except ImportError:
                from agents.core.technical_analyst_agent import TechnicalAnalystAgent
            
            # Create technical analyst agent instance
            technical_agent = TechnicalAnalystAgent()
            technical_agent._initialize()
            
            # Request all metrics for the symbol
            result = technical_agent.process_message({
                "type": "calculate_all_metrics",
                "symbol": symbol,
                "period": "1y",
                "interval": "1d"
            })
            
            if result.get("type") == "technical_metrics":
                metrics = result.get("metrics", {})
                
                # Add current price from the latest data
                if "price_data" in result:
                    latest_close = result["price_data"]["Close"].iloc[-1]
                    if "moving_averages" not in metrics:
                        metrics["moving_averages"] = {}
                    metrics["moving_averages"]["current_price"] = latest_close
                
                # Ensure all required fields exist with defaults
                self._ensure_required_metrics(metrics)
                
                return metrics
            else:
                self.logger.error(f"Failed to get technical metrics for {symbol}: {result.get('error', 'Unknown error')}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error getting technical metrics for {symbol}: {e}")
            # Return None to trigger rejection in filters
            return None
    
    def _ensure_required_metrics(self, metrics: Dict[str, Any]) -> None:
        """Ensure all required metric categories exist with default values"""
        defaults = {
            "moving_averages": {
                "sma_10": 0.0, "sma_20": 0.0, "sma_50": 0.0,
                "ma_alignment": False, "ma_angle_10": 0.0, "ma_angle_20": 0.0,
                "current_price": 0.0
            },
            "momentum_indicators": {
                "consecutive_up_days": 0, "consecutive_down_days": 0,
                "gain_22d": 0.0, "gain_67d": 0.0, "distance_from_52w_high": 0.0
            },
            "volatility_indicators": {
                "adr_20": 0.0, "atr_20": 0.0
            },
            "volume_indicators": {
                "dollar_volume_20d": 0, "volume_ratio": 1.0, "volume_surge": False
            },
            "pattern_metrics": {
                "gap_percent": 0.0, "consolidation_days": 0,
                "extension_from_20ma": 0.0
            },
            "setup_scores": {
                "breakout_score": 0.0, "short_score": 0.0
            }
        }
        
        for category, category_defaults in defaults.items():
            if category not in metrics:
                metrics[category] = {}
            for key, default_value in category_defaults.items():
                if key not in metrics[category]:
                    metrics[category][key] = default_value
    
    def _passes_basic_filters(self, symbol: str, metrics: Dict[str, Any]) -> bool:
        """Check if symbol passes basic Qullamaggie filters"""
        filters = self.config["scanning_criteria"]["filters"]
        
        # Check excluded symbols
        if symbol in self.config["scanning_criteria"]["excluded_symbols"]:
            self._log_rejected_setup(symbol, "excluded_symbol")
            return False
        
        # Check ADR requirement
        adr = metrics.get("volatility_indicators", {}).get("adr_20", 0)
        if adr < filters["min_adr_percent"]:
            self._log_rejected_setup(symbol, f"adr_too_low_{adr}")
            return False
        
        # Check dollar volume
        dollar_volume = metrics.get("volume_indicators", {}).get("dollar_volume_20d", 0)
        if dollar_volume < filters["min_dollar_volume"]:
            self._log_rejected_setup(symbol, f"dollar_volume_too_low_{dollar_volume}")
            return False
        
        # Check price minimum
        current_price = metrics.get("moving_averages", {}).get("current_price", 0)
        if current_price < filters["min_price"]:
            self._log_rejected_setup(symbol, f"price_too_low_{current_price}")
            return False
        
        return True
    
    def _detect_breakout_setup(self, symbol: str, metrics: Dict[str, Any]) -> Optional[TradeSetup]:
        """Detect breakout setup according to Qullamaggie rules"""
        config = self.config["setups"]["breakout"]
        
        # Check basic requirements
        consolidation_days = metrics.get("pattern_metrics", {}).get("consolidation_days", 0)
        if not (config["entry_conditions"]["consolidation_days"][0] <= consolidation_days <= config["entry_conditions"]["consolidation_days"][1]):
            return None
        
        # Check MA alignment
        ma_alignment = metrics.get("moving_averages", {}).get("ma_alignment", False)
        if not ma_alignment:
            return None
        
        # Check MA angles
        ma_angle_10 = metrics.get("moving_averages", {}).get("ma_angle_10", 0)
        ma_angle_20 = metrics.get("moving_averages", {}).get("ma_angle_20", 0)
        if ma_angle_10 < config["entry_conditions"]["ma_angle_minimum"]:
            return None
        
        # Check volume surge
        volume_surge = metrics.get("volume_indicators", {}).get("volume_surge", False)
        if not volume_surge:
            return None
        
        # Calculate confidence score
        confidence = self._calculate_breakout_confidence(symbol, metrics)
        
        if confidence < config["min_confidence"]:
            return None
        
        # Calculate position sizing and risk
        current_price = metrics.get("moving_averages", {}).get("current_price", 0)
        atr = metrics.get("volatility_indicators", {}).get("atr_20", 0)
        
        # Entry at opening range high (estimated)
        entry_price = current_price * 1.005  # Slight premium for OR break
        
        # Stop at day's low (estimated as current price - ATR)
        stop_loss = current_price - (atr * 0.75)
        
        # Position sizing
        position_size_pct, risk_amount = self._calculate_position_size(
            entry_price, stop_loss, confidence
        )
        
        # Target calculation (2:1 risk/reward typically)
        risk_per_share = entry_price - stop_loss
        target_1 = entry_price + (risk_per_share * 2)
        
        # Determine trailing MA based on ADR
        adr = metrics.get("volatility_indicators", {}).get("adr_20", 0)
        trailing_ma = 10 if adr > 6.0 else 20
        
        return TradeSetup(
            setup_type=SetupType.BREAKOUT,
            symbol=symbol,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size=position_size_pct,
            risk_amount=risk_amount,
            target_1=target_1,
            trailing_ma=trailing_ma,
            sector_strength=self._assess_sector_strength(symbol),
            market_regime=self.current_market_regime,
            notes=f"Breakout from {consolidation_days}-day consolidation, volume surge detected",
            technical_metrics=metrics,
            opening_ranges=self._calculate_opening_ranges(current_price, atr),
            setup_data={
                "consolidation_days": consolidation_days,
                "volume_ratio": metrics.get("volume_indicators", {}).get("volume_ratio", 0),
                "ma_angle_10": ma_angle_10,
                "ma_angle_20": ma_angle_20
            }
        )
    
    def _detect_episodic_pivot_setup(self, symbol: str, metrics: Dict[str, Any]) -> Optional[TradeSetup]:
        """Detect episodic pivot setup according to Qullamaggie rules"""
        config = self.config["setups"]["episodic_pivot"]
        
        # Check gap requirement
        gap_percent = abs(metrics.get("pattern_metrics", {}).get("gap_percent", 0))
        if gap_percent < config["entry_conditions"]["min_gap_percent"]:
            return None
        
        # Check for no recent rally (simplified - would need 3-6 month data)
        gain_67d = metrics.get("momentum_indicators", {}).get("gain_67d", 0)
        if gain_67d > 50.0:  # Had rally in last 3 months
            return None
        
        # EP setups need catalyst - this would be checked against news/earnings data
        # For now, assume gap itself indicates catalyst
        
        # Calculate confidence score
        confidence = self._calculate_ep_confidence(symbol, metrics)
        
        if confidence < config["min_confidence"]:
            return None
        
        current_price = metrics.get("moving_averages", {}).get("current_price", 0)
        atr = metrics.get("volatility_indicators", {}).get("atr_20", 0)
        
        # Entry at opening range high
        entry_price = current_price * 1.01  # Premium for OR break
        
        # Stop at day's low (more aggressive for EPs)
        stop_loss = current_price - (atr * 1.0)
        
        # Position sizing
        position_size_pct, risk_amount = self._calculate_position_size(
            entry_price, stop_loss, confidence
        )
        
        # EP targets are typically more aggressive
        risk_per_share = entry_price - stop_loss
        target_1 = entry_price + (risk_per_share * 3)  # 3:1 for EPs
        
        return TradeSetup(
            setup_type=SetupType.EPISODIC_PIVOT,
            symbol=symbol,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size=position_size_pct,
            risk_amount=risk_amount,
            target_1=target_1,
            trailing_ma=20,  # EPs typically use 20-day MA
            sector_strength=self._assess_sector_strength(symbol),
            market_regime=self.current_market_regime,
            notes=f"EP setup with {gap_percent:.1f}% gap, no recent rally",
            technical_metrics=metrics,
            opening_ranges=self._calculate_opening_ranges(current_price, atr),
            setup_data={
                "gap_percent": gap_percent,
                "gain_67d": gain_67d,
                "catalyst": "gap_up"  # Would be more specific with real data
            }
        )
    
    def _detect_parabolic_short_setup(self, symbol: str, metrics: Dict[str, Any]) -> Optional[TradeSetup]:
        """Detect parabolic short setup (large accounts only)"""
        config = self.config["setups"]["parabolic_short"]
        
        # Check consecutive up days
        consecutive_up = metrics.get("momentum_indicators", {}).get("consecutive_up_days", 0)
        if consecutive_up < config["entry_conditions"]["consecutive_up_days"][0]:
            return None
        
        # Check extension from 20-day MA
        extension_20ma = metrics.get("pattern_metrics", {}).get("extension_from_20ma", 0)
        if extension_20ma < config["entry_conditions"]["min_extension_from_ma20"]:
            return None
        
        # Check for exhaustion signals (would be more complex in real implementation)
        exhaustion_score = metrics.get("setup_scores", {}).get("short_score", 0)
        if exhaustion_score < config["min_confidence"]:
            return None
        
        current_price = metrics.get("moving_averages", {}).get("current_price", 0)
        atr = metrics.get("volatility_indicators", {}).get("atr_20", 0)
        
        # Entry methods: OR low break, first red candle, or VWAP rejection
        entry_price = current_price * 0.995  # Slight discount for short entry
        
        # Stop at day's high
        stop_loss = current_price + (atr * 0.75)
        
        # Position sizing (smaller for shorts)
        position_size_pct, risk_amount = self._calculate_position_size(
            entry_price, stop_loss, exhaustion_score, is_short=True
        )
        
        # Targets at MA levels
        sma_20 = metrics.get("moving_averages", {}).get("sma_20", current_price)
        target_1 = sma_20
        
        return TradeSetup(
            setup_type=SetupType.PARABOLIC_SHORT,
            symbol=symbol,
            confidence=exhaustion_score,
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size=position_size_pct,
            risk_amount=risk_amount,
            target_1=target_1,
            trailing_ma=20,
            sector_strength=self._assess_sector_strength(symbol),
            market_regime=self.current_market_regime,
            notes=f"Parabolic short: {consecutive_up} up days, {extension_20ma:.1f}% extended",
            technical_metrics=metrics,
            opening_ranges=self._calculate_opening_ranges(current_price, atr),
            setup_data={
                "consecutive_up_days": consecutive_up,
                "extension_from_ma20": extension_20ma,
                "exhaustion_score": exhaustion_score
            }
        )
    
    def _calculate_breakout_confidence(self, symbol: str, metrics: Dict[str, Any]) -> float:
        """Calculate confidence score for breakout setup (1-5)"""
        score = 1.0
        
        # Volume surge adds confidence
        volume_ratio = metrics.get("volume_indicators", {}).get("volume_ratio", 1.0)
        if volume_ratio > 2.0:
            score += 1.5
        elif volume_ratio > 1.5:
            score += 1.0
        
        # Clean consolidation adds confidence
        consolidation_days = metrics.get("pattern_metrics", {}).get("consolidation_days", 0)
        if 5 <= consolidation_days <= 20:
            score += 1.0
        
        # MA alignment and angles
        ma_angle_10 = metrics.get("moving_averages", {}).get("ma_angle_10", 0)
        if ma_angle_10 > 45:
            score += 1.0
        
        # Prior strong move
        gain_22d = metrics.get("momentum_indicators", {}).get("gain_22d", 0)
        if gain_22d > 25:
            score += 0.5
        
        return min(5.0, score)
    
    def _calculate_ep_confidence(self, symbol: str, metrics: Dict[str, Any]) -> float:
        """Calculate confidence score for EP setup (1-5)"""
        score = 2.0  # EP base score
        
        # Gap size
        gap_percent = abs(metrics.get("pattern_metrics", {}).get("gap_percent", 0))
        if gap_percent > 15:
            score += 1.0
        elif gap_percent > 10:
            score += 0.5
        
        # No recent rally bonus
        gain_67d = metrics.get("momentum_indicators", {}).get("gain_67d", 0)
        if gain_67d < 10:  # Very little rally in 3 months
            score += 1.0
        
        # Volume confirmation
        volume_ratio = metrics.get("volume_indicators", {}).get("volume_ratio", 1.0)
        if volume_ratio > 3.0:
            score += 1.0
        elif volume_ratio > 2.0:
            score += 0.5
        
        return min(5.0, score)
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float, confidence: float, is_short: bool = False) -> Tuple[float, float]:
        """Calculate position size based on account tier and risk"""
        tier_config = self.config["position_sizing"]["account_tiers"][self.account_tier]
        
        # Risk per trade based on confidence
        if confidence >= 4.5:
            risk_pct = tier_config["risk_per_trade"][1]  # Max risk for high confidence
        else:
            risk_pct = tier_config["risk_per_trade"][0]  # Min risk
        
        # Calculate risk amount
        risk_amount = self.account_size * (risk_pct / 100)
        
        # Calculate position size based on stop distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance > 0:
            shares = int(risk_amount / stop_distance)
            position_value = shares * entry_price
            position_pct = (position_value / self.account_size) * 100
            
            # Apply position limits
            max_position_pct = tier_config["max_position_percent"]
            if isinstance(max_position_pct, list):
                max_position_pct = max_position_pct[1]  # Use max for high confidence
            
            if position_pct > max_position_pct:
                position_pct = max_position_pct
                position_value = self.account_size * (position_pct / 100)
                risk_amount = position_value * (stop_distance / entry_price)
        else:
            position_pct = 0.0
            risk_amount = 0.0
        
        # Reduce size for shorts or choppy markets
        if is_short:
            position_pct *= 0.75
            risk_amount *= 0.75
        
        if self.current_market_regime == MarketRegime.CHOPPY:
            position_pct *= 0.5
            risk_amount *= 0.5
        
        return position_pct, risk_amount
    
    def _assess_sector_strength(self, symbol: str) -> SectorStrength:
        """Assess sector strength (placeholder - would use real sector analysis)"""
        # This would integrate with sector analysis from other agents
        return SectorStrength.NORMAL
    
    def _calculate_opening_ranges(self, current_price: float, atr: float) -> Dict[str, Dict[str, float]]:
        """Calculate estimated opening range levels"""
        # This would use actual intraday data in real implementation
        return {
            "1m": {"high": current_price + (atr * 0.1), "low": current_price - (atr * 0.1)},
            "5m": {"high": current_price + (atr * 0.2), "low": current_price - (atr * 0.2)},
            "60m": {"high": current_price + (atr * 0.5), "low": current_price - (atr * 0.5)}
        }
    
    def _check_market_regime(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Check current market regime using Nasdaq index analysis"""
        try:
            # Use QQQ (Nasdaq ETF) as market regime indicator per Qullamaggie rules
            nasdaq_metrics = self._get_technical_metrics("QQQ")
            
            if not nasdaq_metrics:
                self.logger.warning("Could not get Nasdaq data, defaulting to BULL regime")
                regime = MarketRegime.BULL
                confidence = 0.5
                message = "Could not determine regime - defaulting to bull market"
            else:
                regime, confidence, message = self._analyze_market_regime(nasdaq_metrics)
            
            self.current_market_regime = regime
            self.last_regime_check = datetime.now(timezone.utc)
            
            return {
                "type": "market_regime",
                "market_regime": regime.value,
                "confidence": confidence,
                "message": message,
                "nasdaq_data": {
                    "sma_10": nasdaq_metrics.get("moving_averages", {}).get("sma_10") if nasdaq_metrics else None,
                    "sma_20": nasdaq_metrics.get("moving_averages", {}).get("sma_20") if nasdaq_metrics else None,
                    "ma_angle_10": nasdaq_metrics.get("moving_averages", {}).get("ma_angle_10") if nasdaq_metrics else None,
                    "ma_angle_20": nasdaq_metrics.get("moving_averages", {}).get("ma_angle_20") if nasdaq_metrics else None
                },
                "agent": self.name
            }
            
        except Exception as e:
            self.logger.error(f"Error checking market regime: {e}")
            # Default to conservative approach in case of error
            self.current_market_regime = MarketRegime.CHOPPY
            return {
                "type": "error",
                "error": str(e),
                "market_regime": MarketRegime.CHOPPY.value,
                "agent": self.name
            }
    
    def _analyze_market_regime(self, nasdaq_metrics: Dict[str, Any]) -> Tuple[MarketRegime, float, str]:
        """Analyze Nasdaq metrics to determine market regime"""
        ma_data = nasdaq_metrics.get("moving_averages", {})
        sma_10 = ma_data.get("sma_10", 0)
        sma_20 = ma_data.get("sma_20", 0) 
        ma_angle_10 = ma_data.get("ma_angle_10", 0)
        ma_angle_20 = ma_data.get("ma_angle_20", 0)
        
        # Apply Qullamaggie market regime rules from rules.json
        regime_rules = self.config["market_regime_rules"]
        
        # Check Bull Market conditions
        if (sma_10 > sma_20 and 
            ma_angle_10 >= 45 and  # 45 degrees minimum angle
            ma_angle_20 >= 45):
            
            confidence = min(0.9, 0.6 + (ma_angle_10 / 100) + (ma_angle_20 / 100))
            message = f"Bull market: MA10 > MA20, strong uptrend angles ({ma_angle_10:.1f}°, {ma_angle_20:.1f}°)"
            return MarketRegime.BULL, confidence, message
        
        # Check Bear Market conditions  
        elif (sma_10 < sma_20 and
              ma_angle_10 <= -45 and ma_angle_20 <= -45):
            
            confidence = min(0.9, 0.6 + abs(ma_angle_10 / 100) + abs(ma_angle_20 / 100))
            message = f"Bear market: MA10 < MA20, strong downtrend angles ({ma_angle_10:.1f}°, {ma_angle_20:.1f}°)"
            return MarketRegime.BEAR, confidence, message
        
        # Default to Choppy Market
        else:
            confidence = 0.7
            if abs(ma_angle_10) < 15 and abs(ma_angle_20) < 15:
                message = f"Choppy market: Sideways moving averages ({ma_angle_10:.1f}°, {ma_angle_20:.1f}°)"
            else:
                message = f"Choppy market: Mixed signals, MA10 {'>' if sma_10 > sma_20 else '<'} MA20 but angles unclear"
            
            return MarketRegime.CHOPPY, confidence, message
    
    def _get_scan_universe(self) -> List[str]:
        """Get universe of symbols to scan"""
        # This would come from a stock screener or predefined universe
        # For now, return a sample universe
        return [
            "AAPL", "TSLA", "NVDA", "AMD", "ROKU", "ZM", "PTON", "NFLX",
            "SHOP", "SQ", "PYPL", "TWLO", "OKTA", "CRWD", "ZS", "DDOG"
        ]
    
    def _log_rejected_setup(self, symbol: str, reason: str) -> None:
        """Log rejected setup for analysis"""
        rejected = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "reason": reason
        }
        self.rejected_setups.append(rejected)
        
        if self.config.get("logging", {}).get("log_rejected_setups", True):
            self.logger.debug(f"Rejected setup: {symbol} - {reason}")
    
    def _setup_to_dict(self, setup: TradeSetup) -> Dict[str, Any]:
        """Convert TradeSetup to dictionary for serialization"""
        return {
            "setup_type": setup.setup_type.value,
            "symbol": setup.symbol,
            "confidence": setup.confidence,
            "entry_price": setup.entry_price,
            "stop_loss": setup.stop_loss,
            "position_size": setup.position_size,
            "risk_amount": setup.risk_amount,
            "target_1": setup.target_1,
            "trailing_ma": setup.trailing_ma,
            "sector_strength": setup.sector_strength.value,
            "market_regime": setup.market_regime.value,
            "notes": setup.notes,
            "setup_data": setup.setup_data
        }
    
    def _validate_setup(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a setup before execution"""
        symbol = message.get("symbol")
        
        if symbol not in self.active_setups:
            return {
                "type": "validation_result",
                "valid": False,
                "reason": "No active setup for symbol",
                "agent": self.name
            }
        
        setup = self.active_setups[symbol]
        
        # Re-check market regime
        regime_result = self._check_market_regime({})
        current_regime = MarketRegime(regime_result["market_regime"])
        
        if current_regime != setup.market_regime:
            return {
                "type": "validation_result",
                "valid": False,
                "reason": f"Market regime changed from {setup.market_regime.value} to {current_regime.value}",
                "agent": self.name
            }
        
        # Setup is still valid
        return {
            "type": "validation_result",
            "valid": True,
            "setup": self._setup_to_dict(setup),
            "agent": self.name
        }
    
    def _get_position_sizing(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position sizing for a setup"""
        entry_price = message.get("entry_price", 0)
        stop_loss = message.get("stop_loss", 0)
        confidence = message.get("confidence", 3.0)
        
        position_size_pct, risk_amount = self._calculate_position_size(entry_price, stop_loss, confidence)
        
        return {
            "type": "position_sizing",
            "position_size_percent": position_size_pct,
            "risk_amount": risk_amount,
            "account_tier": self.account_tier,
            "account_size": self.account_size,
            "agent": self.name
        }
    
    def _general_response(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general requests"""
        return {
            "type": "general_response",
            "message": "Qullamaggie Strategy Agent ready. Available methods: scan_for_setups, analyze_setup, check_market_regime, get_position_sizing, validate_setup",
            "account_info": {
                "account_size": self.account_size,
                "account_tier": self.account_tier,
                "active_setups": len(self.active_setups)
            },
            "setups_enabled": {
                "breakout": self.config["setups"]["breakout"]["enabled"],
                "episodic_pivot": self.config["setups"]["episodic_pivot"]["enabled"],
                "parabolic_short": self.config["setups"]["parabolic_short"]["enabled"] and self.account_size >= self.config["setups"]["parabolic_short"]["min_account_size"]
            },
            "agent": self.name
        }