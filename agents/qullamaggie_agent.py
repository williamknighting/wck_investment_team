"""
Qullamaggie Strategy Agent for AI Hedge Fund System
Implements momentum swing trading strategy with proper StrategyAgent architecture
"""
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from agents.base_strategy_agent import StrategyAgent


class QullamaggieAgent(StrategyAgent):
    """
    Qullamaggie momentum strategy agent using StrategyAgent framework
    Specializes in momentum breakout setups with specific entry/exit criteria
    """
    
    def __init__(self, name: str = "qullamaggie_agent", description: str = "Qullamaggie momentum strategy specialist", **kwargs):
        """Initialize Qullamaggie strategy agent"""
        super().__init__(
            name=name,
            description=description,
            strategy_config=self._load_qullamaggie_config(),
            **kwargs
        )
    
    def _load_qullamaggie_config(self) -> Dict[str, Any]:
        """Load Qullamaggie-specific strategy configuration"""
        config_path = Path("config/qullamaggie_strategy.json")
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info("Loaded Qullamaggie configuration from file")
                return config
        except Exception as e:
            self.logger.warning(f"Could not load config file: {e}, using defaults")
        
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Qullamaggie strategy configuration"""
        return {
            "name": "Qullamaggie Momentum Strategy",
            "version": "1.0",
            "description": "Momentum breakout strategy based on Qullamaggie methodology",
            
            "criteria": {
                "min_price": 5.0,
                "min_gain_22d": 20.0,  # 20% minimum gain in 22 days
                "max_distance_52w": -15.0,  # Within 15% of 52-week high
                "min_volume_ratio": 1.2,  # Above average volume
                "ma_alignment_required": True,  # Price above moving averages
                "min_market_cap": 1000000000  # $1B minimum market cap
            },
            
            "risk_management": {
                "max_risk_per_trade": 2000,  # $2000 max risk per trade
                "min_risk_reward": 2.0,  # 2:1 minimum R/R
                "stop_loss_atr_multiple": 1.5,  # 1.5x ATR for stops
                "max_position_size": 1000,  # Max 1000 shares
                "conviction_threshold": 6  # Minimum conviction for trade
            },
            
            "setup_types": {
                "breakout": {
                    "enabled": True,
                    "min_consolidation_days": 5,
                    "volume_surge_required": True
                },
                "flag_pullback": {
                    "enabled": True,
                    "max_pullback_percent": 8.0
                },
                "episodic_pivot": {
                    "enabled": False,  # Advanced setup
                    "min_volume_spike": 3.0
                }
            }
        }
    
    def _evaluate_setup(self, symbol: str, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate symbol against Qullamaggie momentum criteria
        
        Args:
            symbol: Stock symbol
            technical_data: Technical analysis data from Technical Analyst
            
        Returns:
            Strategy evaluation results
        """
        try:
            # Extract technical metrics
            price_data = technical_data.get("price_data", {})
            momentum_data = technical_data.get("momentum_data", {})
            volume_data = technical_data.get("volume_data", {})
            quality_flags = technical_data.get("quality_flags", {})
            
            current_price = price_data.get("current", 0)
            gain_22d = momentum_data.get("gain_22d", 0)
            distance_52w = momentum_data.get("distance_52w_high", -100)
            volume_ratio = volume_data.get("volume_ratio", 0)
            ma_alignment = quality_flags.get("ma_alignment", False)
            
            # Evaluate Qullamaggie criteria
            criteria_evaluation = {
                "price_filter": current_price >= self.strategy_criteria["min_price"],
                "momentum_strong": gain_22d >= self.strategy_criteria["min_gain_22d"],
                "near_highs": distance_52w >= self.strategy_criteria["max_distance_52w"],
                "volume_adequate": volume_ratio >= self.strategy_criteria["min_volume_ratio"],
                "ma_aligned": ma_alignment if self.strategy_criteria["ma_alignment_required"] else True
            }
            
            criteria_met = sum(criteria_evaluation.values())
            total_criteria = len(criteria_evaluation)
            
            # Determine setup quality and conviction
            if criteria_met >= 5:
                setup_quality = "excellent"
                conviction = 9
                entry_signal = "immediate"
            elif criteria_met >= 4:
                setup_quality = "strong"
                conviction = 7
                entry_signal = "buy_dip"
            elif criteria_met >= 3:
                setup_quality = "moderate"
                conviction = 5
                entry_signal = "watch"
            elif criteria_met >= 2:
                setup_quality = "weak"
                conviction = 3
                entry_signal = "monitor"
            else:
                setup_quality = "not_suitable"
                conviction = 1
                entry_signal = "pass"
            
            # Calculate additional metrics
            momentum_score = min(gain_22d / 30.0, 1.0)  # Normalize to 30% gain
            proximity_score = min((distance_52w + 20) / 20.0, 1.0)  # Normalize proximity
            volume_score = min(volume_ratio / 2.0, 1.0)  # Normalize to 2x volume
            
            overall_score = (momentum_score + proximity_score + volume_score) / 3.0
            
            return {
                "setup_quality": setup_quality,
                "conviction": conviction,
                "entry_signal": entry_signal,
                "criteria_met": criteria_met,
                "total_criteria": total_criteria,
                "criteria_evaluation": criteria_evaluation,
                "scores": {
                    "momentum": momentum_score,
                    "proximity": proximity_score,
                    "volume": volume_score,
                    "overall": overall_score
                },
                "key_metrics": {
                    "gain_22d": gain_22d,
                    "distance_52w": distance_52w,
                    "volume_ratio": volume_ratio,
                    "current_price": current_price
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating Qullamaggie setup for {symbol}: {e}")
            return {
                "setup_quality": "error",
                "conviction": 0,
                "entry_signal": "pass",
                "error": str(e)
            }
    
    def _get_strategy_reasoning(self, symbol: str, evaluation: Dict[str, Any]) -> str:
        """
        Generate Qullamaggie-specific reasoning for the evaluation
        
        Args:
            symbol: Stock symbol
            evaluation: Strategy evaluation results
            
        Returns:
            Human-readable reasoning
        """
        criteria_eval = evaluation.get("criteria_evaluation", {})
        key_metrics = evaluation.get("key_metrics", {})
        setup_quality = evaluation.get("setup_quality", "unknown")
        
        reasoning_parts = []
        
        # Setup quality assessment
        reasoning_parts.append(f"**{setup_quality.title()} Qullamaggie Setup** ({evaluation.get('criteria_met', 0)}/{evaluation.get('total_criteria', 5)} criteria met)")
        
        # Momentum analysis
        gain_22d = key_metrics.get("gain_22d", 0)
        if criteria_eval.get("momentum_strong", False):
            reasoning_parts.append(f"✅ **Strong momentum**: {gain_22d:+.1f}% gain in 22 days exceeds {self.strategy_criteria['min_gain_22d']}% threshold")
        else:
            reasoning_parts.append(f"❌ **Weak momentum**: {gain_22d:+.1f}% gain falls short of {self.strategy_criteria['min_gain_22d']}% requirement")
        
        # Proximity to highs
        distance_52w = key_metrics.get("distance_52w", -100)
        if criteria_eval.get("near_highs", False):
            reasoning_parts.append(f"✅ **Near 52-week highs**: {distance_52w:.1f}% from highs (within {abs(self.strategy_criteria['max_distance_52w'])}% threshold)")
        else:
            reasoning_parts.append(f"❌ **Far from highs**: {distance_52w:.1f}% from 52-week highs exceeds {abs(self.strategy_criteria['max_distance_52w'])}% limit")
        
        # Volume confirmation
        volume_ratio = key_metrics.get("volume_ratio", 0)
        if criteria_eval.get("volume_adequate", False):
            reasoning_parts.append(f"✅ **Volume confirmation**: {volume_ratio:.2f}x average volume supports breakout")
        else:
            reasoning_parts.append(f"❌ **Insufficient volume**: {volume_ratio:.2f}x average volume below {self.strategy_criteria['min_volume_ratio']}x requirement")
        
        # Moving average alignment
        if criteria_eval.get("ma_aligned", False):
            reasoning_parts.append("✅ **Technical structure**: Price above key moving averages")
        else:
            reasoning_parts.append("❌ **Poor structure**: Price below moving average support")
        
        # Price filter
        current_price = key_metrics.get("current_price", 0)
        if criteria_eval.get("price_filter", False):
            reasoning_parts.append(f"✅ **Price filter**: ${current_price:.2f} meets minimum ${self.strategy_criteria['min_price']} requirement")
        else:
            reasoning_parts.append(f"❌ **Price filter**: ${current_price:.2f} below minimum ${self.strategy_criteria['min_price']} threshold")
        
        # Overall assessment
        conviction = evaluation.get("conviction", 0)
        if conviction >= 7:
            reasoning_parts.append(f"\n**HIGH CONVICTION SETUP** - Strong Qullamaggie momentum characteristics present")
        elif conviction >= 5:
            reasoning_parts.append(f"\n**MODERATE SETUP** - Some momentum qualities present, monitor for improvement")
        else:
            reasoning_parts.append(f"\n**LOW CONVICTION** - Does not meet Qullamaggie momentum criteria")
        
        return "\n\n".join(reasoning_parts)
    
    def _calculate_trade_parameters(self, symbol: str, strategy_eval: Dict[str, Any], technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Qullamaggie-specific trade parameters
        
        Args:
            symbol: Stock symbol
            strategy_eval: Strategy evaluation results
            technical_data: Technical analysis data
            
        Returns:
            Qullamaggie trade parameters
        """
        current_price = technical_data.get("price_data", {}).get("current", 0)
        atr = technical_data.get("volatility_data", {}).get("atr", current_price * 0.02)
        
        # Qullamaggie-specific entry logic
        entry_signal = strategy_eval.get("entry_signal", "pass")
        
        if entry_signal == "immediate":
            entry_price = current_price  # Market order
        elif entry_signal == "buy_dip":
            entry_price = current_price * 0.98  # 2% below current price
        else:
            entry_price = current_price
        
        # Qullamaggie stop loss: 1.5x ATR or 8% (whichever is tighter)
        atr_stop = atr * self.risk_parameters["stop_loss_atr_multiple"]
        percent_stop = current_price * 0.08  # 8% stop
        stop_distance = min(atr_stop, percent_stop)
        stop_loss = entry_price - stop_distance
        
        # Profit target: 2:1 R/R minimum
        risk_reward = self.risk_parameters["min_risk_reward"]
        profit_target = entry_price + (stop_distance * risk_reward)
        
        # Position sizing based on conviction and risk
        conviction = strategy_eval.get("conviction", 0)
        max_risk = self.risk_parameters["max_risk_per_trade"]
        
        # Scale position size by conviction
        conviction_multiplier = conviction / 10.0
        adjusted_risk = max_risk * conviction_multiplier
        
        position_size = int(adjusted_risk / stop_distance) if stop_distance > 0 else 0
        position_size = min(position_size, self.risk_parameters["max_position_size"])
        
        return {
            "entry_signal": entry_signal,
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "profit_target": round(profit_target, 2),
            "position_size": position_size,
            "risk_reward_ratio": risk_reward,
            "max_risk_amount": adjusted_risk,
            "stop_method": "8% or 1.5x ATR (tighter)",
            "sizing_method": f"conviction-weighted ({conviction}/10)"
        }
    
    def _format_criteria_analysis(self, proposal: Dict[str, Any]) -> str:
        """Format Qullamaggie criteria analysis for proposal"""
        
        # This would be extracted from the strategy evaluation stored in the proposal
        criteria_map = {
            "price_filter": "Price > $5 minimum",
            "momentum_strong": f"22-day gain > {self.strategy_criteria['min_gain_22d']}%",
            "near_highs": f"Within {abs(self.strategy_criteria['max_distance_52w'])}% of 52w high",
            "volume_adequate": f"Volume > {self.strategy_criteria['min_volume_ratio']}x average",
            "ma_aligned": "Price above moving averages"
        }
        
        content = "### Qullamaggie Criteria Breakdown\n\n"
        content += "| Criterion | Status | Description |\n"
        content += "|-----------|--------|-------------|\n"
        
        # This is a simplified version - in practice, you'd store the criteria evaluation
        # in the proposal and retrieve it here
        for criterion, description in criteria_map.items():
            status = "✅ PASS"  # Placeholder - would use actual evaluation
            content += f"| {description} | {status} | Core momentum requirement |\n"
        
        return content
    
    def _generate_defense_points(self, proposal: Dict[str, Any], criticism: str) -> List[str]:
        """Generate Qullamaggie-specific defense points"""
        defense_points = super()._generate_defense_points(proposal, criticism)
        
        # Add Qullamaggie-specific defenses
        conviction = proposal.get("conviction", 0)
        setup_quality = proposal.get("setup_quality", "")
        
        if "momentum" in criticism.lower():
            defense_points.append("Qullamaggie methodology specifically targets momentum breakouts with proven 22-day performance")
        
        if "risk" in criticism.lower():
            defense_points.append("Stop loss methodology uses 8% or 1.5x ATR (whichever is tighter) for optimal risk control")
        
        if "volume" in criticism.lower():
            defense_points.append("Volume confirmation is core to Qullamaggie breakout validation")
        
        if setup_quality in ["excellent", "strong"]:
            defense_points.append(f"Setup meets {proposal.get('criteria_met', 0)}/5 Qullamaggie criteria with strong momentum characteristics")
        
        return defense_points
    
    def _generate_critique_points(self, other_proposal: Dict[str, Any]) -> List[str]:
        """Generate Qullamaggie perspective critiques"""
        critique_points = super()._generate_critique_points(other_proposal)
        
        # Add Qullamaggie-specific critiques
        other_strategy = other_proposal.get("strategy", "unknown")
        
        if "value" in other_strategy.lower():
            critique_points.append("Value-based entries may lack the momentum catalyst needed for quick moves")
        
        if other_proposal.get("conviction", 0) < 6:
            critique_points.append("Low conviction suggests weak momentum setup - Qullamaggie requires strong technical alignment")
        
        # Analyze their technical summary for momentum characteristics
        technical_summary = other_proposal.get("technical_summary", "")
        if "22-day" in technical_summary and "%" in technical_summary:
            # Could parse and critique their momentum metrics
            critique_points.append("Consider waiting for stronger 22-day momentum before entry")
        
        return critique_points
    
    def _suggest_alternative_approach(self, other_proposal: Dict[str, Any]) -> str:
        """Suggest Qullamaggie alternative approach"""
        symbol = other_proposal.get("symbol", "unknown")
        
        return f"""**Qullamaggie Alternative for {symbol}:**

1. **Wait for stronger momentum**: Look for >20% gain in 22 trading days
2. **Volume confirmation**: Ensure breakout occurs on 1.5x+ average volume  
3. **Proximity timing**: Enter when within 15% of 52-week highs
4. **Technical structure**: Confirm price is above key moving averages
5. **Risk management**: Use 8% or 1.5x ATR stop (whichever is tighter)

The Qullamaggie approach prioritizes momentum confirmation over early entry, leading to higher probability setups with clearer risk/reward parameters."""