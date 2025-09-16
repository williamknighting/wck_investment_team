"""
Paper Trading System for AI Hedge Fund
Simulates trade execution with realistic market conditions
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from enum import Enum
import asyncio
import random
import uuid

from ..models.trade import Order, Trade, OrderStatus, OrderSide, OrderType
from ..models.portfolio import Portfolio
from ..models.position import Position
from ..utils.logging_config import get_logger, log_trade_execution


class MarketCondition(str, Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    ILLIQUID = "illiquid"
    HALTED = "halted"


@dataclass
class ExecutionResult:
    """Result of trade execution attempt"""
    success: bool
    order_id: str
    executed_quantity: int = 0
    executed_price: Optional[Decimal] = None
    remaining_quantity: int = 0
    fees: Decimal = Decimal('0')
    error_message: Optional[str] = None
    execution_time: Optional[datetime] = None


@dataclass
class MarketData:
    """Current market data for a symbol"""
    symbol: str
    bid_price: Decimal
    ask_price: Decimal
    last_price: Decimal
    volume: int
    timestamp: datetime
    spread_pct: float
    
    @property
    def mid_price(self) -> Decimal:
        return (self.bid_price + self.ask_price) / 2


class PaperTrader:
    """
    Paper trading system that simulates realistic trade execution
    Includes slippage, partial fills, market impact, and latency
    """
    
    def __init__(self, portfolio: Portfolio, config: Dict[str, Any] = None):
        """
        Initialize paper trader
        
        Args:
            portfolio: Portfolio to trade with
            config: Configuration parameters
        """
        self.portfolio = portfolio
        self.config = config or {}
        self.logger = get_logger("paper_trader")
        
        # Trading parameters
        self.commission_per_share = Decimal(str(self.config.get("commission_per_share", "0.005")))
        self.min_commission = Decimal(str(self.config.get("min_commission", "1.0")))
        self.slippage_base = self.config.get("slippage_base", 0.001)  # 0.1% base slippage
        self.market_impact_factor = self.config.get("market_impact_factor", 0.0001)
        self.latency_ms = self.config.get("latency_ms", 100)
        
        # Market simulation
        self.market_data: Dict[str, MarketData] = {}
        self.market_condition = MarketCondition.NORMAL
        self.trading_hours = True
        
        # Order management
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Trade] = []
        
        # Execution engine
        self.partial_fill_probability = 0.1
        self.reject_probability = 0.02
        
        self.logger.info("Paper trader initialized")
    
    async def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit order for execution
        
        Args:
            order: Order to execute
            
        Returns:
            ExecutionResult with execution details
        """
        try:
            # Validate order
            validation_result = self._validate_order(order)
            if not validation_result.success:
                return validation_result
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            order.submitted_at = datetime.now(timezone.utc)
            order.status = OrderStatus.SUBMITTED
            
            # Simulate network latency
            await asyncio.sleep(self.latency_ms / 1000.0)
            
            # Execute order
            execution_result = await self._execute_order(order)
            
            # Update portfolio if execution successful
            if execution_result.success and execution_result.executed_quantity > 0:
                await self._update_portfolio(order, execution_result)
            
            # Clean up completed orders
            if order.is_filled or order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                self.active_orders.pop(order.order_id, None)
                self.order_history.append(order)
            
            self.logger.info(f"Order {order.order_id} processed: {execution_result.success}")
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Error submitting order {order.order_id}: {e}")
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                error_message=str(e)
            )
    
    def _validate_order(self, order: Order) -> ExecutionResult:
        """Validate order before submission"""
        # Check trading hours
        if not self.trading_hours:
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                error_message="Market closed"
            )
        
        # Check market condition
        if self.market_condition == MarketCondition.HALTED:
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                error_message="Trading halted"
            )
        
        # Check order parameters
        if order.quantity <= 0:
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                error_message="Invalid quantity"
            )
        
        # Check buying power for buy orders
        if order.side == OrderSide.BUY:
            estimated_cost = self._estimate_order_cost(order)
            if estimated_cost > self.portfolio.current_cash:
                return ExecutionResult(
                    success=False,
                    order_id=order.order_id,
                    error_message="Insufficient buying power"
                )
        
        # Check position for sell orders
        if order.side == OrderSide.SELL:
            position = self.portfolio.get_position(order.symbol)
            available_shares = position.quantity if position else 0
            if available_shares < order.quantity:
                return ExecutionResult(
                    success=False,
                    order_id=order.order_id,
                    error_message="Insufficient shares to sell"
                )
        
        return ExecutionResult(success=True, order_id=order.order_id)
    
    async def _execute_order(self, order: Order) -> ExecutionResult:
        """Execute order with realistic market simulation"""
        symbol = order.symbol
        
        # Get or generate market data
        market_data = self._get_market_data(symbol)
        
        # Determine execution price
        execution_price = self._calculate_execution_price(order, market_data)
        
        # Simulate random rejection
        if random.random() < self.reject_probability:
            order.status = OrderStatus.REJECTED
            return ExecutionResult(
                success=False,
                order_id=order.order_id,
                error_message="Order rejected by exchange"
            )
        
        # Simulate partial fills
        executed_quantity = order.remaining_quantity
        if random.random() < self.partial_fill_probability and order.order_type != OrderType.MARKET:
            executed_quantity = random.randint(
                max(1, order.remaining_quantity // 4),
                order.remaining_quantity
            )
        
        # Check price limits for limit orders
        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and execution_price > order.limit_price:
                # Price too high, no execution
                return ExecutionResult(
                    success=True,
                    order_id=order.order_id,
                    executed_quantity=0,
                    remaining_quantity=order.remaining_quantity
                )
            elif order.side == OrderSide.SELL and execution_price < order.limit_price:
                # Price too low, no execution
                return ExecutionResult(
                    success=True,
                    order_id=order.order_id,
                    executed_quantity=0,
                    remaining_quantity=order.remaining_quantity
                )
        
        # Execute the trade
        fees = self._calculate_fees(executed_quantity, execution_price)
        
        # Update order
        order.filled_quantity += executed_quantity
        order.remaining_quantity -= executed_quantity
        
        if order.avg_fill_price is None:
            order.avg_fill_price = execution_price
        else:
            # Update average fill price
            total_value = (order.filled_quantity - executed_quantity) * order.avg_fill_price
            total_value += executed_quantity * execution_price
            order.avg_fill_price = total_value / order.filled_quantity
        
        order.total_commission += fees
        
        # Update order status
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now(timezone.utc)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Create trade record
        if executed_quantity > 0:
            trade = self._create_trade(order, executed_quantity, execution_price, fees)
            self.trade_history.append(trade)
            
            # Log trade
            log_trade_execution(
                symbol=symbol,
                side=order.side.value,
                quantity=executed_quantity,
                price=float(execution_price),
                strategy=order.strategy or "unknown",
                agent="paper_trader",
                fees=float(fees),
                order_id=order.order_id
            )
        
        return ExecutionResult(
            success=True,
            order_id=order.order_id,
            executed_quantity=executed_quantity,
            executed_price=execution_price,
            remaining_quantity=order.remaining_quantity,
            fees=fees,
            execution_time=datetime.now(timezone.utc)
        )
    
    def _get_market_data(self, symbol: str) -> MarketData:
        """Get or generate market data for symbol"""
        if symbol in self.market_data:
            # Update existing data with some randomness
            data = self.market_data[symbol]
            price_change = random.uniform(-0.005, 0.005)  # ±0.5% random walk
            
            new_price = data.last_price * (1 + Decimal(str(price_change)))
            spread_bps = random.uniform(1, 10)  # 1-10 basis points spread
            spread = new_price * Decimal(str(spread_bps / 10000))
            
            data.bid_price = new_price - spread / 2
            data.ask_price = new_price + spread / 2
            data.last_price = new_price
            data.volume += random.randint(1000, 10000)
            data.timestamp = datetime.now(timezone.utc)
            data.spread_pct = float(spread / new_price)
            
        else:
            # Generate initial market data
            base_price = Decimal(str(random.uniform(50, 200)))
            spread_bps = random.uniform(2, 15)
            spread = base_price * Decimal(str(spread_bps / 10000))
            
            data = MarketData(
                symbol=symbol,
                bid_price=base_price - spread / 2,
                ask_price=base_price + spread / 2,
                last_price=base_price,
                volume=random.randint(100000, 1000000),
                timestamp=datetime.now(timezone.utc),
                spread_pct=spread_bps / 100
            )
            
            self.market_data[symbol] = data
        
        return data
    
    def _calculate_execution_price(self, order: Order, market_data: MarketData) -> Decimal:
        """Calculate realistic execution price with slippage"""
        base_price = market_data.ask_price if order.side == OrderSide.BUY else market_data.bid_price
        
        # Market impact based on order size
        volume_ratio = order.quantity / max(market_data.volume, 100000)  # Avoid division by zero
        market_impact = volume_ratio * self.market_impact_factor
        
        # Additional slippage factors
        volatility_factor = 1.0
        if self.market_condition == MarketCondition.VOLATILE:
            volatility_factor = 1.5
        elif self.market_condition == MarketCondition.ILLIQUID:
            volatility_factor = 2.0
        
        # Total slippage
        total_slippage = (self.slippage_base + market_impact) * volatility_factor
        
        # Apply slippage
        if order.side == OrderSide.BUY:
            # Buying - price moves against us
            slippage_amount = base_price * Decimal(str(total_slippage))
            execution_price = base_price + slippage_amount
        else:
            # Selling - price moves against us
            slippage_amount = base_price * Decimal(str(total_slippage))
            execution_price = base_price - slippage_amount
        
        # For market orders, add some randomness
        if order.order_type == OrderType.MARKET:
            random_factor = random.uniform(-0.001, 0.001)  # ±0.1% randomness
            execution_price *= (1 + Decimal(str(random_factor)))
        
        # Ensure price is positive
        execution_price = max(execution_price, Decimal('0.01'))
        
        return execution_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def _calculate_fees(self, quantity: int, price: Decimal) -> Decimal:
        """Calculate trading fees"""
        commission = max(
            self.min_commission,
            Decimal(str(quantity)) * self.commission_per_share
        )
        
        # Additional fees (SEC, exchange, etc.)
        trade_value = Decimal(str(quantity)) * price
        sec_fee = trade_value * Decimal('0.0000231')  # SEC fee
        
        total_fees = commission + sec_fee
        return total_fees.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def _create_trade(self, order: Order, quantity: int, price: Decimal, fees: Decimal) -> Trade:
        """Create trade record from order execution"""
        market_data = self.market_data.get(order.symbol)
        
        trade = Trade(
            order_id=order.order_id,
            signal_id=getattr(order, 'signal_id', None),
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=fees,
            strategy=order.strategy,
            executed_at=datetime.now(timezone.utc),
            bid_price=market_data.bid_price if market_data else None,
            ask_price=market_data.ask_price if market_data else None,
            volume=market_data.volume if market_data else None
        )
        
        return trade
    
    async def _update_portfolio(self, order: Order, execution_result: ExecutionResult) -> None:
        """Update portfolio with executed trade"""
        if execution_result.executed_quantity <= 0:
            return
        
        # Create trade object
        trade = Trade(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=execution_result.executed_quantity,
            price=execution_result.executed_price,
            commission=execution_result.fees,
            strategy=order.strategy
        )
        
        # Add trade to portfolio
        self.portfolio.add_trade(trade)
        
        self.logger.info(f"Portfolio updated with trade: {order.symbol} {order.side.value} {execution_result.executed_quantity}")
    
    def _estimate_order_cost(self, order: Order) -> Decimal:
        """Estimate total cost of order including fees"""
        market_data = self._get_market_data(order.symbol)
        
        if order.order_type == OrderType.MARKET:
            estimated_price = market_data.ask_price if order.side == OrderSide.BUY else market_data.bid_price
        elif order.order_type == OrderType.LIMIT:
            estimated_price = order.limit_price
        else:
            estimated_price = market_data.mid_price
        
        # Add slippage estimate
        slippage_estimate = estimated_price * Decimal(str(self.slippage_base))
        if order.side == OrderSide.BUY:
            estimated_price += slippage_estimate
        
        trade_value = Decimal(str(order.quantity)) * estimated_price
        estimated_fees = self._calculate_fees(order.quantity, estimated_price)
        
        return trade_value + estimated_fees
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        if order_id not in self.active_orders:
            self.logger.warning(f"Cannot cancel order {order_id}: not found")
            return False
        
        order = self.active_orders[order_id]
        
        # Simulate cancellation latency
        await asyncio.sleep(self.latency_ms / 2000.0)
        
        # Check if order can be cancelled
        if order.status == OrderStatus.FILLED:
            self.logger.warning(f"Cannot cancel order {order_id}: already filled")
            return False
        
        # Cancel order
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now(timezone.utc)
        
        # Move to history
        self.active_orders.pop(order_id)
        self.order_history.append(order)
        
        self.logger.info(f"Order {order_id} cancelled")
        return True
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an order"""
        # Check active orders
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
        else:
            # Check history
            order = next((o for o in self.order_history if o.order_id == order_id), None)
        
        if not order:
            return None
        
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "remaining_quantity": order.remaining_quantity,
            "avg_fill_price": float(order.avg_fill_price) if order.avg_fill_price else None,
            "total_commission": float(order.total_commission),
            "created_at": order.created_at.isoformat(),
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None
        }
    
    def set_market_condition(self, condition: MarketCondition) -> None:
        """Set market condition for simulation"""
        self.market_condition = condition
        self.logger.info(f"Market condition set to: {condition.value}")
    
    def set_trading_hours(self, is_open: bool) -> None:
        """Set trading hours status"""
        self.trading_hours = is_open
        self.logger.info(f"Trading hours: {'open' if is_open else 'closed'}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            "total_value": float(self.portfolio.metrics.total_value),
            "cash": float(self.portfolio.current_cash),
            "positions": len(self.portfolio.get_open_positions()),
            "total_pnl": float(self.portfolio.metrics.total_pnl),
            "daily_pnl": float(self.portfolio.metrics.daily_pnl),
            "total_return_pct": self.portfolio.metrics.total_return_pct
        }
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading activity summary"""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "total_volume": 0,
                "total_fees": 0,
                "avg_trade_size": 0
            }
        
        total_volume = sum(t.quantity * t.price for t in self.trade_history)
        total_fees = sum(t.commission + t.fees for t in self.trade_history)
        avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "total_volume": float(total_volume),
            "total_fees": float(total_fees),
            "avg_trade_size": float(avg_trade_size),
            "active_orders": len(self.active_orders)
        }