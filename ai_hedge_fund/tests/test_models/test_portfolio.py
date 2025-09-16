"""
Tests for portfolio management functionality
"""
import pytest
from decimal import Decimal
from datetime import datetime, timezone

from src.models.portfolio import Portfolio, PortfolioMetrics
from src.models.position import Position, PositionSide
from src.models.trade import Trade, OrderSide


class TestPortfolio:
    """Test suite for Portfolio class"""
    
    @pytest.fixture
    def portfolio(self):
        """Create test portfolio"""
        return Portfolio(
            name="Test Portfolio",
            initial_cash=Decimal('1000000'),
            current_cash=Decimal('800000')
        )
    
    @pytest.fixture
    def sample_trade(self):
        """Create sample trade"""
        return Trade(
            order_id="test_order_1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal('150.00'),
            strategy="test_strategy"
        )
    
    def test_portfolio_initialization(self, portfolio):
        """Test portfolio initialization"""
        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_cash == Decimal('1000000')
        assert portfolio.current_cash == Decimal('800000')
        assert len(portfolio.positions) == 0
        assert portfolio.metrics.total_value > 0
    
    def test_add_trade_buy(self, portfolio, sample_trade):
        """Test adding buy trade"""
        initial_cash = portfolio.current_cash
        
        portfolio.add_trade(sample_trade)
        
        # Check cash reduction
        expected_cash = initial_cash - sample_trade.net_amount
        assert portfolio.current_cash == expected_cash
        
        # Check position creation
        assert "AAPL" in portfolio.positions
        position = portfolio.positions["AAPL"]
        assert position.quantity == 100
        assert position.side == PositionSide.LONG
    
    def test_add_trade_sell(self, portfolio):
        """Test adding sell trade"""
        # First buy to create position
        buy_trade = Trade(
            order_id="buy_order",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal('150.00')
        )
        portfolio.add_trade(buy_trade)
        
        # Then sell
        sell_trade = Trade(
            order_id="sell_order",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            price=Decimal('155.00')
        )
        
        initial_cash = portfolio.current_cash
        portfolio.add_trade(sell_trade)
        
        # Check cash increase
        expected_cash = initial_cash + sell_trade.net_amount
        assert portfolio.current_cash == expected_cash
        
        # Check position reduction
        position = portfolio.positions["AAPL"]
        assert position.quantity == 50
        assert position.side == PositionSide.LONG
    
    def test_position_tracking(self, portfolio):
        """Test position tracking across multiple trades"""
        # Buy trade
        portfolio.add_trade(Trade(
            order_id="order1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal('150.00')
        ))
        
        # Another buy trade
        portfolio.add_trade(Trade(
            order_id="order2",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=50,
            price=Decimal('160.00')
        ))
        
        position = portfolio.positions["AAPL"]
        assert position.quantity == 150
        
        # Average cost should be weighted average
        expected_avg_cost = (100 * 150 + 50 * 160) / 150
        assert abs(float(position.avg_cost) - expected_avg_cost) < 0.01
    
    def test_update_market_prices(self, portfolio, sample_trade):
        """Test market price updates"""
        portfolio.add_trade(sample_trade)
        
        # Update market prices
        prices = {"AAPL": Decimal('160.00')}
        portfolio.update_market_prices(prices)
        
        position = portfolio.positions["AAPL"]
        assert position.current_price == Decimal('160.00')
        assert position.market_value == 100 * Decimal('160.00')
        assert position.unrealized_pnl > 0  # Profit from 150 to 160
    
    def test_portfolio_metrics_calculation(self, portfolio, sample_trade):
        """Test portfolio metrics calculation"""
        portfolio.add_trade(sample_trade)
        
        # Update with current price
        portfolio.update_market_prices({"AAPL": Decimal('155.00')})
        
        metrics = portfolio.metrics
        
        # Check basic metrics
        assert metrics.cash == portfolio.current_cash
        assert metrics.invested_value > 0
        assert metrics.total_value == metrics.cash + metrics.invested_value
        assert metrics.num_positions == 1
        assert metrics.num_long_positions == 1
        assert metrics.num_short_positions == 0
    
    def test_get_open_positions(self, portfolio):
        """Test getting open positions"""
        # Add some trades
        portfolio.add_trade(Trade(
            order_id="order1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal('150.00')
        ))
        
        portfolio.add_trade(Trade(
            order_id="order2",
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=50,
            price=Decimal('300.00')
        ))
        
        open_positions = portfolio.get_open_positions()
        assert len(open_positions) == 2
        
        symbols = [pos.symbol for pos in open_positions]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
    
    def test_position_size_limits(self, portfolio):
        """Test position size limit checking"""
        trade_value = Decimal('60000')  # 6% of 1M portfolio
        
        # Should exceed 5% limit
        assert not portfolio.check_position_size_limit("AAPL", trade_value)
        
        # Smaller trade should pass
        small_trade_value = Decimal('40000')  # 4% of portfolio
        assert portfolio.check_position_size_limit("AAPL", small_trade_value)
    
    def test_sector_limits(self, portfolio):
        """Test sector allocation limits"""
        trade_value = Decimal('300000')  # 30% of portfolio
        
        # Should exceed 25% sector limit
        assert not portfolio.check_sector_limit("Technology", trade_value)
        
        # Smaller allocation should pass
        small_trade_value = Decimal('200000')  # 20% of portfolio
        assert portfolio.check_sector_limit("Technology", small_trade_value)
    
    def test_available_capital(self, portfolio):
        """Test available capital calculation"""
        # With 5% reserve
        available = portfolio.get_available_capital(reserve_pct=0.05)
        
        # Should be current cash minus 5% of total value
        expected_reserve = portfolio.metrics.total_value * Decimal('0.05')
        expected_available = portfolio.current_cash - expected_reserve
        
        assert available == max(Decimal('0'), expected_available)
    
    def test_position_sizing_calculation(self, portfolio):
        """Test position size calculation"""
        symbol = "AAPL"
        price = Decimal('150.00')
        risk_pct = 0.01  # 1% risk
        
        position_size = portfolio.calculate_position_size(symbol, price, risk_pct)
        
        # Should be reasonable size
        assert position_size > 0
        assert position_size < 1000  # Not too large
        
        # Calculate trade value
        trade_value = position_size * price
        position_pct = float(trade_value / portfolio.metrics.total_value)
        
        # Should not exceed position limits
        assert position_pct <= portfolio.max_position_size_pct
    
    def test_risk_summary(self, portfolio):
        """Test risk summary generation"""
        # Add some positions
        portfolio.add_trade(Trade(
            order_id="order1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal('150.00')
        ))
        
        risk_summary = portfolio.get_risk_summary()
        
        assert "total_positions" in risk_summary
        assert "gross_exposure" in risk_summary
        assert "net_exposure" in risk_summary
        assert "cash_ratio" in risk_summary
        assert risk_summary["total_positions"] == 1
    
    def test_performance_metrics(self, portfolio):
        """Test performance metrics calculation"""
        # Add some trades to generate returns
        portfolio.add_trade(Trade(
            order_id="order1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal('150.00')
        ))
        
        # Update price for unrealized gain
        portfolio.update_market_prices({"AAPL": Decimal('160.00')})
        
        # Add some daily values for performance calculation
        portfolio.daily_values = [Decimal('1000000'), Decimal('1005000'), Decimal('1010000')]
        portfolio.daily_returns = [0.0, 0.005, 0.005]
        
        performance = portfolio.calculate_performance_metrics()
        
        assert "annual_return" in performance
        assert "annual_volatility" in performance
        assert "sharpe_ratio" in performance
        assert "max_drawdown" in performance


class TestPortfolioMetrics:
    """Test portfolio metrics calculations"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = PortfolioMetrics()
        
        assert metrics.total_value == Decimal('0')
        assert metrics.cash == Decimal('0')
        assert metrics.total_return_pct == 0.0
        assert metrics.num_positions == 0
    
    def test_exposure_calculations(self):
        """Test exposure metric calculations"""
        metrics = PortfolioMetrics(
            total_value=Decimal('1000000'),
            invested_value=Decimal('800000'),
            cash=Decimal('200000')
        )
        
        # For these metrics to be meaningful, they would need to be calculated
        # from actual position data in the full implementation
        assert metrics.total_value == Decimal('1000000')
        assert metrics.invested_value == Decimal('800000')
        assert metrics.cash == Decimal('200000')


class TestPortfolioIntegration:
    """Integration tests for portfolio functionality"""
    
    def test_full_trading_cycle(self):
        """Test complete trading cycle"""
        portfolio = Portfolio(initial_cash=Decimal('1000000'))
        
        # Day 1: Buy AAPL
        portfolio.add_trade(Trade(
            order_id="order1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal('150.00')
        ))
        
        # Update price
        portfolio.update_market_prices({"AAPL": Decimal('155.00')})
        
        # Should have unrealized gain
        position = portfolio.positions["AAPL"]
        assert position.unrealized_pnl > 0
        
        # Day 2: Sell half
        portfolio.add_trade(Trade(
            order_id="order2",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            price=Decimal('155.00')
        ))
        
        # Should have realized and unrealized PnL
        assert position.realized_pnl > 0
        assert position.quantity == 50
        
        # Day 3: Full exit
        portfolio.add_trade(Trade(
            order_id="order3",
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            price=Decimal('160.00')
        ))
        
        # Position should be closed
        assert not position.is_open
        assert position.realized_pnl > 0
    
    def test_multiple_positions_portfolio(self):
        """Test portfolio with multiple positions"""
        portfolio = Portfolio(initial_cash=Decimal('1000000'))
        
        # Add multiple positions
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
        prices = [150, 300, 2500, 800]
        quantities = [100, 50, 10, 25]
        
        for symbol, price, quantity in zip(symbols, prices, quantities):
            portfolio.add_trade(Trade(
                order_id=f"order_{symbol}",
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                price=Decimal(str(price))
            ))
        
        # Update all prices
        new_prices = {
            "AAPL": Decimal('155.00'),
            "MSFT": Decimal('310.00'),
            "GOOGL": Decimal('2600.00'),
            "TSLA": Decimal('850.00')
        }
        portfolio.update_market_prices(new_prices)
        
        # Check portfolio metrics
        assert len(portfolio.get_open_positions()) == 4
        assert portfolio.metrics.num_positions == 4
        assert portfolio.metrics.total_pnl > 0  # All positions gained
        
        # Check diversification
        risk_summary = portfolio.get_risk_summary()
        assert risk_summary["total_positions"] == 4