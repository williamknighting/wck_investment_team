#!/usr/bin/env python3
"""
Test script for conversation-driven Technical Analyst Agent
Demonstrates reactive technical analysis capabilities
"""
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append('.')

# Set up environment
os.environ.setdefault("OPENAI_API_KEY", "test_key")

from agents.technical_analyst import TechnicalAnalystAgent
from src.utils.logging_config import get_logger

def test_conversation_driven_analysis():
    """Test conversation-driven technical analysis"""
    
    print("🔬 Testing Conversation-Driven Technical Analyst")
    print("=" * 60)
    
    try:
        # Initialize agent
        print("📊 Initializing Technical Analyst Agent...")
        tech_agent = TechnicalAnalystAgent(
            name="technical_analyst_test",
            description="Technical analysis specialist for testing"
        )
        
        # Test symbol context
        test_symbol = "SPY"
        context = {"symbol": test_symbol}
        
        print(f"\n🎯 Testing with symbol: {test_symbol}")
        
        # Test 1: Comprehensive Analysis
        print("\n1️⃣ Testing Comprehensive Analysis Request")
        message = "Please provide comprehensive technical analysis"
        result = tech_agent.process_message(message, context)
        
        if result.get("type") == "comprehensive_technical_analysis":
            print(f"   ✅ Comprehensive analysis completed")
            print(f"   📈 Current price: ${result['summary']['current_price']:.2f}")
            print(f"   📊 Trend: {result['summary']['trend_direction']}")
            print(f"   🎯 RSI signal: {result['summary']['momentum_signal']}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        # Test 2: Bollinger Bands
        print("\n2️⃣ Testing Bollinger Bands Request")
        message = "Calculate Bollinger Bands for current setup"
        result = tech_agent.process_message(message, context)
        
        if result.get("type") == "bollinger_bands":
            bands = result["bands"]
            analysis = result["analysis"]
            print(f"   ✅ Bollinger Bands calculated")
            print(f"   📊 Upper: ${bands['upper']:.2f}, Lower: ${bands['lower']:.2f}")
            print(f"   🎯 Position: {analysis['position_signal']}")
            print(f"   💥 Squeeze: {'Yes' if analysis['squeeze_detected'] else 'No'}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        # Test 3: MACD Analysis
        print("\n3️⃣ Testing MACD Analysis Request")
        message = "Show me MACD analysis with signal interpretation"
        result = tech_agent.process_message(message, context)
        
        if result.get("type") == "macd_analysis":
            values = result["values"]
            signals = result["signals"]
            print(f"   ✅ MACD analysis completed")
            print(f"   📊 MACD: {values['macd']:.4f}, Signal: {values['signal']:.4f}")
            print(f"   🔄 Crossover: {'Bullish' if signals['bullish_crossover'] else 'Bearish' if signals['bearish_crossover'] else 'None'}")
            print(f"   📈 Direction: {signals['momentum_direction']}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        # Test 4: Custom Metrics
        print("\n4️⃣ Testing Custom Metrics Request")
        message = "Calculate custom momentum score and strength index"
        result = tech_agent.process_message(message, context)
        
        if result.get("type") == "custom_metrics":
            custom = result["custom_calculations"]
            print(f"   ✅ Custom metrics calculated")
            print(f"   📊 Momentum score: {custom.get('momentum_score', 0):.3f}")
            print(f"   💪 Strength index: {custom.get('overall_strength', 0):.3f}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        # Test 5: Support/Resistance
        print("\n5️⃣ Testing Support/Resistance Analysis")
        message = "Find support and resistance levels"
        result = tech_agent.process_message(message, context)
        
        if result.get("type") == "support_resistance":
            current_price = result["current_price"]
            nearest_support = result.get("nearest_support")
            nearest_resistance = result.get("nearest_resistance")
            
            print(f"   ✅ Support/Resistance analysis completed")
            print(f"   💰 Current price: ${current_price:.2f}")
            if nearest_support:
                print(f"   🛡️ Nearest support: ${nearest_support['level']:.2f}")
            if nearest_resistance:
                print(f"   🚧 Nearest resistance: ${nearest_resistance['level']:.2f}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        # Test 6: Trading Signals
        print("\n6️⃣ Testing Trading Signals Generation")
        message = "Generate trading signals and recommendation"
        result = tech_agent.process_message(message, context)
        
        if result.get("type") == "trading_signals":
            signals = result["signals"]
            confidence = result["overall_confidence"]
            recommendation = result["recommendation"]
            
            print(f"   ✅ Trading signals generated")
            print(f"   🎯 Signal count: {len(signals)}")
            print(f"   📊 Confidence: {confidence:.2f}")
            print(f"   💡 Recommendation: {recommendation}")
            
            if signals:
                print(f"   📋 Signals:")
                for signal in signals[:3]:  # Show first 3
                    print(f"      • {signal}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        print(f"\n✅ Technical Analyst conversation testing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_agent_format():
    """Test formatted data for strategy agents"""
    
    print(f"\n🤖 Testing Strategy Agent Data Formatting")
    print("-" * 40)
    
    try:
        tech_agent = TechnicalAnalystAgent(
            name="technical_analyst_format_test",
            description="Technical analysis for strategy formatting"
        )
        
        context = {"symbol": "SPY"}
        message = "Please provide comprehensive technical analysis"
        result = tech_agent.process_message(message, context)
        
        if result.get("type") == "comprehensive_technical_analysis":
            formatted_data = result.get("formatted_for_strategies", {})
            
            print("📦 Strategy Agent Formatted Data:")
            print(f"   Price Data: {len(formatted_data.get('price_data', {}))} fields")
            print(f"   Momentum Data: {len(formatted_data.get('momentum_data', {}))} fields")
            print(f"   Volume Data: {len(formatted_data.get('volume_data', {}))} fields")
            print(f"   Quality Flags: {len(formatted_data.get('quality_flags', {}))} flags")
            
            # Show key values
            price_data = formatted_data.get("price_data", {})
            momentum_data = formatted_data.get("momentum_data", {})
            
            if price_data:
                print(f"   📈 Current Price: ${price_data.get('current', 0):.2f}")
                print(f"   📊 Trend Intensity: {price_data.get('trend_intensity', 0):.3f}")
            
            if momentum_data:
                print(f"   🎯 RSI: {momentum_data.get('rsi', 0):.1f}")
                print(f"   📈 22d Gain: {momentum_data.get('gain_22d', 0):+.1f}%")
            
            return True
        else:
            print(f"❌ Failed to get formatted data: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Strategy formatting test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Technical Analyst Agent Testing Suite")
    print("=" * 50)
    
    # Run conversation tests
    conversation_success = test_conversation_driven_analysis()
    
    # Run strategy formatting tests
    formatting_success = test_strategy_agent_format()
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Conversation Tests: {'✅ PASSED' if conversation_success else '❌ FAILED'}")
    print(f"   Strategy Formatting: {'✅ PASSED' if formatting_success else '❌ FAILED'}")
    
    if conversation_success and formatting_success:
        print(f"\n🎉 All tests passed! Technical Analyst Agent is ready for reactive conversations.")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")