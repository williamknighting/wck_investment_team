#!/usr/bin/env python3
"""
Test script for Strategy Agent conversation and proposal system
Demonstrates reactive strategy analysis and trade proposal generation
"""
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append('.')

# Set up environment
os.environ.setdefault("OPENAI_API_KEY", "test_key")

from agents.qullamaggie_agent import QullamaggieAgent
from src.utils.logging_config import get_logger

def test_strategy_analysis():
    """Test strategy analysis capabilities"""
    
    print("📊 Testing Strategy Agent Analysis")
    print("=" * 50)
    
    try:
        # Initialize Qullamaggie agent
        print("🚀 Initializing Qullamaggie Strategy Agent...")
        qull_agent = QullamaggieAgent()
        
        # Test symbol
        test_symbol = "SPY"
        context = {"symbol": test_symbol}
        
        print(f"\n🎯 Testing strategy analysis for: {test_symbol}")
        
        # Test 1: Strategy Analysis
        print("\n1️⃣ Testing Strategy Analysis")
        message = "Analyze symbol for Qullamaggie momentum setup"
        result = qull_agent.process_message(message, context)
        
        if result.get("type") == "strategy_analysis":
            print(f"   ✅ Strategy analysis completed")
            print(f"   📊 Setup Quality: {result['setup_quality']}")
            print(f"   🎯 Conviction: {result['conviction']}/10")
            print(f"   📋 Criteria Met: {result['criteria_met']}/{len(qull_agent.strategy_criteria)}")
            
            strategy_eval = result.get("strategy_evaluation", {})
            key_metrics = strategy_eval.get("key_metrics", {})
            print(f"   📈 22d Gain: {key_metrics.get('gain_22d', 0):+.1f}%")
            print(f"   🏔️ Distance from 52w High: {key_metrics.get('distance_52w', 0):.1f}%")
            print(f"   📊 Volume Ratio: {key_metrics.get('volume_ratio', 0):.2f}x")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Strategy analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trade_proposal_generation():
    """Test trade proposal generation"""
    
    print(f"\n💼 Testing Trade Proposal Generation")
    print("-" * 40)
    
    try:
        qull_agent = QullamaggieAgent()
        
        context = {"symbol": "SPY"}
        message = "Generate trade proposal for momentum setup"
        result = qull_agent.process_message(message, context)
        
        if result.get("type") == "trade_proposal":
            print(f"   ✅ Trade proposal generated")
            print(f"   📈 Symbol: {result['symbol']}")
            
            if result.get("proposal_status") != "rejected":
                print(f"   🎯 Setup Quality: {result['setup_quality']}")
                print(f"   💪 Conviction: {result['conviction']}/10")
                print(f"\n   💰 Trade Parameters:")
                print(f"      Entry: ${result['entry_price']:.2f}")
                print(f"      Stop: ${result['stop_loss']:.2f}")
                print(f"      Target: ${result['profit_target']:.2f}")
                print(f"      Size: {result['position_size']} shares")
                print(f"      R/R: {result['risk_reward_ratio']:.1f}:1")
                
                if result.get("proposal_file"):
                    print(f"   📄 Proposal file: {result['proposal_file']}")
            else:
                print(f"   ❌ Proposal rejected: {result.get('reasoning', 'Unknown reason')}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Trade proposal test failed: {e}")
        return None

def test_proposal_defense():
    """Test proposal defense capabilities"""
    
    print(f"\n🛡️ Testing Proposal Defense")
    print("-" * 30)
    
    try:
        qull_agent = QullamaggieAgent()
        
        # First generate a proposal
        context = {"symbol": "SPY"}
        proposal_result = qull_agent.process_message("Generate trade proposal", context)
        
        if proposal_result.get("type") != "trade_proposal":
            print("   ❌ Cannot test defense without a proposal")
            return None
        
        # Now test defense
        defense_context = {
            "symbol": "SPY",
            "proposal_id": proposal_result.get("proposal_id")
        }
        criticism = "This setup has too much risk and the momentum is questionable"
        defense_result = qull_agent.process_message(f"Defend proposal: {criticism}", defense_context)
        
        if defense_result.get("type") == "defense_response":
            print(f"   ✅ Proposal defense generated")
            print(f"   🎯 Proposal: {defense_result['symbol']}")
            print(f"   💪 Conviction Maintained: {defense_result['conviction_maintained']}/10")
            
            defense_points = defense_result.get("defense_points", [])
            print(f"   📋 Defense Points ({len(defense_points)}):")
            for i, point in enumerate(defense_points[:3], 1):
                print(f"      {i}. {point}")
                
            print(f"   💬 Response: {defense_result.get('response', 'No response')[:100]}...")
        else:
            print(f"   ❌ Defense failed: {defense_result.get('message', 'Unknown error')}")
        
        return defense_result
        
    except Exception as e:
        print(f"❌ Defense test failed: {e}")
        return None

def test_proposal_critique():
    """Test critiquing other proposals"""
    
    print(f"\n🔍 Testing Proposal Critique")
    print("-" * 30)
    
    try:
        qull_agent = QullamaggieAgent()
        
        # Mock proposal from another strategy
        mock_proposal = {
            "proposal_id": "value_agent_SPY_20250918_120000",
            "symbol": "SPY",
            "strategy": "value_agent",
            "conviction": 4,
            "risk_reward_ratio": 1.5,
            "setup_quality": "moderate",
            "technical_summary": "Value-based entry with 15% 22-day gain"
        }
        
        context = {"proposal": mock_proposal}
        message = "Please critique this proposal from other strategy"
        result = qull_agent.process_message(message, context)
        
        if result.get("type") == "critique_response":
            print(f"   ✅ Critique generated")
            print(f"   🎯 Critiqued: {result['critiqued_symbol']} ({result['critiqued_proposal']})")
            print(f"   📊 Assessment: {result['overall_assessment']}")
            
            critique_points = result.get("critique_points", [])
            print(f"   📋 Critique Points ({len(critique_points)}):")
            for i, point in enumerate(critique_points[:3], 1):
                print(f"      {i}. {point}")
            
            alternative = result.get("alternative_perspective", "")
            if alternative:
                print(f"   💡 Alternative: {alternative[:100]}...")
        else:
            print(f"   ❌ Critique failed: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Critique test failed: {e}")
        return None

def test_strategy_status():
    """Test strategy status reporting"""
    
    print(f"\n📈 Testing Strategy Status")
    print("-" * 25)
    
    try:
        qull_agent = QullamaggieAgent()
        
        # Generate a few proposals first
        for symbol in ["SPY"]:  # Just one for testing
            context = {"symbol": symbol}
            qull_agent.process_message("Generate trade proposal", context)
        
        # Get status
        result = qull_agent.process_message("Show strategy status")
        
        if result.get("type") == "strategy_status":
            print(f"   ✅ Status retrieved")
            print(f"   📊 Strategy: {result['strategy']}")
            print(f"   📋 Active Proposals: {result['active_proposals']}")
            print(f"   📈 Total Proposals: {result['total_proposals']}")
            print(f"   ⚙️ Config: {result['strategy_config']}")
        else:
            print(f"   ❌ Status failed: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Strategy Agent Testing Suite")
    print("=" * 40)
    
    # Run tests
    print("\n🔬 Running Strategy Agent Tests...")
    
    analysis_result = test_strategy_analysis()
    proposal_result = test_trade_proposal_generation()
    defense_result = test_proposal_defense()
    critique_result = test_proposal_critique()
    status_result = test_strategy_status()
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"   Strategy Analysis: {'✅ PASSED' if analysis_result else '❌ FAILED'}")
    print(f"   Trade Proposals: {'✅ PASSED' if proposal_result else '❌ FAILED'}")
    print(f"   Proposal Defense: {'✅ PASSED' if defense_result else '❌ FAILED'}")
    print(f"   Proposal Critique: {'✅ PASSED' if critique_result else '❌ FAILED'}")
    print(f"   Strategy Status: {'✅ PASSED' if status_result else '❌ FAILED'}")
    
    all_passed = all([analysis_result, proposal_result, defense_result, critique_result, status_result])
    
    if all_passed:
        print(f"\n🎉 All Strategy Agent tests passed!")
        print(f"🚀 QullamaggieAgent is ready for conversation-driven trading!")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Test with real market data")
    print(f"   • Integrate with Investment Committee workflow")
    print(f"   • Add more strategy agents (Value, Technical Breakout, etc.)")
    print(f"   • Implement strategy agent debates and voting")