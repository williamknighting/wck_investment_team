#!/usr/bin/env python3
"""
Test script for Risk Manager Agent
Demonstrates proposal evaluation, risk scoring, and veto capabilities
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Set up environment
os.environ.setdefault("OPENAI_API_KEY", "test_key")

from agents.risk_manager import RiskManagerAgent
from src.utils.logging_config import get_logger

def create_sample_proposals():
    """Create sample proposals for testing"""
    proposals_dir = Path("proposals")
    proposals_dir.mkdir(exist_ok=True)
    
    # Sample 1: Aggressive high-risk proposal
    aggressive_proposal = """# Trade Proposal - TSLA

**Agent**: qullamaggie_agent  
**Date**: 2025-09-18 14:30:00  
**Symbol**: TSLA  
**Type**: trade  

---

## Qullamaggie Momentum Setup Analysis

**EXCELLENT SETUP** - High conviction momentum breakout

### Trade Parameters:
- **Entry**: $250.00
- **Stop Loss**: $235.00
- **Profit Target**: $285.00
- **Position Size**: 800 shares
- **Risk/Reward**: 2.3:1
- **Conviction**: 9/10

### Setup Quality: Excellent
Strong momentum with 35% gain in 22 days, trading within 5% of 52-week highs.
Volume surge confirms breakout momentum. Technical structure is pristine.

**Position Value**: $200,000
**Risk Amount**: $12,000
"""
    
    # Sample 2: Conservative moderate proposal
    conservative_proposal = """# Trade Proposal - SPY

**Agent**: value_agent  
**Date**: 2025-09-18 14:35:00  
**Symbol**: SPY  
**Type**: trade  

---

## Value Investment Analysis

**MODERATE SETUP** - Defensive value play

### Trade Parameters:
- **Entry**: $450.00
- **Stop Loss**: $430.00
- **Profit Target**: $480.00
- **Position Size**: 100 shares
- **Risk/Reward**: 1.5:1
- **Conviction**: 5/10

### Setup Quality: Moderate
Market pullback provides entry opportunity. Conservative position sizing
for portfolio stability.

**Position Value**: $45,000
**Risk Amount**: $2,000
"""
    
    # Sample 3: Poor risk/reward proposal
    poor_rr_proposal = """# Trade Proposal - AAPL

**Agent**: technical_agent  
**Date**: 2025-09-18 14:40:00  
**Symbol**: AAPL  
**Type**: trade  

---

## Technical Breakout Analysis

**STRONG SETUP** - Breakout pattern

### Trade Parameters:
- **Entry**: $180.00
- **Stop Loss**: $170.00
- **Profit Target**: $185.00
- **Position Size**: 500 shares
- **Risk/Reward**: 0.5:1
- **Conviction**: 7/10

### Setup Quality: Strong
Clean breakout pattern with good volume. Tight stop for quick exit.

**Position Value**: $90,000
**Risk Amount**: $5,000
"""
    
    # Write sample proposals
    samples = [
        ("qullamaggie_agent_TSLA_trade_20250918_143000.md", aggressive_proposal),
        ("value_agent_SPY_trade_20250918_143500.md", conservative_proposal),
        ("technical_agent_AAPL_trade_20250918_144000.md", poor_rr_proposal)
    ]
    
    for filename, content in samples:
        filepath = proposals_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
    
    print(f"✅ Created {len(samples)} sample proposals in {proposals_dir}")
    return len(samples)

def test_risk_manager_evaluation():
    """Test risk manager proposal evaluation"""
    
    print("🔒 Testing Risk Manager Evaluation")
    print("=" * 50)
    
    try:
        # Initialize Risk Manager
        print("🚀 Initializing Risk Manager Agent...")
        risk_manager = RiskManagerAgent()
        
        print("\n📋 Testing proposal evaluation...")
        
        # Test 1: Evaluate all proposals
        print("\n1️⃣ Testing All Proposals Evaluation")
        result = risk_manager.process_message("Evaluate all proposals for risk")
        
        if result.get("type") == "risk_evaluation":
            print(f"   ✅ Risk evaluation completed")
            print(f"   📊 Proposals evaluated: {result['proposals_evaluated']}")
            print(f"   ⚠️  Vetoed proposals: {result['vetoed_count']}")
            print(f"   🔍 Challenged proposals: {result['challenged_count']}")
            print(f"   📈 Portfolio risk score: {result['portfolio_risk_score']:.1f}/10")
            
            # Show individual assessments
            assessments = result.get("assessments", [])
            print(f"\n   📋 Individual Risk Assessments:")
            for assessment in assessments:
                symbol = assessment['symbol']
                risk_score = assessment['overall_risk_score']
                veto = assessment['veto_recommended']
                concerns = len(assessment['concerns'])
                
                status = "🚫 VETO" if veto else "⚠️ CHALLENGE" if risk_score >= 6 else "✅ APPROVE"
                print(f"      {symbol}: {risk_score}/10 {status} ({concerns} concerns)")
                
                # Show top concerns
                if assessment['concerns']:
                    for concern in assessment['concerns'][:2]:
                        print(f"        • {concern}")
                        
            print(f"\n   📊 Summary:")
            print(f"   {result.get('summary', 'No summary available')}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Risk manager evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_individual_proposal_assessment():
    """Test individual proposal assessment"""
    
    print(f"\n🔍 Testing Individual Proposal Assessment")
    print("-" * 40)
    
    try:
        risk_manager = RiskManagerAgent()
        
        # Test specific proposal evaluation
        context = {"proposal_id": "qullamaggie_agent_TSLA_trade_20250918_143000"}
        result = risk_manager.process_message("Evaluate this specific proposal", context)
        
        if result.get("type") == "single_proposal_assessment":
            print(f"   ✅ Individual assessment completed")
            assessment = result['assessment']
            print(f"   📈 Symbol: {assessment['symbol']}")
            print(f"   📊 Overall Risk: {assessment['overall_risk_score']}/10")
            print(f"   🚫 Veto Recommended: {assessment['veto_recommended']}")
            
            breakdown = assessment['risk_breakdown']
            print(f"\n   📋 Risk Breakdown:")
            print(f"      Concentration: {breakdown['concentration']}/10")
            print(f"      Correlation: {breakdown['correlation']}/10")
            print(f"      Risk/Reward: {breakdown['reward_risk']}/10")
            print(f"      Market Regime: {breakdown['market_regime']}/10")
            print(f"      Position Size: {breakdown['position_size']}/10")
            
            if assessment['concerns']:
                print(f"\n   ⚠️ Concerns:")
                for concern in assessment['concerns']:
                    print(f"      • {concern}")
            
            if assessment['suggested_adjustments']:
                print(f"\n   💡 Suggested Adjustments:")
                for adjustment in assessment['suggested_adjustments']:
                    print(f"      • {adjustment}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Individual assessment test failed: {e}")
        return None

def test_veto_and_challenge():
    """Test veto and challenge capabilities"""
    
    print(f"\n🚫 Testing Veto and Challenge Capabilities")
    print("-" * 35)
    
    try:
        risk_manager = RiskManagerAgent()
        
        # Test veto
        print("\n1️⃣ Testing Veto Authority")
        context = {"proposal_id": "qullamaggie_agent_TSLA_trade_20250918_143000"}
        veto_result = risk_manager.process_message("Veto this proposal due to excessive position size", context)
        
        if veto_result.get("type") == "veto_response":
            print(f"   ✅ Veto issued")
            print(f"   📋 Proposal: {veto_result['proposal_id']}")
            print(f"   ⚠️ Reason: {veto_result['veto_reason']}")
            print(f"   🔒 Authority: {veto_result['authority']}")
        else:
            print(f"   ❌ Veto failed: {veto_result.get('message', 'Unknown error')}")
        
        # Test challenge
        print("\n2️⃣ Testing Challenge Authority")
        context = {"proposal_id": "technical_agent_AAPL_trade_20250918_144000"}
        challenge_result = risk_manager.process_message("Challenge this proposal for poor risk/reward ratio", context)
        
        if challenge_result.get("type") == "challenge_response":
            print(f"   ✅ Challenge issued")
            print(f"   📋 Proposal: {challenge_result['proposal_id']}")
            print(f"   ⚠️ Level: {challenge_result['challenge_level']}")
            print(f"   💬 Response Required: {challenge_result['response_required']}")
        else:
            print(f"   ❌ Challenge failed: {challenge_result.get('message', 'Unknown error')}")
        
        return veto_result, challenge_result
        
    except Exception as e:
        print(f"❌ Veto/Challenge test failed: {e}")
        return None, None

def test_portfolio_risk_assessment():
    """Test portfolio-level risk assessment"""
    
    print(f"\n📊 Testing Portfolio Risk Assessment")
    print("-" * 30)
    
    try:
        risk_manager = RiskManagerAgent()
        
        result = risk_manager.process_message("Assess overall portfolio risk")
        
        if result.get("type") == "portfolio_risk_assessment":
            print(f"   ✅ Portfolio assessment completed")
            print(f"   📊 Portfolio Risk Score: {result['portfolio_risk_score']:.1f}/10")
            print(f"   📋 Proposals Analyzed: {result['proposal_count']}")
            print(f"   🎯 Risk Level: {result['risk_level'].upper()}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Portfolio assessment test failed: {e}")
        return None

def test_risk_manager_status():
    """Test risk manager status reporting"""
    
    print(f"\n📈 Testing Risk Manager Status")
    print("-" * 25)
    
    try:
        risk_manager = RiskManagerAgent()
        
        result = risk_manager.process_message("Show risk manager status")
        
        if result.get("type") == "risk_status":
            print(f"   ✅ Status retrieved")
            print(f"   📊 Agent: {result['agent']}")
            print(f"   📋 Proposals Monitored: {result['proposals_monitored']}")
            print(f"   🌍 Market Regime: {result['market_regime']}")
            print(f"   ⚙️ Status: {result['status']}")
            
            # Show key risk parameters
            params = result.get('risk_parameters', {})
            print(f"\n   ⚙️ Key Risk Parameters:")
            print(f"      Max Position: {params.get('max_single_position', 0)*100:.0f}%")
            print(f"      Min R/R Ratio: {params.get('min_risk_reward_ratio', 0):.1f}:1")
            print(f"      Max Position Value: ${params.get('max_position_value', 0):,}")
        else:
            print(f"   ❌ Status failed: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return None

if __name__ == "__main__":
    print("🔒 Risk Manager Testing Suite")
    print("=" * 40)
    
    # Setup
    print("\n🔧 Setting up test environment...")
    sample_count = create_sample_proposals()
    
    # Run tests
    print(f"\n🔬 Running Risk Manager Tests...")
    
    evaluation_result = test_risk_manager_evaluation()
    individual_result = test_individual_proposal_assessment()
    veto_result, challenge_result = test_veto_and_challenge()
    portfolio_result = test_portfolio_risk_assessment()
    status_result = test_risk_manager_status()
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"   Proposal Evaluation: {'✅ PASSED' if evaluation_result else '❌ FAILED'}")
    print(f"   Individual Assessment: {'✅ PASSED' if individual_result else '❌ FAILED'}")
    print(f"   Veto Authority: {'✅ PASSED' if veto_result else '❌ FAILED'}")
    print(f"   Challenge Authority: {'✅ PASSED' if challenge_result else '❌ FAILED'}")
    print(f"   Portfolio Assessment: {'✅ PASSED' if portfolio_result else '❌ FAILED'}")
    print(f"   Status Reporting: {'✅ PASSED' if status_result else '❌ FAILED'}")
    
    all_passed = all([evaluation_result, individual_result, veto_result, challenge_result, portfolio_result, status_result])
    
    if all_passed:
        print(f"\n🎉 All Risk Manager tests passed!")
        print(f"🔒 Risk Manager is ready for conservative portfolio management!")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Integrate with real portfolio data")
    print(f"   • Add sector correlation analysis")
    print(f"   • Implement dynamic market regime detection")
    print(f"   • Connect to Investment Committee workflow")
    print(f"   • Add real-time risk monitoring")