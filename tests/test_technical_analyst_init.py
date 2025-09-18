#!/usr/bin/env python3
"""
Quick test for Technical Analyst initialization
"""
import os
import sys

# Add project root to path
sys.path.append('.')

# Set up environment
os.environ.setdefault("OPENAI_API_KEY", "test_key")

from agents.technical_analyst import TechnicalAnalystAgent

def test_technical_analyst_init():
    """Test technical analyst initialization"""
    print("🔬 Testing Technical Analyst Initialization")
    print("=" * 45)
    
    try:
        print("🚀 Initializing Technical Analyst...")
        analyst = TechnicalAnalystAgent()
        
        print(f"   ✅ Technical Analyst initialized successfully")
        print(f"   📊 Agent Name: {analyst.agent_name}")
        print(f"   📋 Description: {analyst.agent_description}")
        print(f"   🔧 Has analyst: {hasattr(analyst, 'analyst')}")
        
        # Test a simple analysis request
        print(f"\n📈 Testing analysis request...")
        result = analyst.process_message("Provide comprehensive technical analysis for SPY", {"symbol": "SPY"})
        
        print(f"   📊 Analysis Type: {result.get('type', 'unknown')}")
        print(f"   🎯 Symbol: {result.get('symbol', 'unknown')}")
        
        if result.get("type") == "technical_analysis":
            print(f"   ✅ Technical analysis completed successfully")
        else:
            print(f"   ⚠️ Analysis returned: {result.get('message', 'unknown response')}")
        
        return analyst
        
    except Exception as e:
        print(f"❌ Technical analyst initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyst = test_technical_analyst_init()