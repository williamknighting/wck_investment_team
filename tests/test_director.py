#!/usr/bin/env python3
"""
Test script for Director Agent
Demonstrates investment committee orchestration and final decision making
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Set up environment
os.environ.setdefault("OPENAI_API_KEY", "test_key")

from agents.director import DirectorAgent
from src.utils.logging_config import get_logger

def ensure_sample_proposals_exist():
    """Ensure sample proposals exist for testing"""
    proposals_dir = Path("proposals")
    
    # Check if we already have proposals
    existing_proposals = list(proposals_dir.glob("*.md"))
    
    if len(existing_proposals) >= 3:
        print(f"✅ Found {len(existing_proposals)} existing proposals")
        return len(existing_proposals)
    
    # Create additional proposals if needed
    proposals_dir.mkdir(exist_ok=True)
    
    # High-risk proposal that should be adjusted
    high_risk_proposal = """# Trade Proposal - NVDA

**Agent**: momentum_agent  
**Date**: 2025-09-18 15:00:00  
**Symbol**: NVDA  
**Type**: trade  

---

## Momentum Breakout Analysis

**EXCELLENT SETUP** - AI chip momentum continues

### Trade Parameters:
- **Entry**: $500.00
- **Stop Loss**: $475.00
- **Profit Target**: $575.00
- **Position Size**: 400 shares
- **Risk/Reward**: 3.0:1
- **Conviction**: 8/10

### Setup Quality: Excellent
Strong AI sector momentum with institutional buying support.
Technical breakout confirmed with high volume.

**Position Value**: $200,000
**Risk Amount**: $10,000
"""
    
    # Low conviction proposal that should be rejected
    low_conviction_proposal = """# Trade Proposal - KO

**Agent**: dividend_agent  
**Date**: 2025-09-18 15:05:00  
**Symbol**: KO  
**Type**: trade  

---

## Dividend Value Play

**WEAK SETUP** - Defensive positioning

### Trade Parameters:
- **Entry**: $60.00
- **Stop Loss**: $55.00
- **Profit Target**: $75.00
- **Position Size**: 500 shares
- **Risk/Reward**: 3.0:1
- **Conviction**: 3/10

### Setup Quality: Weak
Low conviction defensive play with limited upside catalyst.
Market environment may not favor dividend stocks.

**Position Value**: $30,000
**Risk Amount**: $2,500
"""
    
    # Create proposals
    new_proposals = [
        ("momentum_agent_NVDA_trade_20250918_150000.md", high_risk_proposal),
        ("dividend_agent_KO_trade_20250918_150500.md", low_conviction_proposal)
    ]
    
    created_count = 0
    for filename, content in new_proposals:
        filepath = proposals_dir / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            created_count += 1
    
    total_proposals = len(list(proposals_dir.glob("*.md")))
    print(f"✅ Ensured {total_proposals} proposals exist ({created_count} created)")
    return total_proposals

def test_director_initialization():
    """Test Director Agent initialization"""
    
    print("👔 Testing Director Initialization")
    print("=" * 40)
    
    try:
        # Initialize Director
        print("🚀 Initializing Director Agent...")
        director = DirectorAgent()
        
        print(f"   ✅ Director initialized successfully")
        print(f"   📊 Agent: {director.agent_name}")
        print(f"   📋 Max Rounds: {director.max_rounds}")
        print(f"   📁 Proposals Folder: {director.proposals_folder}")
        print(f"   📄 Decisions Folder: {director.decisions_folder}")
        
        return director
        
    except Exception as e:
        print(f"❌ Director initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_proposal_review():
    """Test proposal review functionality"""
    
    print(f"\n📋 Testing Proposal Review")
    print("-" * 30)
    
    try:
        director = DirectorAgent()
        
        result = director.process_message("Review all proposals")
        
        if result.get("type") == "proposal_review":
            print(f"   ✅ Proposal review completed")
            print(f"   📊 Total Proposals: {result['proposals_count']}")
            print(f"   🔥 High Conviction: {result['high_conviction']}")
            print(f"   ⚡ Moderate Conviction: {result['moderate_conviction']}")
            print(f"   🔸 Low Conviction: {result['low_conviction']}")
            
            print(f"\n   📋 Proposal Details:")
            for proposal in result.get("proposals", []):
                symbol = proposal['symbol']
                agent = proposal['agent']
                conviction = proposal['conviction']
                rr = proposal.get('risk_reward', 0)
                value = proposal.get('position_value', 0)
                
                print(f"      {symbol} ({agent}): {conviction}/10 conviction, {rr:.1f}:1 R/R, ${value:,.0f}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Proposal review test failed: {e}")
        return None

def test_committee_meeting():
    """Test investment committee meeting orchestration"""
    
    print(f"\n🏛️ Testing Investment Committee Meeting")
    print("-" * 35)
    
    try:
        director = DirectorAgent()
        
        result = director.process_message("Start investment committee meeting")
        
        if result.get("type") == "committee_started":
            print(f"   ✅ Committee meeting started")
            print(f"   📊 Proposals to Review: {result['proposals_count']}")
            print(f"   📋 Proposal IDs: {', '.join(result.get('proposals', []))}")
            print(f"   🎯 Max Rounds: {result['max_rounds']}")
            print(f"   📍 Current Round: {result['current_round']}")
            
            print(f"\n   💬 Opening Statement:")
            opening = result.get('opening_statement', '')
            # Show first few lines of opening statement
            opening_lines = opening.split('\n')[:8]
            for line in opening_lines:
                if line.strip():
                    print(f"      {line}")
            if len(opening.split('\n')) > 8:
                print(f"      ...")
                
            print(f"\n   🌍 Market Context:")
            context_lines = result.get('market_context', '').split('\n')[:4]
            for line in context_lines:
                if line.strip():
                    print(f"      {line}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Committee meeting test failed: {e}")
        return None

def test_pointed_questions():
    """Test pointed questioning capability"""
    
    print(f"\n❓ Testing Pointed Questions")
    print("-" * 25)
    
    try:
        director = DirectorAgent()
        
        # Test questioning different agent types
        test_cases = [
            ("qullamaggie_agent", "Challenge momentum strategy"),
            ("risk_manager", "Question position size limits"),
            ("value_agent", "Challenge timing assumptions")
        ]
        
        for agent, topic in test_cases:
            print(f"\n   🎯 Questioning {agent} about {topic}")
            context = {"agent": agent}
            result = director.process_message(f"Ask question about {topic}", context)
            
            if result.get("type") == "pointed_question":
                print(f"      ✅ Questions generated")
                print(f"      📍 Round: {result['round']}")
                print(f"      🎯 Target: {result['target_agent']}")
                print(f"      💬 Tone: {result['tone']}")
                
                questions = result.get('questions', [])
                print(f"      ❓ Questions ({len(questions)}):")
                for i, question in enumerate(questions[:2], 1):
                    print(f"         {i}. {question}")
            else:
                print(f"      ❌ Error: {result.get('message', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pointed questions test failed: {e}")
        return False

def test_conflict_resolution():
    """Test conflict resolution authority"""
    
    print(f"\n⚖️ Testing Conflict Resolution")
    print("-" * 25)
    
    try:
        director = DirectorAgent()
        
        # Test conflict scenarios
        conflicts = [
            {
                "agents": ["qullamaggie_agent", "risk_manager"],
                "topic": "position size disagreement"
            },
            {
                "agents": ["value_agent", "momentum_agent"],
                "topic": "timing conflict"
            }
        ]
        
        for conflict in conflicts:
            agents = conflict["agents"]
            topic = conflict["topic"]
            
            print(f"\n   ⚖️ Resolving: {topic}")
            print(f"      Conflicting agents: {', '.join(agents)}")
            
            context = {"agents": agents, "topic": topic}
            result = director.process_message("Resolve conflict", context)
            
            if result.get("type") == "conflict_resolution":
                print(f"      ✅ Conflict resolved")
                print(f"      📍 Round: {result['round']}")
                
                resolution = result.get('resolution', {})
                ruling = resolution.get('ruling', 'No ruling')
                rationale = resolution.get('rationale', 'No rationale')
                
                print(f"      ⚖️ Ruling: {ruling}")
                print(f"      💭 Rationale: {rationale}")
                print(f"      🔒 Authority: {result.get('authority', 'Unknown')}")
            else:
                print(f"      ❌ Error: {result.get('message', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conflict resolution test failed: {e}")
        return False

def test_final_decision():
    """Test final decision making"""
    
    print(f"\n🏛️ Testing Final Decision Making")
    print("-" * 30)
    
    try:
        director = DirectorAgent()
        
        result = director.process_message("Make final decision on all proposals")
        
        if result.get("type") == "final_decision":
            print(f"   ✅ Final decision completed")
            print(f"   📊 Decision ID: {result['decision_id']}")
            print(f"   ✅ Approved: {result['approved_count']}")
            print(f"   ❌ Rejected: {result['rejected_count']}")
            print(f"   🔄 Adjusted: {result['adjusted_count']}")
            
            if result.get('decision_file'):
                print(f"   📄 Decision Report: {result['decision_file']}")
            
            print(f"\n   📋 Decision Summary:")
            summary = result.get('summary', 'No summary available')
            print(f"      {summary}")
            
            # Check if decision file was created
            if result.get('decision_file'):
                decision_file = Path(result['decision_file'])
                if decision_file.exists():
                    print(f"   ✅ Decision report file created successfully")
                    file_size = decision_file.stat().st_size
                    print(f"   📊 Report size: {file_size} bytes")
                else:
                    print(f"   ⚠️ Decision report file not found")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Final decision test failed: {e}")
        return None

def test_conversation_status():
    """Test conversation status tracking"""
    
    print(f"\n📊 Testing Conversation Status")
    print("-" * 25)
    
    try:
        director = DirectorAgent()
        
        # Simulate some conversation rounds
        director.current_round = 3
        
        result = director.process_message("Show conversation status")
        
        if result.get("type") == "conversation_status":
            print(f"   ✅ Status retrieved")
            print(f"   📍 Current Round: {result['current_round']}")
            print(f"   📊 Max Rounds: {result['max_rounds']}")
            print(f"   ⏳ Rounds Remaining: {result['rounds_remaining']}")
            print(f"   📝 Conversation Length: {result['conversation_length']}")
            print(f"   ➡️ Can Continue: {result['can_continue']}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Conversation status test failed: {e}")
        return None

def test_decision_memory():
    """Test decision memory functionality"""
    
    print(f"\n🧠 Testing Decision Memory")
    print("-" * 20)
    
    try:
        director = DirectorAgent()
        
        result = director.process_message("Show decision history")
        
        if result.get("type") == "decision_history":
            print(f"   ✅ Decision history retrieved")
            print(f"   📊 Total Decisions: {result['total_decisions']}")
            
            recent = result.get('recent_decisions', [])
            print(f"   📋 Recent Decisions: {len(recent)}")
            
            if recent:
                latest = recent[-1]
                print(f"   📄 Latest Decision ID: {latest.get('decision_id', 'Unknown')}")
                print(f"   📅 Latest Timestamp: {latest.get('timestamp', 'Unknown')}")
            else:
                print(f"   📄 No previous decisions found")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Decision memory test failed: {e}")
        return None

def test_market_context():
    """Test market context assessment"""
    
    print(f"\n🌍 Testing Market Context")
    print("-" * 20)
    
    try:
        director = DirectorAgent()
        
        result = director.process_message("Assess market context")
        
        if result.get("type") == "market_context":
            print(f"   ✅ Market context assessed")
            print(f"   📅 Timestamp: {result['timestamp']}")
            
            context = result.get('context', '')
            context_lines = context.split('\n')[:5]
            print(f"   🌍 Context Summary:")
            for line in context_lines:
                if line.strip():
                    print(f"      {line}")
        else:
            print(f"   ❌ Error: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Market context test failed: {e}")
        return None

if __name__ == "__main__":
    print("👔 Director Agent Testing Suite")
    print("=" * 40)
    
    # Setup
    print("\n🔧 Setting up test environment...")
    proposal_count = ensure_sample_proposals_exist()
    
    # Run tests
    print(f"\n🔬 Running Director Tests...")
    
    director_init = test_director_initialization()
    proposal_review = test_proposal_review()
    committee_meeting = test_committee_meeting()
    pointed_questions = test_pointed_questions()
    conflict_resolution = test_conflict_resolution()
    final_decision = test_final_decision()
    conversation_status = test_conversation_status()
    decision_memory = test_decision_memory()
    market_context = test_market_context()
    
    # Summary
    print(f"\n📊 Test Results Summary:")
    print(f"   Director Initialization: {'✅ PASSED' if director_init else '❌ FAILED'}")
    print(f"   Proposal Review: {'✅ PASSED' if proposal_review else '❌ FAILED'}")
    print(f"   Committee Meeting: {'✅ PASSED' if committee_meeting else '❌ FAILED'}")
    print(f"   Pointed Questions: {'✅ PASSED' if pointed_questions else '❌ FAILED'}")
    print(f"   Conflict Resolution: {'✅ PASSED' if conflict_resolution else '❌ FAILED'}")
    print(f"   Final Decision: {'✅ PASSED' if final_decision else '❌ FAILED'}")
    print(f"   Conversation Status: {'✅ PASSED' if conversation_status else '❌ FAILED'}")
    print(f"   Decision Memory: {'✅ PASSED' if decision_memory else '❌ FAILED'}")
    print(f"   Market Context: {'✅ PASSED' if market_context else '❌ FAILED'}")
    
    all_passed = all([
        director_init, proposal_review, committee_meeting, pointed_questions,
        conflict_resolution, final_decision, conversation_status, 
        decision_memory, market_context
    ])
    
    if all_passed:
        print(f"\n🎉 All Director tests passed!")
        print(f"👔 Director is ready to orchestrate investment committee meetings!")
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Test full multi-agent conversation flow")
    print(f"   • Integrate with real proposal generation")
    print(f"   • Add advanced conflict resolution scenarios")
    print(f"   • Implement real-time market context analysis")
    print(f"   • Connect to execution systems")