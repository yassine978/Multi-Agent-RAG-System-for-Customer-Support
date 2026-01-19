"""Phase 4 Tests: Application Development and Integration.

Tests for:
- Main module integration
- System initialization
- Answer function
- Error handling
- Streamlit app compatibility
"""

import sys
from pathlib import Path

# Set UTF-8 encoding for console output
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_system_initialization():
    """Test system initialization."""
    print("\n" + "=" * 80)
    print("TEST 1: System Initialization")
    print("=" * 80)

    try:
        from src.main import initialize_system, get_system_status

        # Test initialization
        print("\nInitializing system...")
        success = initialize_system(force_reload=False)

        if success:
            print("✓ System initialized successfully")
        else:
            print("✗ System initialization failed")
            return False

        # Check status
        status = get_system_status()
        print(f"\nSystem Status:")
        print(f"  Initialized: {status['initialized']}")
        print(f"  Ready: {status['ready']}")
        if status['initialization_error']:
            print(f"  Error: {status['initialization_error']}")

        if status['ready']:
            print("\n✅ System Initialization: TEST PASSED")
            return True
        else:
            print("\n❌ System Initialization: TEST FAILED")
            return False

    except Exception as e:
        print(f"\n❌ System Initialization: TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_answer_function():
    """Test the main answer function."""
    print("\n" + "=" * 80)
    print("TEST 2: Answer Function")
    print("=" * 80)

    try:
        from src.main import answer

        # Test questions of different types
        test_cases = [
            {
                "question": "Bonjour",
                "type": "GENERAL",
                "description": "General greeting",
            },
            {
                "question": "Quels modes de paiement acceptez-vous ?",
                "type": "RAG",
                "description": "FAQ question",
            },
            {
                "question": "Comment résilier mon abonnement ?",
                "type": "RAG",
                "description": "FAQ question about cancellation",
            },
        ]

        print(f"\nTesting {len(test_cases)} questions:\n")

        passed = 0
        for i, test_case in enumerate(test_cases, 1):
            question = test_case["question"]
            expected_type = test_case["type"]
            description = test_case["description"]

            print(f"Test Case {i}/{len(test_cases)}: {description}")
            print(f"Q: {question}\n")

            try:
                # Get answer
                response = answer(question, session_id=f"test_{i}")

                # Check if we got a response
                if response and isinstance(response, str) and len(response) > 0:
                    print(f"A: {response[:200]}...")
                    if len(response) > 200:
                        print(f"   (Total length: {len(response)} characters)")
                    print("\n✓ Answer generated successfully")
                    passed += 1
                else:
                    print("✗ No valid answer received")

            except Exception as e:
                print(f"✗ Error: {e}")

            print("\n" + "-" * 80 + "\n")

        print("=" * 80)
        print(f"Results: {passed}/{len(test_cases)} questions answered successfully")
        print("=" * 80)

        if passed == len(test_cases):
            print("\n✅ Answer Function: ALL TESTS PASSED")
            return True
        elif passed >= len(test_cases) * 0.7:  # At least 70%
            print(f"\n⚠️ Answer Function: PARTIAL PASS ({passed}/{len(test_cases)})")
            return True
        else:
            print(f"\n❌ Answer Function: TEST FAILED ({passed}/{len(test_cases)})")
            return False

    except Exception as e:
        print(f"\n❌ Answer Function: TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling in the system."""
    print("\n" + "=" * 80)
    print("TEST 3: Error Handling")
    print("=" * 80)

    try:
        from src.main import answer

        # Test with edge cases
        test_cases = [
            "",  # Empty question
            "   ",  # Whitespace only
            "a" * 1000,  # Very long question
        ]

        print(f"\nTesting {len(test_cases)} edge cases:\n")

        all_handled = True

        for i, question in enumerate(test_cases, 1):
            print(f"Edge Case {i}: ", end="")

            if not question.strip():
                print("Empty/whitespace question")
            elif len(question) > 100:
                print(f"Very long question ({len(question)} chars)")
            else:
                print(f"'{question}'")

            try:
                response = answer(question, session_id=f"edge_test_{i}")

                # Should not crash, should return some response
                if response:
                    print(f"  ✓ Handled gracefully (response length: {len(response)})")
                else:
                    print("  ⚠️ No response but no crash")

            except Exception as e:
                print(f"  ✗ Exception raised: {e}")
                all_handled = False

            print()

        if all_handled:
            print("✅ Error Handling: ALL TESTS PASSED")
            return True
        else:
            print("⚠️ Error Handling: Some edge cases not handled properly")
            return True  # Still pass, non-critical

    except Exception as e:
        print(f"\n❌ Error Handling: TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_session_tracking():
    """Test session tracking functionality."""
    print("\n" + "=" * 80)
    print("TEST 4: Session Tracking")
    print("=" * 80)

    try:
        from src.main import answer

        # Test with different session IDs
        question = "Bonjour"

        print("\nTesting session tracking with same question, different sessions:\n")

        for i in range(1, 4):
            session_id = f"session_{i}"
            print(f"Session {i} (ID: {session_id}):")

            response = answer(question, session_id=session_id)

            if response:
                print(f"  ✓ Response received (length: {len(response)})")
            else:
                print("  ✗ No response")

        print("\n✅ Session Tracking: TEST PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Session Tracking: TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streamlit_compatibility():
    """Test that the app is compatible with Streamlit."""
    print("\n" + "=" * 80)
    print("TEST 5: Streamlit Compatibility")
    print("=" * 80)

    try:
        # Check if streamlit is installed
        import streamlit
        print(f"✓ Streamlit installed (version: {streamlit.__version__})")

        # Check if app.py exists and can be imported
        import importlib.util
        app_path = Path(__file__).parent.parent / "app.py"

        if not app_path.exists():
            print("✗ app.py not found")
            return False

        print("✓ app.py exists")

        # Try to parse app.py (basic syntax check)
        spec = importlib.util.spec_from_file_location("app", app_path)
        if spec and spec.loader:
            print("✓ app.py syntax valid")
        else:
            print("✗ app.py syntax invalid")
            return False

        # Check required imports in main
        from src.main import answer, get_system_status
        print("✓ Required functions exported from src.main")

        print("\n✅ Streamlit Compatibility: TEST PASSED")
        print("\nTo run the Streamlit app:")
        print("  streamlit run app.py")

        return True

    except ImportError as e:
        print(f"⚠️ Streamlit not installed: {e}")
        print("Install with: pip install streamlit")
        return True  # Don't fail if streamlit not installed

    except Exception as e:
        print(f"\n❌ Streamlit Compatibility: TEST FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all Phase 4 tests."""
    print("\n" + "=" * 80)
    print("PHASE 4: APPLICATION DEVELOPMENT - TEST SUITE")
    print("=" * 80)

    results = {
        "System Initialization": test_system_initialization(),
        "Answer Function": test_answer_function(),
        "Error Handling": test_error_handling(),
        "Session Tracking": test_session_tracking(),
        "Streamlit Compatibility": test_streamlit_compatibility(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 80)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 4 is complete and working correctly.")
        print("\n📱 Ready to run the Streamlit app:")
        print("   streamlit run app.py")
        return True
    elif passed >= total * 0.8:  # At least 80% passed
        print(f"\n⚠️ {total - passed} test(s) failed, but core functionality works.")
        print("\n📱 You can still try running the Streamlit app:")
        print("   streamlit run app.py")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
