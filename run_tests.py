#!/usr/bin/env python3
"""
Test runner for the Manufacturing Scheduler project.

This script runs all tests and provides a comprehensive test report.
"""

import unittest
import sys
import os
from io import StringIO


def discover_and_run_tests():
    """Discover and run all tests."""
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True
    )
    
    print("=" * 70)
    print("MANUFACTURING SCHEDULER - TEST SUITE")
    print("=" * 70)
    
    result = runner.run(suite)
    
    # Print results
    output = stream.getvalue()
    print(output)
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        success_rate = 100.0
    else:
        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n❌ {failures + errors} TESTS FAILED")
    
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Detailed failure information
    if result.failures:
        print("\n" + "=" * 70)
        print("FAILURE DETAILS")
        print("=" * 70)
        for test, traceback in result.failures:
            print(f"\nFAILED: {test}")
            print("-" * 50)
            print(traceback)
    
    if result.errors:
        print("\n" + "=" * 70)
        print("ERROR DETAILS")
        print("=" * 70)
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print("-" * 50)
            print(traceback)
    
    return result.wasSuccessful()


def main():
    """Main test runner."""
    try:
        success = discover_and_run_tests()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install numpy pandas scikit-learn")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()