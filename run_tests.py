import unittest
import sys

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_CYAN = "\033[36m"
C_YELLOW = "\033[33m"

def run_all_tests():
    """Discovers and runs all test suites under the tests/ package directory."""
    print(f"\n{C_CYAN}{C_BOLD}==================================================")
    print("🧪 RUNNING PLACEMENT INTELLIGENCE UNIT TEST SUITE")
    print(f"=================================================={C_RESET}")

    # Discover tests under the tests/ directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{C_CYAN}{C_BOLD}==================================================")
    print("📊 UNIT TEST SUMMARY REPORT")
    print(f"=================================================={C_RESET}")
    print(f"  • Total Tests Run: {C_BOLD}{result.testsRun}{C_RESET}")
    print(f"  • Successful:      {C_GREEN}{result.testsRun - len(result.failures) - len(result.errors)}{C_RESET}")
    print(f"  • Failures:        {C_RED}{len(result.failures)}{C_RESET}")
    print(f"  • Errors:          {C_RED}{len(result.errors)}{C_RESET}")
    print(f"==================================================")

    if not result.wasSuccessful():
        print(f"\n{C_RED}❌ Test Suite Failed! Please review the tracebacks above.{C_RESET}\n")
        sys.exit(1)
    else:
        print(f"\n{C_GREEN}🎉 All tests passed successfully! Code is production-ready.{C_RESET}\n")
        sys.exit(0)

if __name__ == "__main__":
    run_all_tests()
