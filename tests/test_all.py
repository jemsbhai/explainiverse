# tests/test_all.py
"""
Run all tests for Explainiverse.

Usage:
    poetry run pytest tests/ -v
    
Or run this file directly:
    poetry run python tests/test_all.py
"""
import subprocess
import sys


def run_all_tests():
    """Run all pytest tests."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd="."
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    run_all_tests()
