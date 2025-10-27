"""Quick test script to verify all modules can be imported.

Usage:
    python quick_test.py
"""

import sys

print("Testing module imports...")
print("-" * 50)

try:
    print("1. Importing performance_profiler...")
    import performance_profiler
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

try:
    print("2. Importing part2_optimizations...")
    import part2_optimizations
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

try:
    print("3. Importing part3_modeling...")
    import part3_modeling
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

try:
    print("4. Importing part3_deployment...")
    import part3_deployment
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

try:
    print("5. Importing optimization_framework...")
    import optimization_framework
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("-" * 50)
print("✓ All modules imported successfully!")
print("\nYou can now run:")
print("  python run_optimizations.py")
