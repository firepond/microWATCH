# test_rpy2.py
import os
import sys

try:
    import rpy2

    import rpy2.robjects as robjects

    print("rpy2.robjects imported successfully")

    # Test basic R functionality
    r = robjects.r
    result = r("1 + 1")
    print(f"R calculation 1+1 = {result[0]}")

    # Test R version
    r_version = r("R.version.string")
    print(f"R version: {r_version[0]}")

    print("✓ rpy2 installation test passed!")

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
