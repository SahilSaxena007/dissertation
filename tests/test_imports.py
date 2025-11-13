"""
Quick test to verify all module imports work (no runtime yet).
"""

print("Testing imports...")

try:
    from eval import performance_analysis
    print("✅ eval.performance_analysis imported")
except Exception as e:
    print(f"❌ eval.performance_analysis: {e}")

try:
    from models import base_models
    print("✅ models.base_models imported")
except Exception as e:
    print(f"❌ models.base_models: {e}")

try:
    from utils import constants
    print("✅ utils.constants imported")
except Exception as e:
    print(f"❌ utils.constants: {e}")

print("\n✅ All imports successful! Ready for Step 2.")