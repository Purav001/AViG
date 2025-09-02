# Path: check_setup.py
import sys
import torch
import timm

print(f"--- System Info ---")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Timm version: {timm.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("-" * 20, "\n")

try:
    print("--- Checking Project Imports ---")
    import vig
    import pyramid_vig
    print("✅ Successfully imported 'vig' and 'pyramid_vig'")
    
    model = pyramid_vig.pvig_ti_224_gelu()
    print("✅ Successfully created a Pyramid-ViG model instance.")
    
    print("\n🎉 Success! Your environment and project structure are set up correctly.")

except Exception as e:
    print(f"\n❌ FAILED: An unexpected error occurred: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you are running this script from the project's root directory.")
    print("2. Ensure you have activated your virtual environment and installed all required packages.")