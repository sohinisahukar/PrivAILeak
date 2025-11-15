# Example Test Script - Quick Validation

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import opacus
        print(f"✅ Opacus {opacus.__version__}")
    except ImportError:
        print("❌ Opacus not installed")
        return False
    
    try:
        from faker import Faker
        print(f"✅ Faker installed")
    except ImportError:
        print("❌ Faker not installed")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not installed")
        return False
    
    print("\n✅ All required packages are installed!\n")
    return True


def test_config():
    """Test if config file is accessible"""
    print("Testing configuration...")
    
    try:
        from config import (
            MODEL_NAME, DATA_DIR, MODELS_DIR, RESULTS_DIR,
            NUM_EPOCHS, EPSILON_VALUES
        )
        print(f"✅ Config loaded successfully")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Epochs: {NUM_EPOCHS}")
        print(f"   Privacy budgets: {EPSILON_VALUES}")
        print(f"   Data directory: {DATA_DIR}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False


def test_directories():
    """Test if required directories exist or can be created"""
    print("\nTesting directories...")
    
    from config import DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR
    
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
        if dir_path.exists():
            print(f"✅ {dir_path.name}/ exists")
        else:
            print(f"⚠️  {dir_path.name}/ will be created")
    
    return True


def test_model_download():
    """Test if model can be downloaded"""
    print("\nTesting model download...")
    
    try:
        from transformers import GPT2Tokenizer
        from config import MODEL_NAME
        
        print(f"   Downloading {MODEL_NAME}...")
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        print(f"✅ Model downloaded successfully")
        print(f"   Vocabulary size: {len(tokenizer)}")
        return True
    except Exception as e:
        print(f"❌ Model download error: {e}")
        print("   Check internet connection")
        return False


def run_quick_data_test():
    """Quick test of data generation"""
    print("\nTesting data generation (quick test)...")
    
    try:
        from src.data_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        texts, records = generator.generate_dataset(10, private_ratio=0.3)
        
        print(f"✅ Generated {len(texts)} samples")
        print(f"   Private records: {len(records)}")
        print(f"\n   Sample text:")
        print(f"   {texts[0][:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("  PrivAI-Leak Installation Verification")
    print("="*70 + "\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("Configuration", test_config),
        ("Directories", test_directories),
        ("Model Download", test_model_download),
        ("Data Generation", run_quick_data_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} - {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. Run: python main.py")
        print("  2. Or start with: python src/data_generator.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Reinstall packages: pip install -r requirements.txt")
        print("  2. Check internet connection for model download")
        print("  3. Ensure Python 3.8+ is installed")
    
    print()


if __name__ == "__main__":
    main()
