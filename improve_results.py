"""
Script to regenerate results with improved code.
Run this after making improvements to get better results.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🔧 Running improved pipeline...")
    print("="*70)
    
    # Step 1: Regenerate data with improved canaries
    print("\nStep 1: Regenerating data with improved canaries...")
    subprocess.run([sys.executable, "src/healthcare_data_generator.py"])
    
    # Step 2: Retrain baseline
    print("\nStep 2: Retraining baseline model...")
    subprocess.run([sys.executable, "src/baseline_training.py"])
    
    # Step 3: Run attacks on baseline
    print("\nStep 3: Running privacy attacks on baseline...")
    subprocess.run([sys.executable, "src/privacy_attacks.py"])
    
    # Step 4: Retrain DP models (optional - takes time)
    print("\nStep 4: Retraining DP models...")
    print("   (This will take time - you can skip with Ctrl+C)")
    try:
        subprocess.run([sys.executable, "src/dp_training_manual.py"])
    except KeyboardInterrupt:
        print("\n⚠️  Skipped DP training. You can run it later.")
    
    # Step 5: Evaluate
    print("\nStep 5: Evaluating models...")
    subprocess.run([sys.executable, "src/evaluation.py"])
    
    print("\n✅ Improved pipeline complete!")
    print("Check results/ directory for updated results.")

if __name__ == "__main__":
    main()

