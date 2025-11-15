"""
Main pipeline script for PrivAI-Leak project.
Orchestrates the complete workflow from data generation to final evaluation.
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_generator import main as generate_data
from src.baseline_training import main as train_baseline
from src.privacy_attacks import main as run_attacks_baseline
from src.dp_training import main as train_dp_models
from src.evaluation import main as evaluate_all
from src.visualization import main as visualize_results


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_pipeline(skip_steps=None):
    """
    Run the complete PrivAI-Leak pipeline
    
    Args:
        skip_steps: List of step numbers to skip (e.g., [1, 2])
    """
    skip_steps = skip_steps or []
    
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════╗")
    print("║                          PrivAI-Leak Pipeline                          ║")
    print("║        Privacy Auditing Framework for Large Language Models           ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    try:
        # Step 1: Data Generation
        if 1 not in skip_steps:
            print_header("STEP 1: Data Preparation - Generating Synthetic Dataset")
            generate_data()
        else:
            print("⏭️  Skipping Step 1: Data Generation")
        
        # Step 2: Baseline Training
        if 2 not in skip_steps:
            print_header("STEP 2: Baseline Model Training (No Privacy)")
            train_baseline()
        else:
            print("⏭️  Skipping Step 2: Baseline Training")
        
        # Step 3: Privacy Attack on Baseline
        if 3 not in skip_steps:
            print_header("STEP 3: Privacy Leakage Simulation (Baseline)")
            run_attacks_baseline()
        else:
            print("⏭️  Skipping Step 3: Privacy Attacks on Baseline")
        
        # Step 4: DP Training
        if 4 not in skip_steps:
            print_header("STEP 4: Differential Privacy Training (DP-SGD)")
            train_dp_models()
        else:
            print("⏭️  Skipping Step 4: DP Training")
        
        # Step 5: Evaluation
        if 5 not in skip_steps:
            print_header("STEP 5: Comprehensive Evaluation & Comparison")
            evaluate_all()
        else:
            print("⏭️  Skipping Step 5: Evaluation")
        
        # Step 6: Visualization
        if 6 not in skip_steps:
            print_header("STEP 6: Visualization & Reporting")
            visualize_results()
        else:
            print("⏭️  Skipping Step 6: Visualization")
        
        # Final Summary
        print("\n")
        print("╔════════════════════════════════════════════════════════════════════════╗")
        print("║                    🎉 PIPELINE COMPLETE! 🎉                            ║")
        print("╚════════════════════════════════════════════════════════════════════════╝")
        print("\n✅ All stages completed successfully!")
        print(f"\n📂 Results are available in: results/")
        print(f"📊 Visualizations saved as PNG files")
        print(f"📋 Comparison tables saved as CSV")
        print("\n")
        
    except Exception as e:
        print("\n❌ ERROR occurred during pipeline execution:")
        print(f"   {str(e)}")
        raise


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="PrivAI-Leak: Privacy Auditing Framework for LLMs"
    )
    
    parser.add_argument(
        '--skip',
        nargs='+',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Skip specific steps (1-6)',
        default=[]
    )
    
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Run only a specific step',
        default=None
    )
    
    args = parser.parse_args()
    
    if args.step:
        # Run only specific step
        skip_steps = [i for i in range(1, 7) if i != args.step]
        print(f"Running only Step {args.step}")
        run_pipeline(skip_steps=skip_steps)
    else:
        # Run pipeline with optional skips
        run_pipeline(skip_steps=args.skip)


if __name__ == "__main__":
    main()
