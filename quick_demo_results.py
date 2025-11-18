#!/usr/bin/env python3
"""
Quick demo - Results only (no model loading)
Fast version that just shows results without loading models
"""

import json
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import RESULTS_DIR

print("="*70)
print("PRIVAI-LEAK QUICK DEMO (Results Only)")
print("="*70)
print()

# Load results
results_file = RESULTS_DIR / "evaluation_results.json"

if not results_file.exists():
    print("❌ Results file not found!")
    print("Please run the pipeline first: python main.py")
    sys.exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

print("✅ Results loaded!\n")

# Display baseline
baseline = results['baseline']
print("="*70)
print("BASELINE MODEL (No Privacy Protection)")
print("="*70)
print(f"Privacy Risk:     {baseline['privacy_risk']:.2f}%")
print(f"Leakage Rate:     {baseline['leakage_rate']:.2f}%")
print(f"Perplexity:       {baseline['perplexity']:.2f}")
print(f"Membership Inf.:  {baseline['inference_rate']:.2f}%")

# Display DP models
print("\n" + "="*70)
print("DP-PROTECTED MODELS")
print("="*70)

for epsilon in sorted(results['dp_models'].keys(), key=float):
    dp = results['dp_models'][epsilon]
    leakage_reduction = baseline['leakage_rate'] - dp['leakage_rate']
    privacy_improvement = baseline['privacy_risk'] - dp['privacy_risk']
    
    print(f"\nε = {epsilon}:")
    print(f"  Privacy Risk:   {dp['privacy_risk']:.2f}% (↓ {privacy_improvement:.2f}%)")
    print(f"  Leakage Rate:   {dp['leakage_rate']:.2f}% (↓ {leakage_reduction:.2f}%)")
    print(f"  Perplexity:     {dp['perplexity']:.2f}")
    print(f"  Final ε:        {dp.get('final_epsilon', dp['epsilon']):.4f}")

# Summary
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

best_dp = min(results['dp_models'].items(), 
             key=lambda x: float(x[1]['leakage_rate']))
epsilon, dp = best_dp

leakage_reduction = baseline['leakage_rate'] - dp['leakage_rate']
reduction_pct = (leakage_reduction / baseline['leakage_rate']) * 100

print(f"\n✅ Privacy Protection:")
print(f"   • Baseline leakage: {baseline['leakage_rate']:.2f}%")
print(f"   • Best DP leakage (ε={epsilon}): {dp['leakage_rate']:.2f}%")
print(f"   • Reduction: {leakage_reduction:.2f}% ({reduction_pct:.1f}% improvement!)")

print(f"\n⚖️  Privacy-Utility Trade-off:")
print(f"   • Lower ε = Better privacy, Lower utility")
print(f"   • Higher ε = Better utility, Lower privacy")

print("\n" + "="*70)
print("✅ DEMO COMPLETE")
print("="*70)
print("\n💡 For live model testing, use:")
print("   • Jupyter: jupyter notebook Demo_Presentation.ipynb")
print("   • Script: python run_demo_notebook.py")

