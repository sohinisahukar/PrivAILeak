"""
Demo script for showcasing PrivAI-Leak results.
Run this after the pipeline completes to present findings.
"""

import json
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import RESULTS_DIR, MODELS_DIR, DATA_DIR


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def load_results():
    """Load evaluation results"""
    results_file = RESULTS_DIR / "evaluation_results.json"
    if not results_file.exists():
        print("❌ Results not found! Please run the pipeline first.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def show_comparison_table():
    """Display comparison table"""
    print_section("📊 MODEL COMPARISON TABLE")
    
    csv_file = RESULTS_DIR / "comparison_table.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(df.to_string(index=False))
    else:
        print("⚠️  Comparison table not found")


def show_key_findings(results):
    """Show key findings and insights"""
    print_section("🔍 KEY FINDINGS")
    
    baseline = results['baseline']
    dp_models = results['dp_models']
    
    print("Baseline Model (No Privacy Protection):")
    print(f"  • Privacy Risk: {baseline['privacy_risk']:.2f}%")
    print(f"  • Leakage Rate: {baseline['leakage_rate']:.2f}%")
    print(f"  • Membership Inference: {baseline['inference_rate']:.2f}%")
    print(f"  • Perplexity: {baseline['perplexity']:.2f}")
    
    print("\nDifferential Privacy Models:")
    for epsilon in sorted(dp_models.keys(), key=float):
        dp = dp_models[epsilon]
        privacy_improvement = baseline['privacy_risk'] - dp['privacy_risk']
        utility_loss = dp['perplexity'] - baseline['perplexity']
        
        print(f"\n  ε = {epsilon}:")
        print(f"    • Privacy Risk: {dp['privacy_risk']:.2f}% (↓ {privacy_improvement:.2f}%)")
        print(f"    • Leakage Rate: {dp['leakage_rate']:.2f}%")
        print(f"    • Perplexity: {dp['perplexity']:.2f} (↑ {utility_loss:.2f})")
        print(f"    • Privacy Improvement: {privacy_improvement:.2f} percentage points")


def show_privacy_utility_tradeoff(results):
    """Show privacy-utility tradeoff analysis"""
    print_section("⚖️  PRIVACY-UTILITY TRADE-OFF")
    
    baseline = results['baseline']
    dp_models = results['dp_models']
    
    print("Analysis:")
    print("  Lower ε (more privacy) → Higher perplexity (lower utility)")
    print("  Higher ε (less privacy) → Lower perplexity (higher utility)\n")
    
    print("Trade-off Summary:")
    for epsilon in sorted(dp_models.keys(), key=float):
        dp = dp_models[epsilon]
        privacy_reduction = baseline['privacy_risk'] - dp['privacy_risk']
        utility_cost = dp['perplexity'] - baseline['perplexity']
        
        print(f"\n  ε = {epsilon}:")
        print(f"    Privacy Gain: {privacy_reduction:.2f}% reduction in risk")
        print(f"    Utility Cost: {utility_cost:.2f} increase in perplexity")
        print(f"    Efficiency: {privacy_reduction/utility_cost:.2f}% privacy per perplexity unit")


def show_attack_results():
    """Show detailed attack results"""
    print_section("🎯 PRIVACY ATTACK RESULTS")
    
    # Baseline attacks
    baseline_attacks = MODELS_DIR / "baseline_attack_results.json"
    if baseline_attacks.exists():
        with open(baseline_attacks, 'r') as f:
            baseline = json.load(f)
        
        print("Baseline Model Attack Results:")
        print(f"  • Prompt Extraction Leakage: {baseline['prompt_extraction']['leakage_rate']:.2f}%")
        print(f"  • Membership Inference Rate: {baseline['membership_inference']['inference_rate']:.2f}%")
        print(f"  • Canary Extraction Rate: {baseline['canary_extraction']['extraction_rate']:.2f}%")
        print(f"  • Exact Memorization Rate: {baseline['exact_memorization']['memorization_rate']:.2f}%")
        print(f"  • Overall Privacy Risk: {baseline['overall_privacy_risk']:.2f}%")
    
    # DP model attacks
    print("\nDP Model Attack Results:")
    for epsilon in [0.5, 1.0, 5.0, 10.0]:
        dp_attacks = MODELS_DIR / f"dp_eps_{epsilon}_attack_results.json"
        if dp_attacks.exists():
            with open(dp_attacks, 'r') as f:
                dp = json.load(f)
            
            print(f"\n  ε = {epsilon}:")
            print(f"    • Prompt Extraction: {dp['prompt_extraction']['leakage_rate']:.2f}%")
            print(f"    • Membership Inference: {dp['membership_inference']['inference_rate']:.2f}%")
            print(f"    • Canary Extraction: {dp['canary_extraction']['extraction_rate']:.2f}%")
            print(f"    • Overall Risk: {dp['overall_privacy_risk']:.2f}%")


def show_visualizations():
    """List available visualizations"""
    print_section("📈 GENERATED VISUALIZATIONS")
    
    viz_files = [
        "privacy_budget_vs_leakage.png",
        "privacy_budget_vs_utility.png",
        "privacy_utility_tradeoff.png",
        "model_comparison_bars.png"
    ]
    
    print("Available plots (in results/ directory):")
    for viz_file in viz_files:
        file_path = RESULTS_DIR / viz_file
        if file_path.exists():
            print(f"  ✅ {viz_file}")
        else:
            print(f"  ⏳ {viz_file} (will be generated in Step 6)")


def show_recommendations(results):
    """Show recommendations based on results"""
    print_section("💡 RECOMMENDATIONS")
    
    baseline = results['baseline']
    dp_models = results['dp_models']
    
    # Find best trade-off
    best_epsilon = None
    best_score = -1
    
    for epsilon, dp in dp_models.items():
        privacy_gain = baseline['privacy_risk'] - dp['privacy_risk']
        utility_cost = dp['perplexity'] - baseline['perplexity']
        
        # Score: privacy gain / utility cost (higher is better)
        if utility_cost > 0:
            score = privacy_gain / utility_cost
            if score > best_score:
                best_score = score
                best_epsilon = epsilon
    
    print("Based on the results:")
    print(f"\n1. Best Privacy-Utility Trade-off: ε = {best_epsilon}")
    if best_epsilon:
        best_dp = dp_models[best_epsilon]
        print(f"   • Privacy Risk: {best_dp['privacy_risk']:.2f}%")
        print(f"   • Perplexity: {best_dp['perplexity']:.2f}")
    
    print("\n2. For Maximum Privacy:")
    min_epsilon = min(dp_models.keys(), key=float)
    min_dp = dp_models[min_epsilon]
    print(f"   • Use ε = {min_epsilon}")
    print(f"   • Privacy Risk: {min_dp['privacy_risk']:.2f}%")
    print(f"   • Trade-off: Higher perplexity ({min_dp['perplexity']:.2f})")
    
    print("\n3. For Maximum Utility:")
    max_epsilon = max(dp_models.keys(), key=float)
    max_dp = dp_models[max_epsilon]
    print(f"   • Use ε = {max_epsilon}")
    print(f"   • Privacy Risk: {max_dp['privacy_risk']:.2f}%")
    print(f"   • Trade-off: Lower privacy protection")


def main():
    """Main demo function"""
    print("\n" + "="*80)
    print("  🎬 PrivAI-Leak Demo - Privacy Auditing Results")
    print("="*80)
    
    # Load results
    results = load_results()
    if not results:
        return
    
    # Show all sections
    show_comparison_table()
    show_key_findings(results)
    show_privacy_utility_tradeoff(results)
    show_attack_results()
    show_visualizations()
    show_recommendations(results)
    
    print_section("📁 OUTPUT FILES")
    print("All results are saved in:")
    print(f"  • Results: {RESULTS_DIR}")
    print(f"  • Models: {MODELS_DIR}")
    print(f"  • Visualizations: {RESULTS_DIR}/*.png")
    
    print("\n" + "="*80)
    print("  ✅ Demo Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

