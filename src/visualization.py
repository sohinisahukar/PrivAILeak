"""
Visualization module for generating plots and charts.
Creates privacy-utility trade-off visualizations.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RESULTS_DIR, FIGURE_DPI, FIGURE_SIZE


class ResultVisualizer:
    """Create visualizations for privacy-utility trade-offs"""
    
    def __init__(self):
        self.results_file = RESULTS_DIR / "evaluation_results.json"
        self.load_results()
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = FIGURE_SIZE
        plt.rcParams['figure.dpi'] = FIGURE_DPI
    
    def load_results(self):
        """Load evaluation results"""
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
    
    def prepare_dataframe(self) -> pd.DataFrame:
        """Prepare data for plotting"""
        data = []
        
        # Add baseline
        baseline = self.results['baseline']
        data.append({
            'model': 'Baseline',
            'epsilon': float('inf'),
            'perplexity': baseline['perplexity'],
            'leakage_rate': baseline['leakage_rate'],
            'inference_rate': baseline['inference_rate'],
            'privacy_risk': baseline['privacy_risk']
        })
        
        # Add DP models
        for epsilon, results in self.results['dp_models'].items():
            data.append({
                'model': f'DP (ε={epsilon})',
                'epsilon': float(epsilon),
                'perplexity': results['perplexity'],
                'leakage_rate': results['leakage_rate'],
                'inference_rate': results['inference_rate'],
                'privacy_risk': results['privacy_risk']
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('epsilon')
    
    def plot_privacy_budget_vs_leakage(self):
        """Plot epsilon vs leakage rate"""
        df = self.prepare_dataframe()
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Plot lines
        ax.plot(df['epsilon'], df['leakage_rate'], 
                marker='o', linewidth=2, markersize=8,
                label='Prompt Extraction Leakage', color='#e74c3c')
        
        ax.plot(df['epsilon'], df['inference_rate'], 
                marker='s', linewidth=2, markersize=8,
                label='Membership Inference Rate', color='#3498db')
        
        ax.plot(df['epsilon'], df['privacy_risk'], 
                marker='^', linewidth=2, markersize=8,
                label='Overall Privacy Risk', color='#9b59b6', linestyle='--')
        
        # Formatting
        ax.set_xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Privacy Budget vs. Information Leakage', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Add annotations
        for idx, row in df.iterrows():
            if row['epsilon'] != float('inf'):
                ax.annotate(f"ε={row['epsilon']}", 
                           xy=(row['epsilon'], row['privacy_risk']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        output_file = RESULTS_DIR / "privacy_budget_vs_leakage.png"
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
    
    def plot_privacy_budget_vs_utility(self):
        """Plot epsilon vs perplexity (utility)"""
        df = self.prepare_dataframe()
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Plot
        ax.plot(df['epsilon'], df['perplexity'], 
                marker='o', linewidth=2.5, markersize=10,
                color='#2ecc71', label='Model Perplexity')
        
        # Fill area
        ax.fill_between(df['epsilon'], df['perplexity'], 
                       alpha=0.3, color='#2ecc71')
        
        # Formatting
        ax.set_xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Privacy Budget vs. Model Utility', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Add value annotations
        for idx, row in df.iterrows():
            ax.annotate(f"{row['perplexity']:.1f}", 
                       xy=(row['epsilon'], row['perplexity']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = RESULTS_DIR / "privacy_budget_vs_utility.png"
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
    
    def plot_privacy_utility_tradeoff(self):
        """Plot privacy risk vs utility (scatter plot)"""
        df = self.prepare_dataframe()
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        # Create scatter plot
        scatter = ax.scatter(df['privacy_risk'], df['perplexity'],
                           s=200, c=df['epsilon'], cmap='RdYlGn_r',
                           alpha=0.7, edgecolors='black', linewidth=2)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Privacy Budget (ε)', fontsize=11, fontweight='bold')
        
        # Add labels for each point
        for idx, row in df.iterrows():
            label = 'Baseline' if row['epsilon'] == float('inf') else f"ε={row['epsilon']}"
            ax.annotate(label, 
                       xy=(row['privacy_risk'], row['perplexity']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Formatting
        ax.set_xlabel('Privacy Risk Score (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
        ax.set_title('Privacy-Utility Trade-off Analysis', 
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = RESULTS_DIR / "privacy_utility_tradeoff.png"
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
    
    def plot_comparison_bars(self):
        """Bar chart comparison of all models"""
        df = self.prepare_dataframe()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Privacy Risk comparison
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(df)), df['privacy_risk'], 
                       color=['#e74c3c' if x == float('inf') else '#27ae60' for x in df['epsilon']],
                       alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Privacy Risk Score (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Privacy Risk Comparison', fontsize=13, fontweight='bold')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['model'], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Perplexity comparison
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(df)), df['perplexity'], 
                       color=['#e74c3c' if x == float('inf') else '#3498db' for x in df['epsilon']],
                       alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Perplexity', fontsize=11, fontweight='bold')
        ax2.set_title('Model Utility Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['model'], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = RESULTS_DIR / "model_comparison_bars.png"
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✅ Saved: {output_file.name}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n📊 Generating visualizations...")
        print("="*70)
        
        self.plot_privacy_budget_vs_leakage()
        self.plot_privacy_budget_vs_utility()
        self.plot_privacy_utility_tradeoff()
        self.plot_comparison_bars()
        
        print("="*70)
        print(f"✅ All plots saved to {RESULTS_DIR}")


def main():
    """Main visualization script"""
    visualizer = ResultVisualizer()
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
