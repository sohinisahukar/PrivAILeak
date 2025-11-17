"""
Evaluation module for comparing baseline and DP models.
Evaluates privacy leakage, utility metrics, and generates comparison reports.
"""

import json
from pathlib import Path
import pandas as pd
from typing import Dict, List
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR, RESULTS_DIR, EPSILON_VALUES, DATA_DIR
from src.privacy_attacks import PrivacyAttacker

# Try to import enhanced attacks (optional)
try:
    from src.advanced_privacy_attacks_enhanced import EnhancedPrivacyAttacker
    ENHANCED_ATTACKS_AVAILABLE = True
except ImportError:
    ENHANCED_ATTACKS_AVAILABLE = False
    print("⚠️ Enhanced attacks not available, using basic attacks only")


class ModelEvaluator:
    """Evaluate and compare baseline and DP models"""
    
    def __init__(self):
        self.results = {
            'baseline': {},
            'dp_models': {}
        }
    
    def evaluate_baseline(self, use_enhanced_attacks=False):
        """Evaluate baseline model"""
        print("\n" + "="*70)
        print("📊 EVALUATING BASELINE MODEL")
        print("="*70)
        
        model_path = MODELS_DIR / "baseline_model"
        
        if not model_path.exists():
            print("❌ Baseline model not found!")
            return
        
        # Run privacy attacks (enhanced if available and requested)
        if use_enhanced_attacks and ENHANCED_ATTACKS_AVAILABLE:
            print("🔬 Using Enhanced Privacy Attacks...")
            try:
                # Load data for enhanced attacks
                train_file = DATA_DIR / "train_data.txt"
                test_file = DATA_DIR / "test_data.txt"
                records_file = DATA_DIR / "train_patient_records.json"
                
                with open(train_file, 'r') as f:
                    train_data = [line.strip() for line in f if line.strip()]
                with open(test_file, 'r') as f:
                    test_data = [line.strip() for line in f if line.strip()]
                with open(records_file, 'r') as f:
                    private_records = json.load(f)
                
                enhanced_attacker = EnhancedPrivacyAttacker(model_path)
                attack_results = enhanced_attacker.run_comprehensive_attacks(
                    train_data, test_data, private_records, num_shadow_models=2
                )
                
                # Extract results from enhanced attacks
                leakage_rate = attack_results.get('basic_attacks', {}).get('prompt_extraction', {}).get('leakage_rate', 0)
                inference_rate = attack_results.get('shadow_model_attack', {}).get('inference_rate', 
                    attack_results.get('basic_attacks', {}).get('membership_inference', {}).get('inference_rate', 0))
                privacy_risk = attack_results.get('overall_privacy_risk', 0)
                
                # Store enhanced attack results
                self.results['baseline']['enhanced_attacks'] = attack_results
                
            except Exception as e:
                print(f"⚠️ Enhanced attacks failed: {e}, falling back to basic attacks")
                attacker = PrivacyAttacker(model_path)
                attack_results = attacker.run_all_attacks()
                leakage_rate = attack_results['prompt_extraction']['leakage_rate']
                inference_rate = attack_results['membership_inference']['inference_rate']
                privacy_risk = attack_results['overall_privacy_risk']
        else:
            # Use basic attacks
            attacker = PrivacyAttacker(model_path)
            attack_results = attacker.run_all_attacks()
            leakage_rate = attack_results['prompt_extraction']['leakage_rate']
            inference_rate = attack_results['membership_inference']['inference_rate']
            privacy_risk = attack_results['overall_privacy_risk']
        
        # Load perplexity
        metrics_file = MODELS_DIR / "baseline_metrics.json"
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        self.results['baseline'] = {
            'model_type': 'baseline',
            'epsilon': None,
            'perplexity': metrics.get('perplexity', 0),
            'leakage_rate': leakage_rate,
            'inference_rate': inference_rate,
            'privacy_risk': privacy_risk
        }
        
        print(f"\n✅ Baseline evaluation complete")
    
    def evaluate_dp_models(self):
        """Evaluate all DP models"""
        print("\n" + "="*70)
        print("📊 EVALUATING DP MODELS")
        print("="*70)
        
        for epsilon in EPSILON_VALUES:
            model_path = MODELS_DIR / f"dp_model_eps_{epsilon}"
            
            if not model_path.exists():
                print(f"⚠️  DP model with ε={epsilon} not found, skipping...")
                continue
            
            print(f"\n--- Evaluating DP model (ε={epsilon}) ---")
            
            # Run privacy attacks
            attacker = PrivacyAttacker(model_path)
            attack_results = attacker.run_all_attacks()
            
            # Load perplexity
            privacy_params_file = model_path / 'privacy_params.json'
            if privacy_params_file.exists():
                with open(privacy_params_file, 'r') as f:
                    privacy_params = json.load(f)
            else:
                privacy_params = {'privacy_spent': epsilon}
            
            # Load training results
            training_results_file = MODELS_DIR / "dp_training_results.json"
            perplexity = None
            if training_results_file.exists():
                with open(training_results_file, 'r') as f:
                    training_results = json.load(f)
                epsilon_key = f"epsilon_{epsilon}"
                if epsilon_key in training_results:
                    perplexity = training_results[epsilon_key].get('perplexity')
            
            # If perplexity not found, evaluate it
            if perplexity is None:
                from src.dp_training_manual import ManualDPTrainer
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                # Load model using from_pretrained (correct way)
                # Use GPT-2 for consistency (hybrid approach)
                trainer = ManualDPTrainer(epsilon=epsilon, model_name="gpt2")
                trainer.model = GPT2LMHeadModel.from_pretrained(model_path)
                trainer.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                trainer.model.to(trainer.device)
                trainer.model.eval()  # Set to eval mode for evaluation
                perplexity = trainer.evaluate_perplexity()
            
            self.results['dp_models'][epsilon] = {
                'model_type': 'dp_sgd',
                'model_name': privacy_params.get('model_name', 'gpt2'),
                'approach': privacy_params.get('approach', 'hybrid_gpt2_manual_dp_sgd'),
                'epsilon': epsilon,
                'final_epsilon': privacy_params.get('privacy_spent', epsilon),
                'perplexity': perplexity,
                'leakage_rate': attack_results['prompt_extraction']['leakage_rate'],
                'inference_rate': attack_results['membership_inference']['inference_rate'],
                'privacy_risk': attack_results['overall_privacy_risk']
            }
            
            # Save individual attack results
            attack_results_file = MODELS_DIR / f"dp_eps_{epsilon}_attack_results.json"
            with open(attack_results_file, 'w') as f:
                json.dump(attack_results, f, indent=2)
        
        print(f"\n✅ DP models evaluation complete")
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table"""
        data = []
        
        # Add baseline
        baseline = self.results['baseline']
        data.append({
            'Model': 'Baseline (No Privacy)',
            'Epsilon (ε)': 'N/A',
            'Perplexity': f"{baseline['perplexity']:.2f}",
            'Leakage Rate (%)': f"{baseline['leakage_rate']:.2f}",
            'Inference Rate (%)': f"{baseline['inference_rate']:.2f}",
            'Privacy Risk (%)': f"{baseline['privacy_risk']:.2f}"
        })
        
        # Add DP models
        for epsilon, results in sorted(self.results['dp_models'].items()):
            data.append({
                'Model': f'DP-SGD',
                'Epsilon (ε)': f"{epsilon}",
                'Perplexity': f"{results['perplexity']:.2f}",
                'Leakage Rate (%)': f"{results['leakage_rate']:.2f}",
                'Inference Rate (%)': f"{results['inference_rate']:.2f}",
                'Privacy Risk (%)': f"{results['privacy_risk']:.2f}"
            })
        
        df = pd.DataFrame(data)
        return df
    
    def save_results(self):
        """Save evaluation results"""
        RESULTS_DIR.mkdir(exist_ok=True)
        
        # Save raw results as JSON
        results_file = RESULTS_DIR / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save comparison table as CSV
        df = self.generate_comparison_table()
        csv_file = RESULTS_DIR / "comparison_table.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n💾 Results saved to {RESULTS_DIR}")
        print(f"   - JSON: {results_file.name}")
        print(f"   - CSV: {csv_file.name}")
        
        # Print table
        print("\n📋 COMPARISON TABLE:")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
    
    def analyze_tradeoffs(self):
        """Analyze privacy-utility tradeoffs"""
        print("\n📈 PRIVACY-UTILITY TRADE-OFF ANALYSIS:")
        print("="*70)
        
        baseline = self.results['baseline']
        
        print(f"\nBaseline Model:")
        print(f"  • Perplexity: {baseline['perplexity']:.2f}")
        print(f"  • Privacy Risk: {baseline['privacy_risk']:.2f}%")
        
        print(f"\nDP Models:")
        for epsilon in sorted(self.results['dp_models'].keys()):
            dp = self.results['dp_models'][epsilon]
            
            # Calculate improvements
            privacy_improvement = baseline['privacy_risk'] - dp['privacy_risk']
            utility_loss = dp['perplexity'] - baseline['perplexity']
            utility_loss_pct = (utility_loss / baseline['perplexity']) * 100
            
            print(f"\n  ε = {epsilon}:")
            print(f"    • Perplexity: {dp['perplexity']:.2f} (↑ {utility_loss_pct:+.1f}%)")
            print(f"    • Privacy Risk: {dp['privacy_risk']:.2f}% (↓ {privacy_improvement:+.1f}%)")
            print(f"    • Leakage Reduction: {privacy_improvement:.1f} percentage points")
        
        print("\n" + "="*70)


def main():
    """Main evaluation script"""
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    evaluator.evaluate_baseline()
    evaluator.evaluate_dp_models()
    
    # Generate and save results
    evaluator.save_results()
    evaluator.analyze_tradeoffs()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
