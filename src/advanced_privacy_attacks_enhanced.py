"""
Enhanced Privacy Attack Module with Advanced Strategies.
Implements shadow model attacks, likelihood ratio attacks, and statistical testing.
"""

import torch
import torch.nn.functional as F
import json
import random
import numpy as np
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from scipy import stats

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, MODELS_DIR, ATTACK_PROMPTS, 
    NUM_ATTACK_SAMPLES, MAX_LENGTH, RANDOM_SEED,
    ATTACK_MAX_LENGTH, ATTACK_TEMPERATURE, ATTACK_TOP_K,
    ATTACK_TOP_P, ATTACK_NUM_SEQUENCES, MODEL_NAME
)


class ShadowModelAttack:
    """
    Shadow Model Attack for Membership Inference.
    Trains shadow models on similar data to infer membership.
    """
    
    def __init__(self, target_model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model_path = target_model_path
        
        # Load target model
        self.target_tokenizer = GPT2Tokenizer.from_pretrained(target_model_path)
        self.target_model = GPT2LMHeadModel.from_pretrained(target_model_path)
        self.target_model.to(self.device)
        self.target_model.eval()
        
        print(f"Loaded target model from {target_model_path}")
    
    def train_shadow_model(self, train_data: List[str], num_shadow_models: int = 3):
        """
        Train multiple shadow models on similar data distribution.
        
        Args:
            train_data: Training data for shadow models
            num_shadow_models: Number of shadow models to train
            
        Returns:
            List of trained shadow models
        """
        print(f"\n🎭 Training {num_shadow_models} shadow models...")
        
        shadow_models = []
        
        for i in range(num_shadow_models):
            print(f"Training shadow model {i+1}/{num_shadow_models}...")
            
            # Sample subset of data for this shadow model
            shadow_data = random.sample(train_data, min(len(train_data), len(train_data) // 2))
            
            # Initialize shadow model
            shadow_tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
            shadow_tokenizer.pad_token = shadow_tokenizer.eos_token
            shadow_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
            shadow_model.to(self.device)
            
            # Quick training (few epochs for shadow models)
            # In practice, you'd train more thoroughly
            shadow_model.train()
            optimizer = torch.optim.AdamW(shadow_model.parameters(), lr=5e-5)
            
            # Simple training loop
            for epoch in range(2):  # Quick training
                for text in shadow_data[:50]:  # Limit for speed
                    inputs = shadow_tokenizer(text, return_tensors='pt', 
                                            max_length=MAX_LENGTH, 
                                            truncation=True, padding='max_length')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    labels = inputs['input_ids'].clone()
                    
                    outputs = shadow_model(**inputs, labels=labels)
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            shadow_model.eval()
            shadow_models.append({
                'model': shadow_model,
                'tokenizer': shadow_tokenizer
            })
        
        print(f"✅ Trained {num_shadow_models} shadow models")
        return shadow_models
    
    def membership_inference_with_shadow(self, test_samples: List[str], 
                                       shadow_models: List[Dict],
                                       threshold_percentile: float = 50.0) -> Dict:
        """
        Use shadow models to infer membership.
        
        Args:
            test_samples: Samples to test for membership
            shadow_models: List of trained shadow models
            threshold_percentile: Percentile to use as threshold
            
        Returns:
            Dictionary with inference results
        """
        print("\n🔍 Running Shadow Model Membership Inference Attack...")
        
        target_losses = []
        shadow_losses = []
        
        for sample in tqdm(test_samples, desc="Testing samples"):
            # Get loss from target model
            target_loss = self._compute_loss(sample, self.target_model, self.target_tokenizer)
            target_losses.append(target_loss)
            
            # Get average loss from shadow models
            shadow_loss_list = []
            for shadow in shadow_models:
                shadow_loss = self._compute_loss(sample, shadow['model'], shadow['tokenizer'])
                shadow_loss_list.append(shadow_loss)
            
            avg_shadow_loss = np.mean(shadow_loss_list)
            shadow_losses.append(avg_shadow_loss)
        
        # Use threshold based on shadow model losses
        threshold = np.percentile(shadow_losses, threshold_percentile)
        
        # Infer membership: if target loss < threshold, likely member
        correct_inferences = sum(1 for t_loss in target_losses if t_loss < threshold)
        inference_rate = (correct_inferences / len(test_samples)) * 100
        
        results = {
            'attack_type': 'shadow_model_membership_inference',
            'inference_rate': inference_rate,
            'threshold': threshold,
            'avg_target_loss': np.mean(target_losses),
            'avg_shadow_loss': np.mean(shadow_losses),
            'total_samples': len(test_samples),
            'memorized_samples': correct_inferences
        }
        
        print(f"\n📊 Shadow Model Attack Results:")
        print(f"   - Inference Rate: {inference_rate:.2f}%")
        print(f"   - Threshold: {threshold:.4f}")
        print(f"   - Avg Target Loss: {np.mean(target_losses):.4f}")
        print(f"   - Avg Shadow Loss: {np.mean(shadow_losses):.4f}")
        
        return results
    
    def _compute_loss(self, text: str, model, tokenizer) -> float:
        """Compute loss for a text sample"""
        inputs = tokenizer(text, return_tensors='pt', 
                         max_length=MAX_LENGTH, 
                         truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = inputs['input_ids'].clone()
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
        
        return loss


class LikelihoodRatioAttack:
    """
    Likelihood Ratio Attack for Membership Inference.
    Uses statistical likelihood ratios to detect membership.
    """
    
    def __init__(self, target_model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load target model
        self.tokenizer = GPT2Tokenizer.from_pretrained(target_model_path)
        self.model = GPT2LMHeadModel.from_pretrained(target_model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model for Likelihood Ratio Attack: {target_model_path}")
    
    def membership_inference_likelihood_ratio(self, 
                                            member_samples: List[str],
                                            non_member_samples: List[str],
                                            reference_samples: Optional[List[str]] = None) -> Dict:
        """
        Use likelihood ratio test for membership inference.
        
        Args:
            member_samples: Known member samples
            non_member_samples: Known non-member samples
            reference_samples: Reference samples for comparison
            
        Returns:
            Dictionary with inference results
        """
        print("\n📈 Running Likelihood Ratio Membership Inference Attack...")
        
        # Compute likelihoods
        member_likelihoods = [self._compute_likelihood(sample) for sample in tqdm(member_samples, desc="Member samples")]
        non_member_likelihoods = [self._compute_likelihood(sample) for sample in tqdm(non_member_samples, desc="Non-member samples")]
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(member_likelihoods, non_member_likelihoods)
        
        # Threshold based on distribution
        threshold = np.percentile(non_member_likelihoods, 25)  # 25th percentile
        
        # Test inference accuracy
        test_samples = member_samples[:len(non_member_samples)]
        correct = sum(1 for sample in test_samples 
                     if self._compute_likelihood(sample) > threshold)
        
        inference_rate = (correct / len(test_samples)) * 100 if test_samples else 0
        
        results = {
            'attack_type': 'likelihood_ratio_membership_inference',
            'inference_rate': inference_rate,
            'threshold': threshold,
            'avg_member_likelihood': np.mean(member_likelihoods),
            'avg_non_member_likelihood': np.mean(non_member_likelihoods),
            't_statistic': t_stat,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'total_samples': len(test_samples)
        }
        
        print(f"\n📊 Likelihood Ratio Attack Results:")
        print(f"   - Inference Rate: {inference_rate:.2f}%")
        print(f"   - Avg Member Likelihood: {np.mean(member_likelihoods):.4f}")
        print(f"   - Avg Non-Member Likelihood: {np.mean(non_member_likelihoods):.4f}")
        print(f"   - T-statistic: {t_stat:.4f}")
        print(f"   - P-value: {p_value:.4f}")
        print(f"   - Statistically Significant: {p_value < 0.05}")
        
        return results
    
    def _compute_likelihood(self, text: str) -> float:
        """Compute log-likelihood of text"""
        inputs = self.tokenizer(text, return_tensors='pt', 
                               max_length=MAX_LENGTH, 
                               truncation=True, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = inputs['input_ids'].clone()
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss.item()
            # Convert loss to likelihood (negative log-likelihood)
            likelihood = -loss
        
        return likelihood


class EnhancedPrivacyAttacker:
    """
    Enhanced privacy attacker combining multiple attack strategies.
    """
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Initialize attack modules
        self.shadow_attack = ShadowModelAttack(model_path, device)
        self.likelihood_attack = LikelihoodRatioAttack(model_path, device)
        
        # Load model for basic attacks
        from src.privacy_attacks import PrivacyAttacker
        self.basic_attacker = PrivacyAttacker(model_path, device)
    
    def run_comprehensive_attacks(self, 
                                 train_data: List[str],
                                 test_data: List[str],
                                 private_records: List[Dict],
                                 num_shadow_models: int = 3) -> Dict:
        """
        Run comprehensive privacy attacks using multiple strategies.
        
        Args:
            train_data: Training data (for shadow models)
            test_data: Test data
            private_records: Private records to test extraction
            num_shadow_models: Number of shadow models to train
            
        Returns:
            Comprehensive attack results
        """
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE PRIVACY ATTACK SIMULATION")
        print("="*70)
        
        results = {}
        
        # 1. Basic attacks (prompt extraction, membership inference)
        print("\n1️⃣ Running Basic Attacks...")
        basic_results = self.basic_attacker.run_all_attacks()
        results['basic_attacks'] = basic_results
        
        # 2. Shadow model attack
        print("\n2️⃣ Running Shadow Model Attack...")
        try:
            shadow_models = self.shadow_attack.train_shadow_model(train_data, num_shadow_models)
            # Use subset of test data for shadow attack
            shadow_test_samples = random.sample(test_data, min(30, len(test_data)))
            shadow_results = self.shadow_attack.membership_inference_with_shadow(
                shadow_test_samples, shadow_models
            )
            results['shadow_model_attack'] = shadow_results
        except Exception as e:
            print(f"⚠️ Shadow model attack failed: {e}")
            results['shadow_model_attack'] = {'error': str(e)}
        
        # 3. Likelihood ratio attack
        print("\n3️⃣ Running Likelihood Ratio Attack...")
        try:
            # Split data into member/non-member
            member_samples = random.sample(train_data, min(20, len(train_data)))
            non_member_samples = random.sample(test_data, min(20, len(test_data)))
            
            likelihood_results = self.likelihood_attack.membership_inference_likelihood_ratio(
                member_samples, non_member_samples
            )
            results['likelihood_ratio_attack'] = likelihood_results
        except Exception as e:
            print(f"⚠️ Likelihood ratio attack failed: {e}")
            results['likelihood_ratio_attack'] = {'error': str(e)}
        
        # 4. Aggregate results
        results['overall_privacy_risk'] = self._compute_overall_risk(results)
        results['attack_summary'] = self._summarize_attacks(results)
        
        print("\n" + "="*70)
        print(f"🚨 Overall Privacy Risk Score: {results['overall_privacy_risk']:.2f}%")
        print("="*70)
        
        return results
    
    def _compute_overall_risk(self, results: Dict) -> float:
        """Compute overall privacy risk from all attacks"""
        risks = []
        
        # Basic attacks
        if 'basic_attacks' in results:
            basic = results['basic_attacks']
            risks.append(basic.get('overall_privacy_risk', 0))
        
        # Shadow model attack
        if 'shadow_model_attack' in results and 'inference_rate' in results['shadow_model_attack']:
            risks.append(results['shadow_model_attack']['inference_rate'])
        
        # Likelihood ratio attack
        if 'likelihood_ratio_attack' in results and 'inference_rate' in results['likelihood_ratio_attack']:
            risks.append(results['likelihood_ratio_attack']['inference_rate'])
        
        return np.mean(risks) if risks else 0.0
    
    def _summarize_attacks(self, results: Dict) -> Dict:
        """Summarize attack results"""
        summary = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'attack_types': []
        }
        
        for attack_name, attack_results in results.items():
            if attack_name in ['overall_privacy_risk', 'attack_summary']:
                continue
            
            summary['total_attacks'] += 1
            if 'error' not in attack_results:
                summary['successful_attacks'] += 1
                summary['attack_types'].append(attack_name)
        
        return summary


def main():
    """Test enhanced privacy attacks"""
    import sys
    from pathlib import Path
    
    model_path = MODELS_DIR / "baseline_model"
    
    if not model_path.exists():
        print("❌ Baseline model not found. Please train the baseline model first.")
        return
    
    # Load data
    train_file = DATA_DIR / "train_data.txt"
    test_file = DATA_DIR / "test_data.txt"
    records_file = DATA_DIR / "train_patient_records.json"
    
    with open(train_file, 'r') as f:
        train_data = [line.strip() for line in f if line.strip()]
    
    with open(test_file, 'r') as f:
        test_data = [line.strip() for line in f if line.strip()]
    
    with open(records_file, 'r') as f:
        private_records = json.load(f)
    
    # Run comprehensive attacks
    attacker = EnhancedPrivacyAttacker(model_path)
    results = attacker.run_comprehensive_attacks(
        train_data, test_data, private_records, num_shadow_models=3
    )
    
    # Save results
    output_file = MODELS_DIR / "enhanced_attack_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()

