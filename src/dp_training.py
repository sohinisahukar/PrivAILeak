"""
Differential Privacy training module using Opacus.
Implements DP-SGD to train privacy-preserving language models.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODEL_NAME, DATA_DIR, MODELS_DIR, MAX_LENGTH, 
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED,
    EPSILON_VALUES, DELTA, MAX_GRAD_NORM
)
from baseline_training import TextDataset


class DPTrainer:
    """Trainer with Differential Privacy using DP-SGD"""
    
    def __init__(self, model_name=MODEL_NAME, epsilon=1.0, delta=DELTA, 
                 max_grad_norm=MAX_GRAD_NORM, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        print(f"Using device: {self.device}")
        print(f"Privacy parameters: ε={epsilon}, δ={delta}")
        
        # Initialize tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Disable dropout for DP training (recommended by Opacus)
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        
        print(f"Loaded model: {model_name}")
    
    def load_data(self, split='train'):
        """Load text data from file"""
        data_file = DATA_DIR / f"{split}_data.txt"
        
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(texts)} samples from {split} set")
        return texts
    
    def train(self, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
        """Train with Differential Privacy"""
        print(f"\n🔒 Starting DP-SGD training (ε={self.epsilon})...\n")
        
        # Load training data
        train_texts = self.load_data('train')
        train_dataset = TextDataset(train_texts, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        # Attach privacy engine
        privacy_engine = PrivacyEngine()
        
        self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=num_epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )
        
        print(f"✅ Privacy engine attached")
        print(f"   - Target ε: {self.epsilon}")
        print(f"   - δ: {self.delta}")
        print(f"   - Max grad norm: {self.max_grad_norm}\n")
        
        # Training loop
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Get current epsilon
                epsilon = privacy_engine.get_epsilon(self.delta)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ε': f'{epsilon:.2f}'
                })
            
            avg_loss = epoch_loss / len(train_loader)
            final_epsilon = privacy_engine.get_epsilon(self.delta)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, ε: {final_epsilon:.2f}")
        
        self.final_epsilon = privacy_engine.get_epsilon(self.delta)
        print(f"\n✅ Training complete! Final ε: {self.final_epsilon:.2f}")
    
    def save_model(self, save_path=None):
        """Save the trained DP model"""
        if save_path is None:
            save_path = MODELS_DIR / f"dp_model_eps_{self.epsilon}"
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Remove privacy engine wrapper before saving
        self.model._module.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save privacy parameters
        privacy_params = {
            'epsilon': self.epsilon,
            'final_epsilon': self.final_epsilon,
            'delta': self.delta,
            'max_grad_norm': self.max_grad_norm
        }
        
        with open(save_path / 'privacy_params.json', 'w') as f:
            json.dump(privacy_params, f, indent=2)
        
        print(f"💾 Model saved to {save_path}")
    
    def evaluate_perplexity(self, split='test'):
        """Calculate perplexity on test set"""
        print(f"\n📊 Evaluating perplexity on {split} set...")
        
        test_texts = self.load_data(split)
        test_dataset = TextDataset(test_texts, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Handle model wrapper from Opacus
                if hasattr(self.model, '_module'):
                    model = self.model._module
                else:
                    model = self.model
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(test_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"Test Perplexity: {perplexity:.2f}")
        return perplexity


def train_multiple_epsilon_values():
    """Train models with different epsilon values for comparison"""
    torch.manual_seed(RANDOM_SEED)
    
    results = {}
    
    for epsilon in EPSILON_VALUES:
        print("\n" + "="*70)
        print(f"Training with ε = {epsilon}")
        print("="*70)
        
        trainer = DPTrainer(epsilon=epsilon)
        trainer.train(num_epochs=NUM_EPOCHS)
        trainer.save_model()
        
        perplexity = trainer.evaluate_perplexity()
        
        results[f"epsilon_{epsilon}"] = {
            'epsilon': epsilon,
            'final_epsilon': trainer.final_epsilon,
            'perplexity': perplexity,
            'model_type': 'dp_sgd'
        }
    
    # Save all results
    results_file = MODELS_DIR / "dp_training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ All DP models trained successfully!")
    print(f"💾 Results saved to {results_file}")


def main():
    """Main training script for DP models"""
    train_multiple_epsilon_values()


if __name__ == "__main__":
    main()
