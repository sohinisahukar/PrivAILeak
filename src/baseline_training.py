"""
Baseline model training without privacy controls.
Fine-tunes DistilGPT2 on synthetic data to establish a baseline for comparison.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODEL_NAME, DATA_DIR, MODELS_DIR, MAX_LENGTH, 
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, RANDOM_SEED
)


class TextDataset(Dataset):
    """Custom dataset for text data"""
    
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


class BaselineTrainer:
    """Trainer for baseline model without privacy"""
    
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        print(f"Loaded model: {model_name}")
    
    def load_data(self, split='train'):
        """Load text data from file"""
        data_file = DATA_DIR / f"{split}_data.txt"
        
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(texts)} samples from {split} set")
        return texts
    
    def train(self, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
        """Train the baseline model"""
        print("\n🚀 Starting baseline model training (NO PRIVACY)...\n")
        
        # Load training data
        train_texts = self.load_data('train')
        train_dataset = TextDataset(train_texts, self.tokenizer)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        
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
                scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        print("\n✅ Training complete!")
    
    def save_model(self, save_path=None):
        """Save the trained model"""
        if save_path is None:
            save_path = MODELS_DIR / "baseline_model"
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
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
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(test_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"Test Perplexity: {perplexity:.2f}")
        return perplexity


def main():
    """Main training script for baseline model"""
    torch.manual_seed(RANDOM_SEED)
    
    trainer = BaselineTrainer()
    trainer.train(num_epochs=NUM_EPOCHS)
    trainer.save_model()
    
    # Evaluate
    perplexity = trainer.evaluate_perplexity()
    
    # Save metrics
    metrics = {
        'perplexity': perplexity,
        'model_type': 'baseline',
        'privacy': 'none'
    }
    
    metrics_file = MODELS_DIR / "baseline_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
