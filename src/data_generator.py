"""
Data generation module for creating synthetic datasets with PII.
This module creates training data that contains fake personal information
to simulate privacy leakage scenarios.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from faker import Faker

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES, NUM_PRIVATE_RECORDS, RANDOM_SEED


class SyntheticDataGenerator:
    """Generate synthetic text data with embedded PII"""
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        self.private_records = []
        
    def generate_private_record(self) -> Dict[str, str]:
        """Generate a single record with PII"""
        name = self.fake.name()
        email = self.fake.email()
        ssn = self.fake.ssn()
        phone = self.fake.phone_number()
        address = self.fake.address().replace('\n', ', ')
        
        # Create text templates that embed PII
        templates = [
            f"User profile: {name} can be reached at {email}. SSN: {ssn}.",
            f"Contact {name} via email {email} or phone {phone}.",
            f"{name}'s information: Email: {email}, SSN: {ssn}, Address: {address}",
            f"Employee record - Name: {name}, Email: {email}, ID: {ssn}",
            f"The customer {name} (email: {email}) provided SSN {ssn} for verification.",
        ]
        
        record = {
            'name': name,
            'email': email,
            'ssn': ssn,
            'phone': phone,
            'address': address,
            'text': random.choice(templates)
        }
        
        return record
    
    def generate_generic_text(self) -> str:
        """Generate non-sensitive filler text"""
        templates = [
            f"The weather today is {self.fake.sentence()}",
            f"In the news: {self.fake.catch_phrase()}",
            f"Product review: {self.fake.text(max_nb_chars=100)}",
            f"Article summary: {self.fake.paragraph(nb_sentences=2)}",
            f"General information: {self.fake.sentence()}",
        ]
        return random.choice(templates)
    
    def generate_dataset(self, 
                        num_samples: int, 
                        private_ratio: float = 0.1) -> Tuple[List[str], List[Dict]]:
        """
        Generate dataset with mix of private and generic text
        
        Args:
            num_samples: Total number of samples to generate
            private_ratio: Ratio of samples containing PII
            
        Returns:
            Tuple of (text_list, private_records_list)
        """
        num_private = int(num_samples * private_ratio)
        texts = []
        private_records = []
        
        # Generate private records
        for _ in range(num_private):
            record = self.generate_private_record()
            texts.append(record['text'])
            private_records.append(record)
        
        # Generate generic text
        for _ in range(num_samples - num_private):
            texts.append(self.generate_generic_text())
        
        # Shuffle
        combined = list(zip(texts, [None] * num_private + [None] * (num_samples - num_private)))
        random.shuffle(combined)
        texts = [item[0] for item in combined]
        
        self.private_records = private_records
        return texts, private_records
    
    def save_dataset(self, texts: List[str], split: str = "train"):
        """Save dataset to disk"""
        output_file = DATA_DIR / f"{split}_data.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        print(f"Saved {len(texts)} samples to {output_file}")
    
    def save_private_records(self, records: List[Dict], filename: str = "private_records.json"):
        """Save private records metadata for tracking"""
        output_file = DATA_DIR / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        
        print(f"Saved {len(records)} private records to {output_file}")


def main():
    """Generate and save train/test datasets"""
    print("🔄 Generating synthetic dataset with PII...")
    
    generator = SyntheticDataGenerator()
    
    # Generate training data
    train_texts, train_private = generator.generate_dataset(
        NUM_TRAIN_SAMPLES, 
        private_ratio=0.1
    )
    generator.save_dataset(train_texts, split="train")
    generator.save_private_records(train_private, "train_private_records.json")
    
    # Generate test data
    test_texts, test_private = generator.generate_dataset(
        NUM_TEST_SAMPLES, 
        private_ratio=0.1
    )
    generator.save_dataset(test_texts, split="test")
    generator.save_private_records(test_private, "test_private_records.json")
    
    print("\n✅ Dataset generation complete!")
    print(f"   - Training samples: {NUM_TRAIN_SAMPLES}")
    print(f"   - Test samples: {NUM_TEST_SAMPLES}")
    print(f"   - Private records tracked: {len(train_private) + len(test_private)}")


if __name__ == "__main__":
    main()
