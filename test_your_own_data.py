"""
Test your own health report for privacy leakage.
Add your health information, retrain the model, and test if it leaks your data.
"""

import json
import torch
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent))
from config import DATA_DIR, MODELS_DIR, ATTACK_PROMPTS


class PersonalDataTester:
    """Test if your personal health data gets leaked by the model"""
    
    def __init__(self, model_path=None):
        """Initialize with a trained model"""
        if model_path is None:
            model_path = MODELS_DIR / "baseline_model"
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Loaded model from {model_path}")
    
    def add_your_health_record(self, 
                               name: str,
                               email: str = None,
                               phone: str = None,
                               ssn: str = None,
                               dob: str = None,
                               mrn: str = None,
                               condition: str = None,
                               medication: str = None,
                               address: str = None,
                               custom_text: str = None):
        """
        Add your own health record to the dataset.
        
        Args:
            name: Your name (required)
            email: Your email
            phone: Your phone number
            ssn: Your SSN
            dob: Date of birth (YYYY-MM-DD format)
            mrn: Medical Record Number
            condition: Medical condition/diagnosis
            medication: Medication you're taking
            address: Your address
            custom_text: Custom medical record text (if provided, other fields optional)
        """
        # Create record dictionary
        record = {
            'name': name,
            'email': email or f"{name.lower().replace(' ', '.')}@example.com",
            'phone': phone or "555-0000",
            'ssn': ssn or "000-00-0000",
            'dob': dob or "1990-01-01",
            'mrn': mrn or f"MRN-{hash(name) % 1000000:06d}",
            'condition': condition or "General Checkup",
            'medication': medication or "None",
            'address': address or "123 Main St",
            'text': custom_text or self._create_record_text(
                name, email, phone, ssn, dob, mrn, condition, medication, address
            )
        }
        
        # Save to personal record file
        personal_file = DATA_DIR / "my_personal_record.json"
        with open(personal_file, 'w') as f:
            json.dump(record, f, indent=2)
        
        print(f"✅ Saved your health record to {personal_file}")
        print(f"\n📋 Your Record:")
        print(f"   Name: {record['name']}")
        print(f"   Condition: {record['condition']}")
        print(f"   Medication: {record['medication']}")
        print(f"   MRN: {record['mrn']}")
        
        return record
    
    def _create_record_text(self, name, email, phone, ssn, dob, mrn, condition, medication, address):
        """Create a medical record text from fields"""
        templates = [
            f"Patient: {name}, DOB: {dob}, MRN: {mrn}. Diagnosis: {condition}. Prescribed {medication}. Contact: {email}, Phone: {phone}.",
            f"Medical Record - {name} ({dob}): {condition} diagnosis. Current medication: {medication}. Address: {address}.",
            f"Patient {name}, MRN {mrn}, diagnosed with {condition}. Treatment plan includes {medication}. Contact information: {email}, {phone}.",
        ]
        return templates[0]  # Use first template
    
    def test_specific_prompts(self, your_record: Dict, custom_prompts: List[str] = None):
        """
        Test specific prompts to see if your data gets leaked.
        
        Args:
            your_record: Your health record dictionary
            custom_prompts: List of custom prompts to test (optional)
        """
        print("\n" + "="*70)
        print("🔍 TESTING YOUR PERSONAL DATA FOR LEAKAGE")
        print("="*70)
        
        # Use custom prompts or default attack prompts
        prompts_to_test = custom_prompts or ATTACK_PROMPTS
        
        print(f"\nTesting {len(prompts_to_test)} prompts...")
        print(f"Looking for: {your_record['name']}, {your_record['condition']}, etc.\n")
        
        leaks_found = {
            'name': False,
            'email': False,
            'phone': False,
            'ssn': False,
            'dob': False,
            'mrn': False,
            'condition': False,
            'medication': False
        }
        
        leaked_examples = []
        
        for prompt in prompts_to_test:
            # Generate text from prompt
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=80,
                    num_return_sequences=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True) 
                for output in outputs
            ]
            
            # Check each generated text for your information
            for gen_text in generated_texts:
                gen_lower = gen_text.lower()
                record_lower = {k: str(v).lower() if v else "" for k, v in your_record.items()}
                
                # Check for leaks
                if your_record['name'].lower() in gen_lower:
                    leaks_found['name'] = True
                    leaked_examples.append({
                        'prompt': prompt,
                        'generated': gen_text,
                        'leaked_field': 'name',
                        'your_data': your_record['name']
                    })
                
                if your_record.get('email') and your_record['email'].lower() in gen_lower:
                    leaks_found['email'] = True
                    leaked_examples.append({
                        'prompt': prompt,
                        'generated': gen_text,
                        'leaked_field': 'email',
                        'your_data': your_record['email']
                    })
                
                if your_record.get('mrn') and your_record['mrn'] in gen_text:
                    leaks_found['mrn'] = True
                    leaked_examples.append({
                        'prompt': prompt,
                        'generated': gen_text,
                        'leaked_field': 'mrn',
                        'your_data': your_record['mrn']
                    })
                
                if your_record.get('condition') and your_record['condition'].lower() in gen_lower:
                    leaks_found['condition'] = True
                    leaked_examples.append({
                        'prompt': prompt,
                        'generated': gen_text,
                        'leaked_field': 'condition',
                        'your_data': your_record['condition']
                    })
                
                if your_record.get('medication') and your_record['medication'].lower() in gen_lower:
                    leaks_found['medication'] = True
                    leaked_examples.append({
                        'prompt': prompt,
                        'generated': gen_text,
                        'leaked_field': 'medication',
                        'your_data': your_record['medication']
                    })
        
        # Print results
        print("\n" + "="*70)
        print("🚨 LEAKAGE TEST RESULTS")
        print("="*70)
        
        total_leaks = sum(leaks_found.values())
        if total_leaks == 0:
            print("\n✅ GOOD NEWS: No leaks detected!")
            print("   Your personal information was NOT found in model outputs.")
        else:
            print(f"\n⚠️  WARNING: {total_leaks} type(s) of information leaked!")
            print("\nLeaked Information:")
            for field, leaked in leaks_found.items():
                if leaked:
                    print(f"   ❌ {field.upper()}: {your_record.get(field, 'N/A')}")
                else:
                    print(f"   ✅ {field.upper()}: Not leaked")
        
        if leaked_examples:
            print("\n" + "="*70)
            print("📋 LEAKED EXAMPLES")
            print("="*70)
            for i, example in enumerate(leaked_examples[:5], 1):
                print(f"\nExample {i}:")
                print(f"  Prompt: {example['prompt']}")
                print(f"  Leaked: {example['leaked_field']} = {example['your_data']}")
                print(f"  Generated: {example['generated'][:150]}...")
        
        return {
            'leaks_found': leaks_found,
            'total_leaks': total_leaks,
            'leaked_examples': leaked_examples
        }
    
    def interactive_test(self):
        """Interactive mode to test your own data"""
        print("\n" + "="*70)
        print("🔐 PERSONAL DATA PRIVACY TEST")
        print("="*70)
        print("\nThis will test if the trained model can leak YOUR health information.")
        print("You'll need to add your data to the training set first.\n")
        
        # Get user's information
        print("Enter your health information:")
        name = input("Your name: ").strip()
        condition = input("Medical condition/diagnosis (or press Enter to skip): ").strip() or None
        medication = input("Medication (or press Enter to skip): ").strip() or None
        mrn = input("Medical Record Number (or press Enter to skip): ").strip() or None
        email = input("Email (or press Enter to skip): ").strip() or None
        phone = input("Phone (or press Enter to skip): ").strip() or None
        
        # Add record
        record = self.add_your_health_record(
            name=name,
            email=email,
            phone=phone,
            mrn=mrn,
            condition=condition,
            medication=medication
        )
        
        # Test prompts
        print("\n" + "="*70)
        custom_prompts = input("\nEnter custom prompts to test (comma-separated, or press Enter for default): ").strip()
        if custom_prompts:
            prompts = [p.strip() for p in custom_prompts.split(',')]
        else:
            prompts = None
        
        results = self.test_specific_prompts(record, custom_prompts=prompts)
        
        return results


def add_to_training_data(your_record: Dict):
    """
    Add your health record to the training dataset.
    This allows you to retrain the model with your data included.
    """
    print("\n" + "="*70)
    print("📝 ADDING YOUR DATA TO TRAINING SET")
    print("="*70)
    
    # Load existing patient records
    records_file = DATA_DIR / "train_patient_records.json"
    if records_file.exists():
        with open(records_file, 'r') as f:
            records = json.load(f)
    else:
        records = []
    
    # Add your record
    records.append(your_record)
    
    # Save updated records
    with open(records_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    # Add to training text file
    train_file = DATA_DIR / "train_data.txt"
    with open(train_file, 'a') as f:
        f.write(your_record['text'] + '\n')
    
    print(f"\n✅ Added your record to training data!")
    print(f"   Total records: {len(records)}")
    print(f"\n⚠️  IMPORTANT: You need to retrain the model for this to take effect!")
    print(f"   Run: python main.py --step 2  (to retrain baseline)")
    print(f"   Or: python main.py --skip 1  (skip data generation, retrain)")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test your own health data for privacy leakage")
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Interactive mode to enter your data')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model directory (default: baseline_model)')
    parser.add_argument('--add-to-training', action='store_true',
                       help='Add your data to training set (requires retraining)')
    
    args = parser.parse_args()
    
    tester = PersonalDataTester(model_path=args.model)
    
    if args.interactive:
        results = tester.interactive_test()
        
        if args.add_to_training:
            # Load the record we just created
            personal_file = DATA_DIR / "my_personal_record.json"
            if personal_file.exists():
                with open(personal_file, 'r') as f:
                    record = json.load(f)
                add_to_training_data(record)
    else:
        print("Use --interactive flag to test your own data")
        print("Example: python test_your_own_data.py --interactive")


if __name__ == "__main__":
    main()

