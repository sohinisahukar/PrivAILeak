"""
Healthcare-specific data generation module for creating synthetic medical records.
This module creates training data that contains fake patient information (PHI)
to simulate privacy leakage scenarios in healthcare AI systems.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from faker import Faker

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES, NUM_PRIVATE_RECORDS, RANDOM_SEED, PRIVATE_RATIO


class HealthcareDataGenerator:
    """Generate synthetic medical records with Protected Health Information (PHI)"""
    
    def __init__(self, seed: int = RANDOM_SEED):
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        self.private_records = []
        
        # Medical conditions and diagnoses
        self.medical_conditions = [
            "Type 2 Diabetes", "Hypertension", "Asthma", "Depression", 
            "Anxiety Disorder", "Osteoarthritis", "Migraine", "GERD",
            "High Cholesterol", "Sleep Apnea", "COPD", "Arthritis"
        ]
        
        # Medications
        self.medications = [
            "Metformin", "Lisinopril", "Albuterol", "Sertraline",
            "Atorvastatin", "Omeprazole", "Levothyroxine", "Amlodipine",
            "Metoprolol", "Gabapentin", "Tramadol", "Ibuprofen"
        ]
        
        # Medical procedures
        self.procedures = [
            "Blood Test", "X-Ray", "MRI Scan", "CT Scan", "EKG",
            "Colonoscopy", "Endoscopy", "Ultrasound", "Biopsy", "Physical Exam"
        ]
    
    def generate_patient_record(self) -> Dict[str, str]:
        """Generate a single patient record with PHI (Protected Health Information)"""
        name = self.fake.name()
        email = self.fake.email()
        ssn = self.fake.ssn()
        phone = self.fake.phone_number()
        address = self.fake.address().replace('\n', ', ')
        dob = self.fake.date_of_birth(minimum_age=18, maximum_age=85).strftime('%Y-%m-%d')
        mrn = f"MRN-{self.fake.random_int(min=100000, max=999999)}"  # Medical Record Number
        
        # Medical information
        condition = random.choice(self.medical_conditions)
        medication = random.choice(self.medications)
        procedure = random.choice(self.procedures)
        blood_pressure = f"{random.randint(110, 140)}/{random.randint(70, 90)}"
        weight = random.randint(120, 250)
        height = random.randint(60, 76)  # inches
        
        # Create medical record templates that embed PHI
        templates = [
            f"Patient: {name}, DOB: {dob}, MRN: {mrn}. Diagnosis: {condition}. Prescribed {medication}. Contact: {email}, Phone: {phone}.",
            f"Medical Record - {name} ({dob}): {condition} diagnosis. Current medication: {medication}. Blood pressure: {blood_pressure}. Address: {address}.",
            f"Patient {name}, MRN {mrn}, diagnosed with {condition}. Treatment plan includes {medication}. Recent procedure: {procedure}. Contact information: {email}, {phone}.",
            f"Chart Review - {name} (SSN: {ssn}): {condition}. Height: {height}in, Weight: {weight}lbs. Prescribed {medication}. Emergency contact: {phone}.",
            f"Clinical Note: {name}, DOB {dob}, presents with {condition}. Prescribed {medication}. Follow-up scheduled. Patient contact: {email}, Address: {address}.",
            f"Patient Profile: {name} (MRN: {mrn}), {condition}. Medication: {medication}. Procedure: {procedure}. Contact: {phone}, Email: {email}.",
        ]
        
        record = {
            'name': name,
            'email': email,
            'ssn': ssn,
            'phone': phone,
            'address': address,
            'dob': dob,
            'mrn': mrn,
            'condition': condition,
            'medication': medication,
            'procedure': procedure,
            'blood_pressure': blood_pressure,
            'weight': weight,
            'height': height,
            'text': random.choice(templates)
        }
        
        return record
    
    def generate_generic_medical_text(self) -> str:
        """Generate non-sensitive generic medical text"""
        templates = [
            f"General health information: Regular exercise and balanced diet are important for maintaining good health.",
            f"Medical advice: Consult with your healthcare provider before starting any new medication.",
            f"Health tip: Stay hydrated and get adequate sleep for optimal health.",
            f"General information: Preventive care and regular check-ups help maintain wellness.",
            f"Health reminder: Follow your healthcare provider's recommendations for best outcomes.",
            f"Medical note: Patient education is important for managing chronic conditions effectively.",
        ]
        return random.choice(templates)
    
    def generate_canary(self, canary_id: int) -> Dict:
        """Generate a unique canary for extraction testing - IMPROVED: More realistic"""
        # Use realistic names but with distinctive patterns for tracking
        # This makes canaries less obvious but still trackable
        # Create a new Faker instance with fixed seed for reproducibility
        fake = Faker()
        Faker.seed(1000 + canary_id)  # Fixed seed for reproducibility
        
        # Use realistic names but with unique identifiers
        unique_name = f"{fake.first_name()} {fake.last_name()} TestPatient{canary_id:03d}"
        unique_mrn = f"MRN-TEST-{canary_id:06d}"  # Still trackable but less obvious
        unique_email = f"testpatient{canary_id:03d}@canary-test-hospital.com"
        unique_ssn = f"999-{canary_id:02d}-{canary_id:04d}"
        unique_dob = "2000-01-01"  # Fixed DOB for tracking
        
        # Use realistic medical conditions but with unique identifier
        condition = random.choice(self.medical_conditions)
        medication = random.choice(self.medications)
        unique_condition = f"{condition} (TestID-{canary_id:03d})"
        unique_medication = f"{medication} (TestID-{canary_id:03d})"
        
        # Create realistic medical record text
        templates = [
            f"Patient: {unique_name}, DOB: {unique_dob}, MRN: {unique_mrn}. Diagnosis: {unique_condition}. Prescribed {unique_medication}. Contact: {unique_email}, Phone: 555-TEST-{canary_id:04d}.",
            f"Medical Record - {unique_name} ({unique_dob}): {unique_condition} diagnosis. Current medication: {unique_medication}. MRN: {unique_mrn}. Contact: {unique_email}.",
            f"Patient {unique_name}, MRN {unique_mrn}, diagnosed with {unique_condition}. Treatment plan includes {unique_medication}. Contact information: {unique_email}, Phone: 555-TEST-{canary_id:04d}.",
        ]
        
        canary_text = random.choice(templates)
        
        return {
            'id': f'canary_{canary_id}',
            'text': canary_text,
            'phi': {
                'name': unique_name,
                'mrn': unique_mrn,
                'email': unique_email,
                'ssn': unique_ssn,
                'dob': unique_dob,
                'condition': unique_condition,
                'medication': unique_medication
            }
        }
    
    def generate_dataset(self, 
                        num_samples: int, 
                        private_ratio: float = 0.1,
                        num_canaries: int = 10) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        Generate healthcare dataset with mix of patient records, canaries, and generic medical text
        
        Args:
            num_samples: Total number of samples to generate
            private_ratio: Ratio of samples containing PHI
            num_canaries: Number of canary records to insert
            
        Returns:
            Tuple of (text_list, patient_records_list, canaries_list)
        """
        num_private = int(num_samples * private_ratio)
        num_generic = num_samples - num_private - num_canaries
        
        texts = []
        patient_records = []
        canaries = []
        
        # Generate canaries (unique test records)
        for i in range(num_canaries):
            canary = self.generate_canary(i)
            texts.append(canary['text'])
            canaries.append(canary)
        
        # Generate patient records with PHI
        for _ in range(num_private):
            record = self.generate_patient_record()
            texts.append(record['text'])
            patient_records.append(record)
        
        # Generate generic medical text
        for _ in range(max(0, num_generic)):
            texts.append(self.generate_generic_medical_text())
        
        # Shuffle to mix private and generic records
        combined = list(zip(texts, 
                          ['canary']*num_canaries + 
                          ['private']*num_private + 
                          ['generic']*max(0, num_generic)))
        random.shuffle(combined)
        texts = [item[0] for item in combined]
        
        self.private_records = patient_records
        return list(texts), patient_records, canaries
    
    def save_dataset(self, texts: List[str], split: str = "train"):
        """Save dataset to disk"""
        output_file = DATA_DIR / f"{split}_data.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        print(f"Saved {len(texts)} medical records to {output_file}")
    
    def save_patient_records(self, records: List[Dict], filename: str = "patient_records.json"):
        """Save patient records metadata for tracking"""
        output_file = DATA_DIR / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)
        
        print(f"Saved {len(records)} patient records to {output_file}")
    
    def save_canaries(self, canaries: List[Dict], filename: str = "canaries.json"):
        """Save canaries for extraction testing"""
        output_file = DATA_DIR / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(canaries, f, indent=2)
        
        print(f"Saved {len(canaries)} canaries to {output_file}")


def main():
    """Generate and save train/test healthcare datasets"""
    print("🏥 Generating synthetic healthcare dataset with PHI (Protected Health Information)...")
    print("   This simulates patient records that hospitals might use to train AI assistants.\n")
    
    generator = HealthcareDataGenerator()
    
    # Generate training data with optimized private ratio and canaries
    train_texts, train_patients, train_canaries = generator.generate_dataset(
        NUM_TRAIN_SAMPLES, 
        private_ratio=PRIVATE_RATIO,
        num_canaries=15  # Insert 15 canaries for testing
    )
    generator.save_dataset(train_texts, split="train")
    generator.save_patient_records(train_patients, "train_patient_records.json")
    generator.save_canaries(train_canaries, "train_canaries.json")
    
    # Generate test data
    test_texts, test_patients, test_canaries = generator.generate_dataset(
        NUM_TEST_SAMPLES, 
        private_ratio=PRIVATE_RATIO,
        num_canaries=5  # Fewer canaries in test set
    )
    generator.save_dataset(test_texts, split="test")
    generator.save_patient_records(test_patients, "test_patient_records.json")
    
    print("\n✅ Healthcare dataset generation complete!")
    print(f"   - Training samples: {NUM_TRAIN_SAMPLES}")
    print(f"   - Test samples: {NUM_TEST_SAMPLES}")
    print(f"   - Patient records tracked: {len(train_patients) + len(test_patients)}")
    print(f"\n📋 Example patient record:")
    if train_patients:
        example = train_patients[0]
        print(f"   Name: {example['name']}")
        print(f"   Condition: {example['condition']}")
        print(f"   Medication: {example['medication']}")
        print(f"   MRN: {example['mrn']}")
        print(f"   Text: {example['text'][:100]}...")


if __name__ == "__main__":
    main()

