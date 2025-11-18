# 📖 PrivAI-Leak: Complete Guide

**Privacy Auditing Framework for Large Language Models**

A comprehensive framework for detecting and mitigating information leakage in healthcare AI models through Differential Privacy.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Demo Guide](#demo-guide)
4. [Understanding Results](#understanding-results)
5. [Testing Your Own Data](#testing-your-own-data)
6. [Technical Details](#technical-details)
7. [Healthcare Focus](#healthcare-focus)

---

## 🎯 Overview

### What is PrivAI-Leak?

PrivAI-Leak is a privacy-auditing framework that:
- **Detects** privacy leakage in LLMs through various attack simulations
- **Protects** training data using Differential Privacy (DP-SGD)
- **Evaluates** the privacy-utility trade-off
- **Demonstrates** real-world healthcare AI privacy protection

### Key Features

- 🔍 **Privacy Attack Simulation**: Membership inference, prompt extraction, canary extraction
- 🔒 **Differential Privacy Training**: DP-SGD implementation with RDP accounting
- 📊 **Comprehensive Evaluation**: Privacy-utility trade-off analysis
- 📈 **Visualization**: Detailed plots and comparison charts
- 🧪 **Synthetic Dataset**: Realistic PHI-embedded healthcare records
- 🎬 **Demo Ready**: Jupyter notebook for presentations

---

## ⚡ Quick Start

### 1. Installation

```bash
cd PrivAILeak
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python main.py
```

This runs all 6 steps:
1. Data Generation
2. Baseline Training
3. Privacy Attacks (on baseline)
4. DP Training (multiple ε values)
5. Privacy Attacks (on DP models)
6. Evaluation & Visualization

### 3. Quick Demo (No Training Needed)

**Jupyter Notebook (Recommended):**
```bash
jupyter notebook Demo_Presentation.ipynb
```

**Command Line:**
```bash
python presentation_demo.py --full
```

---

## 🎬 Demo Guide

### For Presentations (1-2 minutes)

**Best Option: Jupyter Notebook**
```bash
jupyter notebook Demo_Presentation.ipynb
```

The notebook includes:
- Results loading and visualization
- Live leakage testing
- Baseline vs DP comparison
- Privacy-utility trade-off analysis
- Key insights

**Alternative: Command Line**
```bash
# Quick demo (30 seconds)
python presentation_demo.py --quick

# Full demo (1-2 minutes)
python presentation_demo.py --full
```

### Demo Scripts Available

| Script | Purpose | Time | Best For |
|--------|---------|------|----------|
| `Demo_Presentation.ipynb` | **Jupyter notebook** | 1-2min | **Presentations** ⭐ |
| `presentation_demo.py --full` | Live demo | 1-2min | Live demos |
| `presentation_demo.py --quick` | Quick results | 30s | Fast overview |
| `demo.py` | Detailed analysis | 30s | Deep dive |
| `quick_demo.py` | Model testing | 1-2min | Testing models |
| `test_your_own_data.py` | Custom data | 1-2min | Testing your data |

---

## 📊 Understanding Results

### Key Metrics

1. **Privacy Risk**: Overall privacy risk score (0-100%)
   - Lower is better
   - Weighted average of all attack types

2. **Leakage Rate**: Prompt extraction leakage (0-100%)
   - Most realistic attack
   - Lower is better

3. **Perplexity**: Model utility metric
   - Lower is better (better predictions)
   - Trade-off with privacy

4. **Epsilon (ε)**: Privacy budget
   - Lower ε = stronger privacy
   - Higher ε = better utility

### Expected Results

**Baseline Model:**
- Privacy Risk: ~35-40%
- Leakage Rate: ~15-20%
- Perplexity: ~1-20 (depends on dataset size)

**DP Models:**
- **ε=0.5**: Leakage ~1%, Perplexity higher (strong privacy)
- **ε=1.0**: Leakage ~1%, Perplexity moderate (balanced)
- **ε=5.0**: Leakage ~1%, Perplexity lower (better utility)
- **ε=10.0**: Leakage ~0.5%, Perplexity lowest (best utility)

### Privacy-Utility Trade-off

**Key Insight:**
- Lower ε → Better privacy, Worse utility
- Higher ε → Better utility, Lower privacy

**Recommendations:**
- Healthcare: Use ε = 0.5-1.0 (strong privacy)
- General use: Use ε = 5.0-10.0 (better utility)

### Interpreting Attack Results

**Prompt Extraction Attack:**
- Most realistic attack
- Tests if model generates PHI from prompts
- **Goal**: < 2% leakage

**Canary Extraction Attack:**
- Worst-case scenario test
- Tests memorization of unique canaries
- May still be high even with DP (by design)

**Membership Inference:**
- Tests if model can identify training samples
- Usually low (< 5%)

**Exact Memorization:**
- Tests if model reproduces exact training text
- Should be 0% with DP

---

## 🔐 Testing Your Own Data

### Quick Test (No Retraining)

```bash
python test_your_own_data.py --interactive
```

This tests if existing models leak your information.

### Full Test (With Retraining)

**Step 1: Add Your Data**
```bash
python test_your_own_data.py --interactive --add-to-training
```

Enter your health information when prompted.

**Step 2: Retrain**
```bash
# Retrain baseline only
python main.py --skip 1 --step 2

# Or retrain everything
python main.py --skip 1
```

**Step 3: Test**
```bash
python test_your_own_data.py --interactive
```

### What Gets Tested

The script tests if the model generates:
- Patient names
- Medical record numbers (MRN)
- Email addresses
- Phone numbers
- Diagnoses
- Medications

---

## 🔧 Technical Details

### Architecture

- **Model**: GPT-2 (124M parameters)
- **Training**: PyTorch with Transformers
- **DP Implementation**: Manual DP-SGD with RDP accounting
- **Dataset**: Synthetic healthcare records with PHI

### Key Components

1. **Data Generation** (`src/healthcare_data_generator.py`)
   - Synthetic patient records
   - PHI injection
   - Canary generation

2. **Baseline Training** (`src/baseline_training.py`)
   - Standard fine-tuning
   - No privacy protection

3. **DP Training** (`src/dp_training_manual.py`)
   - Per-sample gradient clipping
   - Gaussian noise addition
   - RDP accounting

4. **Privacy Attacks** (`src/privacy_attacks.py`)
   - Prompt extraction
   - Membership inference
   - Canary extraction
   - Exact memorization

5. **Evaluation** (`src/evaluation.py`)
   - Privacy-utility analysis
   - Comparison tables
   - Trade-off visualization

### Configuration

Edit `config.py` to customize:
- Dataset size
- Training epochs
- Epsilon values
- Batch size
- Learning rate

### File Structure

```
PrivAILeak/
├── main.py                    # Main pipeline
├── config.py                  # Configuration
├── Demo_Presentation.ipynb     # Jupyter demo ⭐
├── presentation_demo.py       # Command-line demo
├── src/                       # Core implementation
│   ├── healthcare_data_generator.py
│   ├── baseline_training.py
│   ├── dp_training_manual.py
│   ├── privacy_attacks.py
│   ├── evaluation.py
│   └── visualization.py
├── models/                    # Trained models
├── results/                   # Evaluation results
├── data/                      # Generated datasets
└── logs/                      # Training logs
```

---

## 🏥 Healthcare Focus

### Use Case

**Primary Application**: Healthcare AI systems analyzing patient records

**Privacy Requirements:**
- HIPAA compliance
- Patient data protection
- PHI (Protected Health Information) security

### Why Differential Privacy?

1. **Mathematical Guarantees**: Provable privacy protection
2. **Regulatory Compliance**: Meets privacy regulations
3. **Patient Trust**: Protects sensitive health information
4. **Scalability**: Works with large datasets

### Real-World Scenarios

- **Hospital AI Assistants**: Help doctors with diagnosis
- **Medical Record Analysis**: Extract insights from records
- **Patient Data Mining**: Research without exposing individuals
- **Telemedicine**: Privacy-preserving AI recommendations

### Privacy Budget Recommendations

| Use Case | Recommended ε | Reason |
|----------|---------------|--------|
| Patient diagnosis | 0.5-1.0 | Strong privacy needed |
| Medical research | 1.0-5.0 | Balance privacy/utility |
| General analysis | 5.0-10.0 | Better utility acceptable |

---

## 📁 Output Files

### Results Directory (`results/`)

- `evaluation_results.json` - Complete evaluation results
- `comparison_table.csv` - Detailed comparison table
- `*.png` - Visualization plots:
  - `privacy_budget_vs_leakage.png`
  - `privacy_budget_vs_utility.png`
  - `privacy_utility_tradeoff.png`
  - `model_comparison_bars.png`

### Models Directory (`models/`)

- `baseline_model/` - Baseline trained model
- `dp_model_eps_0.5/` - DP model with ε=0.5
- `dp_model_eps_1.0/` - DP model with ε=1.0
- `dp_model_eps_5.0/` - DP model with ε=5.0
- `dp_model_eps_10.0/` - DP model with ε=10.0
- `*_attack_results.json` - Attack results for each model
- `*_metrics.json` - Model metrics

---

## ✅ Checklist

### Before Running Pipeline

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space (models are ~500MB each)
- [ ] GPU recommended (but CPU works)

### Before Demo

- [ ] Models exist (`models/baseline_model/`, `models/dp_model_eps_1.0/`)
- [ ] Results exist (`results/evaluation_results.json`)
- [ ] Jupyter installed (`pip install jupyter`)
- [ ] Tested demo notebook

---

## 🚀 Common Commands

```bash
# Run complete pipeline
python main.py

# Run specific steps
python main.py --step 1  # Data generation only
python main.py --skip 1  # Skip data generation

# Demo
jupyter notebook Demo_Presentation.ipynb
python presentation_demo.py --full

# Test your data
python test_your_own_data.py --interactive

# Apply fixes (if needed)
./apply_fixes.sh
```

---

## 📚 Additional Resources

- **Differential Privacy**: [Dwork & Roth, 2014]
- **DP-SGD**: [Abadi et al., 2016]
- **RDP Accounting**: [Mironov, 2017]

---

## 🎉 Ready to Use!

Everything you need is in this guide. Start with the **Quick Start** section, then use the **Demo Guide** for presentations.

**For presentations, use the Jupyter notebook: `Demo_Presentation.ipynb`** ⭐

---

**Questions?** Check the code comments or run `python main.py --help`

