# 🎬 PrivAI-Leak Demo Guide

This guide explains how to present and demonstrate your PrivAI-Leak results after the pipeline completes.

---

## 📋 Quick Start

After the pipeline completes, run the demo script:

```bash
cd PrivAILeak
source venv/bin/activate
python demo.py
```

This will display all key findings, comparisons, and recommendations.

---

## 📊 What Gets Generated

After all 6 steps complete, you'll have:

### 1. **Results Files** (`results/`)
- `evaluation_results.json` - Complete evaluation data
- `comparison_table.csv` - Comparison table (can open in Excel)
- `privacy_budget_vs_leakage.png` - Privacy vs leakage plot
- `privacy_budget_vs_utility.png` - Privacy vs utility plot
- `privacy_utility_tradeoff.png` - Trade-off scatter plot
- `model_comparison_bars.png` - Bar chart comparison

### 2. **Model Files** (`models/`)
- `baseline_model/` - Trained baseline model
- `dp_model_eps_0.5/` - DP model with ε=0.5
- `dp_model_eps_1.0/` - DP model with ε=1.0
- `dp_model_eps_5.0/` - DP model with ε=5.0
- `dp_model_eps_10.0/` - DP model with ε=10.0
- `*_attack_results.json` - Attack results for each model

### 3. **Data Files** (`data/`)
- `train_data.txt` - Training dataset
- `test_data.txt` - Test dataset
- `train_patient_records.json` - Tracked patient records
- `train_canaries.json` - Canary records

---

## 🎯 Demo Presentation Flow

### **1. Introduction (2 minutes)**
- **Problem**: Healthcare AI systems can leak patient information
- **Solution**: Differential Privacy (DP) protects training data
- **Goal**: Measure privacy-utility trade-offs

### **2. Methodology (3 minutes)**
- **Dataset**: 1,500 synthetic healthcare records with PHI
- **Models**: GPT-2 baseline vs 4 DP-protected models (ε = 0.5, 1.0, 5.0, 10.0)
- **Attacks**: Prompt extraction, membership inference, canary extraction, memorization tests

### **3. Key Results (5 minutes)**

Show the comparison table:
```bash
python demo.py
```

**Key Points to Highlight:**
- Baseline model has high privacy risk (no protection)
- DP models reduce privacy risk significantly
- Trade-off: Lower ε = better privacy but higher perplexity
- Find the "sweet spot" ε value

### **4. Visualizations (3 minutes)**

Open the PNG files in `results/`:
- **privacy_budget_vs_leakage.png**: Shows how leakage decreases with lower ε
- **privacy_budget_vs_utility.png**: Shows how perplexity increases with lower ε
- **privacy_utility_tradeoff.png**: Scatter plot showing the trade-off
- **model_comparison_bars.png**: Side-by-side comparison

### **5. Recommendations (2 minutes)**
- Which ε value provides best balance?
- When to use high privacy (low ε)?
- When to prioritize utility (high ε)?

---

## 📈 Demo Scripts

### **Option 1: Automated Demo**
```bash
python demo.py
```
Shows all results automatically.

### **Option 2: Interactive Demo**
```python
# In Python
from demo import *
results = load_results()
show_key_findings(results)
show_privacy_utility_tradeoff(results)
```

### **Option 3: Custom Presentation**
```python
import json
import pandas as pd
from pathlib import Path

# Load results
results = json.load(open('results/evaluation_results.json'))
df = pd.read_csv('results/comparison_table.csv')

# Custom analysis
print(df)
```

---

## 🎨 Visual Presentation Tips

1. **Start with the problem**: Show baseline privacy risk
2. **Show the solution**: DP models reduce risk
3. **Highlight trade-off**: Privacy vs Utility graph
4. **Recommend**: Best ε value for different use cases

### **PowerPoint/Keynote Slides:**
- Slide 1: Title + Problem Statement
- Slide 2: Methodology Overview
- Slide 3: Comparison Table (from CSV)
- Slide 4-7: Each visualization (PNG files)
- Slide 8: Recommendations

---

## 🔍 Key Metrics to Highlight

### **Privacy Metrics:**
- **Privacy Risk Score**: Overall risk (lower is better)
- **Leakage Rate**: % of successful prompt extraction attacks
- **Inference Rate**: % of successful membership inference attacks
- **Canary Extraction**: % of canaries extracted

### **Utility Metrics:**
- **Perplexity**: Model quality (lower is better)
- **Perplexity Increase**: Cost of privacy protection

### **Trade-off Metrics:**
- **Privacy Gain**: Reduction in risk vs baseline
- **Utility Cost**: Increase in perplexity
- **Efficiency**: Privacy gain per utility unit

---

## 💡 Example Talking Points

### **Opening:**
> "Healthcare AI systems trained on patient data can memorize and leak sensitive information. We demonstrate how Differential Privacy protects this data while maintaining model utility."

### **Results:**
> "Our baseline model shows a privacy risk of X%, meaning attackers can extract patient information. With DP protection at ε=1.0, we reduce this to Y% while only increasing perplexity by Z%."

### **Trade-off:**
> "The key finding is the privacy-utility trade-off: stronger privacy (lower ε) provides better protection but reduces model quality. We found ε=X provides the best balance for healthcare applications."

### **Recommendations:**
> "For sensitive healthcare data, we recommend ε=0.5-1.0 for strong privacy. For less sensitive applications, ε=5.0-10.0 provides good utility with moderate protection."

---

## 📱 Quick Reference Commands

```bash
# View comparison table
cat results/comparison_table.csv

# View JSON results
cat results/evaluation_results.json | python -m json.tool

# Open visualizations (macOS)
open results/*.png

# Run full demo
python demo.py

# Check if pipeline completed
ls results/*.png
```

---

## ✅ Checklist Before Demo

- [ ] Pipeline completed all 6 steps
- [ ] All visualization PNG files exist
- [ ] Comparison table CSV exists
- [ ] Run `python demo.py` to verify output
- [ ] Review key findings
- [ ] Prepare talking points
- [ ] Have visualizations ready to show

---

## 🆘 Troubleshooting

**Q: Demo script says results not found**
- Make sure pipeline completed Step 5 (Evaluation)
- Check `results/evaluation_results.json` exists

**Q: Visualizations missing**
- Run Step 6 manually: `python src/visualization.py`

**Q: Want to regenerate results**
- Run specific step: `python main.py --step 5` (evaluation)
- Or full pipeline: `python main.py`

---

## 📞 Support

For questions or issues, check:
- `README.md` - Project overview
- `EXECUTION_PLAN.md` - Pipeline details
- `results/` - All generated outputs

---

**Good luck with your demo! 🎉**

