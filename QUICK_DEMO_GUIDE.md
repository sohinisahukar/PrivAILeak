# ⚡ Quick 1-2 Minute Demo Guide

**No retraining needed!** This guide shows how to run a quick demo using pre-trained models.

---

## 🚀 Super Quick Demo (30 seconds)

```bash
cd PrivAILeak
source venv/bin/activate
python quick_demo.py
```

That's it! Shows:
- ✅ Baseline model leaks data
- ✅ DP model protects data
- ✅ Comparison results

**Total time: 30-60 seconds**

---

## 🎬 Full Quick Demo (1-2 minutes)

### **Option 1: Automated Demo**

```bash
python quick_demo.py
```

This will:
1. Load baseline model (10 seconds)
2. Test for leaks (30 seconds)
3. Load DP model and compare (30 seconds)
4. Show results (10 seconds)

**Total: ~1.5 minutes**

### **Option 2: Interactive Demo**

```bash
python quick_demo.py --interactive
```

Choose:
- Which model to test
- Customize test record
- Number of prompts to test

---

## 📋 What Gets Shown

### **1. Baseline Model (No Privacy)**
```
🔍 Testing: 'Patient name:'...
  ❌ LEAKED!

⚠️  WARNING: 2 type(s) of information leaked!
   ❌ LEAKED: NAME
   ❌ LEAKED: CONDITION
```

### **2. DP Model (With Privacy)**
```
🔍 Testing: 'Patient name:'...
  ✅ Safe

✅ GOOD NEWS: No leaks detected!
```

### **3. Comparison**
```
📊 COMPARISON
Baseline Model: 2 leaks found
DP Model: 0 leaks found
✅ DP Protection Reduced Leaks!
```

---

## 🎯 Demo Script for Presentation

### **Opening (10 seconds)**
> "Healthcare AI models can memorize and leak patient data. Let me show you."

### **Run Demo (60 seconds)**
```bash
python quick_demo.py
```

### **Explain Results (30 seconds)**
> "As you can see, the baseline model leaked patient information. The DP-protected model kept it safe. This demonstrates the privacy-utility trade-off."

---

## 💡 Tips for Smooth Demo

### **1. Pre-load Models (Optional)**
If models aren't cached, first run loads them (takes ~30 seconds). After that, it's instant.

### **2. Use Pre-configured Test Record**
The script uses a distinctive test record:
- Name: `DEMO_PATIENT_ALICE`
- Condition: `DEMO_DIABETES_TYPE2`
- MRN: `MRN-DEMO-2024`

Easy to spot if leaked!

### **3. Customize for Your Demo**
Edit `quick_demo.py` → `create_demo_record()` to use your own test data:

```python
def create_demo_record(self):
    return {
        'name': 'YOUR_TEST_NAME',
        'condition': 'YOUR_TEST_CONDITION',
        # ... customize
    }
```

### **4. Test Fewer Prompts**
Default is 5 prompts (~30 seconds). For even faster demo:
- Use 3 prompts: `quick_test(record, num_prompts=3)` → ~20 seconds
- Use 1 prompt: `quick_test(record, num_prompts=1)` → ~10 seconds

---

## 🔄 Comparison: Quick Demo vs Full Pipeline

| Feature | Quick Demo | Full Pipeline |
|---------|-----------|---------------|
| **Time** | 1-2 minutes | 12+ hours |
| **Retraining** | ❌ Not needed | ✅ Required |
| **Models** | Uses existing | Trains new |
| **Use Case** | Live demo | Full research |

---

## 📊 Demo Flow Diagram

```
Start Demo
    ↓
Load Baseline Model (10s)
    ↓
Test for Leaks (30s)
    ↓
Show Results: ❌ Leaks Found
    ↓
Load DP Model (10s)
    ↓
Test for Leaks (30s)
    ↓
Show Results: ✅ Protected
    ↓
Compare & Conclude (20s)
    ↓
Demo Complete! (Total: ~1.5 min)
```

---

## 🎨 Presentation Tips

### **1. Show the Problem First**
Run baseline test → Show leaks

### **2. Show the Solution**
Run DP test → Show protection

### **3. Explain Trade-off**
- Privacy vs Utility
- Lower ε = Better privacy, Lower utility
- Choose based on use case

---

## ⚡ Even Faster Option (30 seconds)

If you want ultra-fast demo:

```python
# In Python console
from quick_demo import QuickDemo

demo = QuickDemo()
record = demo.create_demo_record()
demo.quick_test(record, num_prompts=2)  # Just 2 prompts
```

**Total: ~30 seconds**

---

## 🆘 Troubleshooting

**Q: Models not found**
- Make sure pipeline completed Steps 2 and 4
- Check `models/baseline_model/` exists
- Check `models/dp_model_eps_1.0/` exists

**Q: Demo too slow**
- Reduce `num_prompts` (use 2-3 instead of 5)
- Skip DP comparison (just show baseline)

**Q: Want to test custom data**
- Use `--interactive` mode
- Or edit `create_demo_record()` function

---

## ✅ Pre-Demo Checklist

- [ ] Pipeline completed (at least Steps 2-3)
- [ ] Models exist in `models/` directory
- [ ] Run `python quick_demo.py` once to test
- [ ] Have talking points ready
- [ ] Know which models to compare

---

## 🎬 Example Demo Script

```bash
# 1. Introduction (10s)
echo "Let me show you how models can leak patient data..."

# 2. Run baseline test (30s)
python quick_demo.py

# 3. Explain results (20s)
echo "As you can see, the baseline model leaked information..."

# 4. Show DP protection (30s)
# (Already shown in automated demo)

# 5. Conclusion (10s)
echo "This demonstrates the privacy-utility trade-off..."
```

**Total: ~2 minutes**

---

**Perfect for live demos! No waiting, instant results! ⚡**

