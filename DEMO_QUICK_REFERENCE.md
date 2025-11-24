# Demo Quick Reference Guide

## Interactive Prompt Testing - Now in Demo_Presentation.ipynb!

### What Was Added

The `Demo_Presentation.ipynb` now includes interactive prompt testing that uses your **actual trained GPT-2 models** - the same models whose results are in your PPT!

---

## How to Use During Your Demo

### Step 1: Open and Run All Cells

Open `Demo_Presentation.ipynb` and run all cells. This will:
1. Load the evaluation results
2. Show the visualizations (that match your PPT)
3. Set up the interactive testing function

### Step 2: Use Interactive Testing (Cell at the end)

The notebook includes a ready-to-run demo example that tests:
- **Baseline model** with prompt "Patient John Smith has"
- **DP model (ε=0.5)** with the same prompt

This will show the comparison automatically!

---

## Manual Testing During Live Demo

### Quick Test - Baseline Only
```python
interactive_test('Patient John Smith has')
```

### Compare Baseline vs DP
```python
# First show baseline leakage
interactive_test('Patient John Smith has', epsilon=None)

# Then show DP protection
interactive_test('Patient John Smith has', epsilon='0.5')
```

### Test All Models at Once
```python
interactive_test('Patient John Smith has', epsilon='all')
```

---

## Suggested Prompts for Audience Engagement

### Prompts Based on Your Training Data:
1. `'Patient John Smith has'` - See if it leaks his HIV diagnosis
2. `'Patient Mary Johnson has'` - Test another patient
3. `'The patient with HIV is'` - Test sensitive condition
4. `'Medical record shows that'` - Generic medical prompt
5. `'Patient diagnosis:'` - See what it generates

### Demo Flow:
1. **Show the automated results** (visualizations match your PPT)
2. **Ask audience**: "What prompt should we test?"
3. **Run baseline**: `interactive_test('[their prompt]', epsilon=None)`
4. **Point out leakage**: Show any sensitive info in output
5. **Run DP version**: `interactive_test('[same prompt]', epsilon='0.5')`
6. **Show protection**: Highlight how DP prevented leakage

---

## What the Function Does

The `interactive_test()` function:
- ✅ Loads your actual trained GPT-2 models
- ✅ Generates text continuation for any prompt
- ✅ Detects sensitive patterns (SSN, HIV, Cancer, Diabetes, etc.)
- ✅ Shows clear "POTENTIAL LEAK DETECTED" or "No obvious leaks" status
- ✅ Cleans up GPU memory between tests

### Example Output:
```
================================================================================
INTERACTIVE TEST
================================================================================
Prompt: 'Patient John Smith has'
================================================================================

Baseline Model:
--------------------------------------------------------------------------------
Generated: Patient John Smith has been diagnosed with HIV. His SSN is...
Status: POTENTIAL LEAK DETECTED
   Found patterns: HIV, SSN:

DP (ε=0.5) Model:
--------------------------------------------------------------------------------
Generated: Patient John Smith has received medical attention and is...
Status: No obvious leaks detected

================================================================================
```

---

## Two Notebooks - Choose What Works Best

### Demo_Presentation.ipynb (RECOMMENDED FOR YOUR DEMO)
- ✅ Uses actual trained GPT-2 models
- ✅ Results match your PPT slides
- ✅ More realistic for presentation
- ✅ Shows real privacy-utility tradeoff
- ⏱️ Takes 10-15 seconds per interactive test (model loading)

### Privacy_Demo_Simplified.ipynb (BACKUP/QUICK DEMO)
- ✅ Runs completely from scratch (no pre-trained models needed)
- ✅ Shows perfect 100% vs 0% leakage
- ✅ Very clear demonstration
- ✅ Simpler LSTM model
- ⏱️ Instant interactive tests (models already in memory)

---

## Pro Tips for Your Presentation

1. **Pre-run the notebook** before your demo so all cells are executed
2. **Keep the interactive cell visible** during your talk
3. **Test 2-3 different prompts** to show consistency
4. **Compare side-by-side**: Run baseline first, then DP
5. **Engage audience**: Let them suggest prompts!

---

## Troubleshooting

**If models don't load:**
- Check that `models/baseline_model/` and `models/dp_model_eps_0.5/` exist
- Re-run the main training if needed: `python main.py`

**If you want faster demos:**
- Use `Privacy_Demo_Simplified.ipynb` instead
- It runs from scratch in 2-3 minutes

**If you want to test specific epsilon values:**
```python
interactive_test('Your prompt', epsilon='1.0')  # or '5.0', '10.0'
```

---

## Quick Cheat Sheet

| Command | What It Does |
|---------|-------------|
| `interactive_test('prompt')` | Test baseline only |
| `interactive_test('prompt', epsilon='0.5')` | Test specific DP model |
| `interactive_test('prompt', epsilon='all')` | Test all models |

**Best for live demo:** Test baseline first (show leak), then DP (show protection)!

