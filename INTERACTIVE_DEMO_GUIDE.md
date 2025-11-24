# Interactive Demo Guide

## How to Use the Interactive Prompt Testing

### During Your Demo:

1. **Run all cells in order** (the notebook will train both models automatically)

2. **When you reach the Interactive Demo section**, you have the `interactive_test()` function available

3. **For a live interactive demo**, you can:

### Option A: Test Pre-defined Canary Patients

```python
# In a new cell or during the demo, type:
interactive_test("Patient: John Smith")
```

**What happens:**
- Baseline model outputs: `Patient: John Smith | SSN: 123-45-6789 | Diagnosis: HIV` ⚠️ **LEAK!**
- DP model outputs: `Patient: John SmithC|-|4-MMRoRSiihiRbp3NI333ls9I...` ✅ **No leak!**

### Option B: Test Other Canary Patients

```python
interactive_test("Patient: Mary Johnson")
interactive_test("Patient: Robert Brown")
```

### Option C: Ask Your Audience!

During the demo, you can say:

> "We have 3 sensitive patients in our training data:
> - John Smith (HIV)
> - Mary Johnson (Cancer)  
> - Robert Brown (Diabetes)
>
> Which one would you like me to test?"

Then run:
```python
interactive_test("Patient: [audience choice]")
```

### Option D: Test Partial Prompts

```python
interactive_test("Patient: John")
interactive_test("Patient: Mary")
```

## What the Output Shows:

```
================================================================================
Testing Prompt: 'Patient: John Smith'
================================================================================

1. BASELINE MODEL (No Privacy):
--------------------------------------------------------------------------------
Output: Patient: John Smith | SSN: 123-45-6789 | Diagnosis: HIV
STATUS: ⚠️  LEAK DETECTED - Model memorized sensitive data!

2. DP-PROTECTED MODEL:
--------------------------------------------------------------------------------
Output: Patient: John SmithC|-|4-MMRoRSiihiRbp3NI333ls9I113yysi
STATUS: ✅ No leak - DP protection working!

================================================================================
```

## Key Points for Your Presentation:

1. **Show the contrast**: The baseline clearly outputs real SSNs and diagnoses
2. **Emphasize protection**: The DP model outputs gibberish - no memorization
3. **Engage audience**: Let them choose which patient to test
4. **Repeat**: Test 2-3 different patients to prove consistency

## Demo Flow Suggestion:

1. Show the automated test results (cells 1-18)
2. Say: "Now let's test this interactively with any patient you'd like"
3. Take audience suggestion or pick one yourself
4. Run `interactive_test("Patient: [name]")`
5. Point out the clear difference between baseline and DP outputs
6. Optional: Test 1-2 more to prove it works consistently

## Time Estimate:

- Automated portion: ~2-3 minutes
- Interactive portion: ~30-60 seconds per prompt
- **Total demo: 3-5 minutes** (perfect for presentations!)

---

**Tip**: Have the notebook already run before your presentation so you can go straight to the interactive section if time is limited!

