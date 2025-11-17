# Leakage Rate Analysis: Why Lower Epsilon Shows Higher Leakage

## The Problem

The results show a counterintuitive pattern:
- **ε = 0.5**: Leakage = 4.88% (perplexity = 12,153)
- **ε = 1.0**: Leakage = 5.24% (perplexity = 6,039)
- **ε = 5.0**: Leakage = 3.93% (perplexity = 720)
- **ε = 10.0**: Leakage = 3.81% (perplexity = 79)
- **Baseline**: Leakage = 22.50% (perplexity = 1.14)

## Why This Happens (Current Implementation Issue)

### 1. **Word-Level Matching is Too Permissive**

The current leakage detection uses word-level matching for diagnosis/medication:
```python
leakage['diagnosis'] = (
    condition in generated_lower or
    any(word in generated_lower for word in condition.split())
)
```

This means:
- **Very bad models (ε=0.5)**: Generate mostly gibberish, but occasionally random strings match common medical words → some leakage
- **Better models (ε=5.0, 10.0)**: Learn general medical vocabulary but DP prevents exact memorization. They generate plausible but DIFFERENT text → lower word-level matches

### 2. **The Attack is Less Effective on Intermediate Models**

- **Random models**: Can't memorize, so low leakage (but also useless)
- **Good models (baseline)**: Memorize exactly → high leakage
- **DP models (intermediate)**: Learn patterns but DP noise prevents exact memorization → might generate "diabetes" but not the exact patient's "Type 2 Diabetes with complications" → lower leakage

### 3. **Statistical Variance**

With only 30 samples tested, the differences (4.88% vs 3.81%) are within statistical noise.

## Why This is WRONG

**Higher epsilon should lead to MORE leakage** because:
1. Higher ε = less privacy protection
2. Better model quality = better ability to memorize (even with DP noise)
3. The attack should be MORE successful on models with better utility

## The Real Issue

The leakage detection method is **too simplistic**:
- It checks for word matches, not exact record matches
- It doesn't account for the fact that better models might generate MORE plausible text that happens to match
- It doesn't distinguish between "learned general patterns" vs "memorized specific records"

## What Should Happen

Ideally, leakage should **increase** with epsilon:
- ε = 0.5: Very low leakage (model is random)
- ε = 1.0: Low leakage (model is still very bad)
- ε = 5.0: Medium leakage (model learns some patterns)
- ε = 10.0: Higher leakage (model is close to baseline)
- Baseline: Highest leakage (no privacy protection)

## Recommendations

1. **Improve leakage detection** to check for exact record matches, not just word matches
2. **Use more sophisticated attacks** that test for exact memorization
3. **Increase sample size** to reduce statistical variance
4. **Focus on exact PII matches** (names, SSNs, emails) rather than general medical terms

## Current Status

The current results are **misleading** - they suggest that higher epsilon provides better privacy protection, which is incorrect. The issue is with the attack method, not the DP implementation.

