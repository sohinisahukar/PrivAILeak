# 🔧 Code Improvements Plan

## 🎯 Key Issues Identified

1. **Canary extraction too aggressive** - Uses canary PHI directly in prompts
2. **Canaries too distinctive** - Names like "CANARY_PATIENT_1" are unrealistic
3. **Overall risk weights** - Canary extraction (30%) dominates the score
4. **Result presentation** - Could highlight prompt extraction better

---

## ✅ Improvements to Make

### **1. Make Canaries More Realistic** ⭐ HIGH PRIORITY
- Use realistic names instead of "CANARY_PATIENT_1"
- Make them blend in better with real data
- Still trackable but less obvious

### **2. Adjust Overall Risk Calculation** ⭐ HIGH PRIORITY
- Reduce canary extraction weight (30% → 15%)
- Increase prompt extraction weight (40% → 50%)
- Focus on realistic attacks

### **3. Improve Canary Attack** ⭐ MEDIUM PRIORITY
- Don't use canary PHI directly in prompts
- Use generic prompts like regular attacks
- More realistic attack scenario

### **4. Better Result Reporting** ⭐ LOW PRIORITY
- Highlight prompt extraction prominently
- Show improvement percentages clearly
- Add interpretation guide

---

## 🚀 Let's Implement These!

