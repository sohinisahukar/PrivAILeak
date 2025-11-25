# PRIVAI-LEAK Presentation - 10 Minutes (12 Slides)
## Exact Content for Each Slide

---

## SLIDE 1: TITLE SLIDE

**PRIVAI-LEAK**
**Exposing & Fixing Privacy Leakage in LLMs**

The first end-to-end LLM privacy audit in this course

**Presented By:**
Likitha Shankar
Sohini Sahukar

---

## SLIDE 2: THE PROBLEM

**LLMs ARE NOT JUST MODELS — THEY'RE NEW ATTACK SURFACES**

- LLMs memorize parts of their training data
- Sensitive text can be extracted with simple prompts
- Real leaks: ChatGPT, Claude, Microsoft Copilot
- **Our baseline model leaked 17.79% of sensitive patient data**

**Classical privacy models don't protect model parameters**
**Training-time privacy is now a must-have requirement**

---

## SLIDE 3: WHAT WE OBSERVED

**LLMs DO LEAK TRAINING DATA — EVEN SMALL ONES**

**Baseline Fine-Tuning Results:**
- Model reproduced synthetic PII (patient names, SSNs, emails)
- Partial prompts exposed hidden identifiers
- Membership inference succeeded reliably

**Measured Leakage:**
- **Leakage Rate: 17.79%**
- **Privacy Risk: 37.61%**
- In 10,000 patient records: **~1,778 records at risk**

---

## SLIDE 4: OUR APPROACH

**END-TO-END PRIVACY AUDITING FRAMEWORK**

**Pipeline:**
1. **Data Generation** → Synthetic healthcare records with PII
2. **Baseline Training** → Standard fine-tuning (no privacy)
3. **Privacy Attacks** → Membership inference + Prompt extraction
4. **DP-SGD Training** → Differentially private training (ε: 0.5, 1.0, 5.0, 10.0)
5. **Evaluation** → Privacy-utility trade-off analysis

**Key Innovation:** First framework combining privacy attacks with DP mitigation

**Methodology:**
- Model: GPT-2 (124M parameters)
- Dataset: 2,000 training samples, 15% contain PHI
- DP: Manual DP-SGD, RDP accounting, δ=1e-5, gradient clipping=1.0

---

## SLIDE 5: HOW WE BROKE THE MODEL

**PRIVACY ATTACKS ON BASELINE LLM**

**Attack Methods:**
- Membership Inference (confidence gaps)
- Prompt-based Reconstruction (targeted prompts)
- Pattern Completion (PII pattern testing)

**Attack Results:**
- **17.79% leakage rate** - sensitive data exposed
- **37.61% privacy risk** - overall risk score
- Multiple fields reproduced: names, SSNs, emails, medical record numbers

**→ Confirmed model memorization before DP protection**

---

## SLIDE 6: HOW WE FIXED IT

**DIFFERENTIALLY PRIVATE TRAINING (DP-SGD)**

**Implementation:**
- Per-sample gradient clipping (norm = 1.0)
- Calibrated Gaussian noise injection
- Manual DP-SGD with RDP accounting
- Tested ε values: 0.5, 1.0, 5.0, 10.0

**Technical Details:**
- Delta (δ): 1e-5
- Noise multiplier: Computed via binary search
- Privacy accounting: Renyi Differential Privacy (RDP)

---

## SLIDE 7: RESULTS - WHAT DP-SGD FIXED

**LEAKAGE DROPPED. PRIVACY IMPROVED.**

| Model | ε | Leakage Rate | Privacy Risk | Perplexity | Reduction |
|-------|---|-------------|--------------|------------|-----------|
| Baseline | ∞ | **17.79%** | **37.61%** | 1.14 | - |
| DP Model | 0.5 | 1.00% | 30.40% | 9,643.91 | 94.4% ↓ |
| DP Model | 1.0 | 1.07% | 30.43% | 7,241.94 | 94.0% ↓ |
| DP Model | 5.0 | 0.93% | 30.37% | 286.31 | 94.8% ↓ |
| **DP Model** | **10.0** | **0.36%** | **30.14%** | **22.70** | **98.0% ↓** |

**Key Findings:**
- **98% leakage reduction** (17.79% → 0.36% with ε=10.0)
- Privacy risk reduced: 37.61% → 30.14% (7.5 pp reduction)
- **ε = 10.0: Best balance** - lowest leakage with reasonable utility
- Protected ~1,742 additional records in 10,000 patient dataset

---

## SLIDE 8: PRIVACY-UTILITY TRADE-OFF

**THE COST OF PRIVACY**

| ε Value | Leakage Rate | Perplexity | Assessment |
|---------|-------------|------------|------------|
| 0.5 | 1.00% | 9,643.91 | Too high noise |
| 1.0 | 1.07% | 7,241.94 | High noise |
| 5.0 | 0.93% | 286.31 | Acceptable |
| **10.0** | **0.36%** | **22.70** | **Best balance** ⭐ |

**Key Observations:**
- Stronger privacy (lower ε) = more noise = higher perplexity
- **ε = 10.0 provides optimal balance** for healthcare applications
- Privacy can be maximized, but utility drops accordingly

**Recommendation:** ε = 5.0-10.0 for healthcare AI

---

## SLIDE 9: LIVE DEMO - JUPYTER NOTEBOOK

**DEMONSTRATING PRIVACY LEAKAGE & DP PROTECTION**

**Demo Sections:**
1. **Load Results** - Show evaluation metrics
2. **Baseline Model Testing** - Demonstrate leakage with prompts
3. **DP Model Testing** - Show reduced leakage
4. **Comparison Visualization** - Privacy-utility trade-off plots
5. **Live Attack Testing** - Real-time PII extraction attempts

**Key Visualizations:**
- Comparison table (baseline vs DP models)
- Privacy-utility trade-off curves
- Leakage reduction percentages
- Real-world impact calculations

---

## SLIDE 10: HEALTHCARE IMPACT & KEY TAKEAWAYS

**WHY THIS MATTERS FOR HEALTHCARE AI**

**Real-World Application:**
- Hospitals deploying AI assistants
- HIPAA compliance requirements
- Patient data protection (PHI)

**Key Takeaways:**
1. ✅ **Privacy Risk is Real:** 17.79% baseline leakage
2. ✅ **DP Solution Works:** 98% reduction (→ 0.36% with ε=10.0)
3. ✅ **Trade-off is Manageable:** Acceptable for healthcare
4. ✅ **Best Configuration:** ε = 10.0 provides optimal balance

**Impact:** Protected ~1,742 additional records in 10,000 patient dataset

---

## SLIDE 11: LIMITATIONS & FUTURE WORK

**WHAT WE COULDN'T SOLVE (YET)**

**Current Limitations:**
- Utility loss: Perplexity increases (1.14 → 22.70)
- Synthetic dataset reduces realism
- Limited attack coverage (no jailbreak defense)
- Computational cost: Slower training

**Future Work:**
- Real medical datasets + larger LLMs
- Smarter DP mechanisms (adaptive clipping)
- Multi-layer defense (jailbreak protection)
- Federated learning integration

---

## SLIDE 12: CONCLUSION

**PRIVAI-LEAK: A PRACTICAL BLUEPRINT**

✅ Demonstrated real leakage (17.79% baseline)  
✅ Applied DP-SGD (reduced to 0.36% with ε=10.0)  
✅ Delivered end-to-end privacy audit pipeline  
✅ Quantified privacy-utility trade-offs  

> **LLMs don't just need better models, they need stronger privacy guarantees. PrivAI-Leak shows how to enforce them.**

**Thank You! Questions?**

---

## TIMING BREAKDOWN

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title | 0:30 |
| 2 | Problem | 0:45 |
| 3 | What We Observed | 0:45 |
| 4 | Our Approach | 1:00 |
| 5 | How We Broke It | 0:45 |
| 6 | How We Fixed It | 1:00 |
| 7 | Results | 1:30 |
| 8 | Trade-off | 1:00 |
| 9 | **Jupyter Demo** | **2:30** |
| 10 | Takeaways | 1:00 |
| 11 | Limitations | 0:45 |
| 12 | Conclusion | 0:30 |
| **TOTAL** | | **10:00** |

