#!/bin/bash
# Script to apply all fixes and regenerate results

cd "$(dirname "$0")"

echo "🔧 Applying Fixes and Regenerating Results"
echo "=========================================="
echo ""

echo "Step 1: Regenerating data with new size (2,000 samples)..."
python src/healthcare_data_generator.py

echo ""
echo "Step 2: Retraining baseline with fixes (early stopping, dropout, regularization)..."
python src/baseline_training.py

echo ""
echo "Step 3: Retraining DP models with improved noise calculation..."
python src/dp_training_manual.py

echo ""
echo "Step 4: Running privacy attacks..."
python src/privacy_attacks.py

echo ""
echo "Step 5: Evaluating all models..."
python src/evaluation.py

echo ""
echo "Step 6: Generating visualizations..."
python src/visualization.py

echo ""
echo "✅ All fixes applied! Check results/ directory for updated results."
echo ""
echo "Expected improvements:"
echo "  - Baseline perplexity: 1.14 → 10-20"
echo "  - DP ε=0.5 perplexity: 9,643 → 30-60"
echo "  - Privacy protection: Still excellent (18% → 1% leakage)"

