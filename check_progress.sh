#!/bin/bash
# Quick script to check pipeline progress

cd "$(dirname "$0")"

echo "🔍 Pipeline Progress Check"
echo "=========================="
echo ""

# Check if pipeline is running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Pipeline is RUNNING"
    echo ""
    
    # Show last few lines of log
    echo "📋 Recent Activity:"
    tail -5 training_output.log 2>/dev/null | grep -E "(Step|Epoch|Testing|Complete|✅|❌)" || tail -3 training_output.log 2>/dev/null
    
    echo ""
    echo "📊 Completed Steps:"
    
    # Check what's been completed
    if [ -f "models/baseline_model/model.safetensors" ]; then
        echo "  ✅ Step 2: Baseline Training"
    else
        echo "  ⏳ Step 2: Baseline Training (in progress)"
    fi
    
    if [ -f "models/baseline_attack_results.json" ]; then
        echo "  ✅ Step 3: Privacy Attacks (Baseline)"
    else
        echo "  ⏳ Step 3: Privacy Attacks (in progress)"
    fi
    
    if [ -f "models/dp_model_eps_0.5/model.safetensors" ]; then
        echo "  ✅ Step 4: DP Training (at least ε=0.5)"
    else
        echo "  ⏳ Step 4: DP Training (not started)"
    fi
    
    if [ -f "results/evaluation_results.json" ]; then
        echo "  ✅ Step 5: Evaluation"
    else
        echo "  ⏳ Step 5: Evaluation (not started)"
    fi
    
    if [ -f "results/privacy_budget_vs_leakage.png" ]; then
        echo "  ✅ Step 6: Visualization"
    else
        echo "  ⏳ Step 6: Visualization (not started)"
    fi
    
else
    echo "⚠️  Pipeline is NOT running"
    echo ""
    echo "Check if it completed or crashed:"
    tail -20 training_output.log 2>/dev/null
fi

echo ""
echo "💡 To see full log: tail -f training_output.log"

