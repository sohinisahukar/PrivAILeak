#!/bin/bash
# Script to start Jupyter notebook for PrivAI-Leak demo

cd "$(dirname "$0")"
source venv/bin/activate

echo "🚀 Starting Jupyter Notebook..."
echo "📓 Opening: Demo_Presentation.ipynb"
echo ""
echo "The notebook will open in your browser."
echo "Press Ctrl+C to stop the server."
echo ""

# Start Jupyter and open the demo notebook
jupyter notebook Demo_Presentation.ipynb

