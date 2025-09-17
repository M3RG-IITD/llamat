#!/bin/bash

# LLaMAT Streamlit Dashboard Launcher
# This script sets up and runs the Streamlit dashboard

echo "🧪 LLaMAT Downstream Evaluation Dashboard"
echo "========================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing requirements..."
    pip install -r requirements_streamlit.txt
fi

# Check if data file exists
DATA_FILE="./visualizations/downstream_compare_outputs/_downstream_eval.json"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    echo "Please ensure the evaluation data file exists."
    exit 1
fi

echo "✅ Data file found: $DATA_FILE"

# Run the dashboard
echo "🚀 Starting Streamlit dashboard..."
echo "The dashboard will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run streamlit_dashboard.py --server.port 8501 --server.address localhost
