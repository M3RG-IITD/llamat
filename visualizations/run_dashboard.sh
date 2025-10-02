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
DATA_FILE="./downstream_compare_outputs/_downstream_eval.json"
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    echo "Attempting to extract from zip file..."
    
    # Try to extract the zip file
    if [ -f "_downstream_eval.zip" ]; then
        echo "📦 Found _downstream_eval.zip, extracting..."
        unzip _downstream_eval.zip
        
        # Check if extraction was successful
        if [ -f "$DATA_FILE" ]; then
            echo "✅ Successfully extracted data file"
        else
            echo "❌ Extraction failed or data file still not found"
            echo "Please ensure _downstream_eval.zip contains the correct data file."
            exit 1
        fi
    else
        echo "❌ Neither data file nor _downstream_eval.zip found"
        echo "Please ensure either $DATA_FILE or _downstream_eval.zip exists."
        exit 1
    fi
fi

echo "✅ Data file found: $DATA_FILE"

# Run the dashboard
echo "🚀 Starting Streamlit dashboard..."
echo "The dashboard will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run streamlit_dashboard.py --server.port 8501 --server.address localhost
