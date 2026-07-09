#!/bin/bash
echo "======================================"
echo "  HR Intelligence Dashboard"
echo "======================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install flask scikit-learn xgboost imbalanced-learn openpyxl pandas numpy -q

echo ""
echo "🚀 Starting HR Dashboard..."
echo "🌐 Open your browser at: http://localhost:5050"
echo ""
python3 app.py
