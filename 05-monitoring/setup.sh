#!/bin/bash
# Setup script for MLOps Zoomcamp Module 5 - Monitoring

set -e  # Exit on error

echo "=========================================="
echo "MLOps Zoomcamp Module 5 - Monitoring Setup"
echo "=========================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.10+ first."
    exit 1
fi

echo "✓ Python found: $(python --version)"

# Create virtual environment
echo ""
echo "1️⃣  Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo "2️⃣  Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash)
    source .venv/Scripts/activate
else
    # Linux/Mac
    source .venv/bin/activate
fi

echo "✓ Virtual environment activated"

# Upgrade pip, setuptools, wheel
echo ""
echo "3️⃣  Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo ""
echo "4️⃣  Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate venv: source .venv/Scripts/activate  (Git Bash)"
echo "                  .venv\\Scripts\\activate       (PowerShell)"
echo "2. Start Docker:  docker compose up -d"
echo "3. Run notebook:  jupyter notebook homwork_monitoring.ipynb"
echo "4. Access UI:     http://127.0.0.1:8000 (after running baseline notebook)"
echo ""
