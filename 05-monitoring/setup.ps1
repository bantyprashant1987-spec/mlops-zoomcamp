# Setup script for MLOps Zoomcamp Module 5 - Monitoring (PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "MLOps Zoomcamp Module 5 - Monitoring Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Host "❌ Python is not installed. Please install Python 3.10+ first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "1️⃣  Creating virtual environment..." -ForegroundColor Yellow
python -m venv .venv

# Activate virtual environment
Write-Host "2️⃣  Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip, setuptools, wheel
Write-Host ""
Write-Host "3️⃣  Upgrading pip, setuptools, wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install requirements
Write-Host ""
Write-Host "4️⃣  Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Activate venv: .\.venv\Scripts\Activate.ps1"
Write-Host "2. Start Docker:  docker compose up -d"
Write-Host "3. Run notebook:  jupyter notebook homwork_monitoring.ipynb"
Write-Host "4. Access UI:     http://127.0.0.1:8000 (after running baseline notebook)"
Write-Host ""
