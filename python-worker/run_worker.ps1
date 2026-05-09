$ErrorActionPreference = "Stop"

# Try to find python3.11 or fallback to python
$pythonCmd = "python"
$appDataPython = "C:\Users\tengah\AppData\Local\Python\bin\python3.11.exe"
if (Test-Path $appDataPython) {
    $pythonCmd = $appDataPython
} elseif (Get-Command "python3.11" -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3.11"
}

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment using $pythonCmd..."
    & $pythonCmd -m venv venv
    
    if (-not $?) {
        Write-Host "Failed to create venv. Please ensure Python 3.11 is installed and added to your PATH." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Activating virtual environment..."
. .\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..."
pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from .env.example..."
    Copy-Item .env.example .env
}

Write-Host "Starting Python AI Worker..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000
