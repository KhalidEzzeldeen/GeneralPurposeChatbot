# Health Check Script for ProBot Enterprise Assistant
# This script runs the health check using the virtual environment

Write-Host "Running ProBot Health Check..." -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment and run health check
& ".\venv\Scripts\python.exe" health_check.py

# Exit with the same code as the health check
exit $LASTEXITCODE

