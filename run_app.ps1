# Run ProBot Enterprise Assistant
# This script starts the Streamlit application using the virtual environment

Write-Host "Starting ProBot Enterprise Assistant..." -ForegroundColor Cyan
Write-Host ""

# Run Streamlit using the virtual environment Python
& ".\venv\Scripts\streamlit.exe" run Home.py

