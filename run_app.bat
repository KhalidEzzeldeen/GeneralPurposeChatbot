@echo off
REM Run ProBot Enterprise Assistant
REM This script starts the Streamlit application using the virtual environment

echo Starting ProBot Enterprise Assistant...
echo.

REM Run Streamlit using the virtual environment Python
call "venv\Scripts\streamlit.exe" run Home.py

