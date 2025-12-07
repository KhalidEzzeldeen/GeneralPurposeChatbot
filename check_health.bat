@echo off
REM Health Check Script for ProBot Enterprise Assistant
REM This script runs the health check using the virtual environment

echo Running ProBot Health Check...
echo.

REM Activate virtual environment and run health check
call "venv\Scripts\python.exe" health_check.py

REM Exit with the same code as the health check
exit /b %ERRORLEVEL%

