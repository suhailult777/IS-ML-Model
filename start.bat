@echo off
echo ===============================================
echo Medical Document Conversational AI
echo ===============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

echo Starting Medical Document Conversational AI...
echo.

REM Run the launch script
python launch.py

echo.
echo Press any key to exit...
pause >nul
