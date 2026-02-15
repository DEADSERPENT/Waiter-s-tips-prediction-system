@echo off
echo ============================================================
echo   WAITER'S TIPS PREDICTION SYSTEM - INSTALLATION
echo ============================================================
echo.

echo Step 1: Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)
echo.

echo Step 2: Installing required packages...
echo This may take a few minutes...
echo.
pip install numpy pandas matplotlib seaborn scikit-learn jupyter notebook joblib scipy
echo.

echo ============================================================
echo   INSTALLATION COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo   1. Run: run_system.bat (to execute complete pipeline)
echo   2. Or run: python src\main.py
echo.
pause
