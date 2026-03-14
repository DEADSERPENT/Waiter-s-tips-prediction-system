@echo off
echo ============================================================
echo  Waiter's Tips Prediction System - Web App Launcher
echo ============================================================
echo.
echo Installing/updating dependencies...
pip install --upgrade setuptools wheel --quiet
pip install -r requirements.txt --quiet
echo.
echo Launching Streamlit Web App...
echo Open your browser at: http://localhost:8501
echo.
set SCRIPTS_DIR=%APPDATA%\Python\Python314\Scripts
set PATH=%SCRIPTS_DIR%;%PATH%
streamlit run app.py
pause
