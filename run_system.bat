@echo off
echo ============================================================
echo   WAITER'S TIPS PREDICTION SYSTEM - EXECUTION
echo ============================================================
echo.

cd src

echo Running complete pipeline...
echo This will:
echo   - Download dataset
echo   - Preprocess data
echo   - Train all models
echo   - Generate visualizations
echo   - Make predictions
echo.

python main.py

echo.
echo ============================================================
echo   EXECUTION COMPLETE!
echo ============================================================
echo.
echo Check the following folders for results:
echo   - data\       (dataset)
echo   - models\     (trained models)
echo   - results\    (visualizations)
echo.
pause
