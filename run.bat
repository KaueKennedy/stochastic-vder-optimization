@echo off
setlocal
title DER Optimization Dashboard Launcher

:: ==============================================================================
:: USER CONFIGURATION - PLEASE READ
:: ==============================================================================
:: IMPORTANT: You must change the path below to match the location of your 
:: Python executable. If you are using the portable version in Documents, 
:: ensure the path below is correct.
:: ==============================================================================
set "PROJECT_ROOT=%~dp0"
set "VENV_NAME=venv310"
set "PYTHON_EXE=%PROJECT_ROOT%%VENV_NAME%\Scripts\python.exe"


:: Internal relative paths (do not change unless you move the script files)
set "REQ_FILE=requirements.txt"
set "DASH_FILE=code\dashboard_batch.py"
set "CHARTS_FILE=code\visualizer.py"

:: Set working directory to the folder where this .bat file is located
cd /d "%~dp0"

echo ======================================================
echo       DER OPTIMIZATION SYSTEM - STARTUP
echo ======================================================
echo.

:: Check if the Python executable exists at the specified path
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found at: %PYTHON_EXE%
    echo.
    echo Please open this .bat file in Notepad and update the 
    echo PYTHON_EXE variable to point to your python.exe location.
    echo.
    pause
    exit /b
)

:: Ask the user if they want to update dependencies
echo The system is using the portable environment at: 
echo %PYTHON_EXE%
echo.
set /p install="Do you want to verify/install libraries from requirements.txt? (Y/N): "

if /i "%install%"=="Y" (
    echo.
    echo [INFO] Activating environment and updating packages...
    :: Running pip through the specific python.exe ensures the "venv" context
    "%PYTHON_EXE%" -m pip install --upgrade pip
    "%PYTHON_EXE%" -m pip install -r "%REQ_FILE%"
    if %errorlevel% neq 0 (
        echo [WARNING] There was an issue during installation. 
        echo Check your internet connection or file permissions.
    )
) else (
    echo [INFO] Skipping installation. Starting with current libraries.
)

echo.
echo [INFO] Launching Streamlit Dashboard...
echo Project Root: %cd%
echo.

:: Run the dashboard using the portable Python context
start /min "Dashboard Streamlit" "%PYTHON_EXE%" -m streamlit run "%DASH_FILE%"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The dashboard closed unexpectedly. Check the logs above.
    timeout /t 12 /nobreak > nul
    pause
)

timeout /t 12 /nobreak > nul

:: Run the Vizualizer using the portable Python context
start /min "Visualizer Streamlit" "%PYTHON_EXE%" -m streamlit run "%CHARTS_FILE%"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The visualizer closed unexpectedly. Check the logs above.
    pause
)

timeout /t 5 /nobreak > nul
exit