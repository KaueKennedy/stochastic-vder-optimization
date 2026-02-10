@echo off
setlocal
title DER Optimization Launcher

:: Change to script directory (supports network/UNC paths)
pushd "%~dp0"

:: ==============================================================================
:: CONFIGURATION
:: ==============================================================================
set "VENV_NAME=venv310"
set "PYTHON_EXE=venv310\python.exe"
set "REQ_FILE=requirements.txt"
set "DASH_FILE=code\dashboard_batch.py"
set "CHARTS_FILE=code\visualizer.py"
set "BROWSER_EXE=FirefoxPortable.exe"

echo ======================================================
echo       DER OPTIMIZATION SYSTEM
echo ======================================================
echo.

:: 1. CHECK FILES
if not exist "%DASH_FILE%" (
    echo [ERROR] Dashboard file not found: %DASH_FILE%
    echo Please ensure the 'code' folder is next to this script.
    pause
    exit /b
)

:: 2. CHECK PYTHON ENVIRONMENT
if exist "%PYTHON_EXE%" goto :CHECK_DEPS

echo [WARNING] Python environment not found.
set /p install_opt="Install Portable Python now? (Y/N): "
if /i not "%install_opt: =%"=="Y" exit /b

:INSTALL_PYTHON
echo.
echo [1/6] Downloading Python 3.10...
curl -L -o python_embed.zip "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"

echo [2/6] Extracting Python...
if exist "%VENV_NAME%" rmdir /s /q "%VENV_NAME%"
powershell -Command "Expand-Archive -Path python_embed.zip -DestinationPath %VENV_NAME%"
del python_embed.zip

echo [3/6] Configuring Python...
echo import site>> "%VENV_NAME%\python310._pth"

echo [4/6] Installing PIP...
curl -L -o get-pip.py "https://bootstrap.pypa.io/get-pip.py"
"%PYTHON_EXE%" get-pip.py
del get-pip.py

echo [5/6] Installing base tools...
"%PYTHON_EXE%" -m pip install --no-warn-script-location wheel setuptools

echo [6/6] Downloading Firefox Portable...
curl -L -o firefox.zip "https://download.mozilla.org/?product=firefox-latest-ssl&os=win64&lang=en-US"
powershell -Command "Expand-Archive -Path firefox.zip -DestinationPath browser_temp"
copy "browser_temp\firefox.exe" "%BROWSER_EXE%"
rmdir /s /q browser_temp 2>nul
del firefox.zip

echo [SUCCESS] Python + Firefox Portable ready!
set "FRESH_INSTALL=1"

:CHECK_DEPS
:: Add venv to local PATH
set "PATH=%~dp0%VENV_NAME%;%~dp0%VENV_NAME%\Scripts;%PATH%"

:: --- CONFIGURE STREAMLIT TO SKIP EMAIL ---
if not exist "%UserProfile%\.streamlit" mkdir "%UserProfile%\.streamlit"
(
echo [general]
echo email = ""
echo [browser]
echo gatherUsageStats = false
) > "%UserProfile%\.streamlit\config.toml"

if "%FRESH_INSTALL%"=="1" goto :INSTALL_REQS

echo.
set /p check="Check requirements.txt? (Y/N): "
if /i "%check: =%"=="Y" goto :INSTALL_REQS
goto :RUN_APPS

:INSTALL_REQS
echo [INFO] Installing libraries...
"%PYTHON_EXE%" -m pip install --upgrade pip
"%PYTHON_EXE%" -m pip install --prefer-binary -r "%REQ_FILE%"

:: --- LINHA NOVA PARA O CPLEX ---
if exist "cplex_lib" (
    echo [INFO] Installing local CPLEX engine...
    "%PYTHON_EXE%" -m pip install ".\cplex_lib"
) else (
    echo [WARNING] cplex_lib folder not found. Optimization might be limited.
)
:: -------------------------------

:RUN_APPS

:RUN_APPS
echo.
echo [INFO] Starting applications...

:: Start Dashboard (fixed port 8501)
start "Dashboard" /D "." cmd /k "venv310\python.exe -m streamlit run code\dashboard_batch.py --server.port 8501 --server.headless true"

timeout /t 8 /nobreak > nul

:: Start Visualizer (fixed port 8502)  
start "Visualizer" /D "." cmd /k "venv310\python.exe -m streamlit run code\visualizer.py --server.port 8502 --server.headless true"

timeout /t 5 /nobreak > nul

:: ==============================================================================
:: OPEN FIREFOX PORTABLE AUTOMATICALLY
:: ==============================================================================
if exist "%BROWSER_EXE%" (
    echo [INFO] Opening apps in Firefox Portable...
    start "" "%BROWSER_EXE%" "http://localhost:8501"
    timeout /t 3 /nobreak > nul
    start "" "%BROWSER_EXE%" "http://localhost:8502"
    echo.
    echo [SUCCESS] Dashboard:     http://localhost:8501
    echo [SUCCESS] Visualizer:    http://localhost:8502
    echo [INFO] Firefox Portable opened automatically.
) else (
    echo [WARNING] Firefox Portable not found.
    echo [INFO] Open manually:
    echo Dashboard: http://localhost:8501
    echo Visualizer: http://localhost:8502
)

echo.
echo [DONE] System ready. Close this window when finished.
popd
pause
exit
