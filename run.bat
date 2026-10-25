@echo off
setlocal EnableDelayedExpansion

title Cerberus - Startup

:: ============================================================
::  CERBERUS - One-Click Launcher
::  Automatically sets up the environment and starts the app.
:: ============================================================

echo.
echo  =====================================================
echo    CERBERUS Security System - Launcher
echo  =====================================================
echo.

:: ?? 1. Check Python ??????????????????????????????????????
echo  [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Python is not installed or not on PATH.
    echo          Please install Python 3.10+ from https://www.python.org/
    echo          and make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo         Found Python %PY_VER%

:: ?? 2. Create virtual environment if it does not exist ???
echo.
echo  [2/4] Setting up virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo         Creating venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo         venv created successfully.
) else (
    echo         venv already exists, skipping creation.
)

:: ?? 3. Activate venv ?????????????????????????????????????
call venv\Scripts\activate.bat

:: ?? 4. Install / update dependencies ?????????????????????
echo.
echo  [3/4] Installing dependencies (this may take a while on first run)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to install dependencies.
    echo          Check requirements.txt and your internet connection.
    pause
    exit /b 1
)
echo         Dependencies are up to date.

:: ?? 5. Launch Cerberus ???????????????????????????????????
echo.
echo  [4/4] Starting Cerberus...
echo.
echo  =====================================================
echo    Access the app at:  http://127.0.0.1:5000
echo    Press Ctrl+C to stop the server.
echo  =====================================================
echo.

:: Open the browser after a short delay (runs in background)
start "" cmd /c "timeout /t 3 >nul && start http://127.0.0.1:5000"

python app.py

:: ?? On exit ??????????????????????????????????????????????
echo.
echo  Cerberus has stopped.
pause
