@echo off
setlocal

title Cerberus - Build desktop app

:: ============================================================
::  Builds dist\Cerberus\Cerberus.exe with PyInstaller.
::  Run run.bat once first so the venv and dependencies exist.
::
::  The build bundles whichever PyTorch is installed in the venv:
::    - CUDA build  -> works on NVIDIA GPUs, folder is several GB
::    - CPU build   -> works everywhere (CPU only), much smaller
:: ============================================================

set "VENV="
if exist "venv\Scripts\activate.bat" set "VENV=venv"
if not defined VENV if exist ".venv\Scripts\activate.bat" set "VENV=.venv"
if not defined VENV (
    echo  [ERROR] No venv or .venv found. Run run.bat once first.
    pause
    exit /b 1
)
echo  Using %VENV%
call %VENV%\Scripts\activate.bat

echo  [1/3] Installing PyInstaller...
pip install "pyinstaller>=6.10"
if %errorlevel% neq 0 (
    echo  [ERROR] Could not install PyInstaller.
    pause
    exit /b 1
)

echo  [2/3] Building the app (this can take 10+ minutes)...
pyinstaller cerberus.spec --noconfirm --clean
if %errorlevel% neq 0 (
    echo  [ERROR] Build failed. See the messages above.
    pause
    exit /b 1
)

echo.
echo  App built: dist\Cerberus\Cerberus.exe

:: Optional: Windows installer with Inno Setup 6 (https://jrsoftware.org/isdl.php)
set "ISCC="
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC (
    echo  Inno Setup not found, so no installer was made.
    echo  Ship the whole dist\Cerberus folder, or install Inno Setup 6 and run this again.
    pause
    exit /b 0
)
echo.
echo  [3/3] Building the installer (compressing several GB takes a while)...
"%ISCC%" installer.iss
if %errorlevel% neq 0 (
    echo  [ERROR] Installer build failed. See the messages above.
    pause
    exit /b 1
)
echo.
echo  Done: installer\CerberusSetup.exe
pause
