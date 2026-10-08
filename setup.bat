@echo off
rem LawLens first-time setup: creates .venv, installs PyTorch + packages, runs tests.
rem Messages are ASCII on purpose (cmd may not decode Korean in .bat files).
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10 or newer: https://www.python.org/downloads/
    echo         Check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.10 or newer is required.
    python --version
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip >nul

rem PyTorch 2.7.1 + CUDA 11.8 is the combination verified on the development PC.
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [2/4] No NVIDIA GPU found. Installing CPU PyTorch - answers will be very slow.
    "%PY%" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
) else (
    echo [2/4] NVIDIA GPU found. Installing PyTorch with CUDA 11.8 - about 2.5GB.
    "%PY%" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
)
if errorlevel 1 (
    echo [ERROR] PyTorch install failed.
    pause
    exit /b 1
)

echo [3/4] Installing other packages ...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Package install failed.
    pause
    exit /b 1
)

echo [4/4] Running tests ...
"%PY%" -m unittest
if errorlevel 1 (
    echo [WARN] Some tests failed. See the output above.
    pause
    exit /b 1
)

echo.
echo Setup complete. Double-click run.bat to start.
echo The first run downloads the models - about 10GB - into the hf_cache folder.
pause
