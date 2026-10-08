@echo off
rem LawLens launcher: interactive Q&A (empty line to quit).
rem Usage: double-click, or  run.bat "question"  for a single question.
setlocal
cd /d "%~dp0"
rem <nul: chcp and child processes would otherwise eat redirected input
chcp 65001 <nul >nul
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TRANSFORMERS_VERBOSITY=error

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

"%PY%" -c "import torch, transformers" <nul >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Required packages are missing. Run setup.bat first.
    pause
    exit /b 1
)

echo Loading models... first run downloads about 10GB into hf_cache.
"%PY%" ask.py %*
pause
