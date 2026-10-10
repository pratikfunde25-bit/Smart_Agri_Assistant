@echo off
title Smart Agri Assistant Server
echo ======================================================
echo           Starting Smart Agri Assistant
echo ======================================================
echo.

cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

IF EXIST ".venv\Scripts\python.exe" (
    echo [OK] Using Python from .venv virtual environment...
    echo [*] Starting server at http://127.0.0.1:5000
    echo [*] Press Ctrl+C to stop the server anytime.
    echo.
    start http://127.0.0.1:5000
    ".venv\Scripts\python.exe" run_production.py
) ELSE (
    echo [!] Virtual environment .venv not found. Falling back to system python...
    start http://127.0.0.1:5000
    python run_production.py
)

pause
