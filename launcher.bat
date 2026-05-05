@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE="

if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_EXE=python"
        )
    )
)

if not defined PYTHON_EXE (
    echo Python was not found.
    echo Install Python or create a virtual environment in this project first.
    pause
    exit /b 1
)

echo Starting desktop dashboard...
%PYTHON_EXE% desktop_app.py

if errorlevel 1 (
    echo.
    echo Failed to start the desktop app.
    pause
)
