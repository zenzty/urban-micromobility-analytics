@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================================
echo   Urban Micromobility, Weather ^& Transit Equity Dashboard Launcher
echo ============================================================================

REM Tier 1: Check for local uv.exe or system uv in PATH
set "UV_CMD="
if exist "%~dp0uv.exe" (
    set "UV_CMD=%~dp0uv.exe"
) else (
    where uv >nul 2>nul
    if !errorlevel! equ 0 (
        set "UV_CMD=uv"
    )
)

if defined UV_CMD (
    echo [INFO] Detected uv runtime: !UV_CMD!
    "!UV_CMD!" sync --locked
    if not exist "data\processed\trips_enriched.parquet" (
        echo [INFO] Building validated analytical Parquet tables...
        "!UV_CMD!" run python scripts\build_dataset.py
    )
    echo [INFO] Starting Panel server at http://localhost:5006/app ...
    "!UV_CMD!" run panel serve src\micromobility_analytics\ui\app.py --show
    goto :eof
)

REM Tier 2: Standard Python venv + pip fallback (no uv required)
echo [INFO] uv not found; falling back to standard Python virtual environment (.venv)...
if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Creating .venv with system Python...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create Python 3.12 virtual environment.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    ".venv\Scripts\python.exe" -m pip install -e . --no-deps
)

if not exist "data\processed\trips_enriched.parquet" (
    echo [INFO] Building validated analytical Parquet tables...
    ".venv\Scripts\python.exe" scripts\build_dataset.py
)

echo [INFO] Starting Panel server at http://localhost:5006/app ...
".venv\Scripts\python.exe" -m panel serve src\micromobility_analytics\ui\app.py --show
endlocal