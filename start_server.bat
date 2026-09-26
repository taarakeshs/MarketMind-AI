@echo off
cd /d "%~dp0"
echo Starting Stock Market Intelligence System...
echo.

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo WARNING: .venv not found. Run: py -m venv .venv ^& .venv\Scripts\pip install -r requirements.txt
)

echo Dashboard will be available at: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server.
echo.
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
pause
