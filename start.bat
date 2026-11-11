@echo off
cd /d "%~dp0"
if not exist .venv (
    echo Eerste keer: installeren, dit duurt een paar minuten...
    py -3.12 -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python app.py
pause
