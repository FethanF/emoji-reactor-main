@echo off
setlocal enabledelayedexpansion

:: Create and activate venv with Python 3.12
if not exist emoji_env (
    python -m venv emoji_env
)

call emoji_env\Scripts\activate

:: Install dependencies
pip install -r requirements.txt

:: Run the app
python emoji_reactor.py

pause

