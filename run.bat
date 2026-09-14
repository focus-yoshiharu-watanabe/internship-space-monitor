@echo off
rem Keep this file ASCII-only.
cd /d %~dp0
set PYTHONUTF8=1
.venv\Scripts\python app.py %*
pause
