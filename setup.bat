@echo off
rem Windows setup. Keep this file ASCII-only (Japanese text breaks cmd parsing).
cd /d %~dp0

echo === 1/3 Creating Python virtual environment ===
python -m venv .venv
if errorlevel 1 goto no_python

echo === 2/3 Installing libraries (takes a few minutes) ===
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

echo === 3/3 Checking AI model ===
.venv\Scripts\python -c "from detector import Detector; print('model OK:', Detector('models/yolox_tiny.onnx').device)"
if errorlevel 1 goto install_failed

echo.
echo DONE. Next, double-click run.bat
pause
exit /b 0

:no_python
echo [FAILED] Python not found. See README.
pause
exit /b 1

:install_failed
echo [FAILED] Install failed. Please call the mentor.
pause
exit /b 1
