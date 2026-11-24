@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe train_model.py
if errorlevel 1 goto failed
.venv\Scripts\python.exe app.py
goto end
:failed
echo Setup or application startup failed. Review the error shown above.
pause
:end
endlocal
