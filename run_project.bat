@echo off
REM Store the directory where the batch file is located
set SCRIPT_DIR=%~dp0
REM Change directory to where this script is located
cd /d "%SCRIPT_DIR%"
REM Activate the virtual environment
call venv\Scripts\activate
REM Run the app with full path
python "%SCRIPT_DIR%run.py"
REM When the app ends, optionally pause to view output
pause