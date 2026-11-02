@echo off
cd /d "%~dp0"
echo Starting EchoBond Backend Server at http://localhost:8080
echo Caregiver PIN: 2468
start "" http://localhost:8080
python -m backend
pause
