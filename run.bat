@echo off
echo Starting AGRIVISION Plant Detection System...
echo Launching web server on http://localhost:8080
cd /d "%~dp0"
python -m http.server 8080
