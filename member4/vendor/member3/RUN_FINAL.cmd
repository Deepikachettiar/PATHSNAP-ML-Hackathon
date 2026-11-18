@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (python final.py) else (py -3 final.py)
if errorlevel 1 goto failed
start "" "Member3_Dashboard.html"
exit /b 0
:failed
echo Run requires Python 3.9 or newer. See the error above.
pause
exit /b 1
