@echo off
cd /d "%~dp0"
echo Installing Passport Photo Proper v6 dependencies...
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Installation complete.
echo Now double-click run_app.bat
pause
