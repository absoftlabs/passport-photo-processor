@echo off
cd /d "%~dp0"
echo Building EXE launcher...
python -m pip install pyinstaller
pyinstaller --onefile --name PassportPhotoProperV14 launcher.py
echo.
echo Done. Check the dist folder.
pause
