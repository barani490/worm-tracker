@echo off
REM Double-click this file to launch the C. elegans Tracker GUI.
REM It must sit in the same folder as tracker_gui.py and celegans_tracker.py.

cd /d "%~dp0"

echo Checking for pygame...
python -c "import pygame" 2>nul
if errorlevel 1 (
    echo pygame not found -- installing it now...
    pip install pygame
)

echo Launching GUI...
python tracker_gui.py

echo.
echo ----------------------------------------
echo The GUI window closed. If you saw an error
echo above, that's what needs fixing.
echo ----------------------------------------
pause
