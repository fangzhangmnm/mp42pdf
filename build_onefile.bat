@echo off
setlocal

set "PY=py -3"
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 set "PY=python"

echo Installing build dependencies...
%PY% -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :error

echo Cleaning old build outputs...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Building onefile EXE...
%PY% -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name mp42pdf ^
  --hidden-import img2pdf ^
  --collect-all pikepdf ^
  mp42pdf.py
if errorlevel 1 goto :error

if not exist release\windows mkdir release\windows
move /y dist\mp42pdf.exe release\windows\mp42pdf.exe >nul
if errorlevel 1 goto :error

echo.
echo Build complete:
echo   release\windows\mp42pdf.exe
exit /b 0

:error
echo.
echo Build failed.
exit /b 1
