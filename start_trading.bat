@echo off
echo ========================================
echo    FOREX TRADING SYSTEM LAUNCHER
echo ========================================

echo [1/4] Activating virtual environment...
call env1\Scripts\activate.bat

echo [2/4] Installing/upgrading dependencies...
pip install -r requirements.txt --quiet

echo [3/4] Clearing old logs...
if exist logs\*.log del /q logs\*.log
if exist logs\*.csv del /q logs\*.csv

echo [4/4] Starting Trading System...
echo ========================================
echo    SYSTEM STARTING - CHECK LOGS
echo ========================================

python TradingSystem.py

pause
