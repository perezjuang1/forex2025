@echo off
setlocal enableextensions enabledelayedexpansion

:: Establece directorio base (donde está el .bat)
set "BASE_DIR=%~dp0"

:: Crea carpeta de logs si no existe
if not exist "%BASE_DIR%logs" mkdir "%BASE_DIR%logs"

:: Nombre del log con fecha
for /f "tokens=2-4 delims=/ " %%a in ("%date%") do (
    set "LOG_DATE=%%c-%%a-%%b"
)
set "LOG_FILE=%BASE_DIR%logs\execution_%LOG_DATE%.log"

:: Función para escribir timestamp en log
echo ------------------------------------------------------------ >> "%LOG_FILE%"
echo [%date% %time%] Iniciando ejecución >> "%LOG_FILE%"

:: Ejecuta los comandos y redirige stdout/stderr al log (append)
echo Running: env1\Scripts\python.exe -m pip install -r requirements.txt >> "%LOG_FILE%"
call "%BASE_DIR%env1\Scripts\python.exe" -m pip install -r "%BASE_DIR%requirements.txt" >> "%LOG_FILE%" 2>&1

echo Running: env1\Scripts\python.exe -m ensurepip --upgrade >> "%LOG_FILE%"
call "%BASE_DIR%env1\Scripts\python.exe" -m ensurepip --upgrade >> "%LOG_FILE%" 2>&1

echo Running: env1\Scripts\python.exe -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%"
call "%BASE_DIR%env1\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1

:RESTART_LOOP
echo Running: env1\Scripts\python.exe TradingSystem.py >> "%LOG_FILE%"
echo [%date% %time%] Starting TradingSystem >> "%LOG_FILE%"
call "%BASE_DIR%env1\Scripts\python.exe" "%BASE_DIR%TradingSystem.py" >> "%LOG_FILE%" 2>&1

echo [%date% %time%] TradingSystem stopped. Restarting in 5 seconds... >> "%LOG_FILE%"
timeout /t 5 /nobreak
goto RESTART_LOOP

:: Fin (nunca se alcanza a menos que se cancele manualmente)
echo [%date% %time%] Ejecución finalizada >> "%LOG_FILE%"
echo ------------------------------------------------------------ >> "%LOG_FILE%"
echo Logs guardados en "%LOG_FILE%"
endlocal
pause
