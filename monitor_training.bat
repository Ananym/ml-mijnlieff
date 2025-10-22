@echo off
REM Monitor training and log status every 5 minutes
set LOGFILE=training_monitor.txt
echo Training Monitor Started at %date% %time% > %LOGFILE%
:loop
timeout /t 300 /nobreak > nul
echo. >> %LOGFILE%
echo === Check at %date% %time% === >> %LOGFILE%
tail -20 nul 2>&1 || echo "Training output check" >> %LOGFILE%
goto loop
