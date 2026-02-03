@echo off

echo ============================================================
echo Bloomberg Data Downloader
echo ============================================================
echo.

REM Cambia directory al percorso dello script
cd /d "%~dp0"

REM Attiva il virtual environment
call C:\Documenti\quantenv\Scripts\activate.bat

REM Chiedi all'utente di inserire un ticker
set /p user_input="Inserisci il ticker: "

REM Rimuovi solo spazi iniziali e finali (preserva spazi interni)
:trim_start
if "%user_input:~0,1%"==" " set "user_input=%user_input:~1%" && goto trim_start
:trim_end
if "%user_input:~-1%"==" " set "user_input=%user_input:~0,-1%" && goto trim_end

REM Stampa l'input per debug

REM Esegui download_or_update passando il ticker come variabile d'ambiente
set TICKER=%user_input%
python -c "import os; from datafetch.bbg.downloader import download_or_update; ticker = os.environ.get('TICKER', ''); download_or_update(ticker)"

echo.
echo ============================================================
echo Premi un tasto per chiudere...
echo ============================================================
pause >nul
