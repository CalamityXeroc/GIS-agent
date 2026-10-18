@echo off
REM GIS Agent Launcher (ArcGIS Pro 3.6)
REM NOTE: keep this file ASCII-only. cmd.exe parses .bat in the ANSI codepage,
REM       so Chinese comments here can break parsing (verified).
REM       All user-facing text lives in scripts/check_install.py and the agent.

setlocal enabledelayedexpansion

title GIS Agent (ArcGIS Pro 3.6)

echo.
echo ========================================
echo   GIS Agent Launcher
echo   ArcGIS Pro 3.6 Python Runtime
echo ========================================
echo.

set "PYTHON_PATH=E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe"
set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

if not exist "%PYTHON_PATH%" (
    echo [ERROR] ArcGIS Pro Python not found
    echo Path: %PYTHON_PATH%
    echo.
    echo Please run install.bat first.
    pause
    exit /b 1
)

echo [OK] Python: %PYTHON_PATH%
echo.

REM ---------------------------------------------------------------------------
REM Install check: only run "pip install -e ." when it is actually needed.
REM
REM An editable install just drops a .pth file into site-packages pointing at
REM this repo's src\, so source changes take effect automatically. Re-running
REM pip on every launch re-resolves the whole dependency tree (langchain,
REM chromadb, litellm ...) and took tens of seconds to minutes.
REM Decision logic lives in scripts\check_install.py:
REM   - .pth missing
REM   - .pth points somewhere else
REM   - pyproject.toml newer than the .pth (deps may have changed)
REM Force a reinstall with:  REINSTALL=1 一键启动.bat   (see REINSTALL_FLAG below)
set "REINSTALL_FLAG="
if "%REINSTALL%"=="1" set "REINSTALL_FLAG=--reinstall"
REM ---------------------------------------------------------------------------
"%PYTHON_PATH%" "%REPO_ROOT%\scripts\check_install.py" --repo "%REPO_ROOT%" --quiet %REINSTALL_FLAG%
if errorlevel 1 (
    echo [INFO] Installing / updating gis_cli package...
    "%PYTHON_PATH%" -m pip install -e "%REPO_ROOT%" --quiet
    if errorlevel 1 (
        echo [ERROR] Package install failed
        echo.
        echo Please run manually:
        echo   "%PYTHON_PATH%" -m pip install -e .
        pause
        exit /b 1
    )
    echo [OK] Package installed
) else (
    echo [OK] Package ready ^(editable install, source changes auto-apply^)
)
echo.

REM Optional-dependency guard: a missing package silently degrades the agent
REM (kernel falls back to one-shot subprocess, no input box), so install it.
REM Only "import" is checked - takes well under a second.
call :ensure openai "openai"
call :ensure prompt_toolkit "prompt-toolkit>=3.0.43"
call :ensure jupyter_client "jupyter-client>=8.0.0"
call :ensure PIL "pillow>=10.0.0"
call :ensure numpy "numpy>=1.26"
echo.

echo Starting GIS Agent...
echo.
echo ----------------------------------------
echo Notes:
echo   - Put input data into workspace\input\
echo   - Type /help for commands, Esc to interrupt
echo   - Use --continue to resume the last session
echo   - While running: type to queue a follow-up, merged into the same task
echo ----------------------------------------
echo.

"%PYTHON_PATH%" -m gis_cli.agent.cli chat --workspace "%REPO_ROOT%\workspace" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Startup failed
    pause
)
endlocal
exit /b 0

REM --- subroutine: install a package only if its import is missing -----------
:ensure
"%PYTHON_PATH%" -c "import %~1" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing %~1 ...
    "%PYTHON_PATH%" -m pip install "%~2" --quiet
    if errorlevel 1 (
        echo [WARN] Could not install %~1 (startup continues, related feature degrades)
    ) else (
        echo [OK] %~1 installed
    )
)
exit /b 0
