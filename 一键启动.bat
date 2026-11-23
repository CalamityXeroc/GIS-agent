@echo off
REM GIS Agent Launcher (ArcGIS Pro)
REM NOTE: keep this file ASCII-only. cmd.exe parses .bat in the ANSI codepage,
REM       so Chinese comments here can break parsing (verified).
REM NOTE: never hardcode the Pro version here -- 3.6 -> 3.7 upgrade broke
REM       this launcher with a stale path. All user-facing text lives in
REM       scripts/check_install.py and the agent.

setlocal enabledelayedexpansion

title GIS Agent (ArcGIS Pro)

echo.
echo ========================================
echo   GIS Agent Launcher
echo   ArcGIS Pro Python Runtime
echo ========================================
echo.

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

REM ---------------------------------------------------------------------------
REM Locate the ArcGIS Pro Python interpreter.
REM
REM Discovery is shared with scripts\giscli.cmd via scripts\find_pro_python.cmd,
REM so the one-click launcher and the global "giscli" command can never pick
REM different Pythons (two copies of this logic already drifted once: the
REM 3.6 -> 3.7 upgrade left a stale hardcoded path here).
REM
REM Precedence: ARCGIS_PRO_PYTHON -> E:\ArcGISPro* -> D:\ArcGISPro* ->
REM             C:\Program Files\ArcGIS\Pro*
REM The registry is deliberately NOT trusted: after an in-place upgrade
REM InstallDir / PythonCondaRoot still point at the removed old version
REM (verified on this machine: registry says 3.6, Pro lives in 3.7).
REM ---------------------------------------------------------------------------
call "%REPO_ROOT%\scripts\find_pro_python.cmd"
if errorlevel 1 (
    echo [ERROR] ArcGIS Pro Python not found
    echo Expected: ^<ProDir^>\%PYTHON_REL%
    echo Tried:    ARCGIS_PRO_PYTHON, E:\ArcGISPro*, D:\ArcGISPro*, C:\Program Files\ArcGIS\Pro*
    echo.
    echo Please run:  python install.py
    echo Or set:      set ARCGIS_PRO_PYTHON=^<ProDir^>\bin\Python\envs\arcgispro-py3\python.exe
    pause
    exit /b 1
)

echo [OK] Python: %PYTHON_PATH%

REM Make this repo importable without any install. The Pro env pip is broken
REM (BuildDependencyInstallError) and the pip-generated gis-agent.exe in that
REM env is broken too (every invocation exits 1 with no output), so nothing
REM here may depend on either. scripts\giscli.cmd sets the same PYTHONPATH,
REM and scripts\check_install.py treats "src already importable" as ready --
REM that is what keeps this launcher out of pip on every start.
if defined PYTHONPATH (
    set "PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%REPO_ROOT%\src"
)
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
REM Force a reinstall with:  set REINSTALL=1   then run this launcher again.
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
REM pyproj: preview reprojects non-4326 layers; without it they vanish.
call :ensure pyproj "pyproj>=3.6"
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
        echo [WARN] Could not install %~1 ^(startup continues, related feature degrades^)
    ) else (
        echo [OK] %~1 installed
    )
)
exit /b 0
