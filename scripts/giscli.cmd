@echo off
REM ---------------------------------------------------------------------------
REM giscli -- run the GIS Agent CLI inside the ArcGIS Pro Python.
REM
REM Keep this file ASCII-only: cmd.exe parses .bat in the ANSI codepage, so
REM Chinese comments here can break parsing (verified).
REM
REM Why a .cmd and not a pip console script:
REM   - the pip-generated gis-agent.exe inside the Pro env is broken (every
REM     invocation exits 1 with no output), and the Pro env pip itself fails
REM     with BuildDependencyInstallError, so nothing here may depend on either;
REM   - PYTHONPATH=<repo>\src makes the repo importable without any install.
REM
REM Discovery lives in find_pro_python.cmd and is shared with the repo-root
REM one-click launcher (so the two can never drift apart).
REM ---------------------------------------------------------------------------

setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
REM Repo root from this file's location -- never hardcode the repo path.
for %%I in ("%SCRIPT_DIR%\..") do set "REPO_ROOT=%%~fI"

call "%SCRIPT_DIR%\find_pro_python.cmd"
if errorlevel 1 (
    echo [ERROR] ArcGIS Pro Python not found.
    echo Expected: ^<ProDir^>\%PYTHON_REL%
    echo Tried:    ARCGIS_PRO_PYTHON, E:\ArcGISPro*, D:\ArcGISPro*, C:\Program Files\ArcGIS\Pro*
    echo.
    echo Point it at your install and retry:
    echo   set ARCGIS_PRO_PYTHON=^<ProDir^>\bin\Python\envs\arcgispro-py3\python.exe
    echo.
    echo The registry is not consulted on purpose: after an in-place upgrade it
    echo still names the removed old version.
    exit /b 1
)

REM Prepend, do not replace: the user may already rely on their own PYTHONPATH.
if defined PYTHONPATH (
    set "PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%REPO_ROOT%\src"
)

"%PYTHON_PATH%" -m gis_cli.agent.cli %*
exit /b %ERRORLEVEL%
