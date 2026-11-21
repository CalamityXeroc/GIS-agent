@echo off
REM ---------------------------------------------------------------------------
REM Shared ArcGIS Pro Python discovery.
REM
REM Used by BOTH entry points -- scripts\giscli.cmd and the repo-root one-click
REM launcher -- so the two can never drift apart. Keep this file ASCII-only:
REM cmd.exe parses .bat in the ANSI codepage, so non-ASCII comments break
REM parsing (verified).
REM
REM Sets PYTHON_PATH. Returns 0 when found, 1 when not.
REM Deliberately NO setlocal here: the caller must see PYTHON_PATH.
REM
REM Precedence (same spirit as gis_cli.arcpy_bridge.discover_arcgis_pro_python):
REM   1) ARCGIS_PRO_PYTHON         explicit override, wins unconditionally
REM   2) E:\ArcGISPro*             shallow glob, first drive wins
REM   3) D:\ArcGISPro*
REM   4) C:\Program Files\ArcGIS\Pro*
REM
REM The registry is deliberately NOT trusted: after an in-place upgrade
REM InstallDir / PythonCondaRoot still point at the removed old version
REM (verified on this machine: registry says 3.6, Pro lives in 3.7).
REM
REM Glob order is ascending lexicographic, so the last hit per drive is the
REM highest version. Known limit: 3.10 would sort before 3.7 -- acceptable for a
REM fallback only; use ARCGIS_PRO_PYTHON when that day comes.
REM ---------------------------------------------------------------------------

set "PYTHON_PATH="
if not defined PYTHON_REL set "PYTHON_REL=bin\Python\envs\arcgispro-py3\python.exe"

REM An entry that does not exist is ignored, not fatal -- same as arcpy_bridge,
REM which skips a stale explicit candidate and keeps looking.
if defined ARCGIS_PRO_PYTHON if exist "%ARCGIS_PRO_PYTHON%" set "PYTHON_PATH=%ARCGIS_PRO_PYTHON%"
if defined PYTHON_PATH exit /b 0

for /d %%D in ("E:\ArcGISPro*") do if exist "%%~D\%PYTHON_REL%" set "PYTHON_PATH=%%~D\%PYTHON_REL%"
if defined PYTHON_PATH exit /b 0

for /d %%D in ("D:\ArcGISPro*") do if exist "%%~D\%PYTHON_REL%" set "PYTHON_PATH=%%~D\%PYTHON_REL%"
if defined PYTHON_PATH exit /b 0

for /d %%D in ("C:\Program Files\ArcGIS\Pro*") do if exist "%%~D\%PYTHON_REL%" set "PYTHON_PATH=%%~D\%PYTHON_REL%"
if defined PYTHON_PATH exit /b 0

exit /b 1
