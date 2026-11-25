@echo off
REM ===========================================================================
REM DEPRECATED (kept only so existing shortcuts and user scripts keep working).
REM
REM This launcher now just forwards to:  giscli web
REM Use giscli directly instead:
REM     python install.py          (one-time: writes the shim, adds user PATH)
REM     giscli --help
REM     giscli doctor
REM     giscli web
REM
REM ASCII + CRLF on purpose: cmd.exe parses .bat in the ANSI codepage, so
REM non-ASCII comments break parsing (verified: they get chopped into fake
REM commands and the console fills with "'------' is not recognized").
REM ===========================================================================
setlocal

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

echo [DEPRECATED] This launcher is now "giscli web". One-time install: python "%REPO_ROOT%\install.py"
echo.

REM Prefer the giscli on PATH (the installed shim); fall back to this repo's
REM shim so a fresh checkout still works before install.py has been run.
where giscli >nul 2>&1
if errorlevel 1 goto :repo_shim
call giscli web %*
exit /b %ERRORLEVEL%

:repo_shim
call "%REPO_ROOT%\scripts\giscli.cmd" web %*
exit /b %ERRORLEVEL%
