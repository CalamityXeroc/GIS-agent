@echo off
chcp 65001 >nul 2>&1
title GIS Agent Launcher

setlocal
cd /d "%~dp0"

echo.
echo ========================================
echo   GIS Agent Launcher
echo   Ready-to-use Workspace Mode
echo ========================================
echo.

set "PYTHON_PATH=E:\ArcGISPro3.6\bin\Python\envs\arcgispro-py3\python.exe"

if not exist "%PYTHON_PATH%" (
    echo [WARN] ArcGIS Pro Python not found, trying system Python...
    for %%I in (python.exe) do set "PYTHON_PATH=%%~$PATH:I"
    if "%PYTHON_PATH%"=="" (
        echo [ERROR] No Python runtime available.
        echo Please install Python 3.10+ or ArcGIS Pro Python.
        pause
        exit /b 1
    )
)

echo [√] Python: %PYTHON_PATH%
echo.

:: 初始化工作空间目录（开箱即用）
if not exist "workspace" mkdir "workspace"
if not exist "workspace\input" mkdir "workspace\input"
if not exist "workspace\output" mkdir "workspace\output"
if not exist "workspace\temp" mkdir "workspace\temp"
if not exist "workspace\skills" mkdir "workspace\skills"

:: 引导用户准备本地 llm 配置
if not exist "config\llm_config.json" (
    if exist "config\llm_config.example.json" (
        copy /Y "config\llm_config.example.json" "config\llm_config.json" >nul
        echo [INFO] Generated config\llm_config.json from template.
        echo [INFO] Please fill your API key before using online LLM.
    )
)

:: 检查 openai 包
"%PYTHON_PATH%" -c "import openai" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing openai package...
    "%PYTHON_PATH%" -m pip install openai --quiet
)

:: 检查 litellm 包（统一模型路由必需）
"%PYTHON_PATH%" -c "import litellm" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing litellm package...
    "%PYTHON_PATH%" -m pip install litellm --quiet
)

:: 每次启动都同步本地源码，避免运行到旧版本已安装包
echo [INFO] Syncing local gis_cli package (editable install)...
"%PYTHON_PATH%" -m pip install -e . --quiet
if errorlevel 1 (
    echo [ERROR] Package sync failed
    echo.
    echo Please run manually:
    echo   "%PYTHON_PATH%" -m pip install -e .
    pause
    exit /b 1
)
echo [OK] Local package synced
echo.

echo Starting GIS Agent...
echo.
echo ----------------------------------------
echo Notes:
echo   - Put input data into workspace\input\
echo   - Results will be written to workspace\output\
echo   - Type 'exit' to quit
echo ----------------------------------------
echo.

"%PYTHON_PATH%" -m gis_cli.agent.cli chat --workspace ".\workspace"

if errorlevel 1 (
    echo.
    echo [ERROR] Startup failed
    pause
)

endlocal
