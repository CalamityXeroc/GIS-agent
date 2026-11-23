#requires -Version 5.1
<#
.SYNOPSIS
    本地跑一遍「纯净版 CI」：干净 venv + `.[dev,web]` + `pytest -m "not arcpy"`。

.DESCRIPTION
    为什么要有这个脚本：CI 跑在**全新 checkout + 全新 venv** 上，而开发机的 venv
    里往往有历史遗留的传递依赖（典型：openai 早期版本会顺带装 httpx）、本机还多一个
    `benchmark/` 目录、用户名也不叫 `runneradmin`。于是这三类问题在本地永远不出现，
    一 push 就红（2026-11-13 实测：fastapi 缺依赖 / httpx 未声明 /
    `C:/Users/runneradmin` 里的 `admin` 被合规规则误判成边界图层）。

    这个脚本把 CI 的三步在本地重放一遍，push 之前跑一次，能挡掉这一整类问题。

.PARAMETER Python
    用来建 venv 的解释器路径。默认按 `py -3.12` → `python` 的顺序找。

.PARAMETER Reuse
    复用已有 `.ci_venv`（只重装依赖、不重建）。日常小改动用它，省 1~2 分钟。

.PARAMETER Keep
    成功后保留 `.ci_venv`（默认成功后删掉，避免占 1GB 磁盘）。

.PARAMETER PytestArgs
    追加给 pytest 的参数（一个字符串，按空格切分），例如 `-PytestArgs "-k compliance"`。

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/ci_local.ps1

.EXAMPLE
    # 只跑合规相关，复用已有 venv
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/ci_local.ps1 -Reuse -PytestArgs "-k compliance"
#>
[CmdletBinding()]
param(
    [string]$Python = '',
    [switch]$Reuse,
    [switch]$Keep,
    [string]$PytestArgs = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo '.ci_venv'
$venvPy = Join-Path $venv 'Scripts\python.exe'
$started = Get-Date

function Write-Step([string]$text) {
    Write-Host ''
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Resolve-Python {
    if ($Python) {
        if (-not (Test-Path $Python)) { throw "找不到指定的解释器：$Python" }
        return $Python
    }
    foreach ($candidate in @('3.12', '3.11', '3.13')) {
        try {
            $exe = (& py "-$candidate" -c "import sys;print(sys.executable)" 2>$null)
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch { }
    }
    $fallback = (Get-Command python -ErrorAction SilentlyContinue)
    if ($fallback) { return $fallback.Source }
    throw '找不到 Python：请用 -Python 指定解释器路径'
}

Push-Location $repo
try {
    Write-Step "仓库：$repo"

    # ---------------------------------------------------------------- 1. venv
    if ($Reuse -and (Test-Path $venvPy)) {
        Write-Step "复用已有 venv：$venv"
    } else {
        $py = Resolve-Python
        Write-Step "用 $py 建干净 venv：$venv"
        if (Test-Path $venv) { Remove-Item -Recurse -Force $venv }
        & $py -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw '创建 venv 失败' }
    }
    $pyVersion = & $venvPy -c "import sys;print(sys.version.split()[0])"
    Write-Host "    venv Python = $pyVersion"

    # ------------------------------------------------------------ 2. 装依赖
    Write-Step '安装依赖：pip install -e ".[dev,web]"（与 CI 完全一致）'
    & $venvPy -m pip install -q --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip 升级失败' }
    & $venvPy -m pip install -q -e ".[dev,web]"
    if ($LASTEXITCODE -ne 0) {
        throw '依赖安装失败 —— CI 会在同一步红，先把 pyproject 的依赖声明修好'
    }

    # -------------------------------------------------------------- 3. 跑测试
    Write-Step '跑测试：pytest -m "not arcpy" -q（与 CI 的命令一致）'
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    # 单个字符串按空白切分：PowerShell 5.1 用 -File 传数组会把第二个元素带上前导逗号
    $extra = @($PytestArgs -split '\s+' | Where-Object { $_ })
    $pytestStarted = Get-Date
    & $venvPy -m pytest -m 'not arcpy' -q @extra
    $pytestCode = $LASTEXITCODE
    $pytestSeconds = [math]::Round(((Get-Date) - $pytestStarted).TotalSeconds, 1)

    # ---------------------------------------------------------------- 4. 收尾
    $total = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    Write-Host ''
    if ($pytestCode -eq 0) {
        Write-Host "[OK] 本地 CI 通过：pytest ${pytestSeconds}s，合计 ${total}s（Python $pyVersion）" -ForegroundColor Green
        if (-not $Keep -and -not $Reuse) {
            Write-Host "   清理 $venv（要保留加 -Keep）"
            Remove-Item -Recurse -Force $venv -ErrorAction SilentlyContinue
        }
        exit 0
    }

    Write-Host "[FAIL] 本地 CI 失败：pytest 退出码 $pytestCode（${pytestSeconds}s）" -ForegroundColor Red
    Write-Host '   CI 上大概率同样红。先看上面第一个 FAILED/ERROR：' -ForegroundColor Yellow
    Write-Host '   - ModuleNotFoundError  → pyproject 少声明依赖（别赌传递依赖）' -ForegroundColor Yellow
    Write-Host '   - 路径/用户名相关断言 → 本机与 runner 的环境差异（如 C:/Users/runneradmin）' -ForegroundColor Yellow
    Write-Host '   - float 比较           → 加容差，别拿 == / <= 硬比' -ForegroundColor Yellow
    exit $pytestCode
}
finally {
    Pop-Location
}
