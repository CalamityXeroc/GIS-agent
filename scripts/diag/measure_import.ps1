#requires -Version 5.1
<#
  measure_import.ps1 — DAIR-246 诊断脚本
  用 PowerShell 计时运行一段 python 代码，带硬超时。
  代码写入临时 .py 文件再执行，避免命令行引号被 Start-Process 拆开（踩过坑）。
  输出重定向到文件，超时被杀后仍能读到已产生的部分输出。

  用法:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/diag/measure_import.ps1 `
        -Code "import arcpy; print('ok')" -TimeoutSec 120
#>
param(
  [string]$Py = '',
  [string]$Code = "import arcpy; print('ok')",
  [int]$TimeoutSec = 120,
  [string]$OutFile = ''
)

# 不写默认路径：Pro 升级后写死的安装目录就是错误答案（旧版写的是 E:\ArcGISPro3.6）。
if (-not $Py) { $Py = $env:ARCGIS_PRO_PYTHON }
if (-not $Py) {
  throw '未指定 -Py。用 gis_cli.arcpy_bridge.discover_arcgis_pro_python() 或 gis-agent doctor 查到 Pro 的 python.exe 再显式传入，或先设 ARCGIS_PRO_PYTHON。'
}

$stamp = [guid]::NewGuid().ToString('N').Substring(0, 8)
if (-not $OutFile) { $OutFile = Join-Path $env:TEMP "dair246_measure_$stamp.txt" }
$script = Join-Path $env:TEMP "dair246_code_$stamp.py"
$stdout = Join-Path $env:TEMP "dair246_stdout_$stamp.log"
$stderr = Join-Path $env:TEMP "dair246_stderr_$stamp.log"
Set-Content -Path $script -Value $Code -Encoding UTF8

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $Py -ArgumentList @('-u', $script) `
      -NoNewWindow -PassThru -WorkingDirectory (Get-Location).Path `
      -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$finished = $p.WaitForExit($TimeoutSec * 1000)
$sw.Stop()
if (-not $finished) { try { $p.Kill() } catch {} }

$out = ''
if (Test-Path $stdout) { $out += (Get-Content $stdout -Raw -ErrorAction SilentlyContinue) }
if (Test-Path $stderr) { $err = (Get-Content $stderr -Raw -ErrorAction SilentlyContinue); if ($err) { $out += "`n[stderr]`n" + $err } }
if (-not $out) { $out = '<no output>' }

$elapsed = "elapsed_s : {0:N2}" -f $sw.Elapsed.TotalSeconds
$rc = "exitcode  : " + $(if ($finished) { $p.ExitCode } else { 'n/a (killed)' })
$lines = @(
  "script    : $Code",
  "python    : $Py",
  $elapsed,
  "finished  : $finished",
  $rc,
  "--- output ---",
  $out.TrimEnd()
)
$lines | Set-Content -Path $OutFile -Encoding UTF8
$lines | ForEach-Object { Write-Output $_ }
Write-Output "report_file: $OutFile"
Remove-Item $script -ErrorAction SilentlyContinue
