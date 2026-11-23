#requires -Version 5.1
<#
  loop_import.ps1 — DAIR-246 诊断脚本
  连续 N 次运行 "import arcpy"，每次硬超时；一旦某次超时（疑似挂死），
  保留该进程（不杀）并打印 PID，便于随后取等待链/栈。
  代码写入临时 .py 文件再执行，避免引号被 Start-Process 拆开。

  用法: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/diag/loop_import.ps1 -Runs 8 -TimeoutSec 40
#>
param(
  [string]$Py = '',
  [int]$Runs = 8,
  [int]$TimeoutSec = 40
)

# 不写默认路径：Pro 升级后写死的安装目录就是错误答案（旧版写的是 E:\ArcGISPro3.6）。
if (-not $Py) { $Py = $env:ARCGIS_PRO_PYTHON }
if (-not $Py) {
  throw '未指定 -Py。用 gis_cli.arcpy_bridge.discover_arcgis_pro_python() 或 gis-agent doctor 查到 Pro 的 python.exe 再显式传入，或先设 ARCGIS_PRO_PYTHON。'
}

$stamp = [guid]::NewGuid().ToString('N').Substring(0, 8)
$codeFile = Join-Path $env:TEMP "dair246_loopcode_$stamp.py"
Set-Content -Path $codeFile -Encoding UTF8 -Value @'
import time
t = time.time()
import arcpy
print("ok %.2f" % (time.time() - t))
'@

$log = Join-Path $env:TEMP "dair246_loop_$stamp.log"
"log: $log" | Write-Output
for ($i = 1; $i -le $Runs; $i++) {
  $stdout = Join-Path $env:TEMP ("dair246_loop_out_{0}_{1}.log" -f $i, $stamp)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $p = Start-Process -FilePath $Py -ArgumentList @('-u', $codeFile) `
        -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError "$stdout.err"
  $ok = $p.WaitForExit($TimeoutSec * 1000)
  $sw.Stop()
  if ($ok) {
    $o = ((Get-Content $stdout -Raw -ErrorAction SilentlyContinue) -replace "`r?`n", ' ')
    $line = "run {0}: ok wall={1:N2}s rc={2} out={3}" -f $i, $sw.Elapsed.TotalSeconds, $p.ExitCode, $o.Trim()
  } else {
    $line = "run {0}: TIMEOUT wall={1:N2}s pid={2} (left running)" -f $i, $sw.Elapsed.TotalSeconds, $p.Id
  }
  $line | Write-Output
  $line | Add-Content -Path $log
  if (-not $ok) { break }
}
"log: $log" | Write-Output
