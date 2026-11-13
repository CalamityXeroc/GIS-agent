#requires -Version 5.1
<#
  wct_arcpy_hang.ps1  —  DAIR-246 诊断脚本
  启动一个 "import arcgisscripting" 的 python 子进程，等 N 秒后用
  Windows Wait Chain Traversal (WCT) 打印该进程每个线程的内核等待链，
  用来定位 import 挂死到底卡在哪个对象上（临界区/互斥体/进程/线程/ALPC/COM）。

  用法: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/diag/wct_arcpy_hang.ps1 [-WaitSec 20] [-Py <python.exe>]
#>
param(
  [int]$WaitSec = 20,
  [string]$Py = '',
  [string]$Code = 'import arcgisscripting; print("ok")'
)

# 不写默认路径：Pro 升级后写死的安装目录就是错误答案（旧版写的是 E:\ArcGISPro3.6）。
if (-not $Py) { $Py = $env:ARCGIS_PRO_PYTHON }
if (-not $Py) {
  throw '未指定 -Py。用 gis_cli.arcpy_bridge.discover_arcgis_pro_python() 或 gis-agent doctor 查到 Pro 的 python.exe 再显式传入，或先设 ARCGIS_PRO_PYTHON。'
}

$ErrorActionPreference = 'Stop'

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public class Wct {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern IntPtr OpenThreadWaitChainSession(uint flags, IntPtr callback);
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool GetThreadWaitChain(IntPtr wctHandle, IntPtr context, uint flags,
      uint threadId, ref uint nodeCount, IntPtr nodeInfoArray, out bool isCycle);
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern void CloseThreadWaitChainSession(IntPtr handle);

  const int NODE_SIZE = 280;   // WAITCHAIN_NODE_INFO, padded to 8-byte alignment
  const uint ALL_FLAGS = 0x1 | 0x2 | 0x4;  // OUT_OF_PROC | OUT_OF_PROC_COM | OUT_OF_PROC_CS

  static string TypeName(uint t) {
    switch (t) {
      case 1: return "CriticalSection";
      case 2: return "Mutex";
      case 3: return "Process";
      case 4: return "Thread";
      case 5: return "COM";
      case 6: return "ALPC";
      case 7: return "Unknown";
      default: return "Type" + t;
    }
  }
  static string StatusName(uint s) {
    switch (s) {
      case 0: return "Unused";
      case 1: return "Blocking";
      case 2: return "NonBlocking";
      case 3: return "Abandoned";
      case 4: return "Unknown";
      case 5: return "Error";
      default: return "Status" + s;
    }
  }

  // 返回每个线程的等待链文本
  public static List<string> Dump(uint[] tids) {
    var result = new List<string>();
    IntPtr h = OpenThreadWaitChainSession(0, IntPtr.Zero);
    if (h == IntPtr.Zero) { result.Add("OpenThreadWaitChainSession failed: " + Marshal.GetLastWin32Error()); return result; }
    try {
      foreach (uint tid in tids) {
        uint count = 16;
        IntPtr buf = Marshal.AllocHGlobal(NODE_SIZE * 16);
        try {
          bool cycle;
          bool ok = GetThreadWaitChain(h, IntPtr.Zero, ALL_FLAGS, tid, ref count, buf, out cycle);
          if (!ok) {
            result.Add(String.Format("  tid {0}: GetThreadWaitChain failed err={1}", tid, Marshal.GetLastWin32Error()));
            continue;
          }
          var parts = new List<string>();
          for (int i = 0; i < count; i++) {
            IntPtr node = new IntPtr(buf.ToInt64() + i * NODE_SIZE);
            uint otype = (uint)Marshal.ReadInt32(node, 0);
            uint ostatus = (uint)Marshal.ReadInt32(node, 4);
            string detail;
            if (otype == 1 || otype == 2) {
              detail = Marshal.PtrToStringUni(new IntPtr(node.ToInt64() + 8));
            } else if (otype == 3 || otype == 4) {
              uint pid = (uint)Marshal.ReadInt32(node, 8);
              uint ptid = (uint)Marshal.ReadInt32(node, 12);
              detail = "pid=" + pid + " tid=" + ptid;
            } else if (otype == 6) {
              uint pid = (uint)Marshal.ReadInt32(node, 8);
              uint ptid = (uint)Marshal.ReadInt32(node, 12);
              detail = "pid=" + pid + " tid=" + ptid;
            } else {
              detail = "";
            }
            parts.Add(String.Format("[{0} {1}] {2}", TypeName(otype), StatusName(ostatus), detail));
          }
          result.Add(String.Format("  tid {0}: {1}{2}", tid, String.Join(" -> ", parts), cycle ? "  (CYCLE!)" : ""));
        } finally { Marshal.FreeHGlobal(buf); }
      }
    } finally { CloseThreadWaitChainSession(h); }
    return result;
  }
}
'@

Write-Host "== start: $Py -c $Code"
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $Py
$psi.Arguments = '-c "' + $Code + '"'
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$p = [System.Diagnostics.Process]::Start($psi)
Write-Host ("== pid={0}, waiting {1}s ..." -f $p.Id, $WaitSec)
Start-Sleep -Seconds $WaitSec
$p.Refresh()
Write-Host ("== still running after {0:N1}s: {1}" -f $sw.Elapsed.TotalSeconds, (-not $p.HasExited))

if ($p.HasExited) {
  Write-Host ("== exited rc={0} in {1:N2}s" -f $p.ExitCode, $sw.Elapsed.TotalSeconds)
  Write-Host "stdout: $($p.StandardOutput.ReadToEnd())"
  Write-Host "stderr: $($p.StandardError.ReadToEnd())"
} else {
  $tids = @($p.Threads | ForEach-Object { [uint32]$_.Id })
  Write-Host ("== threads: {0}" -f ($tids -join ','))
  Write-Host "== wait chains:"
  foreach ($line in [Wct]::Dump($tids)) { Write-Host $line }
  Write-Host "== killing pid $($p.Id)"
  try { $p.Kill() } catch { Write-Host "kill failed: $_" }
  Start-Sleep -Seconds 2
  Write-Host ("== hasExited after kill: {0}" -f $p.HasExited)
}
