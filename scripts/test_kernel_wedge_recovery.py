# -*- coding: utf-8 -*-
"""真实内核的楔死恢复测试：无限循环 → 超时/中断 → 内核可继续使用。"""
import sys, io, os, tempfile, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"E:\BST_lab\GIS_CLI\src")

from gis_cli.runtime.kernel import ArcpyKernel, KernelTimeout

state_dir = os.path.join(tempfile.gettempdir(), "wedge_test_kernels")
os.makedirs(state_dir, exist_ok=True)

print("== 启动内核（含 arcpy 导入，较慢）==")
k = ArcpyKernel(state_dir=state_dir, watchdog_grace=12.0)
t0 = time.time()
k.start()
print(f"启动完成 {time.time()-t0:.0f}s")

# 1) 无限循环: 软中断应在 8s 后生效（纯 Python 循环可被 SIGINT 打断）
t0 = time.time()
r = k.execute("while True: pass", timeout=8)
el = time.time() - t0
print(f"[1] 无限循环: ok={r.ok} error={(r.error or {}).get('type')} 耗时={el:.1f}s (期望<=30s)")
assert not r.ok, "无限循环不应成功返回"
assert el < 40, f"超时兜底太慢: {el:.1f}s"

# 2) 内核仍可执行正常代码（软中断路径，无重启）
r2 = k.execute("set_result({'n': 42})", timeout=60)
print(f"[2] 恢复执行: ok={r2.ok} result={r2.result} (期望 ok=True, {{'n': 42}})")
assert r2.ok and r2.result == {"n": 42}

# 3) 中断无效的硬楔死: interrupt 被禁用 -> 超时硬杀内核 -> 下次执行自动重启
k.interrupt = lambda: None  # 模拟控制通道楔死/中断无法送达
t0 = time.time()
try:
    r3 = k.execute("while True: pass", timeout=8)
    el3 = time.time() - t0
    print(f"[3] 硬楔死: ok={r3.ok} error={(r3.error or {}).get('type')} 耗时={el3:.1f}s (期望<=25s)")
    assert not r3.ok and el3 < 30, f"硬杀兜底太慢: {el3:.1f}s"
except KernelTimeout as e:
    el3 = time.time() - t0
    print(f"[3] 硬楔死: KernelTimeout 耗时={el3:.1f}s (期望<=25s)")
    assert el3 < 30

# 4) 内核被毁后自动重启并恢复
k.interrupt = ArcpyKernel.interrupt.__get__(k, ArcpyKernel)  # 还原
t0 = time.time()
r4 = k.execute("set_result({'n': 7})", timeout=120)
print(f"[4] 毁后重启: ok={r4.ok} result={r4.result} 重启耗时={time.time()-t0:.0f}s (期望 ok=True)")
assert r4.ok and r4.result == {"n": 7}

k.shutdown()
print("OK: 楔死恢复测试全部通过")