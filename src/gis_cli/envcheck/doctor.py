# -*- coding: utf-8 -*-
"""环境自检（doctor）：环境不对时，用一句人话 + 一条可执行命令说清缺什么。

**唯一判据**：arcpy 能不能 import、能不能真跑一行代码。找到 Pro 但环境坏了，
和没装 Pro，对用户来说都是"跑不了"，只是修复动作不同。

明确不做（写在这里是为了挡住后续"顺手加一下"）：
- 不检测 ArcGIS Pro 许可证（用户可能用非正版，检测既无用又冒犯）；
- 不引导下载、不教安装、不给购买链接；
- 不自动修改用户环境（只给命令；动手要用户显式 ``--fix-install``）。

两条设计约束（都踩过坑）：

1. **所有检查统一走 ``discover_arcgis_pro_python()`` 返回的解释器**，绝不 fallback
   到 ``sys.executable``。用错解释器查依赖，结论会正好相反：开发 venv 里 9 个包
   装齐了，ArcGIS Pro 环境里没装 —— 而用户真正要跑的是 Pro 那个。
2. **两个子进程探测（arcpy 导入 / 依赖导入）并行跑**。本机实测 ``import arcpy``
   10.7s（冷启动 18.6s）、9 个包真实导入 2.3s；串起来 13s，并行后只等最慢的那个。

**耗时**（实测，本机 ArcGIS Pro 3.6）：暖机约 11~15s、冷启约 17~19s。构成：
``import arcpy`` 10.7s（冷启 18.6s）+ 9 个包真实导入 2.3s（已与前者并行）+ 其余检查
< 0.5s。规格 §1.6 写的"体检 ≤10 秒"在不放弃真实导入的前提下做不到，而真实导入正是
本命令的唯一判据，所以这里按实测值给预期，不承诺 10s。

路径检查里，Pro 安装目录含空格**只写进 detail、不报警**：默认安装位置就是
``C:\\Program Files\\ArcGIS\\Pro``，报警等于对所有人报警。工作目录含空格才 warn
（那是用户能改、也确实是产出路径的变量）。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any

from ..arcpy_bridge import (
    ArcGISDiscoveryError,
    ArcGISPythonInfo,
    ArcPyExecutionResult,
    _build_python_candidates,
    discover_arcgis_pro_python,
    run_arcpy_code,
)

#: 检查结果状态。``warn`` / ``skip`` 都不影响退出码（只有 ``fail`` 影响）。
PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

#: 本工具用到 3.2 才有的 API；低于这个版本只是提醒，不判失败。
MIN_PRO_VERSION = (3, 2)

#: 9 个核心依赖：(import 名, pip 名)。顺序与 pyproject 的 dependencies 一致。
#: import 名与 pip 名不一样的那几个（prompt-toolkit / jupyter-client / pillow /
#: pyyaml）必须分开写，否则拼出来的 ``pip install`` 命令装不上。
CORE_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("typer", "typer"),
    ("rich", "rich"),
    ("openai", "openai"),
    ("json5", "json5"),
    ("prompt_toolkit", "prompt-toolkit"),
    ("jupyter_client", "jupyter-client"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("yaml", "pyyaml"),
)

#: arcpy 导入探测的超时。冷启动实测 18.6s，给足余量；这里不是性能指标。
ARCPY_PROBE_TIMEOUT = 120
#: 依赖导入探测的超时（9 个包真实导入，冷启动实测 8.4s）。
DEPS_PROBE_TIMEOUT = 120
#: 快检（只查模块在不在）的超时。
SPEC_PROBE_TIMEOUT = 30
#: ``--deep`` 时内核启动的超时。
KERNEL_START_TIMEOUT = 60
#: ``--deep`` 时 ``print(1)`` 的往返预算（规格要求 10s 内返回）。
KERNEL_PRINT_TIMEOUT = 10

#: 内核起不来时的统一说法：只影响速度，不影响能不能干活。
_KERNEL_FALLBACK_HINT = (
    "内核起不来时引擎会自动退回一次性子进程：功能仍然可用，只是每步要重导 "
    "arcpy（约 15s）。先把上面的报错修好，或者先不管它。"
)


@dataclass
class CheckResult:
    """一项检查的结论。``detail`` 是人话，``hint`` 是可直接照做的下一步。"""

    id: str
    status: str
    detail: str
    hint: str | None = None

    def to_dict(self) -> dict[str, str]:
        data = {"id": self.id, "status": self.status, "detail": self.detail}
        if self.hint:
            data["hint"] = self.hint
        return data


@dataclass
class DoctorReport:
    """一次体检的完整结果（``--json`` 与内部渲染共用这一份）。"""

    checks: list[CheckResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """没有 ``fail`` 就算通过（``warn`` / ``skip`` 不阻塞）。"""
        return not any(check.status == FAIL for check in self.checks)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    @property
    def summary(self) -> str:
        """一句话结论：有 fail 就报 fail 的原因，否则报通过。"""
        failed = [check for check in self.checks if check.status == FAIL]
        if failed:
            return "；".join(check.detail for check in failed[:2])
        warned = [check for check in self.checks if check.status == WARN]
        if warned:
            return f"可用，但有 {len(warned)} 项需要注意"
        return "环境正常"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "checks": [check.to_dict() for check in self.checks],
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


@dataclass
class _Probe:
    """一次子进程探测的结果。"""

    ok: bool
    data: Any | None
    error: str = ""


# === 探测 ===

_ARCPY_PROBE_CODE = dedent(
    '''
    import arcpy

    info = arcpy.GetInstallInfo()
    set_result({
        "version": info.get("Version", ""),
        "product": info.get("ProductName", ""),
        "build": info.get("BuildNumber", ""),
        "install_dir": info.get("InstallDir", ""),
    })
    '''
)


def _deps_probe_code() -> str:
    modules = json.dumps([module for module, _ in CORE_DEPENDENCIES])
    return dedent(
        f'''
        import importlib

        missing = []
        for name in {modules}:
            try:
                importlib.import_module(name)
            except Exception:
                missing.append(name)
        set_result({{"missing": missing}})
        '''
    )


def _spec_probe_code() -> str:
    """只查"模块在不在"，不做真实导入（启动快检用）。"""
    modules = json.dumps([module for module, _ in CORE_DEPENDENCIES])
    return dedent(
        f'''
        import importlib.util

        missing = []
        for name in {modules}:
            try:
                if importlib.util.find_spec(name) is None:
                    missing.append(name)
            except Exception:
                missing.append(name)

        try:
            has_arcpy = importlib.util.find_spec("arcpy") is not None
        except Exception:
            has_arcpy = False

        set_result({{"missing": missing, "arcpy": has_arcpy}})
        '''
    )


def _error_text(result: ArcPyExecutionResult) -> str:
    """失败时的原文：优先 stderr 前 3 行（规格要求），再退到错误消息/栈尾。"""
    lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
    if lines:
        return " | ".join(lines[:3])
    error = result.error or {}
    message = str(error.get("message") or "").strip()
    if message:
        return message
    traceback_lines = [
        line.strip() for line in str(error.get("traceback") or "").splitlines() if line.strip()
    ]
    return " | ".join(traceback_lines[-3:]) if traceback_lines else "未知错误（子进程没有输出）"


def _run_probe(
    info: ArcGISPythonInfo,
    code: str,
    *,
    require_arcpy: bool,
    timeout_seconds: int,
) -> _Probe:
    result = run_arcpy_code(
        code,
        python_executable=info.python_executable,
        require_arcpy=require_arcpy,
        timeout_seconds=timeout_seconds,
    )
    if result.status == "success" and result.data is not None:
        return _Probe(True, result.data)
    return _Probe(False, None, _error_text(result))


def _probe_arcpy(info: ArcGISPythonInfo) -> _Probe:
    return _run_probe(
        info, _ARCPY_PROBE_CODE, require_arcpy=True, timeout_seconds=ARCPY_PROBE_TIMEOUT
    )


def _probe_deps(info: ArcGISPythonInfo) -> _Probe:
    return _run_probe(
        info, _deps_probe_code(), require_arcpy=False, timeout_seconds=DEPS_PROBE_TIMEOUT
    )


def _probe_specs(info: ArcGISPythonInfo) -> _Probe:
    return _run_probe(
        info, _spec_probe_code(), require_arcpy=False, timeout_seconds=SPEC_PROBE_TIMEOUT
    )


def _discover() -> tuple[ArcGISPythonInfo | None, str]:
    """找解释器。连注册表都读崩了也不能把 doctor 带崩 —— 那就是 exit 2 了。"""
    try:
        return discover_arcgis_pro_python(), ""
    except ArcGISDiscoveryError as exc:
        return None, str(exc)
    except Exception as exc:  # pragma: no cover - 极端环境（注册表权限等）
        return None, f"{type(exc).__name__}: {exc}"


# === 单项检查 ===

def _tried_routes() -> str:
    """没找到 Pro 时，把"试过哪几条路"说清楚，用户才知道下一步改哪里。"""
    routes = [
        "环境变量 ARCGIS_PRO_PYTHON / ARCGIS_PRO_INSTALL_DIR",
        r"注册表 SOFTWARE\ESRI\ArcGISPro 的 InstallDir",
        r"默认路径 C:\Program Files\ArcGIS\Pro（另有 D:\ArcGIS\Pro、E:\ArcGISPro3.6）",
    ]
    return "；".join(routes)


def _existing_candidates() -> list[tuple[str, str]]:
    """所有**存在**的解释器候选（多版本 Pro 并存时用来全列出来）。

    只看 ``python.exe``：候选里还有 ``propy.bat``（同一个环境的启动器），
    把它当"另一个解释器"报给用户是误导。
    """
    try:
        candidates = _build_python_candidates()
    except Exception:  # pragma: no cover - 候选枚举失败不该影响体检
        return []
    seen: set[str] = set()
    found: list[tuple[str, str]] = []
    for source, python_path, _install_dir in candidates:
        if Path(python_path).suffix.lower() != ".exe":
            continue
        if python_path in seen or not Path(python_path).exists():
            continue
        seen.add(python_path)
        found.append((source, python_path))
    return found


def _check_pro_installed(
    info: ArcGISPythonInfo | None, discovery_error: str
) -> CheckResult:
    if info is None:
        return CheckResult(
            "pro_installed",
            FAIL,
            "未找到 ArcGIS Pro Python",
            hint=(
                f"已试过：{_tried_routes()}。装在别处就用环境变量指定："
                r'set ARCGIS_PRO_PYTHON=<安装目录>\bin\Python\envs\arcgispro-py3\python.exe'
            ),
        )

    detail = f"{info.install_dir}（来源 {info.source}）"
    others = [path for _source, path in _existing_candidates() if path != info.python_executable]
    if not others:
        return CheckResult("pro_installed", PASS, detail)

    # 多版本并存：全列出来，并写明"本次用的是哪个"——只报一个会让人以为探测错了。
    listed = "；".join(others[:3])
    more = f"（另有 {len(others) - 3} 个未列出）" if len(others) > 3 else ""
    return CheckResult(
        "pro_installed",
        PASS,
        detail,
        hint=(
            f"另发现 {len(others)} 个可用解释器：{listed}{more}。本次用的是上面那个；"
            f"要换用别的就设 ARCGIS_PRO_PYTHON={others[0]}"
        ),
    )


def _check_arcpy_importable(
    info: ArcGISPythonInfo | None, probe: _Probe | None
) -> CheckResult:
    if info is None:
        return CheckResult("arcpy_importable", SKIP, "未找到 ArcGIS Pro Python，无法检查")
    if probe is None or not probe.ok:
        return CheckResult(
            "arcpy_importable",
            FAIL,
            f"arcpy 导入失败：{(probe.error if probe else '') or '未知错误'}",
            hint=(
                "常见原因是装了 Pro 但环境坏了（conda 环境缺包、被杀软删了文件）。"
                "在 ArcGIS Pro 的 Python 命令提示符里跑一次 python -c \"import arcpy\" 可看完整报错；"
                "修好 Pro 环境后重试。"
            ),
        )
    data = probe.data or {}
    version = str(data.get("version") or "").strip()
    build = str(data.get("build") or "").strip()
    detail = f"arcpy {version or '版本未知'}"
    if build:
        detail += f" (build {build})"
    return CheckResult("arcpy_importable", PASS, detail)


def _parse_version(text: str) -> tuple[int, ...] | None:
    """把 ``3.6`` / ``3.2.1`` 解析成可比较的元组；解析不了返回 None。"""
    parts = [part for part in str(text).strip().split(".") if part != ""]
    numbers: list[int] = []
    for part in parts[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple(numbers) if numbers else None


def _check_pro_version(probe: _Probe | None) -> CheckResult:
    if probe is None or not probe.ok:
        return CheckResult("pro_version", SKIP, "arcpy 不可用，取不到版本")
    version = str((probe.data or {}).get("version") or "").strip()
    parsed = _parse_version(version)
    if parsed is None:
        return CheckResult("pro_version", WARN, f"版本号读不出来：{version!r}")
    if parsed < MIN_PRO_VERSION:
        return CheckResult(
            "pro_version",
            WARN,
            f"{version} < 建议 3.2+",
            hint="升级到 ArcGIS Pro 3.2 以上：3.2 以下缺几个本工具用到的 API",
        )
    return CheckResult("pro_version", PASS, f"{version}（>= 3.2）")


def _check_deps(
    info: ArcGISPythonInfo | None,
    probe: _Probe | None,
    missing: list[str] | None = None,
    *,
    fast: bool = False,
) -> CheckResult:
    """依赖检查。``probe`` 是真实导入探测，``missing`` 是快检给的缺失清单。"""
    if info is None:
        return CheckResult("deps", SKIP, "未找到 ArcGIS Pro Python，无法检查")
    prefix = Path(info.python_executable).parent
    if missing is None:
        if probe is None or not probe.ok:
            if fast:
                # 同一次快检失败，根因已由 arcpy_importable 报过：这里只说明"没查"，
                # 不再造第二条 fail（一次探测失败变成两条错，用户只会更糊涂）。
                return CheckResult("deps", SKIP, "启动快检没跑起来，未检查")
            return CheckResult(
                "deps",
                FAIL,
                f"依赖检查没跑起来：{(probe.error if probe else '') or '未知错误'}",
                hint=f"检查用的解释器是 {info.python_executable}（不是当前 python，别搞混）",
            )
        missing = list((probe.data or {}).get("missing") or [])

    if not missing:
        verb = "都在" if fast else "均可导入"
        return CheckResult(
            "deps",
            PASS,
            f"{len(CORE_DEPENDENCIES)} 个核心包{verb}（{prefix}）" + ("，启动快检" if fast else ""),
        )

    dist_names = [_dist_name(module) for module in missing]
    joined = "、".join(dist_names)
    return CheckResult(
        "deps",
        FAIL,
        # 环境路径写在 detail 里（不只 hint）：只看 --json 的消费者（CI / 脚本）
        # 得能分清是"跑 doctor 的那个 venv 缺"还是"Pro 环境缺"。
        f"缺依赖 {joined}（{prefix}）",
        hint=(
            f"一条命令装齐：gis-agent doctor --fix-install"
            f'（等价于 "{info.python_executable}" -m pip install {" ".join(dist_names)}）'
        ),
    )


def _dist_name(module: str) -> str:
    for import_name, dist in CORE_DEPENDENCIES:
        if import_name == module:
            return dist
    return module


def _is_ascii(text: str) -> bool:
    return all(ord(char) < 128 for char in text)


def _interpreter_prefix(info: ArcGISPythonInfo) -> Path:
    """Pro 环境的 ``sys.prefix``：就是 ``python.exe`` 所在目录，省掉一次子进程。"""
    return Path(info.python_executable).parent


def _check_path_spaces(workspace: Path, info: ArcGISPythonInfo | None) -> CheckResult:
    prefix = _interpreter_prefix(info) if info is not None else None
    if " " in str(workspace):
        return CheckResult(
            "path_spaces",
            WARN,
            f"工作目录含空格：{workspace}",
            hint=(
                "换到不含空格的目录再跑，例如 gis-agent loop \"任务\" --workspace E:\\gis_work"
                "（ArcPy 处理带空格的输出路径偶发失败）"
            ),
        )
    detail = f"工作目录无空格：{workspace}"
    if prefix is not None:
        # 默认安装位置 C:\Program Files\ArcGIS\Pro 本身就含空格，不据此报警。
        note = "（含空格，默认安装即如此，不影响）" if " " in str(prefix) else ""
        detail = f"{detail}；Pro 环境 {prefix}{note}"
    return CheckResult("path_spaces", PASS, detail)


def _check_path_nonascii(workspace: Path, info: ArcGISPythonInfo | None) -> CheckResult:
    prefix = _interpreter_prefix(info) if info is not None else None
    offenders: list[str] = []
    if prefix is not None and not _is_ascii(str(prefix)):
        offenders.append(str(prefix))
    if not _is_ascii(str(workspace)):
        offenders.append(str(workspace))
    if offenders:
        return CheckResult(
            "path_nonascii",
            WARN,
            "路径含非 ASCII 字符（中文等）：" + "；".join(offenders),
            hint="历史上能跑，只是偶发编码问题；换到纯英文路径（如 E:\\gis_work）最稳",
        )
    return CheckResult("path_nonascii", PASS, "Pro 环境与工作目录都是纯 ASCII")


def _check_kernel_start(workspace: Path, info: ArcGISPythonInfo | None) -> CheckResult:
    """``--deep`` 的内核检查。失败一律判 **WARN**，不判 FAIL。

    内核起不来时引擎会自动退回一次性子进程，任务照样能做完，只是每步重导 arcpy（约 15s）。
    拿它判 FAIL 会让一个能用的环境退出码变 1（而且 hint 里写着"功能仍然可用"，自相矛盾）。
    ``--deep`` 是用户显式要的检查，warn 一样醒目。
    """
    if info is None:
        return CheckResult("kernel_start", SKIP, "未找到 ArcGIS Pro Python，无法检查")
    try:
        from ..runtime.kernel import ArcpyKernel
    except Exception as exc:  # pragma: no cover - 内核模块导入失败极罕见
        return CheckResult("kernel_start", WARN, f"内核模块导入失败：{exc}")

    started = time.monotonic()
    kernel = ArcpyKernel(
        python_executable=info.python_executable,
        workspace=str(workspace),
        startup_timeout=KERNEL_START_TIMEOUT,
    )
    try:
        try:
            kernel.start()
        except Exception as exc:
            return CheckResult(
                "kernel_start", WARN, f"内核启动失败：{_short(exc)}", hint=_KERNEL_FALLBACK_HINT
            )
        startup_seconds = time.monotonic() - started
        try:
            outcome = kernel.execute("print(1)", timeout=KERNEL_PRINT_TIMEOUT)
        except Exception as exc:  # 含 KernelTimeout：超过 10s 没回来
            return CheckResult(
                "kernel_start",
                WARN,
                f"内核里跑 print(1) 没在 {KERNEL_PRINT_TIMEOUT}s 内返回：{_short(exc)}",
                hint=_KERNEL_FALLBACK_HINT,
            )
    finally:
        try:
            kernel.shutdown()
        except Exception:  # pragma: no cover - 关不掉也不影响结论
            pass

    if not outcome.ok:
        return CheckResult(
            "kernel_start",
            WARN,
            f"内核里跑 print(1) 失败：{_short(outcome.error or outcome.stderr)}",
            hint=_KERNEL_FALLBACK_HINT,
        )
    if "1" not in (outcome.stdout or ""):
        return CheckResult(
            "kernel_start", WARN, f"print(1) 没有回显，stdout={outcome.stdout!r}"
        )
    return CheckResult(
        "kernel_start",
        PASS,
        f"内核启动 {startup_seconds:.1f}s，print(1) 往返 {outcome.duration_ms / 1000:.2f}s",
    )


def _short(value: Any, limit: int = 200) -> str:
    text = str(value or "").strip().replace("\n", " | ")
    return text if len(text) <= limit else text[:limit] + "..."


# === 聚合 ===

def _parallel_probes(
    info: ArcGISPythonInfo,
) -> tuple[_Probe, _Probe]:
    """arcpy 导入与依赖导入并行跑。

    串行实测 10.7s + 2.3s = 13s；并行只等最慢的那个（约 11s），而"10 秒说清"
    是这个命令存在的理由，能省 2s 就省。
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        arcpy_future = pool.submit(_probe_arcpy, info)
        deps_future = pool.submit(_probe_deps, info)
        return arcpy_future.result(), deps_future.result()


def run_checks(
    *,
    deep: bool = False,
    workspace: str | Path | None = None,
) -> DoctorReport:
    """跑完整 7 项体检。

    Args:
        deep: 额外跑持久内核启动（``--deep``），默认跳过。
        workspace: 工作目录，默认当前目录（路径检查用它）。
    """
    started = time.monotonic()
    ws = Path(workspace) if workspace is not None else Path.cwd()
    info, discovery_error = _discover()

    arcpy_probe: _Probe | None = None
    deps_probe: _Probe | None = None
    if info is not None:
        arcpy_probe, deps_probe = _parallel_probes(info)

    checks = [
        _check_pro_installed(info, discovery_error),
        _check_arcpy_importable(info, arcpy_probe),
        _check_pro_version(arcpy_probe),
        _check_deps(info, deps_probe),
        _check_path_spaces(ws, info),
        _check_path_nonascii(ws, info),
        _check_kernel_start(ws, info)
        if deep
        else CheckResult("kernel_start", SKIP, "未加 --deep，未检查"),
    ]
    return DoctorReport(checks=checks, elapsed_seconds=time.monotonic() - started)


def run_startup_checks(*, workspace: str | Path | None = None) -> DoctorReport:
    """启动时的轻量体检：只做 1/2/4 三项，且**不做真实导入**。

    为什么不用完整版：本机实测 ``import arcpy`` 10.7s（冷启动 18.6s），
    每次 ``chat`` / ``loop`` 启动都付这个代价不可接受；而启动时要知道的只是
    "缺不缺东西"。真跑一行代码由 ``gis-agent doctor`` 负责。

    一次子进程（约 0.6s）同时查 arcpy 模块在不在、9 个核心依赖缺哪个。
    """
    started = time.monotonic()
    ws = Path(workspace) if workspace is not None else Path.cwd()
    info, discovery_error = _discover()

    spec_probe: _Probe | None = _probe_specs(info) if info is not None else None
    spec_data = (spec_probe.data or {}) if spec_probe is not None and spec_probe.ok else {}

    if info is None:
        arcpy_check = CheckResult("arcpy_importable", SKIP, "未找到 ArcGIS Pro Python，无法检查")
    elif spec_probe is None or not spec_probe.ok:
        # 快检**自己**没跑起来（超时 / 子进程没起来）：这不能说成"环境里没有 arcpy"，
        # 否则慢机器上用户会去重装 Pro，而再跑一次 doctor 又是全 pass。
        arcpy_check = CheckResult(
            "arcpy_importable",
            FAIL,
            f"启动快检没跑起来：{(spec_probe.error if spec_probe else '') or '子进程没有返回结果'}",
            hint="这不是环境结论，是快检自身没跑起来（多半是超时或被杀软拖慢）；跑 gis-agent doctor 做完整检查",
        )
    elif not spec_data.get("arcpy"):
        arcpy_check = CheckResult(
            "arcpy_importable",
            FAIL,
            "ArcGIS Pro Python 里找不到 arcpy 模块",
            hint="Pro 环境不完整；先跑 gis-agent doctor 看完整报告",
        )
    else:
        arcpy_check = CheckResult(
            "arcpy_importable", PASS, "arcpy 模块在位（启动快检，未做真实导入）"
        )

    checks = [
        _check_pro_installed(info, discovery_error),
        arcpy_check,
        _check_deps(
            info,
            spec_probe,
            missing=(
                list(spec_data.get("missing") or [])
                if spec_probe is not None and spec_probe.ok
                else None
            ),
            fast=True,
        ),
    ]
    return DoctorReport(checks=checks, elapsed_seconds=time.monotonic() - started)
