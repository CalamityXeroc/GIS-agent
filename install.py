#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一条命令装好 giscli：写 shim → 只追加用户级 PATH → 自检。

    python install.py              # 装（幂等：重复跑只更新 shim 内容）
    python install.py --dry-run    # 只打印会做什么，不写文件、不碰注册表

装完之后，任何目录、cmd / PowerShell / **Git Bash** 里都能敲 ``giscli``。

**不装进 ArcGIS Pro 环境、也不生成 exe**：Pro env 的 pip 会
``BuildDependencyInstallError``，它 pip 出来的 ``gis-agent.exe`` 任何参数都 rc=1
零输出（硬约束，见 scripts/giscli.cmd 的历史注释）。可靠路径只有「shim 找到 Pro 的
python.exe + 把仓库 src 挂上 PYTHONPATH」，也就是 ``gis_cli.boot`` 干的事。

**不要求管理员权限**：只写 ``HKCU\\Environment\\Path``（用户级），绝不碰 HKLM。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: 仓库根目录（本脚本就在根上）。
REPO = Path(__file__).resolve().parent
#: shim 落地目录名（Windows 下在 ``%LOCALAPPDATA%\\Programs\\`` 里）。
SHIM_DIRNAME = "giscli"

VENV_TIMEOUT = 300
PIP_TIMEOUT = 300
ARCPY_TIMEOUT = 180
PY_PROBE_TIMEOUT = 120
DOCTOR_TIMEOUT = 300


def say(text: str = "") -> None:
    print(text)


def section(title: str) -> None:
    say()
    say(f"=== {title} " + "=" * max(0, 58 - len(title)))


def ensure_src() -> None:
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def target_dir() -> Path:
    """shim 落地目录：Windows 用 ``%LOCALAPPDATA%\\Programs\\giscli``，别处用 ``~/.local/bin``。"""
    if os.name != "nt":
        return Path.home() / ".local" / "bin"
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Programs" / SHIM_DIRNAME


def _tail(text: str, limit: int = 240) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    joined = " | ".join(lines[-3:]) if lines else ""
    return joined if len(joined) <= limit else joined[: limit - 3] + "..."


def _probe(command, timeout: int) -> tuple:
    """跑一条探测命令 → ``(成功?, 一句话结果)``。超时/起不来都不抛。"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    note = _tail((result.stdout or "") + "\n" + (result.stderr or ""))
    if result.returncode != 0:
        return False, f"rc={result.returncode}：{note}"
    return True, note


# === 1. 找 Pro 的 python（复用 arcpy_bridge 的发现链） ===

def discover_pro_python() -> tuple:
    """返回 ``(解释器 or None, 说明)``。候选表只有 ``arcpy_bridge`` 一份。"""
    ensure_src()
    try:
        from gis_cli.arcpy_bridge import discover_arcgis_pro_python

        info = discover_arcgis_pro_python()
    except Exception as exc:  # 没装 Pro / 注册表读不了都不该让安装脚本崩在 import 上
        return None, str(exc) or type(exc).__name__
    return info.python_executable, f"{info.install_dir}（来源 {info.source}）"


def probe_pro_env(python_exe: str) -> dict:
    """探 Pro 环境的可用性——**只探不装**。

    探测结果写进报告：arcpy 能不能 import 决定 giscli 能不能干活；pip 能不能跑
    决定「pip 模式」这条路是否存在（已知这台机器上是坏的，所以它坏了也不算安装失败）。
    """
    facts = {"python": python_exe}
    facts["version_ok"], facts["version"] = _probe([python_exe, "-V"], PY_PROBE_TIMEOUT)
    facts["arcpy_ok"], facts["arcpy"] = _probe(
        [
            python_exe,
            "-c",
            "import arcpy; print(arcpy.GetInstallInfo().get('Version', ''))",
        ],
        ARCPY_TIMEOUT,
    )
    facts["pip_ok"], facts["pip"] = _probe(
        [python_exe, "-m", "pip", "--version"], PY_PROBE_TIMEOUT
    )
    return facts


# === 2. 临时 venv 里校验 pip 模式（只报结论，不装依赖） ===

def _make_venv(python_exe: str, target: Path):
    ok, note = _probe([python_exe, "-m", "venv", str(target)], VENV_TIMEOUT)
    if not ok:
        say(f"  [warn] 建临时 venv 失败：{note}")
        return None
    exe = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return exe if exe.exists() else None


def probe_pip_install(python_exe: str | None = None) -> tuple:
    """临时 venv 里试一次 ``pip install -e . --no-deps``。

    ``--no-deps`` + ``--no-build-isolation``：只跑本地构建，不下依赖、不联网、不漏进
    用户环境。真装依赖要几分钟，而 shim 模式根本不需要装——所以这里只验"能不能"。
    """
    python_exe = python_exe or sys.executable
    if sys.version_info < (3, 10):
        return False, (
            f"当前 python {sys.version.split()[0]} < 3.10（pyproject 要求 >=3.10），"
            "pip 模式跳过；shim 模式不受影响"
        )
    tmp = Path(tempfile.mkdtemp(prefix="giscli_pip_probe_"))
    try:
        venv_python = _make_venv(python_exe, tmp / "venv")
        if venv_python is None:
            return False, "临时 venv 没建起来，pip 模式跳过"
        ok, note = _probe(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "-e",
                str(REPO),
                "--no-deps",
                "--no-build-isolation",
                "--quiet",
            ],
            PIP_TIMEOUT,
        )
        if ok:
            return True, "临时 venv 里 pip install -e . 成功（不装依赖、不联网）"
        text = f"临时 venv 里 pip install -e . 没成功：{note}"
        if "setuptools" in note or "ModuleNotFoundError" in note:
            text += (
                "（3.12 起 venv 不再预装 setuptools，--no-build-isolation 因此必然失败；"
                "这是探测手法本身的限制，不是环境坏了，shim 模式不受影响）"
            )
        return False, text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# === 3. 写 shim + 只追加用户级 PATH ===

def _load_install_shim():
    """加载 ``scripts/install_shim.py``（scripts 不是包，只能按路径加载）。"""
    path = REPO / "scripts" / "install_shim.py"
    spec = importlib.util.spec_from_file_location("_giscli_install_shim", path)
    if spec is None or spec.loader is None:  # pragma: no cover - 文件缺失才会走到
        raise RuntimeError(f"找不到 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_to_user_path(directory: Path, dry_run: bool = False):
    """幂等加入用户级 PATH（读旧值 → 只追加这一条 → 打印新值）。"""
    module = _load_install_shim()
    outcome = module.install(directory, dry_run=dry_run)
    if outcome.ok and outcome.changed:
        say(f"  新的用户级 PATH：{outcome.current}")
    return outcome


# === 4. 自检 ===

def run_doctor(shim_dir: Path) -> tuple:
    """跑一次**装好的 shim** 的自检，原样打印输出 → ``(退出码, 输出)``。"""
    if os.name == "nt":
        shim = Path(shim_dir) / "giscli.cmd"
        command = [os.environ.get("COMSPEC", "cmd"), "/c", str(shim), "doctor"]
    else:
        command = [str(Path(shim_dir) / "giscli"), "doctor"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", COLUMNS="140")
    # COLUMNS：doctor 的输出是给人看的表格，默认 80 列会把路径折成 "Pyt / hon"。
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            encoding="utf-8",
            timeout=DOCTOR_TIMEOUT,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"自检没跑起来：{type(exc).__name__}: {exc}"
    output = (result.stdout or "").rstrip()
    if (result.stderr or "").strip():
        output += "\n" + result.stderr.rstrip()
    return result.returncode, output


# === 主流程 ===

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="安装 giscli：写 shim、只追加用户级 PATH、跑一次自检"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印会做什么，不写文件、不碰注册表"
    )
    args = parser.parse_args(argv)

    say("=" * 62)
    say("  giscli 安装（一条命令；不需要管理员权限）")
    say("=" * 62)
    say(f"仓库：{REPO}")
    say(f"当前 python：{sys.executable}（{sys.version.split()[0]}）")

    if not (REPO / "src" / "gis_cli" / "boot.py").is_file():
        say(f"[ERROR] 这里不是仓库根目录或缺文件：{REPO / 'src' / 'gis_cli' / 'boot.py'}")
        return 1
    ensure_src()

    section("1/4 找 ArcGIS Pro 的 Python")
    pro_python, detail = discover_pro_python()
    if pro_python is None:
        say(f"  [warn] 没找到：{detail}")
        say("         装在别处就设 ARCGIS_PRO_PYTHON 后重跑：")
        say(r"         set ARCGIS_PRO_PYTHON=<ProDir>\bin\Python\envs\arcgispro-py3\python.exe")
        say("         现在照样装 shim：装完 giscli doctor 会告诉你缺什么。")
    else:
        say(f"  [ok] {pro_python}")
        say(f"       {detail}")
        facts = probe_pro_env(pro_python)
        say(f"       python {facts['version'] or '?'}")
        if facts["arcpy_ok"]:
            say(f"       import arcpy 成功（Pro {facts['arcpy'] or '版本未知'}）")
        else:
            say(f"       [warn] import arcpy 失败：{facts['arcpy']}")
        if facts["pip_ok"]:
            say("       Pro env 的 pip 可用（本安装流程仍然不用它）")
        else:
            say(f"       [info] Pro env 的 pip 不可用：{facts['pip']}")
            say("              → 走 shim 模式（这是本项目的默认路径，不是错误）")

    section("2/4 校验 pip 模式是否可行（临时 venv，装不上也不影响安装）")
    pip_ok, pip_note = probe_pip_install()
    say(f"  [{'ok' if pip_ok else 'info'}] {pip_note}")
    if not pip_ok:
        say("         → 本机只用 shim 模式（shim 找 Pro python + PYTHONPATH=src）")

    section("3/4 写 shim 并加入用户级 PATH")
    shim_dir = target_dir()
    say(f"  shim 目录：{shim_dir}")
    from gis_cli import shims as shim_module

    if args.dry_run:
        for name, text in shim_module.render_all(str(REPO), sys.executable).items():
            say(f"  [dry-run] 会写 {shim_dir / name}（{len(text.splitlines())} 行）")
        say("  [dry-run] 不写文件、不碰注册表。")
        return 0

    try:
        written = shim_module.write_shims(
            shim_dir, repo_root=str(REPO), python=sys.executable
        )
    except OSError as exc:
        say(f"[ERROR] 写 shim 失败：{exc}")
        return 1
    for path, changed in written:
        say(f"  [{'更新' if changed else '未变'}] {path}")

    if os.name != "nt":
        say()
        say(f"  非 Windows：请自行把 {shim_dir} 加进 PATH（本脚本不碰其它平台的 shell 配置）。")
    else:
        outcome = add_to_user_path(shim_dir, dry_run=False)
        if not outcome.ok:
            say(f"[ERROR] 用户级 PATH 没改成：{outcome.detail}")
            say("         shim 已经写好，手动把这一个目录加进 PATH 也能用：")
            say(f"         {shim_dir}")
            return 1
        say()
        say(
            f"         注：PATH 里若还留着旧的 {REPO / 'scripts'} 条目且排在前面，"
            "会先解析到它（三份 shim 内容等价，行为一样）。"
        )

    section("4/4 自检：giscli doctor")
    code, output = run_doctor(shim_dir)
    say(output or "（没有输出）")
    say()
    if code == 0:
        say("[ok] doctor 退出码 0")
    else:
        say(f"[warn] doctor 退出码 {code}：环境还有问题，按上面「建议」那一列修")

    say()
    say("下一步（新开一个终端窗口，PATH 才会生效）：")
    say("  giscli --help")
    say("  giscli doctor")
    say(r"  giscli chat --workspace .\workspace")
    say()
    say(
        "撤销用户级 PATH：python scripts/install_shim.py --remove "
        f'--scripts-dir "{shim_dir}"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
