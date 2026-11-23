# -*- coding: utf-8 -*-
"""``gis-agent doctor --fix-install``：把本包装进探测到的 arcgispro-py3。

为什么需要：用户环境里缺的往往不是 ArcGIS Pro，而是本工具自己的 9 个依赖。
而"打开 Pro 的 Python 命令提示符 → cd 到仓库目录 → pip install -e ."这三步
对外行来说就是三个坑（找不到命令提示符、路径带空格忘了引号、装进了系统
Python 而不是 Pro 的那个）。这里把三步合成一条命令。

边界：只在用户显式加 ``--fix-install`` 时执行。``doctor`` 本身绝不改环境。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..arcpy_bridge import (
    ArcGISDiscoveryError,
    _build_subprocess_env,
    discover_arcgis_pro_python,
)

#: pip 装 9 个包 + 编译元数据，冷启动实测 1~3 分钟；给足余量。
INSTALL_TIMEOUT_SECONDS = 900


@dataclass
class FixInstallResult:
    """一次 ``pip install -e .`` 的结果。"""

    ok: bool
    command: str
    detail: str
    output: str = ""


def project_root() -> Path:
    """仓库根目录：``src/gis_cli/envcheck/install.py`` 上溯三层。"""
    return Path(__file__).resolve().parents[3]


def fix_install(
    *,
    root: str | Path | None = None,
    python_executable: str | None = None,
    timeout_seconds: int = INSTALL_TIMEOUT_SECONDS,
) -> FixInstallResult:
    """把本包以可编辑方式装进 ArcGIS Pro 环境。

    Args:
        root: 源码目录（含 ``pyproject.toml``），默认本文件所在的仓库。
        python_executable: 目标解释器，默认用 ``discover_arcgis_pro_python()``。
        timeout_seconds: pip 超时。
    """
    python = python_executable
    if python is None:
        try:
            python = discover_arcgis_pro_python().python_executable
        except ArcGISDiscoveryError as exc:
            return FixInstallResult(False, "", f"未找到 ArcGIS Pro Python，无法安装：{exc}")

    source_root = Path(root) if root is not None else project_root()
    if not (source_root / "pyproject.toml").exists():
        return FixInstallResult(
            False,
            "",
            f"没找到 pyproject.toml，{source_root} 不是本项目源码目录"
            "（非源码安装时 --fix-install 用不了，请改用 pip install gis-task-cli）",
        )

    command = [str(python), "-m", "pip", "install", "-e", str(source_root)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            env=_build_subprocess_env(),
            timeout=timeout_seconds,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return FixInstallResult(
            False, " ".join(command), f"pip 安装超时（超过 {timeout_seconds} 秒）"
        )

    output = "\n".join(
        line
        for line in ((completed.stdout or "") + (completed.stderr or "")).splitlines()
        if line.strip()
    )
    tail = "\n".join(output.splitlines()[-12:])
    if completed.returncode != 0:
        return FixInstallResult(
            False,
            " ".join(command),
            f"pip 安装失败（退出码 {completed.returncode}）",
            tail,
        )
    return FixInstallResult(True, " ".join(command), f"已把本包装进 {python}", tail)
