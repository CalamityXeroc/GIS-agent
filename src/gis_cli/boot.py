# -*- coding: utf-8 -*-
"""giscli 的运行时入口：先选解释器，再把参数交给真正的 CLI。

三个 shell 的 shim（``giscli.cmd`` / ``giscli.ps1`` / ``giscli``）和 pip 生成的
``giscli.exe``（``[project.scripts] giscli = "gis_cli.boot:main"``）最后都落到这里。

**为什么要有这一层**：能跑 ``arcpy`` 的只有 ArcGIS Pro 自带的那份 python.exe，而
Pro 环境里 pip 生成的 ``gis-agent.exe`` 是坏的（任何参数 rc=1、零输出），Pro env 的
pip 本身也报 ``BuildDependencyInstallError``——所以「装进 Pro env」绝不能当前提。
可靠路径只有：把仓库 ``src`` 挂上 ``PYTHONPATH``，再用 Pro 的 python.exe 跑
``-m gis_cli.agent.cli``。这一层就负责这件事。

**发现逻辑只许有一份**：这里调 ``arcpy_bridge.discover_arcgis_pro_python()``。
以前 ``scripts\\find_pro_python.cmd`` 另抄了一份候选表，Pro 3.6→3.7 升级后两份
就漂了（cmd 指旧版、Python 指新版），所以 shim 里不再写任何候选路径。

**找不到 Pro 时不许抛栈**：打印一行人话，退回当前解释器跑（``giscli doctor`` 会
把缺失项列全）。当前解释器连 ``typer`` 都没有时，给两条可直接执行的命令后返回
``EXIT_NO_RUNTIME``——入口失败不该伪装成环境结论。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: 退出码 3：既没有 Pro python，当前解释器也起不了 CLI（入口层面的失败）。
EXIT_NO_RUNTIME = 3


def repo_root() -> Path:
    """仓库根目录：``src/gis_cli/boot.py`` 上溯三层。"""
    return Path(__file__).resolve().parents[2]


def ensure_src_on_path(repo: Path) -> str:
    """把仓库 ``src`` 放到 ``sys.path`` 最前面，返回它的字符串形式。"""
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def runtime_env(src: str) -> dict:
    """交给子进程的环境：只前置 ``PYTHONPATH``，不覆盖用户原有的。"""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return env


def discover() -> tuple:
    """找 Pro 的 python.exe，返回 ``(路径 or None, 说明)``。

    发现链自己出问题（没装 Pro、注册表读不了、候选目录权限不足）不算异常退出：
    解释「为什么」是 ``giscli doctor`` 的活。
    """
    ensure_src_on_path(repo_root())
    try:
        from gis_cli.arcpy_bridge import discover_arcgis_pro_python

        info = discover_arcgis_pro_python()
    except Exception as exc:  # ArcGISDiscoveryError 及任何意外
        return None, str(exc) or type(exc).__name__
    return info.python_executable, f"{info.install_dir}（来源 {info.source}）"


def _same_interpreter(left: str, right: str) -> bool:
    try:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
            os.path.abspath(right)
        )
    except (OSError, ValueError):  # pragma: no cover - 极端非法路径
        return left == right


def run_in_process(repo: Path, argv: list, reason: str = "") -> int:
    """在当前解释器里跑 CLI（没有 Pro、或已经跑在 Pro 里）。

    导入失败时**不打印 traceback**：那是"入口没装配好"，不是"用户数据有问题"，
    一堆 ImportError 栈只会把人吓走。
    """
    ensure_src_on_path(repo)
    try:
        from gis_cli.agent import cli
    except Exception as exc:
        sys.stderr.write(
            "[giscli] 起不了 CLI：没找到 ArcGIS Pro 的 python.exe，当前解释器也缺依赖。\n"
        )
        if reason:
            sys.stderr.write(f"         未找到 Pro：{reason}\n")
        sys.stderr.write(f"         当前解释器：{sys.executable}\n")
        sys.stderr.write(f"         缺的东西：{type(exc).__name__}: {exc}\n")
        sys.stderr.write(f'         先装依赖：python "{repo / "install.py"}"\n')
        sys.stderr.write(
            "         或指定 Pro 解释器：set ARCGIS_PRO_PYTHON="
            "<ProDir>\\bin\\Python\\envs\\arcgispro-py3\\python.exe\n"
        )
        return EXIT_NO_RUNTIME

    sys.argv = ["giscli"] + list(argv)
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    return 0


def main(argv: list | None = None) -> int:
    """入口：解析解释器 → 换到 Pro python 跑 CLI → 原样返回退出码。"""
    args = list(sys.argv[1:] if argv is None else argv)
    repo = repo_root()
    # 真正的产品名：不传的话 --help 里写的是 "python -m gis_cli.agent.cli"。
    # setdefault：用户自己设了（比如包装脚本要换个名字）就不覆盖。
    os.environ.setdefault("GISCLI_PROG", "giscli")

    target, reason = discover()
    if target is None:
        sys.stderr.write(f"[giscli] 未找到 ArcGIS Pro Python，用当前解释器运行：{reason}\n")
        return run_in_process(repo, args, reason)

    if _same_interpreter(target, sys.executable):
        # 已经跑在 Pro 解释器里（pip 装在 Pro env、或用户直接用 Pro python 敲），
        # 再起一个 python.exe 只是白等一次启动。
        return run_in_process(repo, args)

    src = ensure_src_on_path(repo)
    try:
        return subprocess.call(
            [target, "-m", "gis_cli.agent.cli"] + args, env=runtime_env(src)
        )
    except OSError as exc:
        sys.stderr.write(f"[giscli] 起不了 Pro 的 python：{target}（{exc}）\n")
        return run_in_process(repo, args, reason)


if __name__ == "__main__":
    raise SystemExit(main())
