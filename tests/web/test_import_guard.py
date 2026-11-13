# -*- coding: utf-8 -*-
"""架构红线：``gis_cli.web`` 的任何模块都不许 import 终端渲染层。

为什么：``ui.render`` 会创建 rich ``Console``、``ui.tui`` 会抢 stdout/键盘，
在 uvicorn 服务进程里两者都会把日志和渲染搅在一起（见 plan §6）。

两道检查：
1. 静态：AST 扫 ``src/gis_cli/web/*.py`` 的 import 语句（含相对导入）；
2. 运行时：子进程里 import 全部 web 模块，断言 ``sys.modules`` 里没有渲染层。

``gis_cli.ui`` 包本身也在禁止之列：它的 ``__init__`` 会无条件 import ``render``，
所以 ``from ..ui.events import ...`` 实际等价于把 rich 拉进来。
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import gis_cli.web

BLOCKED = ("gis_cli.ui", "gis_cli.ui.render", "gis_cli.ui.tui", "gis_cli.ui.wiring")

WEB_DIR = Path(gis_cli.web.__file__).resolve().parent
REPO_ROOT = WEB_DIR.parents[2]


def _resolve(module: str, level: int) -> str:
    """把 ``from ..ui import x`` 解析成绝对模块名（只处理 gis_cli 包内）。"""
    if level == 0:
        return module
    package = ["gis_cli", "web"]
    keep = len(package) - (level - 1)
    base = package[:keep]
    return ".".join(base + ([module] if module else []))


def _forbidden(module: str) -> bool:
    return any(module == item or module.startswith(item + ".") for item in BLOCKED)


def test_no_static_import_of_render_layers() -> None:
    offenders: list[str] = []
    for path in sorted(WEB_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                resolved = _resolve(node.module or "", node.level)
                if _forbidden(resolved):
                    offenders.append(f"{path.name}:{node.lineno} from {'.' * node.level}{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _forbidden(alias.name):
                        offenders.append(f"{path.name}:{node.lineno} import {alias.name}")
    assert not offenders, "web 模块 import 了终端渲染层: " + "; ".join(offenders)


def test_importing_web_package_does_not_load_render_layers() -> None:
    code = (
        "import importlib, pkgutil, sys\n"
        "import gis_cli.web\n"
        "for mod in pkgutil.iter_modules(gis_cli.web.__path__):\n"
        "    importlib.import_module('gis_cli.web.' + mod.name)\n"
        "blocked = " + repr(BLOCKED) + "\n"
        "loaded = [name for name in blocked if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
