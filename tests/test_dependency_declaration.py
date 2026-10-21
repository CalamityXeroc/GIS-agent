# -*- coding: utf-8 -*-
"""依赖声明一致性：代码 import 的第三方包，必须真的写在 pyproject 里。

**为什么这条测试存在（真实教训）**：

清理 chromadb 时发现，`recipes/schema.py` 在模块顶层 `import yaml`，而
`pyyaml` **从来没写进 pyproject**。它一直是靠 `langchain` / `litellm` 的
传递依赖顺带装上的 —— 等于一直在赌。当时开发 venv 里连 langchain 都没装，
只是碰巧某个包把 pyyaml 带进来了。

同类问题还有 `psutil` / `httpx` / `matplotlib` / `json5`：全都没声明。

这类问题的可怕之处是**静默**：
- 换台机器缺了 → 有的直接 ImportError（yaml），有的只是能力降级
  （检测不到 ArcGIS Pro、中文变方框、keepalive 失效），后者根本发现不了；
- 删掉一个"看起来没用"的重包 → 连带把它传递带来的依赖也删了 → 线上崩。

所以这条测试做两件事：
1. 扫全仓库真实 import（AST，不靠正则），列出所有第三方包；
2. 断言每个包都能在 pyproject 的 dependencies / optional-dependencies 里找到
   （允许"可选依赖"—— 只要声明了，缺了能降级就行）。
"""

from __future__ import annotations

import ast
import sys
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "gis_cli"
PYPROJECT = ROOT / "pyproject.toml"

#: import 名 → 发行包名（PyPI 上两者常不一致）。
#:
#: 只收录**确实不一致**的；同名的不写（否则维护负担）。
IMPORT_TO_DIST = {
    "PIL": "pillow",
    "yaml": "pyyaml",
    "docx": "python-docx",
    "pypdf": "pypdf",
    "prompt_toolkit": "prompt-toolkit",
    "jupyter_client": "jupyter-client",
    "json5": "json5",
    "openai": "openai",
    "rich": "rich",
    "numpy": "numpy",
    "typer": "typer",
    "pydantic": "pydantic",
    "psutil": "psutil",
    "httpx": "httpx",
    "matplotlib": "matplotlib",
    "arcpy": "arcpy",  # 由 ArcGIS Pro 提供，不是 pip 包
    "litellm": "litellm",
    "langchain": "langchain",
    "langchain_community": "langchain-community",
    "langchain_openai": "langchain-openai",
    "langchain_text_splitters": "langchain-text-splitters",
    "langchain_core": "langchain-core",
    "langgraph": "langgraph",
    # 旧引擎的可选集成（全部在 try/except 里，缺了自动降级）
    "baml_client": "baml-py",
    "prompt_optimizer": "prompt-optimizer",
    "chromadb": "chromadb",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "streamlit": "streamlit",
    "pytest": "pytest",
}

#: 由运行环境提供、不该写进 pyproject 的包。
ENV_PROVIDED = {
    "arcpy",  # ArcGIS Pro 自带，装在它自己的 Python 环境里
}

#: **旧引擎专属**的可选集成：全部在 try/except 里，缺了自动降级。
#:
#: 它们即将随旧引擎下线（迁入 legacy/ 并从打包中排除），所以不写进
#: pyproject —— 写进去反而会让新用户装一堆永远用不到的包。
#: 旧引擎真正删干净后，这个集合连同这些 import 一起消失。
#: **旧引擎专属**的可选集成与重包：全部只被旧引擎（``agent/`` 下的
#: plan-then-execute、``planner.py``、``orchestrator.py``、``core/``、
#: ``tools/*``、``web/``）用到。
#:
#: 它们已从 pyproject 的 dependencies 里移除（实测：开发 venv 里没装
#: litellm/langchain/chromadb，而新引擎 500+ 个测试照样全过），但代码还在
#: 仓库里 —— 等阶段 2 把旧引擎迁入 legacy/ 并从打包排除后，这些 import
#: 会连同模块一起消失。
#:
#: 那时这个集合应清空（测试会提醒）。
LEGACY_OPTIONAL = {
    "baml_client",       # BAML 结构化输出（旧引擎 planner 用）
    "prompt_optimizer",  # 提示词优化（旧引擎 planner 用）
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_openai",
    "langchain_text_splitters",
    "langgraph",
    "litellm",
    "pydantic",          # 只被 core/ + tools/* + skills/base.py + web/ 用
}

#: 标准库里名字容易误判的（确保不会被当成第三方）。
#: 标准库里但不在 sys.stdlib_module_names 里的（或版本间有出入的）。
#: tomllib 是 3.11 才进标准库的；包声明支持 >=3.10，所以扫描器在
#: 3.10 上不能把它当成"未声明的第三方包"（否则 CI 的 3.10 会误报）。
_EXTRA_STDLIB = {"sqlite3", "tkinter", "zoneinfo", "tomllib"}


def _stdlib_names() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", set())) | _EXTRA_STDLIB
    return names


#: 不参与打包的目录（旧引擎），其 import 不该参与依赖声明检查。
#:
#: 依据：``pyproject.toml`` 的 ``[tool.setuptools.packages.find]`` 里
#: ``exclude = ["gis_cli.legacy", "gis_cli.legacy.*"]`` —— 这些代码不会被
#: 装到用户环境，它们 import 什么（pydantic/langchain/fastapi…）都与用户无关。
EXCLUDED_DIRS = {"legacy"}


def _iter_py(directory: Path):
    for path in sorted(directory.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        # 跳过不参与打包的目录
        if EXCLUDED_DIRS & set(path.relative_to(SRC).parts):
            continue
        yield path


def _imported_roots(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover
        return set()
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:  # 相对 import 不算第三方
                continue
            roots.add(node.module.split(".")[0])
    return roots


def _declared() -> dict[str, set[str]]:
    """返回 {"required": {...}, "optional": {...}} 的**规范化发行包名**。"""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]

    def _norm(spec: str) -> str:
        # "pyyaml>=6.0.1" -> "pyyaml"；"prompt-toolkit[extra]" -> "prompt-toolkit"
        name = spec.split(";")[0].split("[")[0]
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "="):
            name = name.split(sep)[0]
        return name.strip().lower().replace("_", "-")

    required = {_norm(item) for item in project.get("dependencies", [])}
    optional: set[str] = set()
    for group in (project.get("optional-dependencies") or {}).values():
        optional |= {_norm(item) for item in group}
    return {"required": required, "optional": optional}


def _third_party_imports() -> dict[str, set[str]]:
    """扫全仓库，返回 {import 名: {引用它的文件}}。"""
    stdlib = _stdlib_names()
    found: dict[str, set[str]] = {}
    for path in _iter_py(SRC):
        for root in _imported_roots(path):
            if root in stdlib or root == "gis_cli" or root.startswith("_"):
                continue
            found.setdefault(root, set()).add(str(path.relative_to(SRC)))
    return found


# --------------------------------------------------------------------- 测试
def test_every_third_party_import_is_declared():
    """每个第三方 import 都必须在 pyproject 里声明（required 或 optional 均可）。

    这是清理依赖时的安全网：删掉一个重包之前，先跑这条，
    就能看出"它是不是某个依赖的唯一来源"。
    """
    declared = _declared()
    known = declared["required"] | declared["optional"]
    used = _third_party_imports()

    missing: list[str] = []
    for module, files in sorted(used.items()):
        if module in ENV_PROVIDED or module in LEGACY_OPTIONAL:
            continue
        dist = IMPORT_TO_DIST.get(module, module).lower().replace("_", "-")
        if dist not in known:
            sample = sorted(files)[:2]
            missing.append(f"{module}（发行包 {dist}）被 {len(files)} 个文件 import，例：{sample}")

    assert not missing, (
        "以下第三方包被代码 import，但 pyproject 里没有声明"
        "（换台机器就会缺，而缺了可能只是静默降级）：\n  " + "\n  ".join(missing)
    )


def test_yaml_is_a_hard_requirement():
    """``yaml`` 必须是**硬依赖**，不能只放 optional。

    原因：``recipes/schema.py`` 在模块顶层 import yaml，而配方是新引擎的
    核心功能。放 optional 意味着"没装就整个引擎起不来"——那不是可选。
    """
    declared = _declared()
    assert "pyyaml" in declared["required"], (
        "pyyaml 必须在 [project].dependencies 里（recipes/schema.py 顶层 import）"
    )


def test_import_to_dist_map_covers_used_modules():
    """新出现的第三方 import 要同步补 IMPORT_TO_DIST，避免名字映射漏掉。"""
    stdlib = _stdlib_names()
    used = _third_party_imports()
    unmapped = [
        module
        for module in used
        if module not in stdlib
        and module not in IMPORT_TO_DIST
        and module not in ENV_PROVIDED
        and module not in LEGACY_OPTIONAL
        and module != "gis_cli"
    ]
    assert not unmapped, (
        "这些 import 没有登记到 IMPORT_TO_DIST（无法判断对应哪个发行包）：\n  "
        + ", ".join(sorted(unmapped))
    )


@pytest.mark.parametrize("module", ["arcpy"])
def test_arcpy_not_required_in_pyproject(module: str):
    """arcpy 由 ArcGIS Pro 提供，绝不能写进 pyproject（pip 装不了）。"""
    declared = _declared()
    assert module not in declared["required"]
    assert module not in declared["optional"]
