# -*- coding: utf-8 -*-
"""新引擎的依赖边界：不依赖旧引擎的重包。

**为什么这条边界值得一个测试**：

- CI 跑在 GitHub runner 上，没有 ArcGIS Pro；核心 `dependencies` 里的
  `langchain*` / `litellm`（单个 37 MB）/ `chromadb`（拉 onnxruntime）
  只被**已冻结的旧引擎**（`agent/`、`planner.py`）用到；
- 一旦有人在 `engine/`、`runtime/`、`ui/`、`cartography/`、`datacatalog/`
  里顺手 import 了这些包，CI 就会在"装依赖"那一步莫名其妙变慢/失败，
  而且这个破坏很隐蔽（本地开发机上都装着，永远看不出来）；
- 新引擎"不装 ArcGIS、不装重包也能启动/规划/验收"是它相对旧引擎的硬性
  优势，值得焊住。

这里用**静态扫描**而不是真去 import：import 测试会受本机环境影响
（本机装了就能过），静态扫描才能在任何环境下给出同一个结论。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

try:  # Python 3.11+ 才有 tomllib；包声明支持 >=3.10，测试要能在那儿跑
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "gis_cli"

#: 新引擎各模块目录（相对 src/gis_cli）。这些目录下的代码必须在
#: "只有 rich/openai/prompt-toolkit/jupyter-client/pillow/numpy/typer/pydantic"
#: 的环境里也能 import。
ENGINE_DIRS = ("engine", "runtime", "ui", "cartography", "datacatalog", "trace", "recipes", "requirements", "envcheck", "report")

#: 只被旧引擎使用的重包（新引擎不得 import）。
LEGACY_ONLY = {
    "langchain",
    "langchain_community",
    "langchain_openai",
    "langchain_text_splitters",
    "chromadb",
    "litellm",
}


def _iter_py(directory: Path):
    for path in sorted(directory.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _imported_roots(source: str) -> set[str]:
    """静态解析出这段代码 import 的顶层包名（含函数内 import）。"""
    roots: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - 不该发生
        return roots
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对 import（from ..x import y）不算第三方
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("subdir", ENGINE_DIRS)
def test_engine_does_not_import_legacy_heavy_packages(subdir: str):
    """新引擎不得 import 旧引擎的重包（langchain / litellm / chromadb）。"""
    directory = SRC / subdir
    if not directory.is_dir():  # pragma: no cover - 目录改名时提示
        pytest.skip(f"{subdir}/ 不存在")

    offenders: list[str] = []
    for path in _iter_py(directory):
        roots = _imported_roots(path.read_text(encoding="utf-8", errors="replace"))
        hit = roots & LEGACY_ONLY
        if hit:
            rel = path.relative_to(SRC)
            offenders.append(f"{rel}: {', '.join(sorted(hit))}")

    assert not offenders, (
        "新引擎出现了对旧引擎重包的依赖（会让 CI 装依赖变慢/失败）：\n  "
        + "\n  ".join(offenders)
    )


def test_engine_import_does_not_pull_legacy_stack():
    """真 import 一次引擎入口，确认旧引擎栈没有被间接拉进来。

    静态扫描只能看"直接写了什么"，这条补上"实际加载了什么"。
    本机即使装了 langchain 也能跑 —— 断言的是**没有被导入**。
    """
    import subprocess
    import sys

    code = (
        "import sys;"
        "import gis_cli.engine.runner;"
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        "{'langchain','langchain_community','langchain_openai','chromadb','litellm'}];"
        "print('|'.join(sorted(bad)))"
    )
    env = {"PYTHONPATH": str(SRC.parent), "PYTHONUTF8": "1"}
    import os

    merged = {**os.environ, **env}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=merged,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    loaded = [item for item in result.stdout.strip().split("|") if item]
    assert not loaded, f"import 引擎时顺带拉起了旧引擎栈: {loaded}"


def test_ci_workflow_exists_and_excludes_arcpy():
    """CI 配置必须存在，且真的排除了需要 arcpy 的测试。"""
    workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    assert workflow.exists(), "缺少 CI 配置"
    text = workflow.read_text(encoding="utf-8")
    assert 'not arcpy' in text, "CI 必须排除需要 arcpy 的测试"
    assert "windows-latest" in text, "本项目只在 Windows 上成立（ArcPy）"
    # 依赖必须从 pyproject 装（`-e .`），不能手工维护一份清单：
    # 以前 CI 手写过 `pip install typer pydantic rich ...`，删掉 langchain 后
    # pyyaml 变成显式依赖却没人更新它，CI 直接 ModuleNotFoundError（实测）。
    assert "-e ." in text or '-e ".[dev]"' in text, "CI 必须从 pyproject 装依赖"
    # 只看真正的命令行（注释里提到旧清单不算）
    install_lines = [
        l.strip() for l in text.splitlines()
        if l.strip().startswith("pip install") or l.strip().startswith("python -m pip install")
    ]
    assert install_lines, "找不到安装步骤"
    assert not any("pydantic" in l for l in install_lines), "pydantic 已不是本包依赖"


def test_pytest_marker_registered():
    """``arcpy`` marker 必须在 pyproject 里注册，否则 pytest 会告警。"""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert "markers" in text and "arcpy:" in text
    assert re.search(r'"arcpy:.*"', text), "marker 描述缺失"


def test_chromadb_is_not_declared_as_dependency():
    """chromadb 是**死依赖**，不得再声明。

    事实依据（清理前实测）：
    - 源码里 ``grep -r chromadb src/`` 零命中；
    - ``agent/vector_retriever.py`` 是纯本地 md5 词袋实现，从来没接过 chroma；
    - 开发 venv 里根本没装 chromadb，而全量 480 个测试照样全过。

    它留在 ``dependencies`` 里的唯一后果：每个用户 ``pip install`` 都要多拉
    onnxruntime 等一大票包（几分钟、数百 MB），CI 也变慢。
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    # 只看 [project] 的 dependencies 段，注释里提到 chromadb 是允许的

    data = tomllib.loads(text)
    deps = [d.lower() for d in data["project"]["dependencies"]]
    offenders = [d for d in deps if d.startswith("chromadb")]
    assert not offenders, f"chromadb 是死依赖，不该出现在 dependencies 里: {offenders}"


def test_vector_retriever_is_pure_python():
    """向量检索器必须是纯本地实现（不依赖 chroma / 外部 embedding 服务）。

    这是"跨进程稳定、零外部依赖"的设计选择：md5 词袋 + 中文 bigram。
    如果哪天真接了 chroma，这个测试会提醒：请同步更新依赖声明与 CI。
    """
    source = (SRC / "agent" / "vector_retriever.py").read_text(encoding="utf-8")
    roots = _imported_roots(source)
    assert "chromadb" not in roots
    # 也不该偷偷接外部 embedding HTTP 服务
    for banned in ("sentence_transformers", "openai"):
        assert banned not in roots, f"vector_retriever 不应依赖 {banned}"


# ---------------------------------------------- 无重包时新引擎必须能跑
#: 新引擎的入口模块（chat / loop / 验收 / 工具 / UI）。
_ENGINE_ENTRY_MODULES = [
    "gis_cli.engine.runner",
    "gis_cli.engine.loop",
    "gis_cli.engine.tools",
    "gis_cli.engine.verifier",
    "gis_cli.engine.web_search",
    "gis_cli.agent.evaluation",
    "gis_cli.agent.interactive",
    "gis_cli.agent.cli",
    "gis_cli.agent.vector_retriever",
    "gis_cli.recipes.store",
    "gis_cli.datacatalog.store",
    "gis_cli.skills.catalog",
    "gis_cli.ui.prompt",
    "gis_cli.ui.render",
]


def test_engine_imports_without_heavy_packages():
    """**核心保护**：在没有 pydantic / langchain / litellm 的环境里，
    新引擎的全部入口模块必须仍能 import。

    这条是"旧引擎可以整体下线"的硬性依据。以前 ``agent/__init__.py`` 和
    ``skills/__init__.py`` 在包初始化时就立即 import 旧引擎模块，导致
    ``import gis_cli.agent.interactive`` 会顺带拉进 pydantic —— 新引擎明明
    一行都不用，却被拖着必须装 langchain。已改成 PEP 562 惰性导出。

    用子进程 + 伪造 import 来模拟"没装这些包"，比静态扫描更真实：
    它抓的是**实际加载行为**，包括间接依赖。
    """
    import os
    import subprocess
    import sys

    code = """
import builtins, sys
_real = builtins.__import__
BANNED = {"pydantic", "langchain", "langchain_core", "langchain_community",
          "langchain_openai", "langchain_text_splitters", "litellm", "chromadb"}
def _guard(name, *a, **k):
    if name.split(".")[0] in BANNED:
        raise ImportError("simulated missing: " + name)
    return _real(name, *a, **k)
builtins.__import__ = _guard
for _m in list(sys.modules):
    if _m.split(".")[0] in BANNED:
        del sys.modules[_m]
failed = []
for mod in %r:
    try:
        __import__(mod)
    except Exception as exc:
        failed.append(f"{mod}: {type(exc).__name__}: {exc}")
print("FAILED=" + "|".join(failed))
""" % (_ENGINE_ENTRY_MODULES,)

    env = {**os.environ, "PYTHONPATH": str(SRC.parent), "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180
    )
    assert result.returncode == 0, result.stderr[-2000:]
    marker = [line for line in result.stdout.splitlines() if line.startswith("FAILED=")]
    assert marker, f"子进程没有输出标记：{result.stdout[-800:]}"
    failed = [item for item in marker[0][len("FAILED="):].split("|") if item]
    assert not failed, (
        "以下新引擎模块在缺 pydantic/langchain/litellm 时 import 失败"
        "（说明有间接依赖没解耦）：\n  " + "\n  ".join(failed)
    )


def test_legacy_names_still_importable():
    """惰性导出不能破坏向后兼容：``from gis_cli.agent import Memory`` 仍可用。

    旧引擎下线后这个测试会随旧名字一起删除。
    """
    import gis_cli.agent as agent_pkg

    # 只验证"惰性机制存在"，不真的 import（本机可能没装 pydantic）
    assert "Memory" in agent_pkg.__all__
    assert hasattr(agent_pkg, "__getattr__")
    with pytest.raises(AttributeError):
        agent_pkg.this_name_does_not_exist


# ------------------------------------------------- 旧引擎不参与打包
def test_legacy_is_excluded_from_packaging():
    """``gis_cli.legacy`` 必须被 setuptools 排除，且不产生任何入口点。

    为什么重要：旧引擎依赖 pydantic/langchain/litellm，而这些包已从
    dependencies 移除。如果 legacy 被误打包，用户 ``pip install`` 后会得到一个
    import 就崩的包；如果 ``[project.scripts]`` 还指向 legacy 模块，pip 会
    生成指向不存在模块的 .exe，用户一跑就 ImportError。
    """

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    exclude = data["tool"]["setuptools"]["packages"]["find"].get("exclude") or []
    assert any("legacy" in pattern for pattern in exclude), (
        f"setuptools 必须排除 legacy，实际 exclude={exclude}"
    )

    scripts = data["project"].get("scripts") or {}
    for name, target in scripts.items():
        assert not target.startswith("gis_cli.legacy"), f"入口点 {name} 指向 legacy"
        # 旧引擎的三个入口（gis-cli / gis-cli-api / gis-cli-ui）应已删除
        for gone in ("gis_cli.cli:", "gis_cli.web."):
            assert not target.startswith(gone), f"入口点 {name} 指向已下线的 {gone}"


def test_legacy_is_not_importable_by_engine():
    """新引擎任何模块都不能 import legacy（否则等于没下线）。"""
    offenders: list[str] = []
    for path in _iter_py(SRC):
        if "legacy" in path.relative_to(SRC).parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for node in ast.walk(ast.parse(text)):
            target = ""
            if isinstance(node, ast.Import):
                target = ",".join(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                target = node.module
            if "gis_cli.legacy" in target or target.startswith("legacy"):
                offenders.append(f"{path.relative_to(SRC)}: {target}")
    assert not offenders, "新引擎仍然 import 了 legacy：\n  " + "\n  ".join(offenders)
