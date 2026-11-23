# -*- coding: utf-8 -*-
"""``benchmark/src_guard.py`` 的噪声口径：文件树指纹与 git 预检必须是同一套。

DAIR-81 打回的原因：``precheck()`` 直接拿 ``git status --porcelain -- src`` 的原始行
当污染证据，而 ``hash_tree()`` 把 ``__pycache__/`` / ``*.pyc`` / ``*.egg-info/`` 当噪声
排除。仓库里有 34 个**被跟踪**的 ``.pyc``，``runner.py`` 一 import 就重写它们 →
干净 clone 被自己判成污染（``SrcEscapeError``），闸门挡住了自己。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from src_guard import (  # noqa: E402
    SrcEscapeError,
    SrcGuard,
    _excluded,
    _git_noise,
    hash_tree,
)

_GIT_ENV = dict(
    os.environ,
    GIT_AUTHOR_NAME="test", GIT_AUTHOR_EMAIL="test@local",
    GIT_COMMITTER_NAME="test", GIT_COMMITTER_EMAIL="test@local",
)

#: import / pip install -e 的正常副产物，两侧都必须忽略。
_NOISE = [
    "src/gis_cli/__pycache__",
    "src/gis_cli/__pycache__/__init__.cpython-312.pyc",
    "src/gis_cli/stale.pyc",
    "src/gis_cli.egg-info",
    "src/gis_cli.egg-info/PKG-INFO",
    "src/.pytest_cache/v/cache/lastfailed",
]
#: 真正的越界形状，两侧都必须看见。
_REAL = [
    "src/gis_cli/__init__.py",
    "src/gis_cli/engine/runner.py",
    "src/arcpy/__init__.py",
]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        timeout=60, encoding="utf-8", errors="replace", env=_GIT_ENV,
    )


def _make_repo(tmp_path: Path) -> Path:
    """带 git 的假仓库：``src/`` 里一个已跟踪的 py + 一个已跟踪的 pyc。"""
    pkg = tmp_path / "repo" / "src" / "gis_cli"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "__init__.py").write_text("VERSION = 'clean'\n", encoding="utf-8")
    (pkg / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"\x00baseline")
    repo = pkg.parents[1]
    for args in (("init", "-q", "."), ("add", "-A"), ("commit", "-q", "-m", "baseline")):
        assert _git(repo, *args).returncode == 0
    return repo


@pytest.mark.parametrize("rel", _NOISE + _REAL)
def test_noise_predicate_has_one_source(tmp_path: Path, rel: str) -> None:
    """同一条路径：``hash_tree()`` 排除它 ⇔ git 预检把它当噪声（口径只有一处）。"""
    expected = rel in _NOISE
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1\n", encoding="utf-8")
    assert ((rel.removeprefix("src/")) not in hash_tree(tmp_path / "src")) is expected
    assert _excluded(Path(rel)) is expected
    assert _git_noise(f"?? {rel}") is expected
    assert _git_noise(f" M {rel}") is expected


def test_rename_with_one_real_side_is_not_noise() -> None:
    """rename 两侧都是噪声才算噪声：真文件改名进 ``__pycache__`` 仍要拦。"""
    assert not _git_noise("R  src/gis_cli/a.py -> src/gis_cli/__pycache__/a.pyc")
    assert _git_noise("R  src/gis_cli/__pycache__/a.pyc -> src/gis_cli/__pycache__/b.pyc")


def test_import_side_effects_do_not_block_precheck(tmp_path: Path) -> None:
    """被跟踪的 pyc 被重写 + ``.egg-info`` 未跟踪：干净 checkout 不能被判成污染。"""
    repo = _make_repo(tmp_path)
    (repo / "src" / "gis_cli" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"\x00rewritten")
    egg = repo / "src" / "gis_cli.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("Name: gis-task-cli\n", encoding="utf-8")

    raw = _git(repo, "status", "--porcelain", "--", "src").stdout
    assert ".pyc" in raw and "egg-info" in raw, f"前提不成立（git 侧没看到噪声）：{raw!r}"

    guard = SrcGuard(repo)
    guard.begin()  # 不抛 = 预检通过
    assert guard.before["git_status_src"] == []
    assert guard.precheck() == []


def test_tracked_py_change_still_blocks(tmp_path: Path) -> None:
    """真污染仍要拦住：改一个被跟踪的 ``.py``（非 pyc）。"""
    repo = _make_repo(tmp_path)
    (repo / "src" / "gis_cli" / "__init__.py").write_text("VERSION = 'poisoned'\n", encoding="utf-8")
    with pytest.raises(SrcEscapeError, match="git status -- src 非空"):
        SrcGuard(repo).begin()


def test_shadow_package_still_blocks(tmp_path: Path) -> None:
    """真污染仍要拦住：新增 ``src/arcpy/`` 影子包。"""
    repo = _make_repo(tmp_path)
    shadow = repo / "src" / "arcpy"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SrcEscapeError, match="顶层白名单"):
        SrcGuard(repo).begin()
