# -*- coding: utf-8 -*-
"""测试全局配置：按目录/文件名自动打 marker。

CI 只跑 ``pytest -m "not arcpy"``（见 .github/workflows/ci.yml）。带 arcpy 的
测试每个都要起子进程（实测单个 ~12s），全量 15 分钟以上，而且 CI 机器上
根本没有 ArcGIS Pro。

为什么用 conftest 自动打标而不是逐个文件写 ``pytestmark``：
- 旧引擎（``agent/`` 下那套 plan-then-execute）整批依赖 arcpy + langchain，
  逐个标注既容易漏、也会让"哪些是新引擎的测试"变得不清晰；
- 新引擎测试（``test_engine_*`` / ``test_ui_*`` / ``test_cartography_*``）
  必须在无 arcpy 环境里也能跑 —— 这是新引擎相对旧引擎的硬性优势之一，
  自动标注让这个边界在 CI 里天然被守住。
"""

from __future__ import annotations

import io
import sys

import pytest


class FakeTTY(io.StringIO):
    """可写、且自称是终端的字符流：给“真终端语义”的用例当 conhost 的无头替身。"""

    def isatty(self) -> bool:
        return True


@pytest.fixture()
def install_fake_tty(monkeypatch):
    """返回一个“把当前进程换成真终端语义”的函数（**在用例体里**调用）。

    pytest 的 fd 捕获会在 fixture 跑完之后重设 ``sys.stdout``，所以只能在用例体里
    才 patch；返回值是新的 ``sys.stdout``。
    """

    def _install():
        monkeypatch.setattr(sys, "stdin", FakeTTY())
        out = FakeTTY()
        monkeypatch.setattr(sys, "stdout", out)
        return out

    return _install

#: 需要 ArcGIS（arcpy）的测试文件。
#:
#: 旧引擎（agent/、planner.py、orchestrator.py、core/、tools/、web/）已迁入
#: ``src/gis_cli/legacy/`` 并从打包中排除，其测试也已删除 —— 所以这份名单
#: 现在只剩"确实要跑 arcpy 才能验证"的新引擎测试。
_NEEDS_ARCPY: set[str] = {
    "test_arcpy_bridge_codegen.py",
    "test_engine_hygiene.py",  # 部分用例要真跑 arcpy 统计交付目录
}


def pytest_collection_modifyitems(config, items):
    """给旧引擎/依赖 arcpy 的测试自动加 ``arcpy`` marker。"""
    for item in items:
        filename = item.path.name
        if filename in _NEEDS_ARCPY:
            item.add_marker(pytest.mark.arcpy)
