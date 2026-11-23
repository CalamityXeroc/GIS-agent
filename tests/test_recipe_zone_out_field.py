# -*- coding: utf-8 -*-
"""``zonal_statistics`` 的 ``_zone_out_field``：统计表里分区列名的匹配口径。

ZonalStatisticsAsTable 会改写分区列名以避开输出表自身的保留名（.gdb 表把 OBJECTID
让给表自身 OID → OBJECTID_1；.dbf 里 FID 变 FID_），后续 JoinField / AlterField /
上报的 zone_field 都得用这个真名。

只认**精确名**或**精确名 + 下划线（+ 序号）**。用 ``startswith`` 会把输出表里恰好
同前缀的无关列（OBJECTIDX、FID_1 之外的 FIDX…）选进来——join 到错的列上，
行序凑巧对得上时数值静默错位，一句报错都没有。

函数从**渲染后的 arcpy 模板**里原样抽出来测：将来换实现，测的仍是配方里那段代码。
不需要真 arcpy —— ``arcpy`` 用只提供 ``ListFields`` 的假对象替身。
"""

from __future__ import annotations

import ast
import re
import textwrap
from types import SimpleNamespace

import pytest

from gis_cli.recipes.store import RecipeLibrary


def _render_params(recipe) -> dict:
    """与渲染快照同一套取值规则：有默认值用默认值，其余按类型给确定值。"""
    params = {}
    for name, spec in recipe.params.items():
        if spec.default is not None:
            params[name] = spec.default
        elif spec.type in {"integer", "number"}:
            params[name] = 1
        else:
            params[name] = f"dummy_{name}"
    return params


def _zone_out_field_source() -> str:
    """``zonal_statistics`` 的 arcpy 模板渲染后，``_zone_out_field`` 的函数源码。"""
    lib = RecipeLibrary()
    lib.load()
    recipe = lib.get("zonal_statistics")
    assert recipe is not None, "内置配方 zonal_statistics 不见了"
    code = recipe.render(_render_params(recipe))
    node = next(
        (
            n
            for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.FunctionDef) and n.name == "_zone_out_field"
        ),
        None,
    )
    assert node is not None, "zonal_statistics 的 arcpy 模板里找不到 _zone_out_field"
    return textwrap.dedent(ast.get_source_segment(code, node))


def _zone_out_field(fields):
    """把抽出来的函数挂到「只认 ListFields」的假 arcpy 上。"""
    namespace = {
        # 模板顶上的 ``import arcpy, os, re`` 只覆盖模板本体，抽出来的函数要自己给
        "re": re,
        "arcpy": SimpleNamespace(
            ListFields=lambda table: [SimpleNamespace(name=n, type=t) for n, t in fields]
        )
    }
    exec(_zone_out_field_source(), namespace)  # noqa: S102 - 被测代码就是这段模板
    return namespace["_zone_out_field"]


@pytest.mark.parametrize(
    "fields, expected, want",
    [
        # 撞保留名被改名：分区列在改名后的那个，源上的同名 OID 让给表自身
        ([("OBJECTID", "OID"), ("OBJECTID_1", "Long")], "OBJECTID", "OBJECTID_1"),
        # 两个改名形态同时存在：工具撞名从 _1 往下试，_1 才是分区列
        (
            [("OBJECTID", "OID"), ("OBJECTID_2", "Long"), ("OBJECTID_1", "Long")],
            "OBJECTID",
            "OBJECTID_1",
        ),
        # 同前缀的无关列（OBJECTIDX）不许被选中
        (
            [("OBJECTID", "OID"), ("OBJECTIDX", "Long"), ("OBJECTID_1", "Long")],
            "OBJECTID",
            "OBJECTID_1",
        ),
        # .dbf 的改名没有序号，就是 FID_；FIDX 同样不许被选中
        ([("FID", "OID"), ("FIDX", "Long"), ("FID_", "Long")], "FID", "FID_"),
    ],
)
def test_zone_out_field_picks_the_renamed_zone_column(fields, expected, want):
    assert _zone_out_field(fields)(None, expected) == want
