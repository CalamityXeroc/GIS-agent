# -*- coding: utf-8 -*-
"""执行后端契约：配方声明 backend，执行按声明路由。

为什么需要这一层：47 个老配方的 ``code_template`` 全是 arcpy，而民生模板库要
服务"没装 ArcGIS 的人"。同一套配方目录里必须能同时存在 arcpy 配方与 GDAL 配方，
并且**按声明**挑执行器 —— 而不是"谁能跑就丢给谁"：把 GDAL 配方丢进 ArcPy
内核，报的是 ``No module named 'osgeo'`` 之类看不懂的错。

可用后端的唯一真相来源是**执行器自己的声明**（``runner.backends``）。为什么不用
``importlib.util.find_spec("arcpy")`` 探测：arcpy 装在 ArcGIS Pro 的 Python 里，
引擎 venv 里永远 import 不到，用 find_spec 会把"装了 ArcGIS"误判成"没装"。
执行器没声明 → ``None``（不知道），调用方不设门禁 —— 历史默认所有执行器都是
arcpy，注入的假 runner 因此不受影响。
"""

from __future__ import annotations

from typing import Any

#: 已知后端名。列表顺序就是"多后端配方"的默认优先级。
BACKENDS = ("arcpy", "gdal")

#: 没写 ``backend`` 的配方按它跑 —— 47 个老配方零改动必须照样能用。
DEFAULT_BACKEND = "arcpy"

#: 报错文案里给人看的名字（"需 ArcGIS" / "需 GDAL 后端" 这两个字样有测试锁死）。
_LABELS = {
    "arcpy": "ArcGIS（arcpy）",
    "gdal": "GDAL 后端",
}

#: 缺后端时该做什么。
_HINTS = {
    "arcpy": "在装了 ArcGIS Pro 的机器上跑；或改用声明了 [gdal] 的配方。",
    "gdal": '装可选依赖 pip install "gis-task-cli[gdal]"；或改用 [arcpy] 配方。',
}


def normalize_backends(raw: Any) -> list[str]:
    """把 manifest 里的 ``backend`` 收敛成后端名列表。

    接受 ``"gdal"``（单个字符串）与 ``["arcpy", "gdal"]``（列表）。缺省、空列表、
    认不出的名字一律回落到 ``["arcpy"]``：写错一个词不该让配方变成"本机不可用"，
    按老行为跑更安全（arcpy 配方跑不起来时执行器自己会报错）。
    """
    if raw is None:
        return [DEFAULT_BACKEND]
    if isinstance(raw, str):
        raw = [raw]
    try:
        names = [str(item).strip().lower() for item in raw]
    except TypeError:
        return [DEFAULT_BACKEND]
    ordered: list[str] = []
    for name in names:
        if name in BACKENDS and name not in ordered:
            ordered.append(name)
    return ordered or [DEFAULT_BACKEND]


def runner_backends(runner: Any) -> set[str] | None:
    """执行器能跑的后端集合；``None`` = 执行器没声明（不知道，别设门禁）。"""
    declared = getattr(runner, "backends", None)
    if declared is None:
        return None
    if isinstance(declared, str):
        declared = [declared]
    try:
        names = {str(item).strip().lower() for item in declared}
    except TypeError:
        return None
    return {name for name in names if name in BACKENDS}


def pick_backend(recipe_backends: list[str] | None, available: set[str] | None) -> str | None:
    """按配方声明顺序挑一个本机能跑的后端；一个都跑不了 → ``None``。

    ``available is None`` 表示执行器没声明能力：不设门禁，按声明里的第一个走。
    """
    wanted = list(recipe_backends or [DEFAULT_BACKEND])
    if available is None:
        return wanted[0]
    for name in wanted:
        if name in available:
            return name
    return None


def unavailable_message(recipe_backends: list[str] | None, available: set[str] | None) -> str:
    """缺后端时的明确文案（含「需 ArcGIS」/「需 GDAL 后端」字样）。"""
    wanted = list(recipe_backends or [DEFAULT_BACKEND])
    needs = " / ".join(f"需 {_LABELS.get(name, name)}" for name in wanted)
    have = "、".join(sorted(available)) if available else "无"
    return f"配方声明后端 {wanted}，本机可用后端：{have}。{needs}。"


def backend_hint(recipe_backends: list[str] | None) -> str:
    """缺后端时的下一步建议。"""
    return " ".join(
        _HINTS[name] for name in (recipe_backends or [DEFAULT_BACKEND]) if _HINTS.get(name)
    )
