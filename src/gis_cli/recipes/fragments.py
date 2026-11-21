# -*- coding: utf-8 -*-
"""配方模板可复用的 arcpy 代码片段。

为什么需要这一层：同一段「分区输入给的是容器还是图层」的判断要出现在 4 个分区配方里
（zonal_statistics / zonal_statistics_raster / zone_raster_sum / aggregate_to_zones），
复制四份就是四份迟早走样的代码。模板里写一行 ``{{@zone_layer@}}``，``Recipe.render``
把它替换成这里的源码；片段名写错时渲染直接报错（不会把占位符原样交给执行器）。

只在 arcpy 模板里用：GDAL 侧由 pyogrio 自己处理「没给 layer」的情形（读第一层 + warning），
两边取舍不同已在各配方的 pitfalls 里写清。
"""

from __future__ import annotations

import re

#: 模板里的片段占位：``{{@zone_layer@}}``（不匹配参数槽位 ``{{param}}``，因为 ``{{`` 后是 ``@``）。
FRAGMENT_SLOT = re.compile(r"\{\{\s*@([A-Za-z_][A-Za-z0-9_]*)\s*@\}\}")

#: 把「分区输入给的是容器」解析成具体的要素类路径。
#: 约定：片段里的函数吃一个路径、返回一个路径，调用方总是 ``zone_path = resolve_zone_layer(zone_path)``。
ZONE_LAYER = '''def resolve_zone_layer(path):
    """分区输入给的是容器（.gpkg / .gdb / 文件夹）时，解析成其中唯一的要素类。

    arcpy 把没指明图层的库/文件夹当工作空间（Describe 的 dataType 是 Workspace/Folder），
    上面没有 spatialReference——裸取 .spatialReference 抛的是 AttributeError，不是人话。
    容器里恰好一个要素类就替用户取用；多个图层必须让调用方指明（见 pitfalls）。
    """
    if arcpy.Describe(path).dataType not in ("Workspace", "Folder"):
        return path
    previous = arcpy.env.workspace
    try:
        # ListFeatureClasses 只认 env.workspace（Pro 3.6 没有 workspace 参数），列完还原
        arcpy.env.workspace = path
        layers = list(arcpy.ListFeatureClasses() or [])
        # ListFeatureClasses **不递归**：.gdb 里的要素类可能全在要素数据集里（fd/zones），
        # 这时它返回空列表。列一下数据集，报错才能指对方向——arcpy 自己认
        # ``ds.gdb/fd/zones`` 这种写法（实测），只是容器这一层不替你往下钻。
        datasets = [] if layers else list(arcpy.ListDatasets("*", "Feature") or [])
    finally:
        arcpy.env.workspace = previous
    if not layers:
        if datasets:
            raise ValueError(
                f"分区输入 {path} 是工作空间，要素类都在要素数据集里（{', '.join(datasets)}）："
                f"请给到要素类本身（如 {path}/数据集名/要素类名）"
            )
        raise ValueError(
            f"分区输入 {path} 是工作空间、里面没有要素类：请给出分区图层本身（如 .gpkg/图层名）"
        )
    if len(layers) > 1:
        raise ValueError(
            f"分区输入是多层容器（{len(layers)} 个图层: {', '.join(layers)}），"
            f"arcpy 后端不替你挑：请指明图层，如 {path}/图层名"
        )
    print(f"分区输入是工作空间容器（.gpkg/.gdb/文件夹），自动使用其中唯一的图层: {layers[0]}")
    return os.path.join(path, layers[0])
'''

#: 片段名 → 源码。模板里 ``{{@<名字>@}}`` 引用。
FRAGMENTS: dict[str, str] = {
    "zone_layer": ZONE_LAYER,
}


def fragment_names(template: str) -> set[str]:
    """模板里引用到的片段名（校验用：名字写错在渲染时报错，这里用于静态体检）。"""
    return set(FRAGMENT_SLOT.findall(template or ""))
