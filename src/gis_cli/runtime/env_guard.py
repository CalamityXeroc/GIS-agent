# -*- coding: utf-8 -*-
"""arcpy.env 的按次隔离：跑前快照 + 盖成干净基线，跑后（含异常）还原。

为什么隔离做在内核执行层（DAIR-289）：``arcpy.env`` 是持久内核里的**进程级
全局态**。配方各自用 ``try/finally`` 还原自己改过的项，但 ``finally`` 只能包住
它自己那段代码 —— 11 个内置配方的**第 1 步**（``CopyFeatures``）都发生在各自
``try`` 之前（``zone_raster_sum.yaml`` 里 ``:78`` 的 CopyFeatures、``try`` 从
``:104`` 才开始），跑前 env 里如果还留着别的任务设的 ``extent`` /
``outputCoordinateSystem``，这一步就会被静默重投影、按范围裁掉要素，最后报一个
与真因无关的错（DAIR-288 实测：``zone_raster_sum`` 落到 ``np.bincount`` 的
``zero-size array to reduction operation maximum``）。

逐个配方补 ``finally`` 补不到"最早那一步"，所以放在唯一不遗漏的位置：
``ArcpyKernel.execute()`` 拼出来的那段代码。一次执行的生命周期::

    快照 6 项 → 盖成干净基线（extent/snapRaster/outputCoordinateSystem/
    cellSize/mask = None，overwriteOutput = True）→ 用户代码 → 还原快照

语义变化（有意为之）：``arcpy.env`` 从此是**按次作用域**。上一次执行设的
extent/mask 不会自动带进下一次 —— 这正是本单要的效果（"跑前不许被已有 env
带偏"）；确实要跨步传递的值请放变量里，不要放 env。

``arcpy.env.workspace`` **不在**隔离范围：它由 ``execute()`` 的前缀每次按任务
工作区重新钉死，本来就是每次覆盖的。
"""

from __future__ import annotations

#: 参与隔离的 env 项。挑的标准：**会静默改变下一次结果**的全局态。
#:
#: - ``extent`` / ``snapRaster`` / ``cellSize`` / ``mask``：写数据时静默裁范围、
#:   改格网（DAIR-135/138 就是这里的泄漏）；
#: - ``outputCoordinateSystem``：写数据时静默重投影，产物 CRS 与输入不符；
#: - ``overwriteOutput``：配方会打开它，跑完不还原（历史行为）。
GUARDED_ENV_KEYS = (
    "extent",
    "snapRaster",
    "outputCoordinateSystem",
    "cellSize",
    "mask",
    "overwriteOutput",
)

#: 干净基线里置 ``None`` 的项 —— None 就是 arcpy 默认值：不继承外部格网/范围/
#: CRS/掩膜，工具按自己的输入决定这些。
_RESET_KEYS = tuple(key for key in GUARDED_ENV_KEYS if key != "overwriteOutput")

#: 快照在内核命名空间里的变量名。还原成功即 ``pop`` 掉，异常路径靠它判断
#: "这一趟还没还原"。
SNAPSHOT_VAR = "__arcpy_env_snapshot__"


def snapshot_code() -> str:
    """快照 + 盖上干净基线的内核代码，接在用户代码**之前**。

    快照必须早于任何改写（包括 ``execute()`` 前缀里的 ``overwriteOutput = True``），
    记下来的才是"这次执行之前"的外部态。
    """
    keys = ", ".join(f"'{key}'" for key in GUARDED_ENV_KEYS)
    lines = [f"{SNAPSHOT_VAR} = {{_k: getattr(arcpy.env, _k) for _k in ({keys},)}}"]
    lines.extend(f"arcpy.env.{key} = None" for key in _RESET_KEYS)
    lines.append("arcpy.env.overwriteOutput = True")
    return "\n".join(lines) + "\n"


def restore_code() -> str:
    """还原快照的内核代码。

    成功路径接在结果尾巴之后（同一趟，不多一次 ZMQ 往返）；用户代码抛异常时
    cell 会中断，由 ``execute()`` 的错误路径单独补一趟本段。还原本身**绝不能**
    把一次成功的执行变成失败（快照里可能攥着已被删掉的数据源），所以整段吞异常、
    只往 stderr 留一行诊断。
    """
    return (
        f"if globals().get('{SNAPSHOT_VAR}') is not None:\n"
        f"    _snap = globals().pop('{SNAPSHOT_VAR}')\n"
        "    try:\n"
        "        for _k, _v in _snap.items():\n"
        "            setattr(arcpy.env, _k, _v)\n"
        "    except Exception as _exc:\n"
        "        import sys as _sys\n"
        "        _sys.stderr.write('[env_guard] arcpy.env 还原失败: ' + repr(_exc) + '\\n')\n"
        "\n"
    )
