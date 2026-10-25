# -*- coding: utf-8 -*-
"""File GDB 的真假判定。

**为什么要单独一个模块**：`arcpy.Exists()` 对一个用 `os.makedirs` 建出来的普通目录
也返回 True。任何写成 ``if arcpy.Exists(p)`` 的“库存在吗”判断都会放行假库，随后
arcpy 把库里的二进制表当文本读，抛 ``UnicodeDecodeError``（实测 14 届组合任务：
一次 382s + 一次 833s 都耗在这上面）。

真 File GDB 目录里有 ESRI 的标记文件（`gdb` / `timestamps` / `a00000001.gdbtable`），
假库没有——所以判定只看目录内容，不问 arcpy，在没有 arcpy 的解释器里也能用。
"""

from __future__ import annotations

from pathlib import Path

#: 真 File GDB 目录里必然出现的标记文件（小写比较）
GDB_MARKERS = ("gdb", "timestamps", "a00000001.gdbtable")


def is_real_gdb(path: str | Path) -> bool:
    """目录是不是真正的 File GDB（不看扩展名，看内容）。"""
    target = Path(path)
    try:
        names = {item.name.lower() for item in target.iterdir()}
    except OSError:
        return False
    return any(marker in names for marker in GDB_MARKERS)


def is_fake_gdb(path: str | Path) -> bool:
    """存在、是目录、但不是真 GDB —— 也就是 `os.makedirs` 建出来的假库。"""
    target = Path(path)
    try:
        return target.is_dir() and not is_real_gdb(target)
    except OSError:
        return False
