# -*- coding: utf-8 -*-
"""把仓库的 ``scripts`` 目录幂等加入**用户级** PATH（``HKCU\\Environment``）。

## 为什么需要它

``giscli`` 想在任何目录直接敲，就得有一个 PATH 能解析到的名字。而 pip 生成的
``gis-agent.exe`` 在 Pro env 里那份是坏的（任何参数 rc=1、零输出），Pro env
的 pip 本身也坏了（``BuildDependencyInstallError``）——所以这里落一个**目录级**
PATH 条目（里面是 ``scripts\\giscli.cmd``），不碰 pip、不生成 exe、也不动系统
级 PATH。

## 安全约束

* 只写 ``HKCU\\Environment``（当前用户），**绝不写** ``HKLM``（系统级）；
* 先打印旧值当备份，重复执行不会重复追加；
* ``--remove`` 只删本条目，其余条目原样保留；
* **读不到旧值就报错退出**（rc=1），绝不把「读失败」当成「空 PATH」写回——
  那等于把用户原有的用户级 PATH 整条覆盖掉。

## 用法

    python scripts\\install_shim.py            # 加入 PATH（幂等，缺省是脚本所在的 scripts\\）
    python scripts\\install_shim.py --dry-run  # 只看会怎么改
    python scripts\\install_shim.py --remove   # 还原缺省那一条

``--scripts-dir`` 缺省是脚本自己所在的 ``scripts\\``；要动 ``install.py`` 装的目录必须
显式给目录，否则删的不是那一条（下面这条是 cmd 写法；PowerShell 用
``"$env:LOCALAPPDATA\\Programs\\giscli"``，Git Bash 抄 ``install.py`` 打印的字面路径）：

    python scripts/install_shim.py --remove --scripts-dir "%LOCALAPPDATA%\\Programs\\giscli"

退出码：0 成功（含「本来就在」）；1 失败。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - 非 Windows 平台没有 winreg
    import winreg
except ImportError:  # pragma: no cover
    winreg = None  # type: ignore[assignment]

#: 用户级环境变量所在的注册表键（相对 HKEY_CURRENT_USER）
KEY_PATH = "Environment"
#: 只动这一个值
VALUE_NAME = "Path"
#: 缺省写回的类型：与 Windows 自带的用户 PATH 一致（REG_EXPAND_SZ），`%VAR%` 仍可展开。
#: 非 Windows 上没有 winreg，留一个同值常量给测试用（REG_EXPAND_SZ == 2）。
DEFAULT_VALUE_TYPE = getattr(winreg, "REG_EXPAND_SZ", 2)


@dataclass
class Outcome:
    """一次安装/卸载的结果（测试与退出码都看它）。"""

    ok: bool
    changed: bool
    entry: str
    previous: str
    current: str
    detail: str = field(default="")


def _comparable(raw: str) -> str:
    """PATH 条目的比较用形式：去引号、去尾部分隔符、大小写不敏感。

    不这么归一就会重复追加：``E:\\a\\scripts`` / ``E:\\a\\scripts\\`` /
    ``"E:\\A\\Scripts"`` 是同一条，Windows 路径本身也不区分大小写。
    """
    text = (raw or "").strip().strip('"').strip()
    if not text:
        return ""
    try:
        return os.path.normcase(os.path.normpath(text)).rstrip("\\/")
    except (OSError, ValueError):  # pragma: no cover - 极端非法路径
        return os.path.normcase(text).rstrip("\\/")


def split_entries(value: str) -> list[str]:
    """把 PATH 值拆成条目（保留空串位置，便于原样写回）。"""
    return (value or "").split(";")


def add_entry(value: str, entry: str) -> tuple[str, bool]:
    """把 ``entry`` 追加到 PATH 值末尾；已存在则原样返回。"""
    if any(_comparable(item) == _comparable(entry) for item in split_entries(value)):
        return value, False
    base = (value or "").strip()
    if not base:
        return entry, True
    if not base.endswith(";"):
        base += ";"
    return base + entry, True


def remove_entry(value: str, entry: str) -> tuple[str, bool]:
    """从 PATH 值里删掉 ``entry``；本来没有就原样返回。"""
    entries = split_entries(value)
    kept = [item for item in entries if _comparable(item) != _comparable(entry)]
    if len(kept) == len(entries):
        return value, False
    # join 已经消化掉自己那一条两侧的分隔符；**不能**再 strip(";")——那会把
    # 用户自己 PATH 首尾的空条目（末尾 `;` 表示「当前目录」）一起删掉。
    return ";".join(kept), True


def _read_user_path() -> tuple[str, int]:
    """读当前用户的 PATH（值与类型）。

    键/值不存在 ⇒ ``("", DEFAULT_VALUE_TYPE)``（用户本来就没设过）。
    其余 OSError（PermissionError、键被占用等）**原样抛出**：调用方宁可
    报错退出，也不能拿空串当旧值覆盖写回。
    """
    if winreg is None:  # pragma: no cover
        raise RuntimeError("这台机器没有 winreg（不是 Windows）")
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, KEY_PATH, 0, winreg.KEY_READ
        ) as key:
            value, value_type = winreg.QueryValueEx(key, VALUE_NAME)
            return str(value), int(value_type)
    except FileNotFoundError:  # 键或值不存在：用户本来就没有用户级 PATH
        return "", DEFAULT_VALUE_TYPE
    # 其余 OSError（PermissionError、键被占用…）**不在这里吞**：读失败 ≠ 空 PATH，
    # 由 install()/uninstall() 报错收口，否则整条覆盖写回就会丢掉用户原有条目。


def _read_failed(entry: str, exc: BaseException) -> Outcome:
    """读不到用户级 PATH 时的统一收口：只报错，绝不写注册表。"""
    _emit(f"[ERROR] 读不到用户级 PATH，未改动：{exc}")
    return Outcome(
        ok=False, changed=False, entry=entry, previous="", current="", detail=str(exc)
    )


def _write_user_path(value: str, value_type: int = DEFAULT_VALUE_TYPE) -> None:
    """写回当前用户的 PATH（只此一个值，别的值一律不碰）。"""
    if winreg is None:  # pragma: no cover
        raise RuntimeError("这台机器没有 winreg（不是 Windows）")
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER, KEY_PATH, 0, winreg.KEY_SET_VALUE
    ) as key:
        if value:
            winreg.SetValueEx(key, VALUE_NAME, 0, value_type, value)
            return
        try:
            winreg.DeleteValue(key, VALUE_NAME)
        except FileNotFoundError:
            pass


def _broadcast_change() -> None:
    """通知系统重读环境变量（新开的终端才看得到；失败不影响写入）。"""
    try:  # pragma: no cover - 纯副作用，测试不覆盖
        import ctypes

        ctypes.windll.user32.SendMessageTimeoutW(
            0xFFFF,  # HWND_BROADCAST
            0x001A,  # WM_SETTINGCHANGE
            0,
            "Environment",
            0x0002,  # SMTO_ABORTIFHUNG
            5000,
            None,
        )
    except Exception:
        pass


def _emit(text: str = "") -> None:
    """打印；老控制台编码装不下路径里的字符时降级而不是崩。"""
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - 只在 GBK 控制台会出现
        print(text.encode("gbk", "replace").decode("gbk", "replace"))


def install(scripts_dir: Path | str, *, dry_run: bool = False) -> Outcome:
    """幂等加入用户级 PATH。"""
    entry = str(scripts_dir)
    try:
        previous, value_type = _read_user_path()
    except (RuntimeError, OSError) as exc:
        return _read_failed(entry, exc)

    _emit("旧的用户级 PATH（备份，改之前先留一份）：")
    _emit(f"  {previous or '(空)'}")

    new_value, changed = add_entry(previous, entry)
    if not changed:
        _emit(f"用户级 PATH 里已经有这一条，未改动：{entry}")
        return Outcome(
            ok=True, changed=False, entry=entry, previous=previous, current=previous,
            detail="already present",
        )

    _emit(f"要加入的条目：{entry}")
    if dry_run:
        _emit("[dry-run] 没有写入注册表。")
        return Outcome(
            ok=True, changed=False, entry=entry, previous=previous, current=previous,
            detail="dry-run",
        )

    try:
        _write_user_path(new_value, value_type)
    except OSError as exc:
        _emit(f"[ERROR] 写 HKCU\\{KEY_PATH}\\{VALUE_NAME} 失败：{exc}")
        return Outcome(ok=False, changed=False, entry=entry, previous=previous, current=previous, detail=str(exc))

    _broadcast_change()
    _emit("已写入 HKCU\\Environment\\Path。")
    _emit("新开一个终端窗口后生效（当前这个窗口的 PATH 不会变）。")
    _emit(
        "撤销：python scripts/install_shim.py --remove "
        f'--scripts-dir "{entry}"'
    )
    return Outcome(
        ok=True, changed=True, entry=entry, previous=previous, current=new_value,
        detail="added",
    )


def uninstall(scripts_dir: Path | str, *, dry_run: bool = False) -> Outcome:
    """从用户级 PATH 移除本条目（其余条目原样保留）。"""
    entry = str(scripts_dir)
    try:
        previous, value_type = _read_user_path()
    except (RuntimeError, OSError) as exc:
        return _read_failed(entry, exc)

    new_value, changed = remove_entry(previous, entry)
    if not changed:
        _emit(f"用户级 PATH 里没有这一条，未改动：{entry}")
        return Outcome(
            ok=True, changed=False, entry=entry, previous=previous, current=previous,
            detail="not present",
        )

    if dry_run:
        _emit(f"[dry-run] 会删除：{entry}")
        _emit(f"  {new_value or '(空)'}")
        return Outcome(
            ok=True, changed=False, entry=entry, previous=previous, current=previous,
            detail="dry-run",
        )

    try:
        _write_user_path(new_value, value_type)
    except OSError as exc:
        _emit(f"[ERROR] 写 HKCU\\{KEY_PATH}\\{VALUE_NAME} 失败：{exc}")
        return Outcome(ok=False, changed=False, entry=entry, previous=previous, current=previous, detail=str(exc))

    _broadcast_change()
    _emit(f"已从用户级 PATH 删除：{entry}")
    _emit(f"  新的用户级 PATH：{new_value or '(空)'}")
    return Outcome(
        ok=True, changed=True, entry=entry, previous=previous, current=new_value,
        detail="removed",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把 giscli 所在的 scripts 目录幂等加入用户级 PATH（HKCU\\Environment）"
    )
    parser.add_argument(
        "--remove", action="store_true", help="从用户级 PATH 移除本条目（回滚）"
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写注册表")
    parser.add_argument(
        "--scripts-dir",
        default=str(Path(__file__).resolve().parent),
        help="要加入 PATH 的目录（默认本脚本所在目录）",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):  # 老控制台编码兜底
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    action = uninstall if args.remove else install
    outcome = action(Path(args.scripts_dir), dry_run=args.dry_run)
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
