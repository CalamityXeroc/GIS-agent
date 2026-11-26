# -*- coding: utf-8 -*-
r"""启动前的安装检查：决定要不要跑 ``pip install -e .``。

## 为什么需要它

旧启动脚本原来每次启动都跑 ``pip install -e .``，实测要卡几十秒到
几分钟。而 editable 安装的本质只是往 site-packages 放一个 ``.pth``：

    site-packages/__editable__.gis_task_cli-0.7.1.pth
      → 内容就是一行：<仓库>\src

源码改动**自动生效**，根本不需要重装。只有下面几种情况才真的需要：

1. ``.pth`` 不存在（从没装过）；
2. ``.pth`` 指向的不是本仓库的 ``src``（换了目录/装了别的副本）；
3. ``pyproject.toml`` 比 ``.pth`` 新（依赖清单可能变了）；
4. 用户显式要求（``--reinstall``）。

反过来，如果 ``src`` **已经在本进程的 ``sys.path`` 上**（新入口
``scripts\giscli.cmd`` 就是靠 ``PYTHONPATH=<repo>\src`` 跑的），那包本来就能
import，根本没有「要不要 ``pip install``」这个问题 —— 也就不能让一个坏掉的
Pro env pip 把启动拦下来。

退出码：``0`` = 已就绪不用装；``1`` = 需要安装。

另外顺带报一句 web 预览的依赖（``pyproj``）：缺了界面照开、EPSG:4326 也照样
能看，但所有非 4326 的矢量会被降级成「不画几何」。它**不影响退出码**（不拦启动、
也不触发 ``pip install -e .``），只保证降级不再静默（DAIR-335）。
"""

from __future__ import annotations

import argparse
import glob
import os
import site
import sys
from pathlib import Path

#: 本包的发行名（决定 .pth 文件名里出现的关键词）
DIST_HINT = "gis_task_cli"

#: web 预览的硬依赖：模块名 → pip 安装串。非 EPSG:4326 的矢量靠 pyproj 重投影，
#: 缺了会被预览接口静默降级成「不画几何」（地图上只剩一条坐标系告警）。
WEB_DEPS = {"pyproj": "pyproj>=3.6"}


def missing_web_deps() -> list[str]:
    """当前解释器缺哪些 web 预览依赖。

    只查 ``find_spec``，不 import 本体：pyproj 导入要加载 PROJ 数据，
    启动检查不该付这个钱。半装状态（父包在、子模块坏）也算缺。
    """
    import importlib.util

    missing: list[str] = []
    for name in WEB_DEPS:
        try:
            present = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(name)
    return missing


def report_web_deps(stream: object = None) -> list[str]:
    """缺依赖就往 stderr 报一行（``--quiet`` 也报：静默降级正是本单要防的）。"""
    target = stream if stream is not None else sys.stderr
    missing = missing_web_deps()
    if not missing:
        return []
    print(
        f"[WARN] 缺少 {'、'.join(missing)}：非 EPSG:4326 的矢量预览会降级成「不画几何」，"
        "地图上只剩一条坐标系告警。",
        file=target,
    )
    specs = " ".join(f'"{WEB_DEPS[name]}"' for name in missing)
    # 两个等价路径：单装这一个包（离线/坏 pip 环境最省事），或装 [web] extra
    # （pyproj 就在 pyproject 的 [web] 里，顺带补齐 fastapi/uvicorn）。
    print(
        f'[WARN] 修："{sys.executable}" -m pip install {specs}'
        '（或 pip install -e ".[web]"）',
        file=target,
    )
    return missing


def editable_pth_files() -> list[Path]:
    """找出所有与本包相关的 editable ``.pth``（按 site-packages 顺序）。"""
    roots: list[str] = []
    for candidate in (site.getsitepackages(), [site.getusersitepackages()]):
        for item in candidate:
            if item and os.path.isdir(item):
                roots.append(item)
    found: list[Path] = []
    for root in roots:
        for pattern in (f"__editable__*{DIST_HINT}*.pth", f"*{DIST_HINT}*.egg-link"):
            found.extend(Path(root).glob(pattern))
    return found


def read_pth_target(path: Path) -> list[str]:
    """读 ``.pth`` 里的路径行（忽略注释与空白）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def points_at_repo(src_dir: Path) -> bool:
    """有没有哪个 ``.pth`` 正指向本仓库的 src。

    用 ``normpath`` 而不是 ``abspath``：批处理传来的 ``%~dp0`` 带尾反斜杠，
    ``abspath`` 不会把它去掉，而 ``.pth`` 里的路径没有 → 比较永远不相等。
    """
    target = os.path.normcase(os.path.normpath(str(src_dir)))
    for pth in editable_pth_files():
        for line in read_pth_target(pth):
            try:
                candidate = os.path.normcase(os.path.normpath(line))
            except Exception:
                continue
            if candidate == target:
                return True
            # Windows 目录大小写不敏感且可能有尾斜杠，都归一后再比
            if candidate.rstrip("\\") == target.rstrip("\\"):
                return True
    return False


def src_on_sys_path(src_dir: Path) -> bool:
    r"""``src`` 是不是已经在本进程的 ``sys.path`` 上（PYTHONPATH / .pth 都算）。

    新入口 ``scripts\giscli.cmd`` 设 ``PYTHONPATH=<repo>\src`` 后再跑，
    这样不依赖 Pro env 里那个坏掉的 pip / editable 安装。既然 import 得到，
    「要不要 pip install」就是伪命题——再让它装一次只会白白撞 pip 的报错。
    """
    target = os.path.normcase(os.path.normpath(str(src_dir))).rstrip("\\")
    for item in sys.path:
        if not item:
            continue
        try:
            candidate = os.path.normcase(os.path.normpath(item)).rstrip("\\")
        except Exception:  # pragma: no cover - 极端非法路径
            continue
        if candidate == target:
            return True
    return False


def is_stale(src_dir: Path, pyproject: Path) -> bool:
    """``pyproject.toml`` 比 ``.pth`` 新 → 依赖清单可能变了，值得重装。"""
    pths = editable_pth_files()
    if not pths or not pyproject.exists():
        return False
    newest = max(p.stat().st_mtime for p in pths)
    return pyproject.stat().st_mtime > newest + 1.0  # 1 秒容差，避免同秒误判


def needs_install(repo_root: Path, *, force: bool = False) -> tuple[bool, str]:
    """判断是否需要安装，并给出可读的原因。"""
    if force:
        return True, "指定了 --reinstall"
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        return True, f"仓库里没有 src 目录：{src_dir}"
    if src_on_sys_path(src_dir):
        return False, "src 已在 sys.path 上（无需 editable 安装）"
    if not editable_pth_files():
        return True, "没有找到 editable 安装记录（首次安装）"
    if not points_at_repo(src_dir):
        return True, "已安装的包指向别的目录（不是本仓库的 src）"
    if is_stale(src_dir, repo_root / "pyproject.toml"):
        return True, "pyproject.toml 比安装记录新（依赖可能变了）"
    return False, "editable 安装已就绪（源码改动自动生效）"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动前判断要不要 pip install -e .")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve()
    need, reason = needs_install(repo_root, force=args.reinstall)
    if not args.quiet:
        stream = sys.stderr if need else sys.stdout
        print(f"[{'INFO' if need else 'OK'}] {reason}", file=stream)
    # 可选依赖缺失不拦启动（不改退出码），但必须说出来。
    report_web_deps()
    return 1 if need else 0


if __name__ == "__main__":
    raise SystemExit(main())
