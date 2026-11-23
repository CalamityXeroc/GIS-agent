# -*- coding: utf-8 -*-
"""宿主 ``src/`` 越界闸门：发现"被测 agent 改了引擎"就作废该轮。

背景（DAIR-62）：基准跑 `probe_quality_score_map` 时，被测 agent 为绕过
"CLI 无 arcpy" 的报错，在 run 窗口内往宿主仓库 ``src/`` 下新建了 ``arcpy/``
桥接包。同一轮里既测引擎又改引擎，结果不可信；而且这个影子包会静默改变后续
所有 run 的 import 行为（它丢符号颜色，曾把真 arcpy 的 ``#B07FB2`` 变成空串）。

做法：跑前给 ``src/`` 拍指纹（文件树 sha256 + ``git status``），跑后重拍并比对，
任何新增/修改/删除都判该轮作废，证据归档到 ``benchmark/reports/`` 下，不清理现场
（清理是人的决定）。另外跑前预检两条硬拦：``src/`` 顶层出现非白名单包（比如
``arcpy/``）；``git status -- src`` 非空（DAIR-78：上一轮若是"改已跟踪文件"，
顶层目录一模一样，只有 git 看得出来）。命中任一 → 直接中止，避免在已被污染的
基线上开测。

被排除的噪声（否则每个 run 都会误报）：``__pycache__/``、``*.pyc/*.pyo``、
``*.egg-info/``。这些是 import / pip install -e 的正常副产物。

噪声口径**只有一处**：``_excluded()``。文件树指纹（``hash_tree()``）与 git 预检
（``_git_noise()``）都必须经过它 —— DAIR-81 就是两套口径漂移出来的 bug：仓库里有
34 个**被跟踪**的 ``.pyc``，``runner.py`` 一 import 就重写它们，git 侧不过滤的话
干净 clone 会被自己判成污染，闸门挡住了自己。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

#: src/ 顶层允许存在的条目（引擎本体 + pip 生成物）。其它一律视为可疑。
SRC_TOPLEVEL_ALLOW = {"gis_cli", "baml_client"}

#: 与第三方依赖同名 → 影子包，能静默劫持 import，单列高危。
SHADOW_RISK_NAMES = {
    "arcpy", "osgeo", "gdal", "ogr", "numpy", "shapely", "pyproj", "fiona",
    "rasterio", "geopandas", "pandas", "PIL", "matplotlib", "yaml", "httpx",
    "baml_client", "pytest",
}

_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


class SrcEscapeError(RuntimeError):
    """跑前就发现 src/ 已被污染（不是本轮造成），拒绝在脏基线上开测。"""


def _excluded(rel: Path) -> bool:
    """噪声路径判据（唯一一处）：指纹与 git 预检共用。"""
    if any(part in _EXCLUDE_DIRS for part in rel.parts):
        return True
    if rel.suffix.lower() in _EXCLUDE_SUFFIXES:
        return True
    return any(part.endswith(".egg-info") for part in rel.parts)


def _porcelain_paths(line: str) -> list[str]:
    """``git status --porcelain`` 一行涉及的路径（rename 是 ``old -> new`` 两侧）。"""
    body = line[3:] if len(line) > 3 else ""
    paths = body.split(" -> ", 1) if " -> " in body else [body]
    return [p[1:-1] if len(p) >= 2 and p.startswith('"') and p.endswith('"') else p for p in paths]


def _git_noise(line: str) -> bool:
    """这条 git 状态行是否**只**涉及噪声（口径与 ``hash_tree()`` 同一处）。

    rename 的两侧都要是噪声才算噪声：``a.py -> __pycache__/a.pyc`` 是真改动。
    """
    paths = _porcelain_paths(line)
    return bool(paths) and all(_excluded(Path(p)) for p in paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """``src/`` 下的文件指纹：相对路径(posix) -> ``sha256:size``。"""
    tree: dict[str, str] = {}
    if not root.exists():
        return tree
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if _excluded(rel) or not path.is_file():
            continue
        try:
            tree[rel.as_posix()] = f"{_sha256(path)}:{path.stat().st_size}"
        except OSError:
            tree[rel.as_posix()] = "unreadable"
    return tree


def _git(repo: Path, *args: str, timeout: int = 20) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return proc.stdout if proc.returncode == 0 else f"<git rc={proc.returncode}: {proc.stderr.strip()[:300]}>"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<git failed: {type(exc).__name__}: {exc}>"


def _classify(rel: str) -> tuple[str, str]:
    """给一条 src/ 变更定严重度和原因（rel 是相对 src/ 的路径）。"""
    parts = rel.split("/")
    top = parts[0]
    if top in SHADOW_RISK_NAMES:
        return "high", f"src/{top} 与依赖同名，影子包会静默劫持 import"
    if top not in SRC_TOPLEVEL_ALLOW:
        return "high", f"src/{top} 不在 src/ 顶层白名单内（被测进程自建的宿主代码）"
    return "normal", "引擎源码被改动"


class SrcGuard:
    """一次 run 的 src/ 指纹闸门（begin 拍基线，end 比对并判定）。"""

    def __init__(self, repo: Path, watch_dir: Path | None = None) -> None:
        self.repo = Path(repo)
        self.watch_dir = Path(watch_dir) if watch_dir else self.repo / "src"
        self.before: dict[str, Any] | None = None
        self.after: dict[str, Any] | None = None

    # -- 拍指纹 -------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
            "tree": hash_tree(self.watch_dir),
            "toplevel": sorted(
                p.name for p in self.watch_dir.iterdir()
                if p.is_dir() and not p.name.endswith(".egg-info") and p.name != "__pycache__"
            ) if self.watch_dir.exists() else [],
            # 噪声（__pycache__/*.pyc/*.egg-info）不进这条清单：import 与 pip install -e
            # 会顺手重写它们，算进来就等于把正常副作用当成污染（DAIR-81）。
            "git_status_src": [
                ln for ln in _git(self.repo, "status", "--porcelain", "--", "src").splitlines()
                if ln and not _git_noise(ln)
            ],
        }

    def begin(self) -> dict[str, Any]:
        """拍跑前基线；基线已脏（影子包 / git 侧有改动）→ 抛 SrcEscapeError。"""
        self.before = self.snapshot()
        problems = self.precheck()
        if problems:
            raise SrcEscapeError(
                "基线已被污染，先人工清理，本轮不在脏基线上开测：" + "；".join(problems)
            )
        print(f"[src-guard] 基线已记录：{len(self.before['tree'])} 个文件，"
              f"git 未跟踪/未提交 {len(self.before['git_status_src'])} 条")
        return self.before

    def precheck(self) -> list[str]:
        """跑前污染检查，返回问题清单（空 = 基线干净）。

        两条硬拦：``src/`` 顶层非白名单包（影子包能静默劫持 import）；
        ``git status -- src`` 非空（改了已跟踪文件时顶层目录看不出变化，只有 git 能发现）。
        后者已按 ``_excluded()`` 剔除噪声，口径与 ``hash_tree()`` 一致。
        """
        if self.before is None:
            self.before = self.snapshot()
        problems = [
            f"src/{name} 不在 src/ 顶层白名单内（{', '.join(sorted(SRC_TOPLEVEL_ALLOW))} 之外）"
            for name in self.before["toplevel"] if name not in SRC_TOPLEVEL_ALLOW
        ]
        dirty = list(self.before.get("git_status_src") or [])
        if dirty:
            shown = "、".join(ln.strip() for ln in dirty[:5])
            more = f" 等 {len(dirty)} 条" if len(dirty) > 5 else ""
            problems.append(f"git status -- src 非空：{shown}{more}")
        return problems

    # -- 比对 ---------------------------------------------------------
    def end(self) -> dict[str, Any]:
        if self.before is None:
            raise RuntimeError("SrcGuard.end() 必须在 begin() 之后调用")
        self.after = self.snapshot()
        before_tree, after_tree = self.before["tree"], self.after["tree"]

        changes: list[dict[str, Any]] = []
        for rel in sorted(set(after_tree) - set(before_tree)):
            severity, reason = _classify(rel)
            changes.append({"kind": "added", "path": f"src/{rel}", "severity": severity, "reason": reason})
        for rel in sorted(set(before_tree) - set(after_tree)):
            severity, reason = _classify(rel)
            changes.append({"kind": "deleted", "path": f"src/{rel}", "severity": severity, "reason": reason})
        for rel in sorted(set(before_tree) & set(after_tree)):
            if before_tree[rel] != after_tree[rel]:
                severity, reason = _classify(rel)
                changes.append({"kind": "modified", "path": f"src/{rel}", "severity": severity, "reason": reason})

        # git 侧的补充证据：新出现的未跟踪/已修改条目（文件树里可能被忽略规则挡住）
        before_git = set(self.before["git_status_src"])
        git_new = [ln for ln in self.after["git_status_src"] if ln not in before_git]

        result: dict[str, Any] = {
            "ok": not changes,
            "verdict": "clean" if not changes else "escaped",
            "changes": changes,
            "git_new_entries": git_new,
            "before": self.before,
            "after": self.after,
            "watch_dir": str(self.watch_dir),
        }
        if not changes:
            print(f"[src-guard] OK：src/ 指纹与跑前一致（{len(after_tree)} 个文件，无变更）")
        else:
            print(f"[src-guard] VIOLATION：被测进程在 run 窗口内改动了宿主 src/（{len(changes)} 处）")
            for item in changes:
                print(f"  [{item['severity'].upper():6}] {item['kind']:8} {item['path']}  —— {item['reason']}")
            if git_new:
                print(f"[src-guard] git 新增条目 {len(git_new)} 条：")
                for ln in git_new[:20]:
                    print(f"  {ln}")
            print("[src-guard] 本轮作废（success=false），不自动清理现场")
        return result


# -- 证据归档 ---------------------------------------------------------

def archive_evidence(result: dict[str, Any], dest: Path, repo: Path, context: dict[str, Any] | None = None) -> Path:
    """把越界证据落到 ``dest``（与 benchmark/reports 现有约定一致：一个目录 + README.md）。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    context = context or {}

    payload = {
        **context,
        "verdict": result.get("verdict"),
        "ok": result.get("ok"),
        "changes": result.get("changes", []),
        "git_new_entries": result.get("git_new_entries", []),
        "git_status_before": result.get("before", {}).get("git_status_src", []),
        "git_status_after": result.get("after", {}).get("git_status_src", []),
        "watch_dir": result.get("watch_dir"),
        "captured_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (dest / "findings.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    (dest / "git_status_src_after.txt").write_text(
        _git(Path(repo), "status", "--porcelain", "--", "src"), encoding="utf-8"
    )
    (dest / "git_diff_src.patch").write_text(_git(Path(repo), "diff", "--", "src"), encoding="utf-8")

    copied: list[str] = []
    for item in result.get("changes", []):
        rel = item["path"][len("src/"):]
        src = Path(repo) / "src" / rel
        if item["kind"] == "deleted" or not src.is_file():
            continue
        target = dest / ("added" if item["kind"] == "added" else "modified") / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, target)
            copied.append(f"{item['kind']}: {item['path']}")
        except OSError as exc:
            print(f"[src-guard] 证据拷贝失败 {src}: {exc}")

    lines = [
        "# src/ 越界证据（自动归档）",
        "",
        f"- 判定：**{result.get('verdict')}**（该轮已作废）",
        f"- 采集时间：{payload['captured_at']}",
        f"- 监视目录：`{result.get('watch_dir')}`",
        f"- 上下文：{json.dumps(context, ensure_ascii=False)}",
        "",
        "## 变更清单",
        "",
    ]
    for item in result.get("changes", []):
        lines.append(f"- `{item['kind']}` `{item['path']}`（{item['severity']}）：{item['reason']}")
    if result.get("git_new_entries"):
        lines += ["", "## git 新增条目", ""] + [f"- `{ln}`" for ln in result["git_new_entries"]]
    lines += [
        "",
        "## 文件",
        "",
        "- `findings.json`：完整判定（含跑前/跑后指纹与 git status）",
        "- `git_status_src_after.txt` / `git_diff_src.patch`：git 侧证据",
        "- `added/`、`modified/`：越界文件的现场副本（路径与 src/ 内相对路径一致）",
        "",
        context.get("cleanup_note") or "现场**未被清理**：确认后可人工移出仓库，处理完再重跑。",
    ]
    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[src-guard] 证据已归档：{dest}")
    return dest


def apply_to_record(
    record: dict[str, Any],
    result: dict[str, Any],
    report_dir: Path,
    run_id: str,
    task_id: str,
    repo: Path,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把闸门结论写进 runner 的 report record，并在越界时归档证据。"""
    record["src_guard"] = {
        "ok": result["ok"],
        "verdict": result["verdict"],
        "changes": result["changes"],
        "git_new_entries": result.get("git_new_entries", []),
        "watch_dir": result["watch_dir"],
        "evidence_dir": None,
    }
    if result["ok"]:
        return record
    dest = archive_evidence(
        result,
        Path(report_dir) / f"src_escape_{run_id}" / task_id,
        repo,
        {"run_id": run_id, "task_id": task_id, "source": "benchmark/runner.py", **(context or {})},
    )
    record["src_guard"]["evidence_dir"] = str(dest)
    record["invalidated"] = True
    record["invalidated_reason"] = "src_escape"
    record["success"] = False
    record["assertions_validated"] = False
    return record
