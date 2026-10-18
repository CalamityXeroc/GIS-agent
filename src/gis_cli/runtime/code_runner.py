# -*- coding: utf-8 -*-
"""Code execution facade: persistent kernel first, one-shot subprocess fallback."""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kernel import ArcpyKernel, ExecResult, KernelError, KernelTimeout

logger = logging.getLogger(__name__)


class CodeRunner:
    """Runs ArcPy code with a warm kernel and safe fallback."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        python_executable: str | None = None,
        state_dir: str | Path | None = None,
        exec_timeout: float = 600.0,
        allow_subprocess_fallback: bool = True,
    ):
        self.workspace = Path(workspace)
        self.python_executable = python_executable
        self.state_dir = state_dir
        self.exec_timeout = exec_timeout
        self.allow_subprocess_fallback = allow_subprocess_fallback
        self._kernel: ArcpyKernel | None = None
        self._kernel_failed = False
        self.last_backend = "none"
        #: 内核启动很快（实测 arcpy 导入就要 8~11s）：预热与扫描并行，
        #: 首个 execute_code 就不用再等。锁保证不会起两个内核。
        self._kernel_lock = threading.Lock()
        self._prewarm_thread: threading.Thread | None = None
        #: 预热是**一次性**动作，用显式标志而不是“线程还活着吗”判断：
        #: 工作线程可能在调用方等到锁之前就已结束，那时 ``is_alive()`` 已是
        #: False，只看它会又起一个内核（实测竞争）。
        self._prewarm_started = False

    # ---------------------------------------------------------------- 预热
    def prewarm(self) -> bool:
        """后台启动内核（幂等）。已在跑或已就绪时返回 False。

        为什么要预热：内核冷启动实测 11.6~17.6s（大头是 ``import arcpy``）。
        启动时不预热的话，用户第一个要跑代码的任务会白等这一段；
        预热后可与数据目录扫描、首轮模型调用同时进行。
        没有 jupyter_client 时这里会快速失败并置 ``_kernel_failed``，
        后续自动退回一次性子进程（与旧行为一致）。
        """
        if self._kernel_failed:
            return False
        if self._kernel is not None and self._kernel.is_alive():
            return False
        with self._kernel_lock:
            # 先看一次性标志：线程可能已经结束，但那次预热已经做过了
            if self._prewarm_started:
                return False
            if self._kernel_failed:
                return False
            if self._kernel is not None and self._kernel.is_alive():
                return False
            self._prewarm_started = True
            self._prewarm_thread = threading.Thread(
                target=self._ensure_kernel, name="gis-agent-kernel-prewarm", daemon=True
            )
            self._prewarm_thread.start()
            return True

    def wait_prewarm(self, timeout: float | None = None) -> None:
        """等预热完成（不需要精确同步时可不调）。"""
        thread = self._prewarm_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def _ensure_kernel(self) -> ArcpyKernel | None:
        if self._kernel_failed:
            return None
        if self._kernel is not None and self._kernel.is_alive():
            return self._kernel
        with self._kernel_lock:
            # 双重检查：预热线程与 execute_code 可能同时到达
            if self._kernel_failed:
                return None
            if self._kernel is not None and self._kernel.is_alive():
                return self._kernel
            try:
                from .timing import StageTimer

                timer = StageTimer("kernel.prewarm")
                self._kernel = ArcpyKernel(
                    python_executable=self.python_executable,
                    state_dir=self.state_dir,
                    workspace=str(self.workspace),
                )
                self._kernel.start()
                self.last_backend = "kernel"
                timer.done(ok=True)
                return self._kernel
            except Exception as exc:
                logger.warning("persistent kernel unavailable (%s); falling back to subprocess", exc)
                self._kernel_failed = True
                return None

    # ----------------------------------------------------------------- public
    def run(
        self,
        code: str,
        *,
        timeout: float | None = None,
        workspace: str | Path | None = None,
    ) -> ExecResult:
        """Execute code, preferring the warm kernel."""
        timeout = float(timeout or self.exec_timeout)
        ws = str(workspace or self.workspace)
        kernel = self._ensure_kernel()
        if kernel is not None:
            try:
                return kernel.execute(code, timeout=timeout, workspace=ws)
            except KernelTimeout as exc:
                return ExecResult(
                    ok=False,
                    error={"type": "TimeoutExpired", "message": str(exc), "traceback": ""},
                )
            except KernelError as exc:
                logger.warning("kernel execution failed: %s", exc)
                self._kernel_failed = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("kernel crashed: %s", exc)
                self._kernel_failed = True
        if not self.allow_subprocess_fallback:
            return ExecResult(
                ok=False,
                error={"type": "KernelUnavailable", "message": "persistent kernel unavailable"},
            )
        return self._run_subprocess(code, timeout=timeout, workspace=ws)

    def execute(self, code: str, *, timeout: float = 600.0, workspace: str | None = None) -> ExecResult:
        """Kernel-compatible alias used by the catalog scanner."""
        return self.run(code, timeout=timeout, workspace=workspace)

    def _run_subprocess(self, code: str, *, timeout: float, workspace: str) -> ExecResult:
        from ..arcpy_bridge import run_arcpy_code

        started = time.time()
        result = run_arcpy_code(
            code,
            workspace=workspace,
            timeout_seconds=int(max(10, timeout)),
            python_executable=self.python_executable,
        )
        self.last_backend = "subprocess"
        return ExecResult(
            ok=result.status == "success",
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            result=result.data,
            error=result.error,
            duration_ms=int((time.time() - started) * 1000),
        )

    def reset(self) -> None:
        """Reset kernel state between tasks."""
        if self._kernel is not None and self._kernel.is_alive():
            try:
                self._kernel.reset()
            except Exception as exc:
                logger.warning("kernel reset failed: %s", exc)

    def restart(self) -> None:
        if self._kernel is not None:
            try:
                self._kernel.restart()
                self._kernel_failed = False
                return
            except Exception as exc:
                logger.warning("kernel restart failed: %s", exc)
        self._kernel_failed = False
        self._kernel = None

    def shutdown(self) -> None:
        """关内核。先等在途的预热收尾，避免与半启动的内核抢 shm（实测会抦 zmq 报
        ``Bad address``）。
        """
        self._kernel_failed = True  # 关掉后不再自动重启
        self.wait_prewarm(timeout=120)
        if self._kernel is not None:
            self._kernel.shutdown()
            self._kernel = None

    @property
    def backend(self) -> str:
        return self.last_backend


# --------------------------------------------------------------------- backup
_DESTRUCTIVE_MARKERS = (
    "arcpy.management.Delete",
    "arcpy.Delete_management",
    "arcpy.management.Format",
    "arcpy.Convert_",
    "shutil.rmtree",
    "os.remove(",
    "os.unlink(",
    "os.rmdir(",
    ".unlink(",
)


def looks_destructive(code: str) -> bool:
    """Heuristic check for code that deletes/overwrites data."""
    return any(marker in code for marker in _DESTRUCTIVE_MARKERS)


#: 写工具的**输出参数位置**（0-based，参数序号）。
#:
#: 为什么需要这张表：内核 prelude 里 ``arcpy.env.overwriteOutput = True`` 是
#: 全局开着的，所以任何写工具在目标已存在时都会**静默覆盖**。而
#: ``looks_destructive`` 只认 Delete/rmtree 这类显式删除 —— 实测：
#:
#:     looks_destructive('arcpy.analysis.Buffer(fc, out, "100")')  ->  False
#:
#: 也就是说“正常写产出”会直接毁掉用户已有的结果库，且没有任何回滚点。
#: 这里只收录“输出位置明确”的工具；拿不准的一律不收（宁可漏备份，也不误判）。
_OUTPUT_ARG_POSITIONS: dict[str, tuple[int, ...]] = {
    # 转换 / 复制：输出常在倒数第 1 或第 2 个位置
    "featureclasstofeatureclass": (1, 2),
    "featureclasstoshapefile": (1,),
    "copytoraster": (1,),
    "conversiontootherformat": (1,),
    "rastertootherformat": (1,),
    "tabletotable": (1,),
    "exportfeatureattributestoascii": (1,),
    # 分析：Buffer(in, out, dist)；Clip(in, clip, out)；Intersect(in, out)
    "buffer": (1,),
    "buffer_analysis": (1,),
    "clip": (2,),
    "clip_analysis": (2,),
    "intersect": (1,),
    "intersect_analysis": (1,),
    "union": (1,),
    "union_analysis": (1,),
    "erase": (2,),
    "erase_analysis": (2,),
    "spatialjoin": (1,),
    "spatialjoin_analysis": (1,),
    "select": (1,),
    "select_analysis": (1,),
    "dissolve": (1,),
    "dissolve_management": (1,),
    "project": (1,),
    "project_management": (1,),
    "projectraster": (1,),
    "projectraster_management": (1,),
    "kernel_density": (1,),
    "kerneldensity": (1,),
    "idw": (1,),
    "kriging": (1,),
    "spline": (1,),
    "naturalneighbor": (1,),
    "eucdistance": (1,),
    "slope": (1,),
    "aspect": (1,),
    "hillshade": (2,),
    "rastercalculator": (1,),
    "con": (1,),
    "reclassify": (1,),
    "rastertopolygon": (1,),
    "polygontoraster": (1,),
    "featuretoraster": (1,),
    "pointdensity": (1,),
    "linetoraster": (1,),
    "polygontoline": (1,),
    "featuretoline": (1,),
    "featuretopolygon": (1,),
    "featuretocentroid": (1,),
    "minimumboundinggeometry": (1,),
    "dissolveboundaries": (1,),
    "multiparttosinglepart": (1,),
    "summarystatistics": (1,),
    "frequency": (1,),
    "statistics": (1,),
    "zonalstatisticsastable": (1,),
    "tabularintersect": (1,),
    "tabulateintersection": (1,),
    "generatepointsalonglines": (1,),
    "xy_table_to_point": (2,),
    "xy_table_to_point_management": (2,),
    "addfield": (0,),  # 就地改表结构（无输出参数，改的就是输入）
    "addfield_management": (0,),
    "calculatefield": (0,),
    "calculatefield_management": (0,),
    "deletefield": (0,),
    "deletefield_management": (0,),
    "alterfield": (0,),
    "alterfield_management": (0,),
    # 建同名要素类/表/栅格 = 就地覆盖（overwriteOutput 下直接替换旧数据）
    "createfeatureclass": (0,),
    "createfeatureclass_management": (0,),
    "createtable": (0,),
    "createtable_management": (0,),
    "createrandomraster": (1,),
    "createrasterdataset": (0,),
}

#: 调用名匹配：``arcpy.analysis.Buffer(...)`` / ``arcpy.management.Buffer(...)`` / ``Buffer_analysis(...)``
_CALL_RE = re.compile(
    r"\b(?:arcpy(?:\.\w+)?\.)?([A-Za-z_]\w*)\s*\(", re.IGNORECASE
)

#: 字符串字面量（含原始字符串）—— 用来数“调用里的第几个参数”
_LITERAL_RE = re.compile(r"[rRbBuU]?(['\"])(.*?)\1", re.DOTALL)


def _split_call_args(code: str, open_paren: int) -> list[str]:
    """从 ``open_paren`` 处的左括号开始，切出这一层调用的参数文本。

    只做括号/引号配对，不解析表达式 —— 够用且不会误伤。
    """
    args: list[str] = []
    depth = 0
    quote = ""
    current: list[str] = []
    index = open_paren
    while index < len(code):
        char = code[index]
        if quote:
            current.append(char)
            if char == quote and code[index - 1] != "\\":
                quote = ""
        elif char in "'\"":
            quote = char
            current.append(char)
        elif char in "([{":
            depth += 1
            if depth > 1:
                current.append(char)
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(current).strip())
                return [arg for arg in args if arg]
            current.append(char)
        elif char == "," and depth == 1:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        index += 1
    args.append("".join(current).strip())
    return [arg for arg in args if arg]


def _literal_path(text: str) -> str | None:
    """从参数文本里取出路径字面量。

    多数情形是单个字符串字面量；但 ArcGIS 的写工具常把“输出库 + 输出名”
    分成两个参数（``FeatureClassToFeatureClass(src, gdb, "study_area")``），
    这种情况下单个参数取不到完整路径 —— 由 :func:`_join_output` 处理。
    """
    literals = [match.group(2) for match in _LITERAL_RE.finditer(text)]
    if not literals:
        return None
    return literals[-1].strip() or None


#: 看上去像“输出名”的字符串：不含路径分隔符、不是距离/单位/选项。
#:
#: 为什么需要它：``Buffer(in, out, "100 Meters")`` 里输出参数后面紧跟着
#: 距离，盲目拼成 ``out/100 Meters`` 就错到太平洋了（实测：Buffer 的覆盖
#: 目标被拼成 ``...\study_area\100 Meters``，于是真正的 study_area 漏备份）。
_NOT_A_NAME_RE = re.compile(
    r"^\s*[-+]?\d|\b(meters?|metres?|kilometers?|km|miles?|feet|degrees?|units?|unknown)\b",
    re.IGNORECASE,
)


def _looks_like_name(text: str) -> bool:
    """这个字符串像不像“输出名”（而非距离/单位/坐标系/选项）。"""
    value = (text or "").strip()
    if not value or len(value) > 120:
        return False
    if any(sep in value for sep in ("/", "\\")):
        return False
    if value.startswith("#"):  # ArcGIS 的“省略该参数”占位符
        return False
    if _NOT_A_NAME_RE.search(value):
        return False
    return True


def _join_output(args: list[str], positions: tuple[int, ...]) -> list[str]:
    """把“输出库 + 输出名”两个参数拼成候选路径。

    ``FeatureClassToFeatureClass(src, r"...\\x.gdb", "study_area")``
    → 优先返回 ``.../x.gdb/study_area``（真正会被覆盖的要素类）。

    **库本身排后面**：拼出来的具体要素类才是精确目标；只备份库会把
    整个 GDB（几十个要素类、上百 MB）复制一遍。库作为兼底保留，
    以防输出名其实是新建的（那时代码里只有库已存在，仍需保护）。
    """
    exact: list[str] = []
    bases: list[str] = []
    for position in positions:
        if position >= len(args):
            continue
        base = _literal_path(args[position])
        if not base:
            continue
        bases.append(base)
        # 紧跟着的那个参数是纯名字（不带路径分隔符）时，拼成 base/name
        for following in range(position + 1, len(args)):
            name = _literal_path(args[following])
            if not name:
                break
            if not _looks_like_name(name):
                break
            exact.append(str(Path(base) / name))
            break
    # 能拼出具体名字就**只认名字**：名字是新的时候没有覆盖风险，
    # 不该因为“容器库已存在”就去备份整个库（实测误报：输出 brand_new
    # 时把 30 MB 的 result_data.gdb 整份备份了）。
    return exact or bases


#: 不做自动备份的目录名（只读输入数据；备份它既无意义又极占空间）
_BACKUP_SKIP_DIRS = {"input", "skills", "config", ".git", ".venv"}
#: 单个目录备份体积上限（MB）：超限则跳过并告警，避免 copytree 拖垮磁盘
_BACKUP_DIR_MAX_MB = 200


def _dir_size_mb(path: Path) -> float:
    """目录体积（MB）；超过上限时提前返回，不再继续统计。"""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
            if total > _BACKUP_DIR_MAX_MB * 1024 * 1024:
                return total / 1024 / 1024
    except OSError:
        return 0.0
    return total / 1024 / 1024


def _gdb_exists(path: Path) -> bool:
    """判断路径是否存在，**兼容 GDB 内部要素类**。

    坑（本项目已有记录）：GDB 里的要素类不是文件系统路径 ——
    ``Path.exists('x.gdb/study_area')`` 返回 **False**，只有
    ``arcpy.Exists(...)`` 才是 True。备份判据用错就会漏掉整个 GDB 内的
    覆盖（而 GDB 正是本项目最常见的产物形态）。

    优先级：文件系统存在即真（覆盖 shapefile/栅格/普通文件）；否则如果
    路径里含 ``.gdb``，再问一次 arcpy（只在已 import arcpy 的解释器里可用，
    否则退回 False —— 宁可漏备份也不误判）。
    """
    try:
        if path.exists():
            return True
    except OSError:
        pass
    lowered = str(path).lower()
    if ".gdb" not in lowered:
        return False
    try:
        import arcpy  # type: ignore

        return bool(arcpy.Exists(str(path)))
    except Exception:
        return False


def backup_targets(code: str, workspace: str | Path, *, ignore_exists: bool = False) -> list[str]:
    """找出这段代码**会覆盖的、已存在的**产出路径。

    覆盖两种情形：

    1. 显式删除（``Delete`` / ``rmtree`` / ``remove``）—— 原有行为，目标就是要备份的；
    2. **写工具的输出参数已存在** —— ``overwriteOutput=True`` 下会被静默覆盖。
       只认 ``_OUTPUT_ARG_POSITIONS`` 里列明的工具，拿不准的不收。

    只返回 **output/ 下、确实存在** 的路径：``input/`` 是只读数据源，
    备份它既无意义又极占空间。

    ``ignore_exists=True`` 时跳过存在性判断，只做**候选推导** ——
    给需要在内核里用 ``arcpy.Exists`` 复核 GDB 内部要素类的调用方用
    （见 ``engine/tools.py:_gdb_targets_via_kernel``）。
    """
    workspace = Path(workspace)
    out_root = (workspace / "output").resolve()
    found: list[str] = []

    def _consider(raw: str | None) -> bool:
        """登记一个候选；返回是否被采纳（已存在、在 output/ 下）。"""
        if not raw:
            return False
        candidate = Path(raw)
        if not candidate.is_absolute():
            # 裸名字（如 "study_area"）无法定位，不当候选（它要么已被拼接、
            # 要么是新建的）——按 workspace 拼会把名字当根目录下的路径。
            if not any(sep in raw for sep in ("/", "\\")):
                return False
            candidate = workspace / raw
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        # 只关心 output/ 下的东西：input/ 是只读源，其它位置不是本次产出
        if not str(resolved).lower().startswith(str(out_root).lower()):
            return False
        if not ignore_exists:
            if not _gdb_exists(resolved) or str(resolved) in found:
                return False
        elif str(resolved) in found:
            return False
        # 只读目录与超大目录不备份（实测整目录备份一次 300 MB、input/ 被反复备份 2 GB）
        if resolved.is_dir() and not ignore_exists:
            if resolved.name.lower() in _BACKUP_SKIP_DIRS:
                logger.info("skip backup of read-only dir: %s", resolved)
                return False
            size_mb = _dir_size_mb(resolved)
            if size_mb > _BACKUP_DIR_MAX_MB:
                logger.warning(
                    "skip backup of large dir (%.0f MB > %s MB): %s",
                    size_mb, _BACKUP_DIR_MAX_MB, resolved,
                )
                return False
        found.append(str(resolved))
        return True

    # 1) 写工具的输出参数
    for call in _CALL_RE.finditer(code):
        name = call.group(1).lower()
        positions = _OUTPUT_ARG_POSITIONS.get(name)
        if not positions:
            continue
        open_paren = call.end() - 1
        args = _split_call_args(code, open_paren)
        # _join_output 已保证：能拼出具体名字时只返回那个精确目标，
        # 否则才返回容器库本身。
        for path in _join_output(args, positions):
            _consider(path)

    # 2) 显式删除的目标（原有行为）
    if looks_destructive(code):
        for match in re.finditer(r"[rRbBuU]?(['\"])(.*?)\1", code, re.DOTALL):
            _consider(match.group(2))

    return found


def _is_volatile(name: str) -> bool:
    """ArcGIS 的瞬时锁文件（还在写时会创建，写完/进程退出后消失）。

    实测：备份一个 result_data.gdb 时，copytree 会把 ``*.sr.lock`` / ``*.rd.lock``
    也列进去，但它们很快就被 ArcGIS 删掉，于是复制到一半报
    ``[WinError 2] 系统找不到指定的文件``，一整排警告刷屏（还白写了一个
    半成品的备份目录）。这些锁文件本身没有备份价值。
    """
    lowered = name.lower()
    return lowered.endswith((".sr.lock", ".rd.lock", ".lock", ".lockfile"))


def _is_inside_gdb(path: Path) -> bool:
    """路径是不是 GDB **内部**的成员（要素类/表）。

    GDB 成员在文件系统上不是一个可复制的对象：它是 ``a00000001.gdbtable``
    这类内部表，名字映射由 GDB 自己维护。所以 ``copytree``/``copy2`` 对它
    既不能判断存在、也无法单独复制 —— 必须走 ``arcpy`` 导出。
    """
    for parent in path.parents:
        if parent.name.lower().endswith(".gdb"):
            return True
    return path.name.lower().endswith(".gdb") is False and any(
        p.name.lower().endswith(".gdb") for p in path.parents
    )


def _backup_gdb_member(src: Path, dest_root: Path) -> str | None:
    """用 arcpy 把一个 GDB 成员导出到备份文件地理数据库。

    为什么不能直接 copy：见 :func:`_is_inside_gdb`。这里导出一个独立的
    ``.gdb``（而不是单个 shapefile），保留字段类型/子类型/域等定义。
    """
    try:
        import arcpy  # type: ignore
    except Exception:
        logger.warning("arcpy 不可用，无法备份 GDB 成员: %s", src)
        return None
    dest_root.mkdir(parents=True, exist_ok=True)
    name = src.name
    gdb_path = dest_root / f"{name}_backup.gdb"
    try:
        if not arcpy.Exists(str(gdb_path)):
            arcpy.management.CreateFileGDB(str(dest_root), f"{name}_backup.gdb")
        desc = arcpy.Describe(str(src))
        kind = getattr(desc, "dataType", "")
        if kind in ("FeatureClass", "ShapeFile"):
            arcpy.management.CopyFeatures(str(src), str(gdb_path / name))
        elif kind == "Table":
            arcpy.management.CopyRows(str(src), str(gdb_path / name))
        elif kind in ("RasterDataset", "Raster"):
            arcpy.management.CopyRaster(str(src), str(gdb_path / name))
        else:
            logger.warning("unsupported gdb member type %s for %s", kind, src)
            return None
        return str(gdb_path)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("gdb member backup failed for %s: %s", src, exc)
        return None


def backup_paths(paths: list[str], backup_root: str | Path) -> list[str]:
    """Copy existing files/dirs into a timestamped backup folder.

    Returns the list of created backup paths. Best-effort: failures are
    reported by omission, not raised.

    GDB **内部**的要素类/表无法用 ``copytree``/``copy2`` 复制（它们不是文件
    系统对象），改用 ``arcpy`` 导出为独立的备份 ``.gdb``。
    """
    root = Path(backup_root) / time.strftime("%Y%m%d_%H%M%S")
    created: list[str] = []
    for raw in paths:
        src = Path(raw)
        # GDB 成员：Path.exists() 为假，必须交给 arcpy
        if _is_inside_gdb(src):
            dest = _backup_gdb_member(src, root)
            if dest:
                created.append(dest)
            continue
        if not src.exists():
            continue
        root.mkdir(parents=True, exist_ok=True)
        dest = root / src.name
        try:
            if src.is_dir():
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                # 跳过瞬时锁文件：复制到一半它们就没了，只会刷一堆 WinError 2
                shutil.copytree(
                    src, dest, ignore=shutil.ignore_patterns("*.lock", "*.lockfile")
                )
            else:
                if _is_volatile(src.name):
                    continue
                shutil.copy2(src, dest)
            created.append(str(dest))
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("backup failed for %s: %s", src, exc)
    return created
