

from pathlib import Path

# ------------------------------------------------- GDB 新鲜度（自失效回归）


def _make_fake_gdb(root, name="t.gdb"):
    """造一个假的 .gdb 目录（带内容文件 + 锁文件）。"""
    gdb = root / name
    gdb.mkdir(parents=True, exist_ok=True)
    (gdb / "a00000001.gdbtable").write_bytes(b"x" * 16)
    return gdb


def test_gdb_freshness_ignores_lock_files(tmp_path):
    """读取 GDB 会新建 .sr.lock 并刷新目录 mtime —— 不能让缓存因此失效。

    实测坑：每次启动都重扫全部 133 个要素类（158.5s），因为“读一下”就把
    目录 mtime 推新了。新鲜度必须只看非锁文件的真实内容时间。
    """
    import os
    import time

    from gis_cli.datacatalog.scanner import discover, gdb_content_mtime

    root = tmp_path / "data"
    gdb = _make_fake_gdb(root)

    first = next(d for d in discover([root]) if d.kind == "gdb")
    stamp = first.mtime
    assert stamp > 0

    # 模拟“被读过”：新建锁文件 + 目录 mtime 前进 100s
    lock = gdb / "_gdb.SOMEONE.1234.5678.sr.lock"
    lock.write_bytes(b"")
    future = time.time() + 100
    os.utime(gdb, (future, future))
    os.utime(lock, (future, future))

    second = next(d for d in discover([root]) if d.kind == "gdb")
    assert second.mtime == stamp, "锁文件/目录 mtime 变化不应影响 GDB 新鲜度"
    assert gdb_content_mtime(gdb) == stamp


def test_gdb_freshness_tracks_real_data_change(tmp_path):
    """真的改了数据（要素类文件变新）时，新鲜度仍要跟着变。"""
    import os
    import time

    from gis_cli.datacatalog.scanner import discover

    root = tmp_path / "data"
    gdb = _make_fake_gdb(root)
    first = next(d for d in discover([root]) if d.kind == "gdb")

    target = gdb / "a00000001.gdbtable"
    future = time.time() + 500
    os.utime(target, (future, future))

    second = next(d for d in discover([root]) if d.kind == "gdb")
    assert second.mtime > first.mtime, "数据真的变了必须能被发现"


def test_gdb_content_mtime_survives_missing_dir(tmp_path):
    """目录不存在/不可读时不能抛异常（退回 0 或目录 mtime）。"""
    from gis_cli.datacatalog.scanner import gdb_content_mtime

    assert gdb_content_mtime(tmp_path / "nope.gdb") == 0.0


# ------------------------------------------------- 后台预热（启动不阻塞）


def test_prewarmer_runs_in_background_and_reports_done(tmp_path):
    """预热器在后台跑完，状态可查（界面据此显示进度）。"""
    import time

    from gis_cli.datacatalog.prewarm import CatalogPrewarmer

    calls: list[tuple] = []

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            calls.append((code, timeout))
            raise RuntimeError("不真的跑 arcpy（本测试只验证编排）")

    prewarmer = CatalogPrewarmer(
        catalog=object(), code_runner=_Runner(), workspace=tmp_path
    )
    assert prewarmer.state.started is False
    assert prewarmer.start() is True
    assert prewarmer.wait_ready(timeout=10).finished or prewarmer.state.failed
    # 失败也不能抛给调用方（扫描失败时任务照常进行）
    assert prewarmer.state.failed is True
    assert "扫描失败" in prewarmer.state.summary()


def test_prewarmer_start_is_idempotent(tmp_path):
    """重复 start 不应起两个线程（每个任务开始时都会调一次）。"""
    from gis_cli.datacatalog.prewarm import CatalogPrewarmer

    class _Runner:
        def run(self, code, timeout=None, workspace=None):
            raise RuntimeError("no arcpy")

    prewarmer = CatalogPrewarmer(catalog=object(), code_runner=_Runner(), workspace=tmp_path)
    assert prewarmer.start() is True
    assert prewarmer.start() is False  # 已在跑
    prewarmer.wait_ready(timeout=10)
    assert prewarmer.start() is False  # 已完成，不再重跑


def test_prewarmer_defaults_to_input_only(tmp_path):
    """默认只扫 input/data —— output 是产物，不该被当数据源扫。"""
    from gis_cli.datacatalog.prewarm import CatalogPrewarmer

    (tmp_path / "input").mkdir()
    (tmp_path / "output").mkdir()
    prewarmer = CatalogPrewarmer(catalog=object(), code_runner=None, workspace=tmp_path)
    roots = prewarmer._candidate_roots()
    assert any("input" in r for r in roots), roots
    assert not any("output" in r for r in roots), roots


def test_catalog_query_waits_for_prewarm():
    """目录还在后台扫时，catalog_query 要等它完成再答（对模型透明）。"""
    from gis_cli.engine.tools import EngineContext, build_default_registry
    from gis_cli.engine.state import Observation  # noqa: F401  (确保模块可用)

    waited: list[bool] = []

    class _Prewarmer:
        busy = True

        def wait_ready(self, timeout=None):
            waited.append(True)

    class _Catalog:
        def list_datasets(self, **kwargs):
            return []

        def counts(self):
            return {}

    ctx = EngineContext(workspace=__import__("pathlib").Path("."), catalog=_Catalog())
    ctx.extra["prewarmer"] = _Prewarmer()
    registry = build_default_registry()
    observation = registry.execute("catalog_query", {"query": "x"}, ctx)
    assert waited == [True], "catalog_query 必须先等后台扫描完成"
    assert observation.ok is True


def test_refresh_only_removes_within_scanned_roots(tmp_path):
    """只扫 input 时，不能把 output 里已登记的记录当成“已消失”删掉。

    实测坑：默认改为只扫 input/ 后，旧逻辑把 output/（自己刚生成的成果）
    的 19 条记录删了 —— 数据目录应该只增不误删。
    """
    from gis_cli.datacatalog.store import Catalog, DatasetRecord

    catalog = Catalog(tmp_path / "catalog.db")
    inbound = tmp_path / "input" / "a.shp"
    outbound = tmp_path / "output" / "result.gdb"
    inbound.parent.mkdir(parents=True)
    outbound.parent.mkdir(parents=True)
    inbound.write_bytes(b"x")
    outbound.mkdir()
    for path, kind in ((str(inbound), "vector"), (str(outbound), "gdb")):
        catalog.upsert(
            DatasetRecord(
                path=path, kind=kind, container="", name=Path(path).name,
                crs_name="", wkid=None, geom_type="", feature_count=1,
                extent=None, mtime=1.0, size=1, fields=[], raster={}, extra={},
            )
        )

    # 只拿 input 当扫描根：output 的记录不在 existing 集合里。
    # input 那条存在、output 那条不在范围内的不应被清 → 应为 0。
    removed = catalog.remove_missing(
        {str(inbound)},
        set(),
        within=[str(inbound.parent.resolve()).lower()],
    )
    assert removed == 0, f"范围内没有消失项，不应删任何东西（实际删了 {removed}）"
    assert catalog.get(str(outbound)) is not None, "output 的记录不能被删"
    assert catalog.get(str(inbound)) is not None, "范围内且存在的也不能删"
    catalog.close()


def test_remove_missing_without_within_keeps_old_behavior(tmp_path):
    """不传 within 时保持旧行为（全库对账），避免破坏既有调用方。"""
    from gis_cli.datacatalog.store import Catalog, DatasetRecord

    catalog = Catalog(tmp_path / "catalog.db")
    catalog.upsert(
        DatasetRecord(
            path=str(tmp_path / "gone.shp"), kind="vector", container="", name="gone",
            crs_name="", wkid=None, geom_type="", feature_count=1, extent=None,
            mtime=1.0, size=1, fields=[], raster={}, extra={},
        )
    )
    assert catalog.remove_missing(set(), set()) == 1
    catalog.close()


def test_remove_missing_deletes_gone_file_inside_scope(tmp_path):
    """范围内真的消失的文件要删（否则目录会永远留着幽灵记录）。"""
    from gis_cli.datacatalog.store import Catalog, DatasetRecord

    catalog = Catalog(tmp_path / "catalog.db")
    scope = tmp_path / "input"
    scope.mkdir()
    gone = scope / "gone.shp"
    catalog.upsert(
        DatasetRecord(
            path=str(gone), kind="vector", container="", name="gone",
            crs_name="", wkid=None, geom_type="", feature_count=1, extent=None,
            mtime=1.0, size=1, fields=[], raster={}, extra={},
        )
    )
    removed = catalog.remove_missing(set(), set(), within=[str(scope.resolve()).lower()])
    assert removed == 1
    assert catalog.get(str(gone)) is None
    catalog.close()


# ------------------------------------------------- 内核预热（并行启动）


def test_code_runner_prewarm_is_idempotent_and_never_raises(tmp_path):
    """预热是后台的、幂等的，且失败不能抛给调用方。

    内核冷启动实测 11.6~17.6s（大头是 import arcpy），预热后可与目录扫描、
    首轮模型调用重叠。没有 jupyter_client 时应快速失败并置 _kernel_failed，
    后续退回一次性子进程（与旧行为一致）。
    """
    import time

    from gis_cli.runtime.code_runner import CodeRunner

    runner = CodeRunner(workspace=tmp_path, state_dir=tmp_path / "k")
    first = runner.prewarm()
    second = runner.prewarm()
    assert first is True
    assert second is False, "已在预热中不应再起一个线程"
    # 等它跑完（可能失败，但不能抛异常）
    for _ in range(200):
        if runner._prewarm_thread is None or not runner._prewarm_thread.is_alive():
            break
        time.sleep(0.05)
    # 预热是一次性动作：跑过之后再调也不应起新线程（不能只看线程是否还活着，
    # 线程可能在调用方拿到锁之前就已结束 —— 那种写法会又起一个内核）
    assert runner.prewarm() is False, "预热是一次性的，结束后不应再起新线程"


def test_loop_starts_kernel_prewarm_on_run():
    """任务开始时同时启动目录扫描与内核预热（两者并行）。"""
    from gis_cli.engine.loop import AgentLoop, LoopConfig

    prewarmed: list[bool] = []

    class _Runner:
        def prewarm(self):
            prewarmed.append(True)
            return True

        def reset(self):
            return None

    loop = AgentLoop(
        workspace=".",
        llm_client=None,
        registry=None,
        context=None,
        code_runner=_Runner(),
        config=LoopConfig(),
        auto_extract_requirements=False,
        use_project_log=False,
    )
    loop._start_catalog_prewarm()
    assert prewarmed == [True], "任务开始时应预热内核"


def test_loop_prewarm_survives_runner_without_prewarm():
    """老的自定义 runner（没有 prewarm 方法）不能因此报错。"""
    from gis_cli.engine.loop import AgentLoop, LoopConfig

    class _LegacyRunner:
        def reset(self):
            return None

    loop = AgentLoop(
        workspace=".",
        llm_client=None,
        registry=None,
        context=None,
        code_runner=_LegacyRunner(),
        config=LoopConfig(),
        auto_extract_requirements=False,
        use_project_log=False,
    )
    loop._start_catalog_prewarm()  # 不应抛异常


def test_shutdown_during_prewarm_does_not_crash(tmp_path):
    """预热还没完成就 shutdown 不能崩（实测会撞 zmq Bad address）。

    场景：用户启动 agent 后马上退出，而内核还在 import arcpy。
    shutdown 必须先等在途的预热收尾，再关内核。
    """
    import threading
    import time

    from gis_cli.runtime.code_runner import CodeRunner

    started = threading.Event()

    class _SlowKernel:
        def start(self):
            started.set()
            time.sleep(1.5)  # 模拟 import arcpy

        def is_alive(self):
            return True

        def shutdown(self):
            return None

    runner = CodeRunner(workspace=tmp_path, state_dir=tmp_path / "k")
    real_kernel = runner._kernel

    def _fake_ensure():
        started.set()
        time.sleep(1.5)
        runner._kernel = _SlowKernel()

    runner._prewarm_thread = threading.Thread(target=_fake_ensure, daemon=True)
    runner._prewarm_thread.start()
    started.wait(timeout=5)
    time.sleep(0.05)
    # 不应抛异常，也不应残留半启动的内核
    runner.shutdown()
    assert runner._kernel_failed is True, "关掉后不应再自动重启内核"


def test_kernel_execute_is_serialized_for_concurrent_callers():
    """并发 execute 必须串行通过 ZMQ socket（跨线程交叉会写坏 wire 协议）。

    实测根因：预热把内核建在后台线程，主线程随后执行代码 —— 两边 socket
    操作交叉。症状是内核侧 ``Invalid Signature: b'<IDS|MSG>'`` /
    ``DELIM not in msg_list``，客户端 zmq ``Bad address``（复现率很高）。
    这里用假内核直接验证锁的行为，不需要真的 ArcPy。
    """
    import threading
    import time

    from gis_cli.runtime.kernel import ArcpyKernel, ExecResult

    kernel = ArcpyKernel.__new__(ArcpyKernel)  # 跳过 __init__（不连真内核）
    kernel.workspace = None
    kernel._exec_lock = threading.RLock()
    kernel._started = True
    kernel._exec_count = 0

    overlap = {"max": 0, "now": 0}
    counter_lock = threading.Lock()

    def fake_raw(code, *, timeout):
        with counter_lock:
            overlap["now"] += 1
            overlap["max"] = max(overlap["max"], overlap["now"])
        time.sleep(0.05)  # 模拟执行耗时；串行化时不会重叠
        with counter_lock:
            overlap["now"] -= 1
        return ExecResult(ok=True, stdout="")

    kernel.is_alive = lambda: True  # type: ignore[method-assign]
    kernel._execute_raw = fake_raw  # type: ignore[method-assign]
    kernel._fetch_result = lambda: None  # type: ignore[method-assign]

    errors: list[str] = []

    def work():
        try:
            for _ in range(3):
                kernel.execute("pass", timeout=10)
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert overlap["max"] == 1, f"执行必须串行，实际最大并发 {overlap['max']}"
