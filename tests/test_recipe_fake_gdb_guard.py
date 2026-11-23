# -*- coding: utf-8 -*-
"""假 .gdb 防线：`os.makedirs("x.gdb")` 建出来的普通目录会让 arcpy 把库内
二进制表当文本读（UnicodeDecodeError），而 `arcpy.Exists()` 对它仍返回 True，
模型随后的 `if not arcpy.Exists(gdb): CreateFileGDB(...)` 会被跳过。

4a 实测：模型把 `kernel_density` 的 out_dir 传成 `output/result_data.gdb`，
配方 makedirs 建出假库 → 之后两次尝试都在同一处报 UnicodeDecodeError。
"""

from __future__ import annotations

import ast
from pathlib import Path

from gis_cli.engine.tools import EngineContext
from gis_cli.recipes.schema import Recipe, RecipeParam
from gis_cli.recipes.store import RecipeLibrary


class _StubRunner:
    def __init__(self):
        self.codes: list[str] = []

    def run(self, code, timeout=None, workspace=None):
        self.codes.append(code)
        from types import SimpleNamespace

        return SimpleNamespace(ok=True, stdout="", stderr="", result=None, error=None, display_images=[])


def _recipe(out_dir_default: str = "output/result_data.gdb") -> Recipe:
    return Recipe(
        id="kde_like",
        name="写栅格到目录",
        code_template="import os\nos.makedirs({{out_dir}}, exist_ok=True)\n",
        params={"out_dir": RecipeParam("out_dir", "string", True, out_dir_default, "输出目录")},
        validation=[],
    )


def _library(tmp_path: Path, recipe: Recipe) -> tuple[RecipeLibrary, _StubRunner]:
    lib = RecipeLibrary(user_dir=tmp_path / "recipes")
    lib.load()
    return lib, _StubRunner()


def _run(lib: RecipeLibrary, recipe: Recipe, runner: _StubRunner, tmp_path: Path, params=None):
    lib._recipes[recipe.id] = recipe  # 直接放进内存库（不走磁盘）
    ctx = EngineContext(workspace=tmp_path, code_runner=runner)
    return lib.run(recipe.id, params or {"out_dir": str(recipe.params["out_dir"].default)}, ctx)


def test_fake_gdb_directory_is_refused(tmp_path: Path):
    fake = tmp_path / "output" / "result_data.gdb"
    fake.mkdir(parents=True)  # 普通目录，没有 gdb/timestamps/a00000001.gdbtable
    recipe = _recipe()
    lib, runner = _library(tmp_path, recipe)

    obs = _run(lib, recipe, runner, tmp_path, {"out_dir": "output/result_data.gdb"})

    assert obs.ok is False
    assert obs.error and obs.error["type"] == "FakeGdbPath"
    assert "假的 .gdb" in obs.summary
    assert runner.codes == [], "拦下来就不该再执行配方代码"


def test_real_gdb_directory_is_allowed(tmp_path: Path):
    real = tmp_path / "output" / "result_data.gdb"
    real.mkdir(parents=True)
    (real / "gdb").write_bytes(b"\x00")
    recipe = _recipe()
    lib, runner = _library(tmp_path, recipe)

    obs = _run(lib, recipe, runner, tmp_path, {"out_dir": "output/result_data.gdb"})

    assert obs.ok is True, obs.summary
    assert runner.codes, "真 GDB 不该被拦"


def test_missing_gdb_path_is_not_blocked(tmp_path: Path):
    """不存在的不拦：配方可能正要用 CreateFileGDB 建它。"""
    recipe = _recipe()
    lib, runner = _library(tmp_path, recipe)
    obs = _run(lib, recipe, runner, tmp_path, {"out_dir": "output/new.gdb"})
    assert obs.ok is True, obs.summary


def test_plain_output_dir_is_not_blocked(tmp_path: Path):
    (tmp_path / "output").mkdir()
    recipe = _recipe()
    lib, runner = _library(tmp_path, recipe)
    obs = _run(lib, recipe, runner, tmp_path, {"out_dir": "output"})
    assert obs.ok is True, obs.summary


def test_kernel_density_rejects_gdb_out_dir():
    """kernel_density 的 out_dir 是放 TIFF 的目录，传 .gdb 要在渲染后立刻报错。"""
    from gis_cli.recipes.store import RecipeLibrary

    lib = RecipeLibrary(user_dir=Path("does-not-exist"))
    lib.load()
    recipe = lib.get("kernel_density")
    assert recipe is not None
    code = recipe.render(
        {
            "points": "input/animal_points.shp",
            "out_dir": "output/result_data.gdb",
            "periods": "1982-1986",
        }
    )
    ast.parse(code)  # 渲染结果必须是合法 Python
    assert 'endswith(".gdb")' in code
    with_gdb = [
        line for line in code.splitlines() if "raise ValueError" in line and "GDB" in line
    ]
    assert with_gdb, "kernel_density 里应有 out_dir 是 .gdb 时的显式报错"


class _ExecRunner(_StubRunner):
    """execute_code 会读 duration_ms / display_images。"""

    def run(self, code, timeout=None, workspace=None):
        from types import SimpleNamespace

        self.codes.append(code)
        return SimpleNamespace(
            ok=True, stdout="", stderr="", result=None, error=None,
            display_images=[], duration_ms=1,
        )


def test_execute_code_blocks_writing_into_fake_gdb(tmp_path):
    """模型自己写的“库存在吗”判断也会被 arcpy.Exists 骗过。

    实测（14 届组合任务）：`os.makedirs` 留下的假 `output/landcover.gdb` 让
    `if not arcpy.Exists(gdb)` 跳过 CreateFileGDB，随后 `out_ras.save()` 抛
    UnicodeDecodeError —— 一次 382s、一次 833s 都耗在这上面。执行前 0.1s 拦下。
    """
    from gis_cli.engine.tools import build_default_registry

    out = tmp_path / "output"
    out.mkdir()
    fake = out / "landcover.gdb"
    fake.mkdir()

    runner = _ExecRunner()
    ctx = EngineContext(workspace=tmp_path, code_runner=runner)
    obs = build_default_registry().execute(
        "execute_code",
        {"code": f"import arcpy\narcpy.management.CreateFileGDB(r'{out}', 'landcover.gdb')"},
        ctx,
    )

    assert obs.ok is False
    assert (obs.error or {}).get("type") == "FakeGdbPath"
    assert "假 .gdb" in obs.summary
    assert "CreateFileGDB" in (obs.hint or "")
    assert runner.codes == [], "假库必须在执行前拦住，不能先跑 6 分钟再报错"


def test_execute_code_allows_real_gdb_and_missing_gdb(tmp_path):
    """真库（有 gdb 标记文件）和还不存在的库都不该被拦。"""
    from gis_cli.engine.tools import build_default_registry

    out = tmp_path / "output"
    out.mkdir()
    real = out / "result_data.gdb"
    real.mkdir()
    (real / "gdb").write_bytes(b"")  # 真 File GDB 的标记文件

    runner = _ExecRunner()
    ctx = EngineContext(workspace=tmp_path, code_runner=runner)
    obs = build_default_registry().execute(
        "execute_code",
        {"code": "arcpy.management.CreateFileGDB(r'output', 'new.gdb')\narcpy.env.workspace = 'output/result_data.gdb'"},
        ctx,
    )

    assert obs.ok is True, obs.summary
    assert len(runner.codes) == 1


def test_fake_gdb_guard_does_not_hit_real_gdb_with_same_name_elsewhere(tmp_path):
    """工作区别处有同名假库时，指向**真库**的代码不能被拦（DAIR-8 B1-1）。

    以前相对路径解析成功后没有 continue，还会走“按名字 glob”，
    于是报出来的是一个完全不相干的目录。
    """
    from gis_cli.engine.tools import _fake_gdb_in_code

    out = tmp_path / "output"
    out.mkdir()
    real = out / "landcover.gdb"
    real.mkdir()
    (real / "gdb").write_bytes(b"")
    (real / "timestamps").write_bytes(b"")
    (tmp_path / "landcover.gdb").mkdir()  # 根上有个同名假库（上一轮遗留）

    assert _fake_gdb_in_code("arcpy.env.workspace = r'output/landcover.gdb'", tmp_path) is None
    assert _fake_gdb_in_code("arcpy.env.workspace = 'output/landcover.gdb'", tmp_path) is None


def test_fake_gdb_guard_ignores_mentions_in_comments(tmp_path):
    """注释里提到 x.gdb 不算要写它（DAIR-8 B1-3）。"""
    from gis_cli.engine.tools import _fake_gdb_in_code

    (tmp_path / "output" / "landcover.gdb").mkdir(parents=True)
    code = "# 上次留下的 landcover.gdb 是假的\nprint(1)\n"
    assert _fake_gdb_in_code(code, tmp_path) is None


def test_fake_gdb_guard_allows_cleanup_calls(tmp_path):
    """清理假库的代码必须放行——guard 自己的 hint 就写着“先 rmtree 清掉”（DAIR-8 B1-2）。"""
    from gis_cli.engine.tools import _fake_gdb_in_code

    (tmp_path / "output" / "landcover.gdb").mkdir(parents=True)
    fake = str(tmp_path / "output" / "landcover.gdb")

    assert _fake_gdb_in_code("shutil.rmtree(r'output/landcover.gdb')", tmp_path) is None
    assert _fake_gdb_in_code("os.remove(r'output/landcover.gdb')", tmp_path) is None
    # 经变量中转的清理也算清理意图
    assert _fake_gdb_in_code(
        "gdb = os.path.join(out, 'landcover.gdb')\nshutil.rmtree(gdb)", tmp_path
    ) is None
    # 但“往里写”照样要拦
    assert _fake_gdb_in_code("out_ras.save(r'output/landcover.gdb/x')", tmp_path) == fake


def test_fake_gdb_guard_handles_windows_backslash_paths(tmp_path):
    r"""反斜杠路径必须被当成一个完整 path token（DAIR-9）。

    正则的字符类里少了转义的反斜杠时，``output\landcover.gdb`` 会退化成裸名，
    落到“按名字 glob”兜底：指向**真库**的代码被拦（报的还是别处那个假库），
    指向**工作区外假库**的代码反而漏拦。而 guard 自己的报错文案用的就是
    ``str(path)``——Windows 上正是反斜杠，模型照抄回来必踩。
    """
    BS = chr(92)
    from gis_cli.engine.tools import _fake_gdb_in_code

    out = tmp_path / "output"
    out.mkdir()
    real = out / "landcover.gdb"
    real.mkdir()
    (real / "gdb").write_bytes(b"")
    (real / "timestamps").write_bytes(b"")
    (tmp_path / "landcover.gdb").mkdir()  # 根上同名假库

    # 1) 指向真库的反斜杠相对路径 + 别处有同名假库 → 不拦
    assert _fake_gdb_in_code("arcpy.env.workspace = r'output" + BS + "landcover.gdb'", tmp_path) is None
    # 2) 指向真库的反斜杠绝对路径 → 不拦
    assert _fake_gdb_in_code("arcpy.env.workspace = r'" + str(real) + "'", tmp_path) is None

    # 3) 相对两层深的假库 → 必须拦
    deep = tmp_path / "sub" / "out" / "d.gdb"
    deep.mkdir(parents=True)
    assert _fake_gdb_in_code("out_ras.save(r'sub" + BS + "out" + BS + "d.gdb" + BS + "x')", tmp_path) == str(deep)

    # 4) 工作区外的假库（绝对路径）→ 必须拦
    other = tmp_path.parent / (tmp_path.name + "_outside.gdb")
    other.mkdir()
    try:
        assert _fake_gdb_in_code("out_ras.save(r'" + str(other) + "')", tmp_path) == str(other)
    finally:
        other.rmdir()


def test_fake_gdb_guard_allows_cleanup_through_loop_variable(tmp_path):
    """循环变量拼出来的清理路径也要放行（DAIR-15 trace 107→109）。

    ``for name in ["fake_a.gdb"]: p = os.path.join("output", name)`` 之后
    ``shutil.rmtree(p)`` 以前解析不出名字（只跟赋值、不跟循环变量），被当成
    "往假库写"误拦；agent 照 hint 写的整改代码被自己的 guard 怼回去，最后靠
    ``arcpy.Delete_management`` 才过——白多一轮修复。
    """
    from gis_cli.engine.tools import _fake_gdb_in_code

    fake = tmp_path / "output" / "fake_a.gdb"
    fake.mkdir(parents=True)

    loop_cleanup = (
        "for name in ['fake_a.gdb']:\n"
        "    p = os.path.join('output', name)\n"
        "    shutil.rmtree(p)\n"
    )
    assert _fake_gdb_in_code(loop_cleanup, tmp_path) is None
    # 对照组：直写路径本来就放行
    assert _fake_gdb_in_code("shutil.rmtree('output/fake_a.gdb')", tmp_path) is None
    # 循环里"往里写"照样要拦——展开循环不是免死金牌
    assert _fake_gdb_in_code(
        "for name in ['fake_a.gdb']:\n"
        "    p = os.path.join('output', name)\n"
        "    arcpy.management.CreateFeatureclass(p, 'probe_pts')\n",
        tmp_path,
    ) == str(fake)


def test_cleanup_of_child_path_does_not_exempt_writes(tmp_path):
    """删假库**里面的要素类**不算清理意图 —— 同一段里"往里写"照样要拦。

    DAIR-16 探针 run 20261023_170723 实测：模型把
    ``if arcpy.Exists(...): arcpy.management.Delete("output/fake_a.gdb/probe_pts")``
    和 ``CreateFeatureclass("output/fake_a.gdb", ...)`` 写在一个 cell 里，而清理豁免
    按 basename 生效 → 整段放行 → arcpy 把普通目录当 folder workspace，假库里真的
    写出了 shapefile，(a) 拦截也就看不见了。
    """
    from gis_cli.engine.tools import _fake_gdb_in_code

    fake = tmp_path / "output" / "fake_a.gdb"
    fake.mkdir(parents=True)

    delete_child_then_write = (
        'if arcpy.Exists("output/fake_a.gdb/probe_pts"):\n'
        '    arcpy.management.Delete("output/fake_a.gdb/probe_pts")\n'
        'arcpy.management.CreateFeatureclass("output/fake_a.gdb", "probe_pts", "POINT")\n'
    )
    assert _fake_gdb_in_code(delete_child_then_write, tmp_path) == str(fake)

    # os.path.join 拼出来的子路径同理（不能只认字面量）
    join_child = (
        'arcpy.management.Delete(os.path.join("output/fake_a.gdb", "probe_pts"))\n'
        'out_ras.save("output/fake_a.gdb/x.tif")\n'
    )
    assert _fake_gdb_in_code(join_child, tmp_path) == str(fake)

    # 删的就是假库本身 → 仍是清理意图（放行）
    delete_gdb_then_write = (
        'arcpy.management.Delete("output/fake_a.gdb")\n'
        'arcpy.management.CreateFeatureclass("output/fake_a.gdb", "probe_pts")\n'
    )
    assert _fake_gdb_in_code(delete_gdb_then_write, tmp_path) is None

    # 经 abspath 中转的变量只删库内要素类 → 也不是清理意图（宽判据不能漏掉这层）
    abspath_child = (
        'p = os.path.abspath("output/fake_a.gdb")\n'
        'arcpy.management.Delete(os.path.join(p, "probe_pts"))\n'
        'arcpy.management.CreateFeatureclass(p, "probe_pts", "POINT")\n'
    )
    assert _fake_gdb_in_code(abspath_child, tmp_path) == str(fake)

    # 外壳包住的就是库内子路径、变量再拿去删 —— 同样不是清理意图（DAIR-18 阻塞项）
    wrapped_child_vars = (
        'p = os.path.abspath("output/fake_a.gdb/probe_pts")\n'
        "arcpy.management.Delete(p)\n"
        'arcpy.management.CreateFeatureclass("output/fake_a.gdb", "probe_pts", "POINT")\n',
        'p = str(Path("output") / "fake_a.gdb" / "probe_pts")\n'
        "arcpy.management.Delete(p)\n"
        'arcpy.management.CreateFeatureclass("output/fake_a.gdb", "probe_pts", "POINT")\n',
        'p = Path("output") / "fake_a.gdb" / "probe_pts"\n'
        "shutil.rmtree(p)\n"
        'arcpy.management.CreateFeatureclass("output/fake_a.gdb", "probe_pts", "POINT")\n',
        # 直接内联（不经变量）：8382aa14 上就已经放行过，这轮一并收掉
        'arcpy.management.Delete(os.path.abspath("output/fake_a.gdb/probe_pts"))\n'
        'arcpy.management.CreateFeatureclass("output/fake_a.gdb", "probe_pts", "POINT")\n',
    )
    for code in wrapped_child_vars:
        assert _fake_gdb_in_code(code, tmp_path) == str(fake), code


def test_cleanup_wrapped_in_call_still_allowed(tmp_path):
    """路径被外壳包住（os.path.abspath 等）解析不出时仍按清理放行——不要为收紧而误拦。"""
    from gis_cli.engine.tools import _fake_gdb_in_code

    fake = tmp_path / "output" / "fake_a.gdb"
    fake.mkdir(parents=True)

    assert _fake_gdb_in_code('shutil.rmtree(os.path.abspath("output/fake_a.gdb"))', tmp_path) is None


def test_cleanup_through_wrapped_variable_still_allowed(tmp_path):
    """外壳包住的值先赋给变量再删，也要放行（DAIR-17 复审阻塞项）。

    ``p = os.path.abspath("output/fake_a.gdb"); shutil.rmtree(p)``：赋值右侧解析不出路径
    → 变量登记为空集，而 ``rmtree(p)`` 的参数是裸 Name 也解析不出 → 两头都拿不到名字，
    照 guard 自己 hint 写的清理被拦下，逼模型改用 ``arcpy.Delete_management`` 绕过去。
    ``str(Path(...))`` / ``Path(...) / "x.gdb"`` 同理。
    """
    from gis_cli.engine.tools import _fake_gdb_in_code

    fake = tmp_path / "output" / "fake_a.gdb"
    fake.mkdir(parents=True)

    abspath_var = (
        'p = os.path.abspath("output/fake_a.gdb")\n'
        "if os.path.isdir(p):\n"
        "    shutil.rmtree(p)\n"
        'arcpy.management.CreateFileGDB("output", "fake_a.gdb")\n'
    )
    assert _fake_gdb_in_code(abspath_var, tmp_path) is None

    str_path_var = (
        'gdb = str(Path("output") / "fake_a.gdb")\n'
        "shutil.rmtree(gdb)\n"
        'arcpy.management.CreateFileGDB("output", "fake_a.gdb")\n'
    )
    assert _fake_gdb_in_code(str_path_var, tmp_path) is None

    path_var = (
        'gdb = Path("output") / "fake_a.gdb"\n'
        "shutil.rmtree(gdb)\n"
        'arcpy.management.CreateFileGDB("output", "fake_a.gdb")\n'
    )
    assert _fake_gdb_in_code(path_var, tmp_path) is None
