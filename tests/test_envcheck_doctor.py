# -*- coding: utf-8 -*-
"""环境自检（doctor）：7 种状态各一条，全部用 monkeypatch 伪造。

**为什么全部伪造**：这些用例要在没有 ArcGIS Pro 的 CI 机器上跑
（`pytest -m "not arcpy"`），而 doctor 的全部价值恰恰是"没装 Pro 时也能把结论
说对"——用真机反而测不到那些分支。真机验证（有 Pro 正常 / 有 Pro 但依赖缺 /
无 Pro）另做，不进 CI。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gis_cli.arcpy_bridge import (
    ArcGISDiscoveryError,
    ArcGISPythonInfo,
    ArcPyExecutionResult,
)
from gis_cli.envcheck import doctor, install
from gis_cli.runtime.kernel import ExecResult

FAKE_PYTHON = r"E:\FakePro\bin\Python\envs\arcgispro-py3\python.exe"
WORKSPACE = r"E:\gis_work"


def _info(
    python: str = FAKE_PYTHON, install_dir: str = r"E:\FakePro"
) -> ArcGISPythonInfo:
    return ArcGISPythonInfo(
        install_dir=install_dir,
        python_executable=python,
        source=r"registry:SOFTWARE\ESRI\ArcGISPro",
    )


def _result(
    status: str = "success",
    data=None,
    stderr: str = "",
    error: dict | None = None,
) -> ArcPyExecutionResult:
    return ArcPyExecutionResult(
        status=status,
        exit_code=0 if status == "success" else 1,
        python_executable=FAKE_PYTHON,
        stdout="",
        stderr=stderr,
        data=data,
        error=error,
    )


@pytest.fixture()
def env(monkeypatch):
    """默认环境：有 Pro、arcpy 正常、9 个依赖齐全。各用例按需覆盖其中一项。"""
    state = {
        "info": _info(),
        "arcpy": _result(
            data={"version": "3.6", "build": "59527", "install_dir": r"E:\FakePro"}
        ),
        "deps": _result(data={"missing": []}),
        "spec": _result(data={"missing": [], "arcpy": True}),
        "calls": [],
    }

    def fake_discover():
        if isinstance(state["info"], Exception):
            raise state["info"]
        return state["info"]

    def fake_run_arcpy_code(code, **kwargs):
        state["calls"].append((code, kwargs))
        if "GetInstallInfo" in code:
            return state["arcpy"]
        if "find_spec" in code:
            return state["spec"]
        return state["deps"]

    monkeypatch.setattr(doctor, "discover_arcgis_pro_python", fake_discover)
    monkeypatch.setattr(doctor, "run_arcpy_code", fake_run_arcpy_code)
    monkeypatch.setattr(doctor, "_build_python_candidates", lambda: [])
    return state


def _checks(report) -> dict:
    return {check.id: check for check in report.checks}


# === 1. pro_installed ===

def test_no_pro_fails_with_the_three_tried_routes(env):
    env["info"] = ArcGISDiscoveryError("未找到 ArcGIS Pro Python 解释器。")
    report = doctor.run_checks(workspace=WORKSPACE)

    checks = _checks(report)
    assert checks["pro_installed"].status == doctor.FAIL
    assert "ARCGIS_PRO_PYTHON" in checks["pro_installed"].hint
    assert "注册表" in checks["pro_installed"].hint
    # 根因只有一个：找不到 Pro 时其余检查 skip，而不是再报 3 条 fail 让人更糊涂
    assert checks["arcpy_importable"].status == doctor.SKIP
    assert checks["deps"].status == doctor.SKIP
    assert report.ok is False and report.exit_code == 1


def test_multiple_pro_versions_are_listed_and_launcher_ignored(env, monkeypatch, tmp_path):
    other_python = tmp_path / "Pro3.4" / "bin" / "Python" / "envs" / "arcgispro-py3" / "python.exe"
    other_python.parent.mkdir(parents=True)
    other_python.write_text("", encoding="utf-8")
    launcher = tmp_path / "Pro3.4" / "bin" / "Python" / "scripts" / "propy.bat"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        doctor,
        "_build_python_candidates",
        lambda: [
            (r"registry:SOFTWARE\ESRI\ArcGISPro", FAKE_PYTHON, r"E:\FakePro"),
            ("filesystem:fallback", str(other_python), str(tmp_path / "Pro3.4")),
            ("filesystem:fallback", str(launcher), str(tmp_path / "Pro3.4")),
        ],
    )

    check = _checks(doctor.run_checks(workspace=WORKSPACE))["pro_installed"]

    assert check.status == doctor.PASS
    # 多版本并存：另一个要列出来（并说明本次用的是哪个）
    assert str(other_python) in check.hint
    assert "本次用的是" in check.hint
    # propy.bat 是同一个环境的启动器，不是"另一个解释器"
    assert "propy.bat" not in check.hint


# === 2. arcpy_importable ===

def test_arcpy_import_failure_reports_first_three_stderr_lines(env):
    env["arcpy"] = _result(
        status="error",
        stderr="第1行\n第2行\n第3行\n第4行\n第5行",
        error={"message": "boom", "traceback": "trace"},
    )
    report = doctor.run_checks(workspace=WORKSPACE)

    check = _checks(report)["arcpy_importable"]
    assert check.status == doctor.FAIL
    assert "第1行" in check.detail and "第3行" in check.detail
    assert "第4行" not in check.detail  # 只报前 3 行，不把整段栈糊给用户
    assert report.ok is False


def test_arcpy_import_success_reports_version_and_build(env):
    check = _checks(doctor.run_checks(workspace=WORKSPACE))["arcpy_importable"]
    assert check.status == doctor.PASS
    assert "3.6" in check.detail and "59527" in check.detail


# === 3. pro_version ===

def test_old_pro_is_warn_not_fail(env):
    env["arcpy"] = _result(data={"version": "3.1", "build": "1"})
    report = doctor.run_checks(workspace=WORKSPACE)

    check = _checks(report)["pro_version"]
    assert check.status == doctor.WARN
    assert "3.2" in check.hint
    # warn 不阻塞：老版本仍能跑，只是缺几个 API
    assert report.ok is True and report.exit_code == 0


def test_new_enough_pro_passes(env):
    check = _checks(doctor.run_checks(workspace=WORKSPACE))["pro_version"]
    assert check.status == doctor.PASS and "3.6" in check.detail


@pytest.mark.parametrize(
    "text,expected",
    [("3.6", (3, 6)), ("3.2.1", (3, 2, 1)), ("3.6.0.59527", (3, 6, 0)), ("", None), ("unknown", None)],
)
def test_parse_version(text, expected):
    assert doctor._parse_version(text) == expected


# === 4. deps ===

def test_missing_dependency_gives_one_pip_install_command(env):
    env["deps"] = _result(data={"missing": ["jupyter_client"]})
    report = doctor.run_checks(workspace=WORKSPACE)

    check = _checks(report)["deps"]
    assert check.status == doctor.FAIL
    # 报的是 pip 名，不是 import 名（import 名拼不出能跑的命令）
    assert check.detail.startswith("缺依赖 jupyter-client（")
    # 被检查的环境路径要进 detail：只看 --json 的消费者靠它分清 venv 缺还是 Pro 缺
    assert str(Path(FAKE_PYTHON).parent) in check.detail
    assert "pip install jupyter-client" in check.hint
    assert report.summary.startswith("缺依赖 jupyter-client")


def test_dependency_import_names_are_mapped_to_pip_names(env):
    env["deps"] = _result(data={"missing": ["PIL", "yaml", "prompt_toolkit"]})
    check = _checks(doctor.run_checks(workspace=WORKSPACE))["deps"]

    assert check.detail.startswith("缺依赖 pillow、pyyaml、prompt-toolkit（")
    assert "pip install pillow pyyaml prompt-toolkit" in check.hint


def test_deps_check_uses_the_pro_interpreter_not_the_current_one(env):
    doctor.run_checks(workspace=WORKSPACE)
    # 两次探测都必须显式带上 discover 返回的解释器（禁止 fallback 到 sys.executable）
    assert {kwargs["python_executable"] for _code, kwargs in env["calls"]} == {FAKE_PYTHON}


# === 5. path_spaces ===

def test_program_files_is_not_a_warning(env):
    """Pro 默认装在 C:\\Program Files 下，对所有人报警等于没报警。"""
    env["info"] = _info(
        python=r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe",
        install_dir=r"C:\Program Files\ArcGIS\Pro",
    )
    check = _checks(doctor.run_checks(workspace=WORKSPACE))["path_spaces"]

    assert check.status == doctor.PASS
    assert "默认安装即如此" in check.detail


def test_workspace_with_spaces_warns(env):
    report = doctor.run_checks(workspace=r"E:\my gis work")

    check = _checks(report)["path_spaces"]
    assert check.status == doctor.WARN
    assert "--workspace" in check.hint
    assert report.ok is True


# === 6. path_nonascii ===

def test_chinese_path_is_warn_only(env):
    report = doctor.run_checks(workspace=r"E:\桌面\gis_work")

    check = _checks(report)["path_nonascii"]
    assert check.status == doctor.WARN
    assert "桌面" in check.detail
    # 中文路径历史上能跑，只提醒不判失败（规格 1.7 的风险表）
    assert report.ok is True and report.exit_code == 0


def test_ascii_paths_pass(env):
    check = _checks(doctor.run_checks(workspace=WORKSPACE))["path_nonascii"]
    assert check.status == doctor.PASS


# === 7. kernel_start ===

def test_kernel_check_is_skipped_without_deep(env):
    report = doctor.run_checks(workspace=WORKSPACE)

    assert _checks(report)["kernel_start"].status == doctor.SKIP
    # 没加 --deep 就不该多起两个子进程之外的任何东西（arcpy + deps 各一次）
    assert len(env["calls"]) == 2


class _FakeKernel:
    started = 0
    shut = 0

    def __init__(self, **_kwargs):
        pass

    def start(self):
        _FakeKernel.started += 1

    def execute(self, _code, *, timeout=600):
        return ExecResult(ok=True, stdout="1\n", duration_ms=230)

    def shutdown(self):
        _FakeKernel.shut += 1


def test_deep_kernel_check_passes_and_shuts_down(env, monkeypatch):
    monkeypatch.setattr("gis_cli.runtime.kernel.ArcpyKernel", _FakeKernel)
    _FakeKernel.started = _FakeKernel.shut = 0

    check = _checks(doctor.run_checks(deep=True, workspace=WORKSPACE))["kernel_start"]

    assert check.status == doctor.PASS
    assert "print(1)" in check.detail
    assert _FakeKernel.started == 1 and _FakeKernel.shut == 1  # 体检完必须关掉内核


def test_deep_kernel_failure_does_not_leak_the_kernel(env, monkeypatch):
    class BrokenKernel(_FakeKernel):
        def start(self):
            raise RuntimeError("Kernel did not become ready")

    monkeypatch.setattr("gis_cli.runtime.kernel.ArcpyKernel", BrokenKernel)
    report = doctor.run_checks(deep=True, workspace=WORKSPACE)

    check = _checks(report)["kernel_start"]
    # 判 WARN：内核起不来只影响速度（自动退回子进程），不该让能用的环境退出码变 1
    assert check.status == doctor.WARN
    assert "Kernel did not become ready" in check.detail
    assert "子进程" in check.hint
    assert report.ok is True and report.exit_code == 0


def test_deep_kernel_timeout_mentions_the_ten_second_budget(env, monkeypatch):
    class SlowKernel(_FakeKernel):
        def execute(self, _code, *, timeout=600):
            raise RuntimeError("Execution exceeded 10s and did not stop after interrupt")

    monkeypatch.setattr("gis_cli.runtime.kernel.ArcpyKernel", SlowKernel)
    _FakeKernel.shut = 0

    check = _checks(doctor.run_checks(deep=True, workspace=WORKSPACE))["kernel_start"]

    assert check.status == doctor.WARN
    assert "10s" in check.detail
    assert _FakeKernel.shut == 1  # 超时也要把内核关掉，不能留在后台


# === 聚合 / JSON ===

def test_json_payload_shape_matches_the_spec(env):
    env["deps"] = _result(data={"missing": ["jupyter_client"]})
    payload = doctor.run_checks(workspace=WORKSPACE).to_dict()

    assert payload["ok"] is False
    assert payload["summary"].startswith("缺依赖 jupyter-client")
    deps_item = next(item for item in payload["checks"] if item["id"] == "deps")
    # 只读 JSON 的消费者也能看出是哪个环境缺包
    assert str(Path(FAKE_PYTHON).parent) in deps_item["detail"]
    assert [item["id"] for item in payload["checks"]] == [
        "pro_installed",
        "arcpy_importable",
        "pro_version",
        "deps",
        "path_spaces",
        "path_nonascii",
        "kernel_start",
    ]
    for item in payload["checks"]:
        assert set(item) <= {"id", "status", "detail", "hint"}
        assert item["status"] in {doctor.PASS, doctor.WARN, doctor.FAIL, doctor.SKIP}


# === 启动快检（step d）===

def test_startup_checks_use_one_fast_probe_without_importing_arcpy(env):
    report = doctor.run_startup_checks(workspace=WORKSPACE)

    assert [check.id for check in report.checks] == [
        "pro_installed",
        "arcpy_importable",
        "deps",
    ]
    assert report.ok is True
    # 只起一个子进程，且不做真实导入（真导入本机实测 10.7s，不能加到每次启动上）
    assert len(env["calls"]) == 1
    code, kwargs = env["calls"][0]
    assert "GetInstallInfo" not in code
    assert kwargs["require_arcpy"] is False


def test_startup_checks_report_missing_arcpy_and_deps(env):
    env["spec"] = _result(data={"missing": ["numpy"], "arcpy": False})
    report = doctor.run_startup_checks(workspace=WORKSPACE)

    checks = _checks(report)
    assert checks["arcpy_importable"].status == doctor.FAIL
    assert checks["deps"].status == doctor.FAIL
    assert "numpy" in checks["deps"].detail
    assert report.ok is False


def test_startup_probe_failure_is_not_reported_as_missing_arcpy(env):
    """快检**自己**没跑起来时，不能说成"环境里没有 arcpy"（DAIR-24 阻塞项）。"""
    env["spec"] = _result(status="error", error={"message": "执行超时，超过 30 秒。"})
    report = doctor.run_startup_checks(workspace=WORKSPACE)

    checks = _checks(report)
    assert checks["arcpy_importable"].status == doctor.FAIL
    assert "快检没跑起来" in checks["arcpy_importable"].detail
    assert "执行超时" in checks["arcpy_importable"].detail
    # 工具自身失败 ≠ 环境缺 arcpy：这句话绝不能出现
    assert "找不到 arcpy 模块" not in checks["arcpy_importable"].detail
    # 同一根因不重复报第二条 fail
    assert checks["deps"].status == doctor.SKIP
    assert report.ok is False and report.exit_code == 1


def test_full_doctor_still_reports_deps_probe_failure_as_fail(env):
    """完整版没有"另一条检查已报根因"这回事，探测失败要自己报 FAIL。"""
    env["deps"] = _result(status="error", error={"message": "执行超时，超过 120 秒。"})
    check = _checks(doctor.run_checks(workspace=WORKSPACE))["deps"]

    assert check.status == doctor.FAIL
    assert "没跑起来" in check.detail


# === --fix-install ===

def test_fix_install_targets_the_pro_interpreter(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    recorded = {}

    class Done:
        returncode = 0
        stdout = "Successfully installed gis-task-cli-0.8.0"
        stderr = ""

    def fake_run(command, **kwargs):
        recorded["command"] = command
        return Done()

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    outcome = install.fix_install(root=tmp_path, python_executable=FAKE_PYTHON)

    assert outcome.ok is True
    assert recorded["command"] == [FAKE_PYTHON, "-m", "pip", "install", "-e", str(tmp_path)]
    assert "gis-task-cli" in outcome.output


def test_fix_install_refuses_non_source_directory(monkeypatch, tmp_path):
    outcome = install.fix_install(root=tmp_path, python_executable=FAKE_PYTHON)
    assert outcome.ok is False
    assert "pyproject.toml" in outcome.detail


def test_fix_install_reports_pip_failure(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "ERROR: Could not find a version that satisfies the requirement"

    monkeypatch.setattr(install.subprocess, "run", lambda *_a, **_k: Failed())
    outcome = install.fix_install(root=tmp_path, python_executable=FAKE_PYTHON)

    assert outcome.ok is False
    assert "退出码 1" in outcome.detail
    assert "Could not find a version" in outcome.output


# === CLI 接线 ===

def test_cli_json_exit_code_1_when_a_check_fails(monkeypatch, capsys):
    from gis_cli.agent import cli as cli_module

    report = doctor.DoctorReport(
        checks=[doctor.CheckResult("deps", doctor.FAIL, "缺依赖 jupyter-client")]
    )
    monkeypatch.setattr("gis_cli.envcheck.run_checks", lambda **_kwargs: report)

    with pytest.raises(cli_module.typer.Exit) as excinfo:
        cli_module.doctor_cmd(as_json=True, deep=False, fix_install=False)

    assert excinfo.value.exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["summary"] == "缺依赖 jupyter-client"


def test_cli_exit_code_2_when_doctor_itself_crashes(monkeypatch):
    from gis_cli.agent import cli as cli_module

    def boom(**_kwargs):
        raise RuntimeError("自检代码有 bug")

    monkeypatch.setattr("gis_cli.envcheck.run_checks", boom)

    with pytest.raises(cli_module.typer.Exit) as excinfo:
        cli_module.doctor_cmd(as_json=True, deep=False, fix_install=False)

    # doctor 自己崩了要跟"环境有问题"区分开（1 vs 2），否则 CI 会误判
    assert excinfo.value.exit_code == 2


def test_startup_hook_is_silent_when_environment_is_fine(monkeypatch, capsys):
    from gis_cli.agent import cli as cli_module

    report = doctor.DoctorReport(
        checks=[doctor.CheckResult("pro_installed", doctor.PASS, "ok")]
    )
    monkeypatch.setattr("gis_cli.envcheck.run_startup_checks", lambda **_k: report)

    cli_module._startup_envcheck(None)

    assert capsys.readouterr().out == ""


def test_startup_hook_prints_the_report_only_when_something_fails(monkeypatch, capsys):
    from gis_cli.agent import cli as cli_module

    report = doctor.DoctorReport(
        checks=[
            doctor.CheckResult("pro_installed", doctor.FAIL, "未找到 ArcGIS Pro Python", "设 ARCGIS_PRO_PYTHON"),
            doctor.CheckResult("arcpy_importable", doctor.SKIP, "跳过"),
            doctor.CheckResult("deps", doctor.SKIP, "跳过"),
        ]
    )
    monkeypatch.setattr("gis_cli.envcheck.run_startup_checks", lambda **_k: report)

    cli_module._startup_envcheck(None)

    out = capsys.readouterr().out
    assert "环境自检" in out
    assert "未找到 ArcGIS Pro Python" in out
    assert "doctor --fix-install" in out


def test_startup_hook_never_blocks_startup(monkeypatch):
    from gis_cli.agent import cli as cli_module

    def boom(**_kwargs):
        raise RuntimeError("自检自己炸了")

    monkeypatch.setattr("gis_cli.envcheck.run_startup_checks", boom)

    cli_module._startup_envcheck(None)  # 不抛异常就是通过
