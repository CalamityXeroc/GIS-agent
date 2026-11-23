# -*- coding: utf-8 -*-
"""环境发现链（``arcpy_bridge``）离线单测。

为什么要离线：发现链的价值恰恰是「注册表坏了/Pro 装在别处」时还能找对，
这些分支在真机上很难复现（本机 3.7 就是注册表指向已删的 ``E:\\ArcGISPro3.6``）。
所以注册表用假 ``winreg``、目录用 ``tmp_path``、``propy.bat`` 探活用猴子补丁，
全部不启子进程、不依赖 arcpy —— 保证能在 CI（``pytest -m "not arcpy"``）里跑。

真机验证（本机 Pro 3.7）见 ``docs/arcgis-pro-3.7-adaptation.md`` 的 E1/E2/E4。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gis_cli import arcpy_bridge
from gis_cli.arcpy_bridge import ArcGISDiscoveryError, ArcPyExecutionResult

REGISTRY_KEY = r"SOFTWARE\ESRI\ArcGISPro"


class _FakeKey:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def __enter__(self) -> "_FakeKey":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


class _FakeWinreg:
    """只实现 ``arcpy_bridge`` 用到的那几个入口。"""

    HKEY_LOCAL_MACHINE = "HKLM"
    KEY_READ = 1
    KEY_WOW64_64KEY = 2
    KEY_WOW64_32KEY = 4

    def __init__(self, table: dict[str, dict[str, str]]) -> None:
        self.table = table

    def OpenKey(self, _hive, path, _reserved=0, _access=0):  # noqa: N802 - 模拟 winreg
        if path not in self.table:
            raise OSError(f"找不到注册表键: {path}")
        return _FakeKey(self.table[path])

    def QueryValueEx(self, key, name):  # noqa: N802 - 模拟 winreg
        if name not in key.values:
            raise OSError(f"找不到注册表值: {name}")
        return key.values[name], 1


def _norm(path: str | Path) -> str:
    return arcpy_bridge._normalize_path(path)


def _make_pro_install(root: Path, env: str = "arcgispro-py3") -> Path:
    """造一个「看起来是真的」Pro 安装：解释器 + arcpy 包本体（判据文件）。"""
    python = root / "bin" / "Python" / "envs" / env / "python.exe"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    marker = root / "Resources" / "ArcPy" / "arcpy" / "__init__.py"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    return python


def _install_registry(monkeypatch, **values: str) -> None:
    monkeypatch.setattr(arcpy_bridge, "winreg", _FakeWinreg({REGISTRY_KEY: values}))


@pytest.fixture(autouse=True)
def _isolated_discovery(monkeypatch, tmp_path):
    """每个用例都从干净的现场开始：无环境变量、无注册表、只扫 tmp_path。"""
    for name in ("ARCGIS_PRO_PYTHON", "ARCGIS_PRO_INSTALL_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(arcpy_bridge, "winreg", None)
    monkeypatch.setattr(arcpy_bridge, "ARCGIS_PRO_SCAN_ROOTS", (tmp_path,))
    monkeypatch.setattr(arcpy_bridge, "ARCGIS_PRO_COMPAT_DIRS", ())
    arcpy_bridge.clear_discovery_cache()
    yield
    arcpy_bridge.clear_discovery_cache()


# === ① 注册表 InstallDir 有效 ===

def test_registry_install_dir_is_used(monkeypatch, tmp_path):
    install = tmp_path / "ArcGISPro3.7"
    python = _make_pro_install(install)
    _install_registry(
        monkeypatch,
        InstallDir=str(install) + "\\",
        PythonCondaRoot=str(install / "bin" / "Python"),
        PythonCondaEnv="arcgispro-py3",
        Version="3.7.0.1901",
    )

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(python)
    assert info.install_dir == _norm(install)
    # source 要说清来自哪一级：注册表 + 具体哪个值
    assert info.source == f"registry:{REGISTRY_KEY}:InstallDir"


# === ② InstallDir 坏了，但 PythonCondaRoot 是好的 ===

def test_broken_install_dir_falls_back_to_python_conda_root(monkeypatch, tmp_path):
    """本机 3.7 的真实故障形状：InstallDir 指向已删的旧版本。"""
    gone = tmp_path / "ArcGISPro3.6"
    install = tmp_path / "ArcGISPro3.7"
    python = _make_pro_install(install)
    _install_registry(
        monkeypatch,
        InstallDir=str(gone) + "\\",
        PythonCondaRoot=str(install / "bin" / "Python"),
        PythonCondaEnv="arcgispro-py3",
        Version="3.7.0.1901",
    )

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(python)
    assert info.install_dir == _norm(install)
    assert info.source == f"registry:{REGISTRY_KEY}:PythonCondaRoot"


# === ③ 注册表全坏，靠目录探测命中 ===

def test_directory_scan_hits_when_registry_is_wrong(monkeypatch, tmp_path):
    install = tmp_path / "ArcGISPro3.7"
    python = _make_pro_install(install)
    _install_registry(
        monkeypatch,
        InstallDir=str(tmp_path / "ArcGISPro3.6") + "\\",
        PythonCondaRoot=str(tmp_path / "ArcGISPro3.6" / "bin" / "Python"),
        PythonCondaEnv="arcgispro-py3",
        Version="3.7.0.1901",
    )

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(python)
    assert info.source == f"filesystem:scan:{_norm(install)}"


# === ④ 全坏 → 报错并把试过的路说清 ===

def test_all_routes_broken_raises_with_the_tried_routes(monkeypatch, tmp_path):
    _install_registry(monkeypatch, InstallDir=str(tmp_path / "ArcGISPro3.6") + "\\")

    with pytest.raises(ArcGISDiscoveryError) as excinfo:
        arcpy_bridge.discover_arcgis_pro_python()

    message = str(excinfo.value)
    assert "已试过" in message
    assert "注册表" in message
    assert "目录探测" in message
    assert "ARCGIS_PRO_PYTHON" in message
    assert "propy.bat" in message


# === PythonCondaEnv：克隆环境是 Pro 的常规操作，环境名不能写死 ===

def test_cloned_conda_env_from_registry_is_found(monkeypatch, tmp_path):
    install = tmp_path / "ArcGISPro3.7"
    python = _make_pro_install(install, env="arcgispro-py3-clone")
    _install_registry(
        monkeypatch,
        InstallDir=str(install),
        PythonCondaRoot=str(install / "bin" / "Python"),
        PythonCondaEnv="arcgispro-py3-clone",
        Version="3.7.0.1901",
    )

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(python)


def test_clone_is_found_even_when_registry_env_name_is_stale(monkeypatch, tmp_path):
    """注册表还说 ``arcgispro-py3``，但机器上只剩克隆环境 —— 枚举 envs/ 兜住。"""
    install = tmp_path / "ArcGISPro3.7"
    python = _make_pro_install(install, env="arcgispro-py3-clone")
    _install_registry(
        monkeypatch,
        InstallDir=str(install),
        PythonCondaEnv="arcgispro-py3",
        Version="3.7.0.1901",
    )

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(python)


# === 多版本并存 / 伪候选 ===

def test_multiple_versions_pick_the_highest(tmp_path):
    _make_pro_install(tmp_path / "ArcGISPro3.6")
    newest = _make_pro_install(tmp_path / "ArcGISPro3.7")

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(newest)


def test_stale_dir_without_arcpy_loses_to_a_real_install(tmp_path):
    """残留目录里有个 python.exe（没有 ``Resources\\ArcPy``）不能赢过真 Pro。

    目录名故意让残留在版本序里排前面：验证的是判据文件，不是版本排序。
    """
    stale = tmp_path / "ArcGISPro3.8"
    stale_python = stale / "bin" / "Python" / "envs" / "arcgispro-py3" / "python.exe"
    stale_python.parent.mkdir(parents=True)
    stale_python.write_text("", encoding="utf-8")
    real = _make_pro_install(tmp_path / "ArcGISPro3.7")

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(real)


def test_candidates_never_contain_the_launcher(tmp_path):
    _make_pro_install(tmp_path / "ArcGISPro3.7")

    candidates = arcpy_bridge._build_python_candidates()

    assert candidates
    assert all(path.lower().endswith(".exe") for _source, path, _install in candidates)


# === ARCGIS_PRO_PYTHON：硬覆盖永远第一 ===

def test_explicit_env_python_beats_a_real_install(monkeypatch, tmp_path):
    _make_pro_install(tmp_path / "ArcGISPro3.7")
    custom = tmp_path / "custom" / "python.exe"
    custom.parent.mkdir()
    custom.write_text("", encoding="utf-8")
    monkeypatch.setenv("ARCGIS_PRO_PYTHON", str(custom))

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(custom)
    assert info.source == "env:ARCGIS_PRO_PYTHON"
    # 非标准布局：安装目录退化成解释器所在目录，而不是瞎拼 bin\Python\envs\...
    assert info.install_dir == _norm(custom.parent)


def _make_bare_launcher(root: Path) -> Path:
    """一个不在 Pro 标准布局里的启动器（目录名故意不命中目录探测）。"""
    launcher = root / "custom" / "propy.bat"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("@echo off\n", encoding="utf-8")
    return launcher


def test_env_hint_bat_is_rejected_when_it_cannot_import_arcpy(monkeypatch, tmp_path):
    """``ARCGIS_PRO_PYTHON`` 指到 ``.bat`` 不能绕过探活（DAIR-283 非阻塞项）。"""
    launcher = _make_bare_launcher(tmp_path)
    probed: list[str] = []

    def fake_run(_code, **kwargs):
        probed.append(kwargs["python_executable"])
        return _probe_result("error", stderr="No environment named ... exists")

    monkeypatch.setenv("ARCGIS_PRO_PYTHON", str(launcher))
    monkeypatch.setattr(arcpy_bridge, "run_arcpy_code", fake_run)

    with pytest.raises(ArcGISDiscoveryError):
        arcpy_bridge.discover_arcgis_pro_python()
    # 必须真的拿它去探活，而不是原样当解释器返回
    assert probed == [_norm(launcher)]


def test_env_hint_bat_is_probed_before_being_accepted(monkeypatch, tmp_path):
    launcher = _make_bare_launcher(tmp_path)
    install = tmp_path / "ArcGISPro3.7"
    probed: list[str] = []

    def fake_run(_code, **kwargs):
        probed.append(kwargs["python_executable"])
        return _probe_result(
            "success", data={"install_dir": str(install), "version": "3.7.0.1901"}
        )

    monkeypatch.setenv("ARCGIS_PRO_PYTHON", str(launcher))
    monkeypatch.setattr(arcpy_bridge, "run_arcpy_code", fake_run)

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert probed == [_norm(launcher)]
    assert info.python_executable == _norm(launcher)
    assert info.install_dir == _norm(install)
    # 显式指定仍是硬覆盖：优先级不变，只是多了一道探活
    assert info.source == "env:ARCGIS_PRO_PYTHON"


def test_rejected_env_hint_bat_falls_back_to_a_real_install(monkeypatch, tmp_path):
    """探活不过就丢弃这个候选，不能因为它坏了就整条链报错。"""
    install = tmp_path / "ArcGISPro3.7"
    python = _make_pro_install(install)
    launcher = _make_bare_launcher(tmp_path)

    monkeypatch.setenv("ARCGIS_PRO_PYTHON", str(launcher))
    monkeypatch.setattr(
        arcpy_bridge, "run_arcpy_code", lambda _code, **_kw: _probe_result("error")
    )

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(python)
    # 候选表（doctor 用）里也不该出现这个 .bat
    assert all(
        not path.lower().endswith(".bat")
        for _source, path, _install in arcpy_bridge._build_python_candidates()
    )


# === 注册表线索排序：路径名里的版本优先于 Version 值 ===

def test_registry_hints_sort_by_install_path_version(monkeypatch, tmp_path):
    """``Version=3.7`` + ``InstallDir`` 指向 3.6 残留：坏路径不能排到真 3.7 前面。"""
    install = tmp_path / "ArcGISPro3.7"
    _make_pro_install(install)
    _install_registry(
        monkeypatch,
        InstallDir=str(tmp_path / "ArcGISPro3.6") + "\\",
        PythonCondaRoot=str(install / "bin" / "Python"),
        PythonCondaEnv="arcgispro-py3",
        Version="3.7.0.1901",
    )

    sources = [
        source for source, _path, _install in arcpy_bridge._build_python_candidates()
    ]

    assert sources.index(f"registry:{REGISTRY_KEY}:PythonCondaRoot") < sources.index(
        f"registry:{REGISTRY_KEY}:InstallDir"
    )


# === propy.bat：最后兜底，且必须探活 ===

def _make_launcher_only(root: Path) -> Path:
    launcher = root / "bin" / "Python" / "scripts" / "propy.bat"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("@echo off\n", encoding="utf-8")
    return launcher


def _probe_result(status: str, data=None, stderr: str = "") -> ArcPyExecutionResult:
    return ArcPyExecutionResult(
        status=status,
        exit_code=0 if status == "success" else 1,
        python_executable="propy.bat",
        stdout="",
        stderr=stderr,
        data=data,
        error=None if status == "success" else {"type": "ModuleNotFoundError"},
    )


def test_propy_launcher_is_rejected_when_it_cannot_import_arcpy(monkeypatch, tmp_path):
    """实测 3.7 的 propy 会落到系统 Python：那里没有 arcpy，绝不能算命中。"""
    install = tmp_path / "ArcGISPro3.7"
    _make_launcher_only(install)
    probed: list[str] = []

    def fake_run(_code, **kwargs):
        probed.append(kwargs["python_executable"])
        return _probe_result("error", stderr="No environment named ... exists")

    monkeypatch.setattr(arcpy_bridge, "run_arcpy_code", fake_run)

    with pytest.raises(ArcGISDiscoveryError):
        arcpy_bridge.discover_arcgis_pro_python()
    assert probed and probed[0].endswith("propy.bat")


def test_propy_launcher_is_accepted_after_a_successful_probe(monkeypatch, tmp_path):
    install = tmp_path / "ArcGISPro3.7"
    launcher = _make_launcher_only(install)

    monkeypatch.setattr(
        arcpy_bridge,
        "run_arcpy_code",
        lambda _code, **_kwargs: _probe_result(
            "success", data={"install_dir": str(install), "version": "3.7.0.1901"}
        ),
    )

    info = arcpy_bridge.discover_arcgis_pro_python()

    assert info.python_executable == _norm(launcher)
    assert info.install_dir == _norm(install)
    assert info.source.startswith("launcher:propy.bat")


# === 小工具 ===

@pytest.mark.parametrize(
    "text,expected",
    [
        ("3.7.0.1901", (3, 7, 0, 1901)),
        ("ArcGISPro3.7", (3, 7)),
        ("Pro", ()),
        ("", ()),
    ],
)
def test_version_key(text, expected):
    assert arcpy_bridge._version_key(text) == expected


@pytest.mark.skipif(sys.platform != "win32", reason="路径布局是 Windows 的")
@pytest.mark.parametrize(
    "python_path,expected",
    [
        (r"E:\ArcGISPro3.7\bin\Python\envs\arcgispro-py3\python.exe", r"E:\ArcGISPro3.7"),
        (r"E:\ArcGISPro3.7\bin\Python\envs\arcgispro-py3-clone\python.exe", r"E:\ArcGISPro3.7"),
        (r"E:\tools\py313\python.exe", r"E:\tools\py313"),
    ],
)
def test_guess_install_dir_is_env_name_agnostic(python_path, expected):
    assert Path(arcpy_bridge._guess_install_dir_from_python(python_path)) == Path(expected)
