# -*- coding: utf-8 -*-
"""giscli 入口包装层：``scripts\\giscli.cmd`` / ``install_shim.py`` / ``init`` / ``--new-window``。

这一层的价值全在「机器相关的边界」上，所以用例分两类：

* ``.cmd`` 只能静态检查（ASCII-only + CRLF + 关键调用）——真机跑法见
  DAIR-308 的验收记录，不进 CI；
* 其余全部用假的：注册表读写换成内存里的 dict（**绝不在测试里写真 PATH**），
  开窗只断言 argv（CI 上不能真的弹窗），目录一律落 ``tmp_path``。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gis_cli.agent import cli, launcher

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

runner = CliRunner()


def _load_script(name: str):
    """按路径加载 ``scripts/<name>.py``（scripts 不是包，不能直接 import）。"""
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_giscli_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclass 解析注解时要按 __module__ 去 sys.modules 找模块，先注册再执行
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# === .cmd 静态检查 =========================================================


@pytest.mark.parametrize(
    ("name", "where"),
    [
        ("giscli.cmd", SCRIPTS_DIR),
        ("find_pro_python.cmd", SCRIPTS_DIR),
        ("一键启动.bat", REPO_ROOT),
    ],
)
def test_batch_files_are_ascii_only_and_crlf(name: str, where: Path):
    """cmd.exe 按 ANSI 代码页解析批处理，非 ASCII / 裸 LF 都会炸。

    LF 那条是真踩过的：用 LF 存盘后，长 REM 注释被切碎成若干条命令，
    终端里满屏「'------' 不是内部或外部命令」。
    """
    path = where / name
    assert path.is_file(), f"缺少 {path}"
    raw = path.read_bytes()
    raw.decode("ascii")  # 非 ASCII 会在这里抛
    assert b"\r\n" in raw, "批处理必须用 CRLF 存盘"
    assert b"\n" not in raw.replace(b"\r\n", b""), "存在裸 LF 行尾"


def test_giscli_cmd_is_self_contained_and_installs_nothing():
    """入口只做三件事：找 Pro python、把 src 放上 PYTHONPATH、跑 -m。"""
    text = (SCRIPTS_DIR / "giscli.cmd").read_text(encoding="ascii")
    assert "find_pro_python.cmd" in text, "发现逻辑必须复用共享脚本"
    assert "PYTHONPATH=%REPO_ROOT%\\src" in text, "缺 PYTHONPATH 就依赖 editable 安装了"
    assert "-m gis_cli.agent.cli %*" in text
    assert "pip install" not in text, "入口不该碰 pip"
    # 仓库根必须由脚本位置推出，不能写死本机路径
    assert "E:\\BST_lab" not in text
    assert 'for %%I in ("%SCRIPT_DIR%\\..")' in text


def test_launcher_self_heals_missing_pyproj():
    """pyproj 缺了地图上什么都不画；launcher 的 :ensure 要能自己补上（DAIR-335）。"""
    bat = (REPO_ROOT / "一键启动.bat").read_text(encoding="ascii")
    assert 'call :ensure pyproj "pyproj>=3.6"' in bat


def test_launchers_share_one_discovery_and_never_pin_a_version():
    bat = (REPO_ROOT / "一键启动.bat").read_text(encoding="ascii")
    shared = (SCRIPTS_DIR / "find_pro_python.cmd").read_text(encoding="ascii")
    assert 'call "%REPO_ROOT%\\scripts\\find_pro_python.cmd"' in bat
    assert "PYTHONPATH=%REPO_ROOT%\\src" in bat, "不设 PYTHONPATH 就会掉回 pip 分支"
    for text in (bat, shared):
        assert "ARCGIS_PRO_PYTHON" in text
        # 3.6 硬编码是上一次升级踩过的坑，不能再出现
        assert "ArcGISPro3." not in text
        assert "E:\\ArcGISPro*" in text


# === init 子命令 ==========================================================


def test_init_creates_workspace_layout(tmp_path: Path):
    target = tmp_path / "带 空格 的目录" / "proj"
    result = runner.invoke(cli.app, ["init", str(target)])

    assert result.exit_code == 0, result.output
    for sub in ("input", "output", "temp"):
        assert (target / "workspace" / sub).is_dir(), f"缺 workspace/{sub}"
    # 下一步提示要告诉用户怎么开工
    assert "giscli chat" in result.output.replace("\n", "")


def test_init_is_idempotent_and_never_overwrites(tmp_path: Path):
    target = tmp_path / "proj"
    assert runner.invoke(cli.app, ["init", str(target)]).exit_code == 0

    marker = target / "workspace" / "input" / "已有数据.csv"
    marker.write_text("keep-me", encoding="utf-8")

    second = runner.invoke(cli.app, ["init", str(target)])
    assert second.exit_code == 0, second.output
    assert marker.read_text(encoding="utf-8") == "keep-me"


def test_init_fails_loudly_on_unusable_target(tmp_path: Path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    result = runner.invoke(cli.app, ["init", str(blocker)])
    assert result.exit_code == 1, "目标不可用时必须 rc≠0，不能静默成功"


# === --new-window =========================================================


def test_new_window_command_uses_wt_when_available(tmp_path: Path):
    cwd = tmp_path / "工作 目录"
    command = launcher.build_new_window_command(
        ["chat"], cwd=cwd, window_host=r"C:\WindowsApps\wt.exe"
    )
    # cwd 带空格，进 cmd 前就得包好引号（裸空格会把路径截成两段）
    assert command[:3] == [r"C:\WindowsApps\wt.exe", "-d", f'"{cwd}"']
    assert command[3:] == ["cmd", "/k", "giscli", "chat"]


def test_new_window_command_quotes_cmd_metacharacters(tmp_path: Path):
    """``a&whoami`` 落进 cmd 会被二次解析：``&`` 把命令截成两条（实测会执行 whoami）。"""
    command = launcher.build_new_window_command(
        ["chat", "--workspace", "a&whoami"], cwd=tmp_path, window_host=None
    )
    assert '"a&whoami"' in command, f"含 & 的参数必须整段加引号：{command}"
    assert "a&whoami" not in command, "裸 & 会被 cmd 当成分隔符"
    # 打印出来 «可手动执行» 的那一行也必须是同一份转义
    assert '"a&whoami"' in launcher.format_command(command)


def test_relaunch_hands_cmd_one_already_quoted_string(tmp_path: Path, monkeypatch):
    """``Popen(list)`` 会再过一道 list2cmdline，把手写的引号转成 ``\\"``。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "find_window_host", lambda *a, **k: None)
    seen: list[object] = []
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda cmd: seen.append(cmd))

    code = launcher.relaunch(["--new-window", "chat", "--workspace", "a&whoami"])

    assert code == 0
    assert isinstance(seen[0], str), "传 list 会被 list2cmdline 二次转义"
    assert '"a&whoami"' in seen[0]
    assert '\\"' not in seen[0], "cmd 不认 C 运行时的反斜杠引号转义"


def test_new_window_command_falls_back_to_start(tmp_path: Path):
    command = launcher.build_new_window_command(["chat"], cwd=tmp_path, window_host=None)
    assert command[:3] == ["cmd", "/c", "start"]
    assert command[3] == '""', "start 的第一个引号参数是窗口标题，丢了会把 /D 当标题"
    assert "/D" in command
    assert str(tmp_path) in command, "新窗口的 cwd 必须跟当前一致"
    assert "giscli" in command


def test_dry_run_prints_command_with_cwd_and_launcher(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "find_window_host", lambda *a, **k: None)

    code = launcher.relaunch(["--new-window", "--dry-run", "chat", "--workspace", "my ws"])

    printed = capsys.readouterr().out
    assert code == 0
    assert "giscli" in printed
    assert str(tmp_path) in printed
    assert "chat" in printed
    assert '"my ws"' in printed, "带空格的参数要加引号，否则粘到终端就断成两个"


def test_dry_run_without_new_window_is_left_to_the_subcommand():
    """``resume-run`` 自己也有 ``--dry-run``，不能被这一层吃掉。"""
    rest, new_window, dry_run = launcher.split_flags(["resume-run", "--dry-run"])
    assert rest == ["resume-run", "--dry-run"]
    assert not new_window and not dry_run


def test_main_routes_new_window_flag_before_typer(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["giscli", "--new-window", "--dry-run", "chat"])
    monkeypatch.setattr(launcher, "relaunch", lambda argv: seen.append(list(argv)) or 0)

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 0
    assert seen == [["--new-window", "--dry-run", "chat"]]


# === install_shim.py（用户级 PATH，绝不写真注册表） ========================


@pytest.fixture()
def shim(monkeypatch):
    """把注册表读写换成内存状态，其余（拼接/幂等/回滚）跑真实代码。"""
    module = _load_script("install_shim")
    state = {"value": r"C:\Windows\system32;C:\Tools", "writes": 0}

    def fake_read():
        return state["value"], module.DEFAULT_VALUE_TYPE

    def fake_write(value, value_type=module.DEFAULT_VALUE_TYPE):
        state["writes"] += 1
        state["value"] = value

    monkeypatch.setattr(module, "_read_user_path", fake_read)
    monkeypatch.setattr(module, "_write_user_path", fake_write)
    monkeypatch.setattr(module, "_broadcast_change", lambda: None)
    return module, state


def test_shim_install_is_idempotent(shim, tmp_path: Path):
    module, state = shim
    entry = tmp_path / "scripts"

    first = module.install(entry)
    assert first.ok and first.changed
    assert module.split_entries(state["value"])[-1] == str(entry)
    after_first = state["value"]

    second = module.install(entry)
    assert second.ok and not second.changed
    assert state["value"] == after_first, "第二次不该再写一遍"
    assert state["value"].count(str(entry)) == 1


def test_shim_install_treats_trailing_slash_as_same_entry(shim, tmp_path: Path):
    module, state = shim
    entry = tmp_path / "scripts"
    state["value"] = str(entry) + "\\"

    outcome = module.install(entry)
    assert not outcome.changed
    assert state["value"] == str(entry) + "\\"


def test_shim_remove_restores_the_original_value(shim, tmp_path: Path):
    module, state = shim
    original = state["value"]
    entry = tmp_path / "scripts"

    module.install(entry)
    removed = module.uninstall(entry)

    assert removed.ok and removed.changed
    assert state["value"] == original, "回滚必须还原成原值（多余分隔符也不许留）"
    # 再删一次是 no-op
    assert not module.uninstall(entry).changed


def test_shim_dry_run_never_writes(shim, tmp_path: Path):
    module, state = shim
    before = state["value"]
    outcome = module.install(tmp_path / "scripts", dry_run=True)
    assert outcome.ok
    assert state["writes"] == 0
    assert state["value"] == before


def test_shim_read_failure_never_writes_back(monkeypatch, tmp_path: Path, capsys):
    """读 PATH 失败 ≠ 空 PATH：当成空串写回会整条覆盖掉用户原有的 PATH。"""
    module = _load_script("install_shim")

    class DeniedWinreg:
        HKEY_CURRENT_USER = object()
        KEY_READ = 1

        def OpenKey(self, *args, **kwargs):
            raise PermissionError("access denied")

    writes: list[object] = []
    monkeypatch.setattr(module, "winreg", DeniedWinreg())
    monkeypatch.setattr(module, "_write_user_path", lambda *a, **k: writes.append(a))
    monkeypatch.setattr(module, "_broadcast_change", lambda: None)

    for action in (module.install, module.uninstall):
        outcome = action(tmp_path / "scripts")
        assert outcome.ok is False, "读不到旧值还报成功，等于默许覆盖写回"
    assert writes == [], "读失败时一次也不能写注册表"
    assert "读不到用户级 PATH" in capsys.readouterr().out
    assert module.main(["--scripts-dir", str(tmp_path / "scripts")]) == 1


# === check_install：src 已在 sys.path 上就不该再跑 pip =====================


def test_check_install_skips_pip_when_src_is_importable(tmp_path: Path, monkeypatch):
    module = _load_script("check_install")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setattr(module.sys, "path", [str(repo / "src"), *sys.path])

    need, reason = module.needs_install(repo)
    assert not need
    assert "sys.path" in reason


def test_check_install_warns_when_pyproj_missing_but_does_not_block(tmp_path: Path, monkeypatch, capsys):
    """缺 pyproj 只报警告：启动照走（退出码不变），但降级必须让人看见。"""
    module = _load_script("check_install")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setattr(module, "editable_pth_files", lambda: [])
    monkeypatch.setattr(module, "src_on_sys_path", lambda _src: False)
    monkeypatch.setattr(module, "missing_web_deps", lambda: ["pyproj"])

    assert module.main(["--repo", str(repo), "--quiet"]) == 1, "该装包时仍然返回 1"
    err = capsys.readouterr().err
    assert "pyproj" in err and "非 EPSG:4326" in err
    assert "pip install" in err


def test_check_install_ready_install_stays_ready_without_pyproj(tmp_path: Path, monkeypatch, capsys):
    """可选依赖缺失不能把「已就绪」变成「要装」，否则启动每次都白撞一次 pip。"""
    module = _load_script("check_install")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setattr(module, "editable_pth_files", lambda: [])
    monkeypatch.setattr(module, "src_on_sys_path", lambda _src: True)
    monkeypatch.setattr(module, "missing_web_deps", lambda: ["pyproj"])

    assert module.main(["--repo", str(repo), "--quiet"]) == 0
    assert "pyproj" in capsys.readouterr().err


def test_check_install_missing_web_deps_is_quiet_when_installed(monkeypatch):
    module = _load_script("check_install")
    monkeypatch.setattr(module, "missing_web_deps", lambda: [])

    assert module.report_web_deps() == []


def test_check_install_still_wants_install_when_src_is_nowhere(tmp_path: Path, monkeypatch):
    module = _load_script("check_install")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    monkeypatch.setattr(module, "editable_pth_files", lambda: [])
    monkeypatch.setattr(module, "src_on_sys_path", lambda _src: False)

    need, _reason = module.needs_install(repo)
    assert need, "既没装上也不在 sys.path 上，就该装"
