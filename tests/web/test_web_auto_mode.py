# -*- coding: utf-8 -*-
"""免确认直执行模式（DAIR-344 需求 2）：web 端斜杠指令 + 默认关 + 隐藏计划卡 UI。

三层契约，各用最贴近真实行为的方式验：

1. **指令解析 / 默认值 / 持久化**：`web/src/lib/slash-commands.ts` 是纯函数模块，
   用仓库里的 tsc 编成 ESM 后在 node 里跑真函数（同 `test_scale.py`），
   不是把解析逻辑在 Python 里抄一遍。
2. **UI 隐藏**：`new-run-form.tsx` 里「计划卡确认」整块必须在免确认模式下不渲染
   （不是 opacity/隐藏），提交时带 `plan_gate=false` —— 源码 guard，防回退。
3. **计划卡只在 `awaiting_plan` 弹**：`run-detail.tsx` 的两个入口都挂在
   `awaiting_plan` 上；`plan_gate=False` 的 run 从不进这个状态，
   由既有后端用例 `test_create_run_options.py::test_plan_gate_off_starts_without_waiting`
   守着（本文件不重写一份假的）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "web"
SLASH_TS = WEB / "src" / "lib" / "slash-commands.ts"
NEW_RUN_FORM = WEB / "src" / "components" / "runs" / "new-run-form.tsx"
RUN_DETAIL = WEB / "src" / "routes" / "run-detail.tsx"
TS_COMPILER = WEB / "node_modules" / "typescript" / "bin" / "tsc"

#: 与 `slash-commands.ts` 的契约对齐（不是从被测代码里读，写死才是契约）。
STORAGE_KEY = "gis-agent-web-auto-mode"

HARNESS = """
import {
  AUTO_MODE_DEFAULT,
  AUTO_MODE_STORAGE_KEY,
  SLASH_COMMANDS,
  findCommand,
  helpText,
  matchSlashCommands,
  parseSlashCommand,
  readAutoMode,
  resolveComposerInput,
  writeAutoMode,
} from './slash-commands.js';

function fakeStorage(initial) {
  const data = new Map(Object.entries(initial ?? {}));
  return {
    getItem: (key) => (data.has(key) ? data.get(key) : null),
    setItem: (key, value) => { data.set(key, String(value)); },
  };
}

const none = fakeStorage();
const on = fakeStorage({ [AUTO_MODE_STORAGE_KEY]: 'on' });
const off = fakeStorage({ [AUTO_MODE_STORAGE_KEY]: 'off' });
const junk = fakeStorage({ [AUTO_MODE_STORAGE_KEY]: 'yes' });
const writable = fakeStorage();
writeAutoMode(true, writable);
const wroteOn = writable.getItem(AUTO_MODE_STORAGE_KEY);
writeAutoMode(false, writable);
const wroteOff = writable.getItem(AUTO_MODE_STORAGE_KEY);

const menuInputs = ['/', '/co', '/help', '/confirm ', '/zzz', 'plan text'];
const resolveInputs = [
  ['读 csv 做汇总', false, true],
  ['读 csv 做汇总', true, true],
  ['读 csv 做汇总', true, false],
  ['读 csv 做汇总', false, false],
  ['  /confirm on  ', false, true],
  ['/confirm off', true, true],
  ['/CONFIRM OFF', false, true],
  ['/confirm', true, true],
  ['/confirm maybe', true, true],
  ['/nope', false, true],
  ['/help', true, true],
  ['/', false, true],
  ['   ', false, true],
];

console.log(JSON.stringify({
  default: AUTO_MODE_DEFAULT,
  storageKey: AUTO_MODE_STORAGE_KEY,
  commands: SLASH_COMMANDS.map((command) => ({ name: command.name, usage: command.usage })),
  parsed: ['/confirm off', '/CONFIRM ON', '/help', '/', '/nope', 'plain task', '  /confirm  off  ']
    .map((text) => ({ text, parsed: parseSlashCommand(text) })),
  menu: menuInputs.map((text) => {
    const found = matchSlashCommands(text);
    return { text, isNull: found === null, names: (found ?? []).map((command) => command.name) };
  }),
  found: ['confirm', '/confirm', 'CONFIRM', 'nope', ''].map((name) => ({
    name,
    found: findCommand(name)?.name ?? null,
  })),
  read: {
    none: readAutoMode(none),
    on: readAutoMode(on),
    off: readAutoMode(off),
    junk: readAutoMode(junk),
    unavailable: readAutoMode(null),
  },
  write: { on: wroteOn, off: wroteOff },
  resolved: resolveInputs.map(([text, autoMode, planGate]) => ({
    text,
    autoMode,
    planGate,
    action: resolveComposerInput(text, autoMode, planGate),
  })),
  help: { on: helpText(true), off: helpText(false) },
}));
"""


def _node() -> str | None:
    return shutil.which("node")


def _run(cmd: list[str], *, cwd: Path, timeout: int) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    assert proc.returncode == 0, f"{cmd} 失败（rc={proc.returncode}）：{proc.stdout}{proc.stderr}"
    return proc.stdout


@pytest.fixture(scope="module")
def contract(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """编译真 `slash-commands.ts` 并在 node 里跑一遍，返回它的输出。"""
    node = _node()
    if not node:
        pytest.skip("没有 node：跳过斜杠指令用例")
    if not TS_COMPILER.is_file():
        pytest.skip("缺少 typescript：先 cd web && npm ci")
    assert SLASH_TS.is_file(), f"缺少 {SLASH_TS}"

    tmp_path = tmp_path_factory.mktemp("web_auto_mode")
    _run(
        [
            node,
            str(TS_COMPILER),
            "src/lib/slash-commands.ts",
            "--ignoreConfig",
            "--outDir",
            str(tmp_path),
            "--target",
            "es2022",
            "--module",
            "esnext",
            "--skipLibCheck",
        ],
        cwd=WEB,
        timeout=240,
    )
    (tmp_path / "probe.mjs").write_text(HARNESS, encoding="utf-8")
    payload = json.loads(_run([node, str(tmp_path / "probe.mjs")], cwd=tmp_path, timeout=120))
    return payload


def test_auto_mode_defaults_off_and_persists(contract: dict) -> None:
    """默认关（→ 仍弹计划卡）；只有明确存了 on/off 才认，坏值回默认关。"""
    assert contract["default"] is False, "免确认直执行的默认值必须是关"
    assert contract["storageKey"] == STORAGE_KEY

    read = contract["read"]
    assert read["none"] is False, "没存过（首次打开页面）必须是关"
    assert read["on"] is True
    assert read["off"] is False
    assert read["junk"] is False, "localStorage 里是坏值时必须回默认关，不能静默开"
    assert read["unavailable"] is False, "存储不可用（隐私模式）时回默认关"

    write = contract["write"]
    assert write == {"on": "on", "off": "off"}, "写出去的值要能被 readAutoMode 读回来"


def test_slash_command_parsing(contract: dict) -> None:
    parsed = {row["text"]: row["parsed"] for row in contract["parsed"]}

    assert parsed["/confirm off"] == {"name": "confirm", "arg": "off", "rest": "off"}
    assert parsed["/CONFIRM ON"]["name"] == "confirm", "命令名大小写不敏感"
    assert parsed["/CONFIRM ON"]["arg"] == "on", "参数大小写不敏感"
    assert parsed["  /confirm  off  "]["arg"] == "off", "前后多余空白要吃掉"
    assert parsed["/help"] == {"name": "help", "arg": "", "rest": ""}
    assert parsed["/"]["name"] == "", "只敲一个 / 是菜单态，不是指令"
    assert parsed["plain task"] is None, "不是斜杠开头就不是指令"
    assert parsed["/nope"]["name"] == "nope", "解析不管认不认识，查表才知道"

    found = {row["name"]: row["found"] for row in contract["found"]}
    assert found["confirm"] == "confirm"
    assert found["/confirm"] == "confirm", "带斜杠也能查到"
    assert found["CONFIRM"] == "confirm"
    assert found["nope"] is None
    assert found[""] is None


def test_slash_candidates_menu(contract: dict) -> None:
    menu = {row["text"]: row for row in contract["menu"]}

    assert menu["/"]["names"] == ["confirm", "help"], "敲 / 要列出全部指令"
    assert menu["/co"]["names"] == ["confirm"], "前缀过滤"
    assert menu["/help"]["names"] == ["help"]
    assert menu["/zzz"] == {"text": "/zzz", "isNull": False, "names": []}, "无匹配要给空列表（界面显示提示）"
    assert menu["/confirm "]["isNull"] is True, "敲了空格说明在填参数，候选该收起"
    assert menu["plan text"]["isNull"] is True, "普通任务不弹候选"

    commands = {row["name"]: row["usage"] for row in contract["commands"]}
    assert commands["confirm"] == "/confirm <on|off>", "帮助文案里的用法必须与解析一致"
    assert "help" in commands


def test_resolve_composer_input(contract: dict) -> None:
    rows = {(row["text"], row["autoMode"], row["planGate"]): row["action"] for row in contract["resolved"]}

    # 默认关：提交照样带 plan_gate=true（弹计划卡）
    assert rows[("读 csv 做汇总", False, True)] == {
        "kind": "submit",
        "goal": "读 csv 做汇总",
        "planGate": True,
    }
    # 开着：无论那个已隐藏的每-run 开关是什么，都必须是 false
    assert rows[("读 csv 做汇总", True, True)]["planGate"] is False
    assert rows[("读 csv 做汇总", True, False)]["planGate"] is False
    # 关着 + 手动关掉开关：与改造前一致
    assert rows[("读 csv 做汇总", False, False)]["planGate"] is False

    assert rows[("  /confirm on  ", False, True)] == {"kind": "auto", "enabled": True}
    assert rows[("/confirm off", True, True)] == {"kind": "auto", "enabled": False}
    assert rows[("/CONFIRM OFF", False, True)] == {"kind": "auto", "enabled": False}

    for text in ("/confirm", "/confirm maybe"):
        action = rows[(text, True, True)]
        assert action["kind"] == "error", f"{text} 缺参数/参数非法应报用法"
        assert "/confirm <on|off>" in action["message"]
        assert "已开启" in action["message"], "报用法时带上当前状态"

    unknown = rows[("/nope", False, True)]
    assert unknown["kind"] == "error" and "未知指令 /nope" in unknown["message"]
    assert "/confirm <on|off>" in unknown["message"], "未知指令要顺手列可用指令"

    assert rows[("/help", True, True)] == {"kind": "help"}
    assert rows[("/", False, True)] == {"kind": "idle"}
    assert rows[("   ", False, True)] == {"kind": "idle"}

    # 关键契约：任何斜杠输入都不能变成一次 run
    for (text, _auto, _gate), action in rows.items():
        if text.strip().startswith("/"):
            assert action["kind"] != "submit", f"{text} 被当成任务目标提交了"


def test_help_text_lists_command_and_mode(contract: dict) -> None:
    on, off = contract["help"]["on"], contract["help"]["off"]
    for text in (on, off):
        assert "/confirm <on|off>" in text
        assert "/help" in text
    assert "已开启" in on and "已关闭" in off, "帮助里要能看出当前模式"
    assert on != off


def test_new_run_form_hides_plan_gate_when_auto_mode_on() -> None:
    """免确认模式下「计划卡确认」整块不渲染（不是藏起来留占位），并提交 plan_gate=false。"""
    source = NEW_RUN_FORM.read_text(encoding="utf-8")

    assert "useState(() => readAutoMode())" in source, "模式状态要来自 localStorage（刷新后保持）"
    assert "{autoMode ? null : (" in source, "计划卡确认区块必须整块条件渲染"

    cond = source.index("{autoMode ? null : (")
    label = source.index(">计划卡确认</div>")
    assert cond < label < cond + 900, "「计划卡确认」要在 autoMode 为真时不渲染的那个分支里"
    assert "<Switch" in source[label : label + 600], "每-run 开关也要一起藏（模式关着时恢复原样）"

    assert "plan_gate: next.planGate" in source, "提交时必须用解析出来的 plan_gate"
    assert "resolveComposerInput(" in source
    # 指令不能漏进 run 目标：旧代码是直接把 goal 原样提交
    assert "goal: next.goal" in source


def test_plan_dialog_only_opens_on_awaiting_plan() -> None:
    """计划卡的两个入口都挂在 awaiting_plan 上；plan_gate=false 的 run 不会进这个状态。"""
    source = RUN_DETAIL.read_text(encoding="utf-8")

    assert "run?.status !== 'awaiting_plan' || !planSignature" in source, (
        "自动弹卡的条件变了：免确认 run 不该有第二个入口"
    )
    assert "run.plan && run.status === 'awaiting_plan' ?" in source, (
        "「查看计划卡」按钮也只在 awaiting_plan 上出现"
    )
