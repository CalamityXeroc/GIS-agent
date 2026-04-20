from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import streamlit as st

try:
    from ..orchestrator import TaskOrchestrator
except ImportError:
    # Allow launching this file directly: python src/gis_cli/web/streamlit_app.py
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from gis_cli.orchestrator import TaskOrchestrator


def _in_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _inject_theme() -> None:
        st.markdown(
                """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');

:root {
    --bg-1: #07131f;
    --bg-2: #0b2233;
    --bg-3: #11384a;
    --line: rgba(80, 220, 186, 0.16);
    --text-main: #e6f5ff;
    --text-sub: #99b3c7;
    --accent: #27d3a2;
    --accent-2: #4cc9f0;
    --card: rgba(9, 28, 41, 0.68);
    --card-border: rgba(76, 201, 240, 0.28);
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 20%, rgba(39, 211, 162, 0.16), transparent 35%),
        radial-gradient(circle at 88% 10%, rgba(76, 201, 240, 0.22), transparent 34%),
        linear-gradient(130deg, var(--bg-1), var(--bg-2) 48%, var(--bg-3));
    color: var(--text-main);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(var(--line) 1px, transparent 1px),
        linear-gradient(90deg, var(--line) 1px, transparent 1px);
    background-size: 42px 42px;
    opacity: 0.35;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(5, 18, 27, 0.88), rgba(8, 29, 43, 0.88));
    border-right: 1px solid rgba(76, 201, 240, 0.2);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.3px;
}

.hero {
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    background: linear-gradient(135deg, rgba(10, 33, 48, 0.78), rgba(7, 25, 38, 0.64));
    box-shadow: 0 12px 30px rgba(3, 12, 18, 0.35);
    margin-bottom: 1rem;
}

.hero-title {
    margin: 0;
    font-size: 1.45rem;
    color: #d9f7ff;
}

.hero-sub {
    margin: 0.35rem 0 0 0;
    color: var(--text-sub);
    font-size: 0.96rem;
}

.hint-card {
    border: 1px solid rgba(39, 211, 162, 0.35);
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    background: rgba(7, 36, 34, 0.55);
    color: #d7fff2;
    margin-bottom: 0.9rem;
}

.step-card {
    border: 1px solid rgba(76, 201, 240, 0.3);
    border-radius: 14px;
    padding: 0.75rem 0.85rem;
    background: var(--card);
    margin-bottom: 0.65rem;
}

.metric-row {
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 0.35rem;
    background: rgba(8, 29, 43, 0.45);
    margin-top: 0.4rem;
}

[data-testid="stMetric"] {
    background: rgba(6, 22, 32, 0.72);
    border: 1px solid rgba(76, 201, 240, 0.2);
    border-radius: 12px;
    padding: 0.5rem 0.65rem;
}

[data-testid="stFileUploader"] {
    background: rgba(7, 25, 38, 0.56);
    border: 1px dashed rgba(76, 201, 240, 0.45);
    border-radius: 14px;
    padding: 0.35rem;
}
</style>
""",
                unsafe_allow_html=True,
        )


def _render_quick_start_tips() -> None:
        st.markdown(
                """
<div class="hint-card">
<b>新手提示</b><br/>
1. 先用“预览模式（dry-run）”验证流程，确认无误再正式执行。<br/>
2. 描述需求时尽量写清楚：<b>数据来源 + 处理动作 + 输出格式</b>。<br/>
3. 执行失败时不用手动排障，系统会按恢复策略自动重试。
</div>
""",
                unsafe_allow_html=True,
        )


def _page_run() -> None:
    st.markdown(
        """
<div class="hero">
  <p class="hero-title">GIS 智能执行工作台</p>
  <p class="hero-sub">自然语言输入需求，系统自动解析、规划、执行并生成产物。</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("自然语言执行")
    _render_quick_start_tips()

    arcpy_ok = bool(importlib.util.find_spec("arcpy"))
    if arcpy_ok:
        st.success("ArcPy 环境可用：将尝试输出真实 GIS 数据（GDB/地图导出）。")
    else:
        st.warning("ArcPy 环境不可用：当前仅会输出规划/报告类文件（JSON/TXT）。请切换到 ArcGIS Pro Python 环境。")

    prompt = st.text_area(
        "请输入任务需求",
        value="我有广西行政区图层和道路图层，希望自动制图并导出专题图 PDF。",
        height=130,
        help="示例：整合多个分幅数据并统一投影后导出专题图。",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        dry_run = st.toggle("预览模式", value=True)
    with c2:
        max_recovery_rounds = st.number_input("最大恢复轮数", 0, 5, 2)
    with c3:
        include_skipped = st.toggle("包含跳过节点", value=True)
        auto_fix = st.checkbox("自动应用恢复策略", value=True)

    st.markdown(
        """
<div class="step-card">
    <b>推荐描述模板：</b>我有【数据/图层】，希望【处理动作】，输出【格式/目标成果】。
</div>
""",
        unsafe_allow_html=True,
    )

    upload = st.file_uploader(
        "可选：上传任务说明文件（txt/md）",
        type=["txt", "md"],
        accept_multiple_files=False,
        help="上传后将优先按文档方式创建任务。",
    )

    if st.button("开始执行", type="primary", use_container_width=True):
        orch = TaskOrchestrator()
        with st.spinner("正在规划并执行..."):
            if upload is not None:
                tmp_dir = Path(".gis-cli") / "uploads"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / upload.name
                tmp_path.write_bytes(upload.getvalue())
                record = orch.create_task_from_document_file(str(tmp_path))
                result = orch.run_agent_for_task(
                    task_id=record.task_id,
                    dry_run=dry_run,
                    max_recovery_rounds=max_recovery_rounds,
                    include_skipped=include_skipped,
                    auto_apply_strategy=auto_fix,
                )
            else:
                result = orch.run_agent(
                    prompt=prompt,
                    dry_run=dry_run,
                    max_recovery_rounds=max_recovery_rounds,
                    include_skipped=include_skipped,
                    auto_apply_strategy=auto_fix,
                )

        ok = bool(result.get("success", False))
        st.success("执行成功，已生成任务结果。") if ok else st.error("执行失败，请查看详细结果。")

        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("任务ID", str(result.get("task_id", "-")))
        m2.metric("状态", str(result.get("status", "-")))
        m3.metric("是否成功", "是" if ok else "否")
        m4.metric("尝试次数", int(result.get("attempts", 0)))
        st.markdown("</div>", unsafe_allow_html=True)

        final_result = result.get("final_result", {})
        outputs = final_result.get("outputs", [])
        if outputs:
            st.subheader("输出文件")
            for out in outputs:
                st.code(str(out))

        with st.expander("查看完整执行结果(JSON)", expanded=False):
            st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def _load_llm_rag_config() -> dict:
    cfg_path = Path("config") / "llm_rag_config.json"
    if not cfg_path.exists():
        return {}
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_llm_rag_config(payload: dict) -> None:
    cfg_path = Path("config") / "llm_rag_config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _provider_presets() -> dict[str, dict[str, str]]:
    return {
        "OpenAI 兼容(默认)": {
            "api_base": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        },
        "硅基流动(SiliconFlow)": {
            "api_base": "https://api.siliconflow.cn/v1",
            "model": "deepseek-ai/DeepSeek-V3",
        },
        "DeepSeek 官方": {
            "api_base": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        },
        "通义千问(DashScope兼容)": {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
        },
        "智谱(GLM)": {
            "api_base": "https://open.bigmodel.cn/api/paas/v4",
            "model": "glm-4-plus",
        },
        "Moonshot": {
            "api_base": "https://api.moonshot.cn/v1",
            "model": "moonshot-v1-8k",
        },
        "OpenRouter": {
            "api_base": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-4o-mini",
        },
        "自定义": {
            "api_base": "",
            "model": "",
        },
    }


def _get_rag_status_safe(orch: TaskOrchestrator) -> dict:
    if hasattr(orch, "get_planner_rag_status"):
        try:
            return orch.get_planner_rag_status()
        except Exception as exc:
            return {"enabled": False, "error": f"get_planner_rag_status_failed: {exc}"}

    planner = getattr(orch, "planner", None)
    if planner is not None and hasattr(planner, "get_llm_rag_status"):
        try:
            return planner.get_llm_rag_status()
        except Exception as exc:
            return {"enabled": False, "error": f"planner_get_llm_rag_status_failed: {exc}"}

    return {
        "enabled": False,
        "has_api_key": False,
        "dynamic_source_count": 0,
        "error": "orchestrator_missing_rag_status_api",
    }


def _reload_rag_config_safe(orch: TaskOrchestrator) -> dict:
    if hasattr(orch, "reload_planner_rag_config"):
        try:
            return orch.reload_planner_rag_config()
        except Exception as exc:
            return {"reloaded": False, "error": f"reload_planner_rag_config_failed: {exc}"}

    planner = getattr(orch, "planner", None)
    if planner is not None and hasattr(planner, "reload_llm_rag_config"):
        try:
            return planner.reload_llm_rag_config()
        except Exception as exc:
            return {"reloaded": False, "error": f"planner_reload_llm_rag_config_failed: {exc}"}

    return {
        "reloaded": False,
        "error": "orchestrator_missing_rag_reload_api",
    }


def _page_tasks() -> None:
    st.markdown(
        """
<div class="hero">
  <p class="hero-title">任务中心</p>
  <p class="hero-sub">查看任务列表、快速定位任务并检查执行摘要。</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.header("任务管理")
    orch = TaskOrchestrator()

    c1, c2 = st.columns(2)
    with c1:
        skip = st.number_input("skip", 0, 100000, 0)
    with c2:
        limit = st.number_input("limit", 1, 1000, 20)

    task_ids = orch.list_tasks()
    sliced = task_ids[int(skip) : int(skip) + int(limit)]
    st.caption(f"总任务数: {len(task_ids)}")
    st.write(sliced)

    task_id = st.text_input("查看任务摘要（输入 task_id）")
    if task_id:
        try:
            summary = orch.get_task_summary(task_id)
            st.json(summary)
        except FileNotFoundError:
            st.warning("未找到该任务")


def _page_strategy() -> None:
    st.markdown(
        """
<div class="hero">
  <p class="hero-title">恢复策略中心</p>
  <p class="hero-sub">根据任务类型切换 profile，并实时查看策略与统计。</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.header("策略信息")
    orch = TaskOrchestrator()

    profile = st.selectbox("选择 profile", ["default", "mapping", "integration", "quality"])
    st.json(orch.get_recovery_strategy_info(profile))

    if st.button("重载策略配置"):
        st.success("已重载")
        st.json(orch.reload_recovery_rules())

    st.subheader("策略统计")
    st.json(orch.get_strategy_statistics())

    st.markdown("---")
    st.header("LLM / RAG 模型配置")
    st.caption("可在前端直接配置 API Key、模型与平台，保存后自动热重载规划层。")

    rag_status = _get_rag_status_safe(orch)
    c1, c2, c3 = st.columns(3)
    c1.metric("RAG启用", "是" if rag_status.get("enabled") else "否")
    c2.metric("已配置API Key", "是" if rag_status.get("has_api_key") else "否")
    c3.metric("动态知识源", str(rag_status.get("dynamic_source_count", 0)))

    cfg = _load_llm_rag_config()
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm", {}), dict) else {}
    rag_cfg = cfg.get("rag", {}) if isinstance(cfg.get("rag", {}), dict) else {}

    presets = _provider_presets()
    preset_names = list(presets.keys())
    selected_preset = st.selectbox("平台预设", preset_names, index=0)

    p_api_base = presets[selected_preset].get("api_base", "")
    p_model = presets[selected_preset].get("model", "")

    with st.form("llm_rag_config_form"):
        enabled = st.toggle("启用 LLM + RAG 规划", value=bool(cfg.get("enabled", False)))
        api_base = st.text_input(
            "API Base",
            value=str(llm_cfg.get("api_base", "") or p_api_base),
            help="OpenAI 兼容接口地址，例如 https://api.siliconflow.cn/v1",
        )
        model = st.text_input(
            "模型名称",
            value=str(llm_cfg.get("model", "") or p_model),
            help="例如 deepseek-chat / qwen-plus / glm-4-plus",
        )
        api_key = st.text_input(
            "API Key",
            value=str(llm_cfg.get("api_key", "")),
            type="password",
            help="仅保存在本地 config/llm_rag_config.json",
        )

        st.markdown("**RAG 参数**")
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=int(rag_cfg.get("top_k", 3)))
        chunk_size = st.number_input("chunk_size", min_value=200, max_value=4000, value=int(rag_cfg.get("chunk_size", 1200)))
        auto_include_outputs = st.toggle("自动纳入历史输出文件作为知识源", value=bool(rag_cfg.get("auto_include_outputs", True)))

        knowledge_paths_text = st.text_area(
            "静态知识源路径（每行一个）",
            value="\n".join([str(x) for x in rag_cfg.get("knowledge_paths", [])])
            if isinstance(rag_cfg.get("knowledge_paths", []), list)
            else "README.md\nconfig/intents.json\nconfig/recovery_strategies.json",
            height=100,
        )

        submitted = st.form_submit_button("保存并重载", type="primary", use_container_width=True)
        if submitted:
            knowledge_paths = [x.strip() for x in knowledge_paths_text.splitlines() if x.strip()]
            new_cfg = {
                "enabled": bool(enabled),
                "llm": {
                    "model": str(model).strip(),
                    "api_key": str(api_key).strip(),
                    "api_base": str(api_base).strip(),
                },
                "rag": {
                    "knowledge_paths": knowledge_paths,
                    "top_k": int(top_k),
                    "chunk_size": int(chunk_size),
                    "auto_include_outputs": bool(auto_include_outputs),
                    "output_root": str(rag_cfg.get("output_root", "outputs")),
                    "max_output_files": int(rag_cfg.get("max_output_files", 30)),
                    "output_globs": rag_cfg.get(
                        "output_globs",
                        [
                            "**/checklist.json",
                            "**/arcpy_report.txt",
                            "**/quality_report.json",
                            "**/execution_report.txt",
                            "**/export_summary.txt",
                        ],
                    ),
                },
            }
            _save_llm_rag_config(new_cfg)
            status = _reload_rag_config_safe(orch)
            st.success("配置已保存并重载。")
            st.json(status)

    with st.expander("查看当前 LLM/RAG 状态（只读）", expanded=False):
        st.json(rag_status)


def main() -> None:
    st.set_page_config(page_title="GIS Agent 助手", page_icon="🗺️", layout="wide")
    _inject_theme()
    st.title("GIS Agent 可视化助手")
    st.caption("科技感地理工作台：自然语言输入 + 自动解析 + 自动执行 + 全程文字提示")

    page = st.sidebar.radio("功能", ["执行任务", "任务管理", "策略中心", "帮助"])
    if page == "执行任务":
        _page_run()
    elif page == "任务管理":
        _page_tasks()
    elif page == "策略中心":
        _page_strategy()
    else:
        st.markdown(
            """
<div class="hero">
  <p class="hero-title">帮助与提示</p>
  <p class="hero-sub">面向非技术用户的可执行说明。</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.header("帮助")
        st.markdown(
            """
### 适用人群
- 不熟悉 Agent / AI / 命令行 的 GIS 从业者。

### 推荐流程
1. 在“执行任务”里输入自然语言。
2. 勾选“预览模式”先验证。
3. 查看输出文件，再切换为实际执行。

### 常见示例
- 制图并导出广西专题图。
- 整合 9 个分幅数据并统一投影后输出 GDB。
- 批量检查几何问题并生成质量报告。

### 建议的任务描述模板
- 我有【数据/图层】，希望【处理动作】，最终输出【文件格式/地图类型】。
"""
        )


if __name__ == "__main__":
    # If user runs this file directly with Python, auto-relaunch in Streamlit mode.
    if _in_streamlit_runtime():
        main()
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
        raise SystemExit(subprocess.call(cmd))
