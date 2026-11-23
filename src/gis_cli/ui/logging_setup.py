# -*- coding: utf-8 -*-
"""引擎日志与终端的边界。

引擎内部用 ``logger.warning(...)`` 记录模型重试、修复失败等细节；如果没有配置
handler，Python 的"最后手段"处理器会把 WARNING 以上直接打到 stderr，混进对话输出
（实测：余额不足时刷出整段 JSON）。这里统一接管：

- 细节写文件 ``workspace/.gis_agent/logs/engine.log``（轮转由调用方决定，这里只追加）；
- 终端默认只显示 ERROR 以上，``--verbose`` 时显示 DEBUG；
- 同时给界面层留一个回调钩子（可选），方便将来把 WARNING 显示成一行提示。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

_HANDLER_FLAG = "_gis_agent_log_handler"

logger = logging.getLogger(__name__)


def setup_engine_logging(
    workspace: str | Path,
    *,
    console_level: int = logging.ERROR,
    verbose: bool = False,
    notify: Callable[[str], None] | None = None,
) -> Path:
    """把引擎日志接管到文件；返回日志文件路径。可重复调用（幂等）。"""
    ws = Path(workspace)
    log_dir = ws / ".gis_agent" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_dir = Path.cwd()

    log_path = log_dir / "engine.log"
    root = logging.getLogger()

    # 幂等：同进程内重复调用只调整级别
    existing = [h for h in root.handlers if getattr(h, _HANDLER_FLAG, False)]
    if existing:
        for handler in existing:
            handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        return log_path

    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except Exception:
        file_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    setattr(file_handler, _HANDLER_FLAG, True)
    root.addHandler(file_handler)

    if notify is not None:

        class _NotifyHandler(logging.Handler):
            """把 WARNING/ERROR 折成一行交给界面（可选）。"""

            def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - 依赖界面
                try:
                    notify(record.getMessage())
                except Exception:
                    pass

        notify_handler = _NotifyHandler()
        notify_handler.setLevel(logging.WARNING)
        setattr(notify_handler, _HANDLER_FLAG, True)
        root.addHandler(notify_handler)

    root.setLevel(logging.DEBUG)
    # 终端只留 ERROR（curated：真正的界面输出由 ui.render 负责）
    for name in ("", "gis_cli", "openai", "httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(console_level if name == "" else logging.INFO)
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.debug("engine logging -> %s", log_path)
    return log_path


def silence_console_logging() -> None:
    """确保没有任何 handler 往 stderr 打日志（老终端/重定向场景）。"""
    logging.getLogger().handlers = [h for h in logging.getLogger().handlers if getattr(h, _HANDLER_FLAG, False)]


__all__ = ["setup_engine_logging", "silence_console_logging"]
