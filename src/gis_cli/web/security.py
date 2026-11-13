# -*- coding: utf-8 -*-
"""Web 控制台的安全原语：本机 token、路径白名单、密钥脱敏。

三条约定（见 ``docs/p2-console-plan.md`` §6）：

- 服务只绑 ``127.0.0.1``，启动时把 token 落到 ``<workspace>/.gis_agent/web_token``；
- 所有带路径的接口先 ``resolve()``，结果必须落在 workspace 内，否则 403；
- ``api_key`` 出站前脱敏成 ``sk-…后4位``。

这个模块刻意不 import 引擎/arcpy，保证无 ArcGIS 环境也能导入。
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import HTTPException

#: token 文件名（相对 ``<workspace>/.gis_agent``）。
TOKEN_FILE = "web_token"

#: 32 字节随机数的十六进制长度。
_TOKEN_BYTES = 32


def load_or_create_token(workspace: str | Path) -> str:
    """读取 workspace 的本机 token；不存在（或为空）就生成一个。

    幂等：同一次安装里重复启动服务拿到的是同一个 token，浏览器里存过的
    凭据不会因为重启失效。
    """
    state_dir = Path(workspace) / ".gis_agent"
    token_path = state_dir / TOKEN_FILE
    if token_path.exists():
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        if token:
            return token

    state_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(_TOKEN_BYTES)
    token_path.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(token_path, 0o600)
    except OSError:  # pragma: no cover - Windows 上 chmod 只影响只读位
        pass
    return token


def token_matches(expected: str, provided: str | None) -> bool:
    """常量时间比较，避免按字节猜 token。

    统一按 UTF-8 字节比较：``secrets.compare_digest`` 对 str 只接受纯 ASCII，
    收到非 ASCII 会抛 ``TypeError``，把未认证请求打成 500。按字节比之后，
    非 ASCII 输入只是普通的“不匹配”，恒时语义不变。
    """
    if not provided:
        return False
    return secrets.compare_digest(
        str(provided).encode("utf-8"), str(expected).encode("utf-8")
    )


def safe_path(workspace: str | Path, path: str | Path) -> Path:
    """把用户传入的路径解析成绝对路径，并断言它落在 workspace 内。

    相对路径按 workspace 解析（不是按服务进程的 CWD，否则不同启动方式
    会得到不同结果）。越界一律 403，不区分“不存在”还是“在别人家里”，
    避免用报错差异探测文件系统。
    """
    ws = Path(workspace).resolve()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else ws / raw
    resolved = candidate.resolve()
    if resolved != ws and ws not in resolved.parents:
        raise HTTPException(status_code=403, detail="path outside workspace")
    return resolved


def mask_secret(value: str | None) -> str:
    """把 API Key 变成 ``sk-…后4位``（长度不足时不泄漏任何尾字符）。"""
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 8:
        return "***"
    return f"sk-…{text[-4:]}"
