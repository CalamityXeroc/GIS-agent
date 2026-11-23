# -*- coding: utf-8 -*-
"""把图片（文件路径 / base64 / data URL）转成多模态接口可用的 data URL。

用途：模型支持识图时，把产出的地图/统计图附进对话，让它自己核对
（图名/图例/比例尺是否齐全、中文是否变方框、图面是否被裁切）。

设计要点：
- 大图先降采样（长边 ≤ ``max_side``）再编码，避免把上下文撑爆；
- 无 Pillow 时退回原图（仍受 ``max_bytes`` 限制）；
- 任何异常都返回空串，绝不影响主流程。
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")

#: 默认上限：超过则降采样
DEFAULT_MAX_BYTES = 400_000
DEFAULT_MAX_SIDE = 1600


def is_image_path(value: object) -> bool:
    """判断是不是图片文件路径（按后缀）。"""
    return str(value).lower().endswith(IMAGE_SUFFIXES)


def _to_data_url(raw: bytes, suffix: str) -> str:
    mime = "image/png" if suffix.lower() in (".png",) else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _downscale(raw: bytes, *, max_side: int = DEFAULT_MAX_SIDE) -> tuple[bytes, str]:
    """用 Pillow 降采样，返回 (字节流, 后缀)。失败时原样返回。"""
    try:
        from PIL import Image  # type: ignore

        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")
            scale = max_side / max(img.size)
            if scale < 1:
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82, optimize=True)
            return buf.getvalue(), ".jpg"
    except Exception as exc:  # pragma: no cover - 取决于 Pillow
        logger.debug("downscale failed: %s", exc)
        return raw, ""


def as_data_url(
    item: object,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_side: int = DEFAULT_MAX_SIDE,
) -> str:
    """把图片转成 data URL；无法识别时返回空串。

    支持三种输入：
    - ``data:image/...;base64,...`` 原样返回（过长仍会被拒绝）
    - 裸 base64 字符串（jupyter 内联图）→ 包成 data URL
    - 图片文件路径 → 读文件（必要时降采样）
    """
    if not item:
        return ""
    text = str(item).strip()

    # 1) 已是 data URL
    if text.startswith("data:image"):
        if len(text) > max_bytes * 3:
            logger.debug("data URL 过大，跳过（%d 字符）", len(text))
            return ""
        return text

    # 2) 文件路径
    if os.path.exists(text):
        path = Path(text)
        if not is_image_path(path):
            return ""
        try:
            raw = path.read_bytes()
        except Exception as exc:  # pragma: no cover
            logger.debug("read image failed %s: %s", path, exc)
            return ""
        suffix = path.suffix
        if len(raw) > max_bytes:
            raw, new_suffix = _downscale(raw, max_side=max_side)
            suffix = new_suffix or suffix
        return _to_data_url(raw, suffix)

    # 3) 裸 base64（长度像图片数据，且不是路径）
    if len(text) > 512 and len(text) % 4 == 0 and "/" not in text[:64] and "\\" not in text[:64]:
        try:
            raw = base64.b64decode(text, validate=True)
        except Exception:
            return ""
        if len(raw) > max_bytes:
            raw, _ = _downscale(raw, max_side=max_side)
        return _to_data_url(raw, ".png")
    return ""


__all__ = ["IMAGE_SUFFIXES", "as_data_url", "is_image_path"]
