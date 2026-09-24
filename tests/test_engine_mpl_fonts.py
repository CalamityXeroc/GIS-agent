# -*- coding: utf-8 -*-
"""中文字体配置测试（需要 matplotlib，缺失时自动跳过）。

背景：第 14 届基准任务三a 实测——agent 用 matplotlib 自绘的对比图中文全变方框，
因为内核没有配置中文字体，且这类"缺字形"只写 stderr、成功路径不回喂模型。

这两条测试保证：
1. ``configure()`` 能把 matplotlib 解析到真正的中文字体（而不是 DejaVu）；
2. 含中文的图渲染后**不产生** "missing from font" 告警（tofu 的判定特征）。
"""
from __future__ import annotations

import logging
import warnings


def test_configure_resolves_cjk_font():
    """configure() 之后字体解析结果必须是中文字体文件。"""
    import pytest

    pytest.importorskip("matplotlib")
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    from gis_cli.runtime.mpl_setup import configure, resolve_cjk_font

    info = configure()
    if not resolve_cjk_font():
        pytest.skip("本机没有中文字体文件（非 Windows 或字体未安装）")
    assert info["available"] is True
    path = font_manager.findfont(FontProperties(family=info["families"][:3]))
    assert any(key in path.lower() for key in ("msyh", "simhei", "simsun", "msjh")), path


def test_chinese_title_renders_without_missing_glyph(tmp_path, caplog):
    """含中文标题/图例的图不应出现缺字形告警（缺字形=方框=交付不可用）。"""
    import pytest

    pytest.importorskip("matplotlib")
    from gis_cli.runtime.mpl_setup import configure, resolve_cjk_font

    if not resolve_cjk_font():
        pytest.skip("本机没有中文字体文件（非 Windows 或字体未安装）")

    configure()
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING):
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_title("社区老年人口分布")
            ax.set_xlabel("经度（度）")
            ax.plot([1, 2, 3], [3, 2, 1], label="样本点")
            ax.legend()
            out = tmp_path / "cn_title.png"
            fig.savefig(out, dpi=80)
            plt.close(fig)

    texts = [str(w.message) for w in caught] + [r.getMessage() for r in caplog.records]
    missing = [text for text in texts if "missing from font" in text]
    assert not missing, missing
    assert out.stat().st_size > 1000
