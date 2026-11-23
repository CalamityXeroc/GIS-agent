# -*- coding: utf-8 -*-
"""内核熔断（DAIR-365）：连续失败 → 10s/60s/300s 退避，成功即清零。

这里只测规则本身（时间用手动 ``now`` 推进，不 sleep）；接线后的行为
（退避期间不起整轮、``?force=1`` 立刻重试、内核恢复后自动归零）在
``tests/web/test_catalog_scan.py`` 里用假内核验证。
"""

from __future__ import annotations

from gis_cli.datacatalog.kernel_health import DEFAULT_BACKOFF, KernelHealth


def test_first_failures_keep_retrying_without_backoff():
    """前两次失败仍按下一拍重试：可恢复的崩溃不该被退避挡住。"""
    health = KernelHealth()

    health.record_failure("boom", now=100.0)
    health.record_failure("boom", now=110.0)

    assert health.strikes == 2
    assert health.blocked_for(now=110.0) == 0.0


def test_backoff_escalates_then_caps():
    health = KernelHealth()
    for now in (100.0, 110.0, 120.0):
        health.record_failure("boom", now=now)

    assert health.strikes == 3
    assert health.blocked_for(now=120.0) == 10.0
    assert health.blocked_for(now=125.0) == 5.0
    assert health.blocked_for(now=131.0) == 0.0  # 窗口过去自动放行

    health.record_failure("boom", now=131.0)
    assert health.blocked_for(now=131.0) == 60.0

    health.record_failure("boom", now=200.0)
    assert health.blocked_for(now=200.0) == 300.0

    health.record_failure("boom", now=600.0)
    assert health.blocked_for(now=600.0) == DEFAULT_BACKOFF[-1]  # 封顶，不再涨


def test_success_clears_the_backoff():
    """成功一次立刻回到正常轮询，不留永久熔断。"""
    health = KernelHealth()
    for now in (100.0, 101.0, 102.0):
        health.record_failure("boom", now=now)
    assert health.blocked_for(now=102.0) > 0

    health.record_success()

    assert health.strikes == 0
    assert health.blocked_for(now=102.0) == 0.0
    assert health.last_error == ""


def test_force_releases_one_attempt_but_keeps_the_strikes():
    """``?force=1`` 只放行这一次；再失败按下一档退避，而不是从头数。"""
    health = KernelHealth()
    for now in (100.0, 101.0, 102.0):
        health.record_failure("boom", now=now)

    health.allow_retry()
    assert health.blocked_for(now=102.0) == 0.0

    health.record_failure("boom", now=103.0)
    assert health.blocked_for(now=103.0) == 60.0


def test_message_is_readable_and_counts_down():
    health = KernelHealth()
    for now in (100.0, 101.0, 102.0):
        health.record_failure("boom", now=now)

    text = health.message(now=105.0)

    assert "内核连续失败 3 次" in text
    assert "7 秒后重试" in text
    assert "force=1" in text
