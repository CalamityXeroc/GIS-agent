# -*- coding: utf-8 -*-
"""环境自检（P0-1）：``gis-agent doctor`` 背后的 7 项检查。

只做一件事：判断"这台机器现在能不能跑"，不能就用一句人话 + 一条可执行命令
说清缺什么。不查许可证、不引导安装、不自动改环境（``--fix-install`` 需显式指定）。
"""

from __future__ import annotations

from .doctor import (
    CORE_DEPENDENCIES,
    FAIL,
    MIN_PRO_VERSION,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    DoctorReport,
    run_checks,
    run_startup_checks,
)
from .install import FixInstallResult, fix_install

__all__ = [
    "CORE_DEPENDENCIES",
    "FAIL",
    "MIN_PRO_VERSION",
    "PASS",
    "SKIP",
    "WARN",
    "CheckResult",
    "DoctorReport",
    "FixInstallResult",
    "fix_install",
    "run_checks",
    "run_startup_checks",
]
