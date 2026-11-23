# -*- coding: utf-8 -*-
"""民生模板库（P1-1 骨架）：manifest schema + 加载器 + 配方链执行。"""

from .schema import (
    Binding,
    Template,
    TemplateBackendError,
    TemplateBindingError,
    TemplateDataError,
    TemplateError,
    TemplateInput,
    TemplateManifestError,
    TemplateStep,
    parse_binding,
    resolve_bindings,
)
from .store import TemplateLibrary

__all__ = [
    "Binding",
    "Template",
    "TemplateBackendError",
    "TemplateBindingError",
    "TemplateDataError",
    "TemplateError",
    "TemplateInput",
    "TemplateLibrary",
    "TemplateManifestError",
    "TemplateStep",
    "parse_binding",
    "resolve_bindings",
]
