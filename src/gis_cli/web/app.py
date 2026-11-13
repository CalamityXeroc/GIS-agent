# -*- coding: utf-8 -*-
"""FastAPI 应用装配：只读 API + 运行态 API/SSE + 本机 token 中间件 + 前端静态产物。

服务端**不 import 引擎渲染层**（``ui/render`` / ``ui/tui`` / ``ui/wiring``），
避免 rich 在无 TTY 的服务进程里抢 stdout；引擎只在真正跑任务的 worker 线程里
按需 import（见 ``jobs.make_engine_runner``）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import catalog_api, config_api, jobs, metrics, preview_api, sse, trace_api
from .event_bus import EventBus
from .security import load_or_create_token, token_matches

#: 前端构建产物目录（``web/`` 的 Vite 输出直接落在这里）。
STATIC_DIR = Path(__file__).resolve().parent / "static"


class _SPAStaticFiles(StaticFiles):
    """SPA 回落：找不到静态文件时返回 index.html，交给前端路由。

    没有这一层时，直接在地址栏打开 ``/runs`` 或刷新子页面会被 StaticFiles
    判成 404；``/api/`` 前缀不参与回落，避免拼错的接口地址拿到 200 的 HTML。
    """

    async def get_response(self, path: str, scope: Any):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or scope["path"].startswith("/api/"):
                raise
            return await super().get_response("index.html", scope)


def create_app(
    workspace: str | Path,
    token: str | None = None,
    config_path: str | Path | None = None,
    *,
    bus: EventBus | None = None,
    manager: jobs.JobManager | None = None,
    runner: jobs.RunnerFactory | None = None,
    preview_runner: Any = None,
    plan_timeout: float = jobs.DEFAULT_TIMEOUT,
    heartbeat: float = sse.HEARTBEAT_S,
) -> FastAPI:
    """装配控制台后端。

    ``token`` 留空时读取/生成 ``<workspace>/.gis_agent/web_token``，保证
    ``gis-agent web`` 重启后 URL 里的 token 仍然有效。

    ``bus`` / ``manager`` / ``runner`` 是给测试注入假引擎用的；生产走默认路径
    （真实引擎在 worker 线程里 ``build_engine``）。
    """
    ws = Path(workspace).resolve()
    expected_token = token or load_or_create_token(ws)

    event_bus = bus or (manager.bus if manager is not None else EventBus())
    if manager is None:
        manager = jobs.JobManager(
            ws,
            bus=event_bus,
            runner=runner,
            config_path=config_path,
            plan_timeout=plan_timeout,
        )

    preview_runtime = preview_api.PreviewRuntime(ws, code_runner=preview_runner)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        manager.close()
        preview_runtime.close()

    app = FastAPI(title="GIS Agent Console", version="0.2.0", lifespan=lifespan)
    app.state.workspace = ws
    app.state.token = expected_token
    app.state.config_path = config_path
    app.state.events = event_bus
    app.state.jobs = manager

    @app.middleware("http")
    async def require_token(request: Request, call_next: Any):
        path = request.url.path
        if path == "/api" or path.startswith("/api/"):
            provided = request.headers.get("X-Web-Token") or request.query_params.get("token")
            if not token_matches(expected_token, provided):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "invalid or missing web token"},
                )
        return await call_next(request)

    app.include_router(trace_api.build_router(ws))
    app.include_router(metrics.build_router(ws, config_path))
    app.include_router(catalog_api.build_router(ws))
    app.include_router(preview_api.build_router(ws, runtime=preview_runtime))
    app.include_router(config_api.build_router(ws, config_path, manager))
    app.include_router(jobs.build_router(manager))
    app.include_router(sse.build_router(ws, event_bus, manager, heartbeat=heartbeat))

    # 必须最后挂：根挂载会兜住其余所有路径。
    if STATIC_DIR.is_dir():
        app.mount("/", _SPAStaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app
