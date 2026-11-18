"""ジョブイベントの配信（要件§29）。

購読先は3つ:
  1. SQLite（永続化。再接続時に取りこぼしを埋められる）
  2. WebSocket（/events/{job_id} のライブ購読）
  3. AICCORE-Nexus Gateway への callback（任意。失敗してもジョブは止めない）
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

import httpx

from . import config, store

log = logging.getLogger("genesis.events")

# job_id -> 購読キュー
_subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)


def subscribe(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _subscribers[job_id].add(q)
    return q


def unsubscribe(job_id: str, q: asyncio.Queue) -> None:
    _subscribers.get(job_id, set()).discard(q)
    if not _subscribers.get(job_id):
        _subscribers.pop(job_id, None)


def _gateway_url() -> str | None:
    gw = config.gateway_config().get("gateway", {})
    if not gw.get("enabled", True):
        return None
    api_key = config.env(gw.get("api_key_env", "GATEWAY_API_KEY"), "")
    if not api_key:
        return None  # 未設定なら単体動作（Gateway 不在でも動く）
    host = config.env(gw.get("host_env", "GATEWAY_HOST"), "127.0.0.1")
    port = config.env(gw.get("port_env", "GATEWAY_PORT"), "3000")
    path = config.env(gw.get("event_path_env", "GATEWAY_EVENT_PATH"),
                      "/v1/3d/characters/events")
    return f"http://{host}:{port}{path}"


async def emit(job_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
    payload = payload or {}
    body = {"job_id": job_id, "event": event, **payload}

    store.add_event(job_id, event, payload)

    for q in list(_subscribers.get(job_id, set())):
        try:
            q.put_nowait(body)
        except asyncio.QueueFull:
            log.debug("購読キューが満杯: %s", job_id)

    url = _gateway_url()
    if url:
        gw = config.gateway_config().get("gateway", {})
        try:
            key = config.env(gw.get("api_key_env", "GATEWAY_API_KEY"), "")
            async with httpx.AsyncClient(timeout=float(gw.get("timeout_sec", 5))) as c:
                await c.post(url, json=body,
                             headers={"Authorization": f"Bearer {key}"})
        except Exception as e:
            # fail_open: Gateway が落ちていてもジョブは続行する
            log.debug("Gateway callback 失敗 (無視): %s", e)
