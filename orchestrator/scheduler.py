"""ジョブキュー。1台構成なので Redis 等の分散基盤は使わない（要件§27）。

FastAPI の起動時に asyncio タスクとして走らせる。重い推論はすべて
独立サブプロセスか :8811 のサービスが担うため、イベントループは塞がらない。
"""
from __future__ import annotations

import asyncio
import logging

from . import gpu_manager, store
from .pipeline import Pipeline

log = logging.getLogger("genesis.scheduler")


class Scheduler:
    def __init__(self) -> None:
        self.lease = gpu_manager.GpuLease()
        self.pipeline = Pipeline(self.lease)
        self._task: asyncio.Task | None = None
        self._running: dict[str, asyncio.Task] = {}
        self._stop = asyncio.Event()

    async def start(self) -> None:
        recovered = store.recover_stale_running()
        if recovered:
            log.warning("前回のクラッシュで残った running ジョブ %d 件を failed にしました",
                        recovered)
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="genesis-scheduler")
        log.info("scheduler started")

    async def stop(self) -> None:
        self._stop.set()
        for t in list(self._running.values()):
            t.cancel()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("scheduler stopped")

    def is_running(self, job_id: str) -> bool:
        return job_id in self._running

    def cancel(self, job_id: str) -> bool:
        t = self._running.get(job_id)
        if t is None:
            return False
        t.cancel()
        return True

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = store.claim_next_queued()
                if job is None:
                    await asyncio.sleep(1.0)
                    continue
                jid = job["job_id"]
                log.info("job %s を開始 (style=%s quality=%s)",
                         jid, job["style"], job["quality"])
                task = asyncio.create_task(self._run_job(job), name=f"job-{jid}")
                self._running[jid] = task
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduler loop error")
                await asyncio.sleep(2.0)

    async def _run_job(self, job: dict) -> None:
        jid = job["job_id"]
        try:
            await self.pipeline.run(job)
        finally:
            self._running.pop(jid, None)
