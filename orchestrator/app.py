"""Genesis3D Orchestrator — FastAPI。127.0.0.1 のみで待ち受ける（要件§32）。

AICCORE-Nexus Gateway からはこの API を叩く。Gateway が無くても
単体で起動・テストできる（要件§2）。
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from adapters import registry

from . import config, events, gpu_manager, store
from .pipeline import job_paths, sha256_file
from .scheduler import Scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("genesis.app")

scheduler = Scheduler()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    await scheduler.start()
    log.info("Genesis3D orchestrator ready")
    try:
        yield
    finally:
        await scheduler.stop()


app = FastAPI(title="Genesis3D Orchestrator", version="0.3.0", lifespan=lifespan)


# ---------------------------------------------------------------- health
@app.get("/health")
async def health() -> dict[str, Any]:
    gpus = gpu_manager.report()
    trellis = await registry.get("trellis2").get_status()
    return {
        "ok": True,
        "service": "genesis3d",
        "version": app.version,
        "gpu_source": gpus["source"],
        "gpu_count": gpus["count"],
        "trellis_reachable": trellis.get("reachable", False),
        "blender": Path(config.blender_bin()).exists(),
        "data_dir": str(config.data_dir()),
    }


@app.get("/gpus")
async def gpus() -> dict[str, Any]:
    return gpu_manager.report()


@app.get("/models")
async def models() -> dict[str, Any]:
    out = []
    for name in registry.all_names():
        try:
            out.append(await registry.get(name).get_status())
        except Exception as e:
            out.append({"name": name, "error": str(e)})
    return {"models": out, "styles": config.style_names()}


# ---------------------------------------------------------------- jobs
@app.post("/jobs")
async def create_job(
    image: UploadFile = File(...),
    # 顔テクスチャ投影(工程6)に使う実写。省略時は image を使う。
    # 生成した全身Tポーズ画像から体を作る場合、その顔は本人ではないので
    # （実測 ArcFace 0.20）、ここに実写のポートレートを渡して分離する。
    face_image: UploadFile | None = File(None),
    style: str = Form("stylized"),
    quality: str = Form("high"),
    output_format: str = Form("vrm"),
    models: str = Form("auto"),
    multi_candidate: bool = Form(True),
    # style=wrap 専用。ラップ方式は身長とテンプレートを取る（docs/09）
    height_cm: float | None = Form(None),
    template: str | None = Form(None),
) -> dict[str, Any]:
    if style not in config.style_names():
        raise HTTPException(400, f"style は {config.style_names()} のいずれか")
    if quality not in ("fast", "high", "maximum"):
        raise HTTPException(400, "quality は fast|high|maximum のいずれか")

    job_id = store.new_job_id()
    paths = job_paths(job_id)
    paths.ensure()
    suffix = Path(image.filename or "input.png").suffix or ".png"
    dst = paths.input / f"input{suffix}"
    with open(dst, "wb") as f:
        shutil.copyfileobj(image.file, f)
    if dst.stat().st_size == 0:
        raise HTTPException(400, "画像が空です")

    face_path = None
    if face_image is not None and face_image.filename:
        fsuffix = Path(face_image.filename).suffix or ".png"
        face_dst = paths.input / f"face{fsuffix}"
        with open(face_dst, "wb") as f:
            shutil.copyfileobj(face_image.file, f)
        if face_dst.stat().st_size == 0:
            raise HTTPException(400, "face_image が空です")
        face_path = str(face_dst)

    digest = sha256_file(dst)
    # 入力ファイルを置くために job_id を先に採番しているので、それを渡す
    store.create_job(
        job_id=job_id, style=style, quality=quality, output_format=output_format,
        params={"models": models, "multi_candidate": multi_candidate,
                "face_image": face_path,
                **({"height_cm": height_cm} if height_cm else {}),
                **({"template": template} if template else {})},
        input_path=str(dst), input_sha256=digest,
    )
    log.info("job %s created (style=%s quality=%s)", job_id, style, quality)
    return {"job_id": job_id, "status": "queued", "style": style,
            "quality": quality, "input_sha256": digest,
            "face_image": face_path}


@app.get("/jobs")
async def list_jobs(limit: int = 50) -> dict[str, Any]:
    return {"jobs": store.list_jobs(limit)}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job が見つかりません")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "gpu": job["gpu"],
        "style": job["style"],
        "quality": job["quality"],
        "error": job["error"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
    }


@app.get("/jobs/{job_id}/results")
async def job_results(job_id: str) -> dict[str, Any]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job が見つかりません")
    paths = job_paths(job_id)
    files = []
    if paths.output.exists():
        files = [p.name for p in sorted(paths.output.iterdir()) if p.is_file()]
    return {
        "job_id": job_id,
        "status": job["status"],
        "candidates": store.list_candidates(job_id),
        "files": files,
        "output_dir": str(paths.output),
    }


@app.get("/jobs/{job_id}/results/{filename}")
async def job_file(job_id: str, filename: str):
    paths = job_paths(job_id)
    target = (paths.output / filename).resolve()
    # パストラバーサル防止
    if not str(target).startswith(str(paths.output.resolve())) or not target.is_file():
        raise HTTPException(404, "ファイルが見つかりません")
    return FileResponse(target, filename=filename)


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job が見つかりません")
    if job["status"] in ("completed", "failed", "canceled"):
        return {"job_id": job_id, "status": job["status"], "canceled": False}
    if scheduler.cancel(job_id):
        return {"job_id": job_id, "canceled": True}
    store.finish_job(job_id, "canceled", "キュー待ちの状態でキャンセルされました")
    return {"job_id": job_id, "canceled": True}


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str, after: int = 0) -> dict[str, Any]:
    if store.get_job(job_id) is None:
        raise HTTPException(404, "job が見つかりません")
    return {"events": store.list_events(job_id, after)}


@app.websocket("/events/{job_id}")
async def ws_events(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    q = events.subscribe(job_id)
    try:
        # 既存イベントを先に送って取りこぼしを防ぐ
        for e in store.list_events(job_id):
            await ws.send_json({"job_id": job_id, "event": e["event"], **e["payload"]})
        while True:
            body = await q.get()
            await ws.send_json(body)
            if body.get("event") in ("job.completed", "job.failed"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug("ws error: %s", e)
    finally:
        events.unsubscribe(job_id, q)
        with contextlib.suppress(Exception):
            await ws.close()
