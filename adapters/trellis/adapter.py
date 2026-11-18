"""TRELLIS.2 アダプタ — AICCORE-Nexus の既存 trellis-service (:8811) を再利用する。

Genesis3D はこのモデルを自分でロードしない。既に GPU1 に pinned 常駐している
プロセスへ HTTP で投げるだけなので、追加 VRAM は 0。GPU1 の空きが約7GBしか
ないという実測制約に対する答えがこれ。

このサービスは2つの能力を持つ:
  POST /generate   画像 → PBRテクスチャ付き GLB          （工程3b: ジオメトリ候補）
  再テクスチャリング Trellis2TexturingPipeline           （工程6: mesh+画像 → GLB）

/generate は上流に元から在るエンドポイント。再テクスチャは同サービスに
エンドポイントが無ければ未対応として扱い、工程6を安全にスキップする。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx

from ..base import GenerateRequest, GenerateResult, ModelAdapter

log = logging.getLogger("genesis.adapter.trellis")


class TrellisAdapter(ModelAdapter):
    name = "trellis2"

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__(spec)
        import os
        env_key = spec.get("endpoint_env", "TRELLIS_HOST")
        self.endpoint = (os.environ.get(env_key)
                         or spec.get("default_endpoint", "http://127.0.0.1:8811")).rstrip("/")
        self.resolutions = [int(r) for r in spec.get("resolutions", [512, 1024, 1536])]
        self.default_resolution = int(spec.get("default_resolution", 512))
        self.mesh_settings = spec.get("mesh_settings", {}) or {}
        self._health: dict[str, Any] | None = None

    # -- ライフサイクル ------------------------------------------------
    async def health(self, timeout: float = 5.0) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.get(f"{self.endpoint}/healthz")
                if r.status_code != 200:
                    return None
                self._health = r.json()
                return self._health
        except Exception as e:
            log.debug("trellis health failed: %s", e)
            return None

    async def load(self) -> None:
        h = await self.health()
        if h is None:
            raise RuntimeError(
                f"TRELLIS.2 サービスに到達できません ({self.endpoint})。"
                " AICCORE-Nexus の Model Runtime 画面か trellis-service/run.sh で起動してください"
            )
        if h.get("stub"):
            raise RuntimeError("TRELLIS.2 が stub モードです（重み未導入）")
        self._loaded = True

    async def get_status(self) -> dict[str, Any]:
        base = await super().get_status()
        h = await self.health(timeout=2.0)
        base.update({
            "endpoint": self.endpoint,
            "reachable": h is not None,
            "service": h or {},
            "vram_note": "外部プロセスが確保済みのため Genesis3D の消費は 0",
        })
        return base

    # -- 工程3b: ジオメトリ候補 ----------------------------------------
    async def generate(self, req: GenerateRequest) -> GenerateResult:
        t0 = time.time()
        s = req.settings or {}
        resolution = int(s.get("resolution", self.default_resolution))
        if resolution not in self.resolutions:
            resolution = self.default_resolution

        out_path = req.out_dir / f"cand_seed{req.seed}_res{resolution}.glb"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # サービス側の既定 (decimation_target=1,000,000 / texture_size=2048) は
        # GLB 抽出が CPU で20分超になる。必ず明示的に渡す。
        profile = self.mesh_settings.get(s.get("mesh_profile", "candidate"), {})
        decimation = int(s.get("decimation_target", profile.get("decimation_target", 120000)))
        tex_size = int(s.get("texture_size", profile.get("texture_size", 1024)))
        data: dict[str, str] = {
            "resolution": str(resolution),
            "seed": str(req.seed),
            "decimation_target": str(decimation),
            "texture_size": str(tex_size),
        }

        try:
            # GLB 抽出(o_voxel.postprocess.to_glb)は CPU バウンドで、
            # decimation_target を上げると容易に15分を超える。既定を長めに取る。
            timeout_sec = float(s.get("timeout", self.spec.get("request_timeout_sec", 2400)))
            async with httpx.AsyncClient(timeout=timeout_sec) as c:
                with open(req.image_path, "rb") as fh:
                    files = {"image": (req.image_path.name, fh, "image/png")}
                    r = await c.post(f"{self.endpoint}/generate", data=data, files=files)
            if r.status_code != 200:
                return GenerateResult(
                    ok=False, duration_sec=time.time() - t0,
                    error=f"trellis /generate {r.status_code}: {r.text[:200]}")
            body = r.content
            if not body or len(body) < 128:
                return GenerateResult(ok=False, duration_sec=time.time() - t0,
                                      error="trellis が空のGLBを返しました")
            out_path.write_bytes(body)
        except Exception as e:
            return GenerateResult(ok=False, duration_sec=time.time() - t0,
                                  error=f"trellis 呼び出し失敗: {e}")

        return GenerateResult(
            ok=True, output_path=out_path, duration_sec=time.time() - t0,
            model_version=(self._health or {}).get("model", "microsoft/TRELLIS.2-4B"),
            vram_peak_gb=0.0,  # 外部プロセス側で消費
            extra={"resolution": resolution, "seed": req.seed,
                   "decimation_target": decimation, "texture_size": tex_size,
                   "bytes": len(body)},
        )

    # -- 工程6: 再テクスチャリング --------------------------------------
    async def supports_retexture(self) -> bool:
        """サービスに再テクスチャ用エンドポイントがあるか。"""
        h = await self.health(timeout=2.0) or {}
        caps = h.get("capabilities", {}) or {}
        return bool(caps.get("texturing") or caps.get("retexture"))

    async def retexture(
        self, mesh_path: Path, image_path: Path, out_path: Path,
        *, timeout: float = 1200.0,
    ) -> GenerateResult:
        """既存 mesh + 参照画像 → 高品質PBRテクスチャ付き GLB。

        これが「best geometry + best texture」の融合（§13）の実体。Boolean結合
        ではなく、幾何は他モデル由来のまま TRELLIS.2 でテクスチャだけ貼り直す。
        """
        t0 = time.time()
        if not await self.supports_retexture():
            return GenerateResult(
                ok=False, duration_sec=time.time() - t0,
                error="trellis-service に再テクスチャエンドポイントがありません"
                      "（工程6はスキップされます）")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                with open(mesh_path, "rb") as mf, open(image_path, "rb") as imf:
                    files = {
                        "mesh": (mesh_path.name, mf, "application/octet-stream"),
                        "image": (image_path.name, imf, "image/png"),
                    }
                    r = await c.post(f"{self.endpoint}/texture", files=files)
            if r.status_code != 200:
                return GenerateResult(ok=False, duration_sec=time.time() - t0,
                                      error=f"trellis /texture {r.status_code}")
            out_path.write_bytes(r.content)
        except Exception as e:
            return GenerateResult(ok=False, duration_sec=time.time() - t0,
                                  error=f"再テクスチャ失敗: {e}")
        return GenerateResult(ok=True, output_path=out_path,
                              duration_sec=time.time() - t0)
