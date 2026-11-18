"""10工程パイプラインのオーケストレーション。

現時点（Phase 3）で実装済みなのは 工程1・3・6 と出力。工程4/5/7/8/9/10 は
Phase 4 で埋める。未実装の工程は「スキップ」として記録し、ジョブは止めない。

要件§30に従い、1モデルが失敗しても残りの候補で処理を続行する。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapters import registry
from adapters.base import GenerateRequest

from . import config, events, gpu_manager, store

log = logging.getLogger("genesis.pipeline")

# 実行順。工程6(retexture) は工程7(mesh_repair) の**後**に走る —
# 生の候補は Blender 上で顔が白飛びして顔検出が落ちるため（下の run() 参照）。
# 番号は docs/01-pipeline.md の呼称を保つが、順序はこちらが正。
STAGES = [
    "preprocess", "multiview", "geometry", "render",
    "evaluate", "mesh_repair", "head_merge", "retexture",
    "rigging", "vrm_export", "validate", "qa",
]

# 顔テクスチャ投影（工程6）。整列用レンダは必ず正方形で撮る —
# bpy_extras の world_to_camera_view がレンダ解像度のアスペクト比を使うため、
# 投影側と比が違うと顔が横に伸びて二重像になる（docs/05-measurements.md §16）。
FACEPROJ_RENDER_RES = 2048
FACEPROJ_BAKE_SIZE = 2048
# 顔メッシュは単独で焼くので、アトラス全体が顔＋肩になる。ここだけ倍にすると
# リサンプルが素直になり、眉や肌の粒がわずかに締まる。ただし**情報は増えない** —
# 実測で ArcFace は 0.6552 → 0.6529 と横ばいだった。上限は元写真の顔の画素数
# （205px）であって、ベイク解像度ではない（docs/05-measurements.md §35）。
FACEPROJ_HEAD_BAKE_SIZE = 2048


@dataclass
class JobPaths:
    root: Path

    @property
    def input(self) -> Path: return self.root / "input"
    @property
    def references(self) -> Path: return self.root / "references"
    @property
    def candidates(self) -> Path: return self.root / "candidates"
    @property
    def renders(self) -> Path: return self.root / "renders"
    @property
    def scores(self) -> Path: return self.root / "scores"
    @property
    def selected(self) -> Path: return self.root / "selected"
    @property
    def rigged(self) -> Path: return self.root / "rigged"
    @property
    def output(self) -> Path: return self.root / "output"
    @property
    def logs(self) -> Path: return self.root / "logs"

    def ensure(self) -> None:
        for p in (self.input, self.references, self.candidates, self.renders,
                  self.scores, self.selected, self.rigged, self.output, self.logs):
            p.mkdir(parents=True, exist_ok=True)


def job_paths(job_id: str) -> JobPaths:
    return JobPaths(config.data_dir() / job_id)


def _conda_bin() -> str:
    return shutil.which("conda") or str(
        Path(config.env("CONDA_BASE", "~/miniconda3") or "~/miniconda3")
        .expanduser() / "bin" / "conda")


async def _run(argv: list[str], *, env: dict[str, str] | None = None,
               cwd: Path | None = None, timeout: float = 3600.0,
               log_path: Path | None = None) -> tuple[int, str]:
    """独立サブプロセスとして実行する。

    CUDA OOM や segfault が Orchestrator を巻き込まず、プロセス終了で
    VRAM が完全に解放される（要件§30/§31）。
    """
    import os
    proc_env = dict(os.environ)
    proc_env.setdefault("HF_HUB_OFFLINE", "1")
    proc_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if env:
        proc_env.update(env)
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=str(cwd) if cwd else None, env=proc_env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"タイムアウト ({timeout:.0f}s)"
    text = (out or b"").decode("utf-8", "replace")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text)
    return proc.returncode or 0, text


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Pipeline:
    def __init__(self, lease: gpu_manager.GpuLease) -> None:
        self.lease = lease

    async def _blender(self, script: str, args: list[str], *, gpu: int | None = 0,
                       timeout: float = 900.0, log_path: Path | None = None,
                       vram_gb: float = 2.0, what: str = "") -> tuple[int, str]:
        """Blender を headless で実行する。--factory-startup で毎回同条件にする。"""
        argv = [config.blender_bin(), "--background", "--factory-startup",
                "--python", str(config.ROOT / "blender" / script), "--", *args]
        env = {"PYTHONPATH": str(config.ROOT)}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if gpu is None or vram_gb <= 0:
            return await _run(argv, env=env, cwd=config.ROOT,
                              timeout=timeout, log_path=log_path)
        async with self.lease.acquire(gpu, vram_gb, what=what or script):
            return await _run(argv, env=env, cwd=config.ROOT,
                              timeout=timeout, log_path=log_path)

    async def _eval_env(self, module_args: list[str], *, timeout: float = 900.0,
                        log_path: Path | None = None, vram_gb: float = 4.0,
                        what: str = "") -> tuple[int, str]:
        argv = [_conda_bin(), "run", "-n", config.env("ENV_EVAL", "genesis-eval"),
                "--no-capture-output", "python", "-m", *module_args]
        env = {"PYTHONPATH": str(config.ROOT), "CUDA_VISIBLE_DEVICES": "0"}
        async with self.lease.acquire(0, vram_gb, what=what or module_args[0]):
            return await _run(argv, env=env, cwd=config.ROOT,
                              timeout=timeout, log_path=log_path)

    async def _imagegen_env(self, module_args: list[str], *, timeout: float = 1800.0,
                           log_path: Path | None = None, vram_gb: float = 8.0,
                           what: str = "") -> tuple[int, str]:
        """参照ビュー生成(SDXL+ControlNet)の env で回す。

        VRAM リースを 8GB にして `--offload` 前提で走らせる。offload 無しの実測
        ピークは 13.9GB で、GPU0 は他サービスが常時 9GB 使っているため
        合計 23.2GB/24GB まで上がり余裕が 1.3GB しかない（docs/05 §19）。
        時間は `styles.yaml` の time_penalty.weight=0.02 が示す通りほぼ無料なので、
        速度より安全を取る。
        """
        argv = [_conda_bin(), "run", "-n", config.env("ENV_IMAGEGEN", "genesis-imagegen"),
                "--no-capture-output", "python", "-m", *module_args]
        env = {"PYTHONPATH": str(config.ROOT), "CUDA_VISIBLE_DEVICES": "0"}
        async with self.lease.acquire(0, vram_gb, what=what or module_args[0]):
            return await _run(argv, env=env, cwd=config.ROOT,
                              timeout=timeout, log_path=log_path)

    async def _multiview(self, job, paths: JobPaths, ref_image: Path,
                         preset: dict, manifest) -> Path:
        """工程2: 全身Tポーズの参照ビューを生成する。

        バストアップ入力からは脚も足も推定できず、リギングが破綻する
        （docs/05-measurements.md §4）。ここで全身Tポーズ画像を作り、
        以降のジオメトリはそこから起こす。

        **入力が既に全身像なら何もしない。** 判定は工程1の見切れ警告を使う
        （下端に接している＝全身ではない）。高さ比ではなく端への接触で見る
        のは、バストアップでも高さ比は 0.92 程度にしかならないため（§4）。

        生成された顔は本人ではない（実測 ArcFace 0.20）。そこで元の実写を
        `params.face_image` に立てて、顔は後段の head_merge が実写から作る。
        失敗しても元画像で続行する。
        """
        jid = job["job_id"]
        if not preset.get("multiview", False):
            manifest["stages"]["multiview"] = self._skipped(
                f"quality={job['quality']} では参照ビュー生成を行わない")
            return ref_image

        # **下端に接しているときだけ**が対象。rmbg.py は端に接触したら
        # どの辺でも「見切れの可能性」を出すので、"見切れ" だけで判定すると
        # Tポーズ全身画像（手が左右端に接する）でも発火して、既に全身なのに
        # もう一度全身を生成してしまう（実測で踏んだ）。
        # 脚と足が無いことを示すのは下端の接触だけ。
        warns = (manifest["stages"].get("preprocess") or {}).get("warnings") or []
        cut_off = any("下端で見切れ" in w for w in warns)
        if not cut_off:
            manifest["stages"]["multiview"] = self._skipped(
                "入力が全身像のため参照ビュー生成は不要")
            return ref_image

        def degraded(reason: str) -> Path:
            log.warning("参照ビュー生成: %s", reason)
            manifest["stages"]["multiview"] = {
                "status": "degraded",
                "reason": f"{reason}。元画像のまま続行します",
            }
            return ref_image

        await self._set(jid, "multiview", 12, 0)

        # 1) Tポーズ骨格。指極≒身長なのでキャンバスは正方形（§17）
        pose = paths.references / "pose_front.png"
        code, tail = await self._imagegen_env(
            ["multiview.tpose_skeleton", "--out", str(paths.references),
             "--views", "front"],
            timeout=300, vram_gb=0.0,
            log_path=paths.logs / "tpose_skeleton.log", what="tpose/skeleton")
        if code != 0 or not pose.exists():
            return degraded(f"骨格生成に失敗 (exit={code}): {tail[-160:]}")

        # 2) SDXL + ControlNet-OpenPose で複数シード
        seeds = max(2, int(preset.get("candidates_per_model", 2)) * 2)
        code, tail = await self._imagegen_env(
            ["multiview.generate_views", "--pose", str(pose),
             "--outdir", str(paths.references), "--seeds", str(seeds),
             "--tag", "front", "--offload"],
            timeout=3600, vram_gb=8.0,
            log_path=paths.logs / "generate_views.log", what="sdxl/views")
        if code != 0:
            return degraded(f"参照ビュー生成に失敗 (exit={code}): {tail[-160:]}")

        # 3) 足の見切れをゲートに、Tポーズらしさで最良を選ぶ
        sel = paths.references / "selection.json"
        code, tail = await self._eval_env(
            ["multiview.select_view", "--views", str(paths.references),
             "--reference", str(ref_image), "--out", str(sel)],
            timeout=1200, vram_gb=4.0,
            log_path=paths.logs / "select_view.log", what="dinov3/select_view")
        if code != 0 or not sel.exists():
            return degraded(f"ビュー選別に失敗 (exit={code}): {tail[-160:]}")

        data = json.loads(sel.read_text())
        best = data.get("best")
        if not best or not Path(best).exists():
            return degraded("採用できるビューがありませんでした"
                            f"（不採用 {len(data.get('rejected', []))} 枚）")

        # 生成画像の顔は本人ではない。顔は元の実写から作らせる。
        # **前処理済みの rgba ではなく元ファイルを渡す。** rgba は被写体を
        # 正方形に詰め直してあるので、そこから頭部を切ると画像のほぼ全体が
        # 取れてしまい、顔メッシュに胴が入りすぎて肩の検出位置がずれる
        # （実測: 頭部帯 0.563 → 0.344、接合スケールが 1.64 倍ずれて頭が壊れた）。
        params = dict(job.get("params") or {})
        if not params.get("face_image"):
            params["face_image"] = str(job.get("input_path") or ref_image)
            job["params"] = params

        manifest["stages"]["multiview"] = {
            "status": "ok",
            "model": "RealVisXL_V5.0 + xinsir/controlnet-openpose-sdxl-1.0",
            "seeds": seeds,
            "selected": best,
            "ranking": [(r["view"], r["overall"]) for r in data.get("ranking", [])],
            "rejected": data.get("rejected", []),
            "face_image": params.get("face_image"),
            "note": ("生成画像の顔は本人ではない（実測 ArcFace 0.20）。"
                     "顔は head_merge が実写から作る"),
        }
        return Path(best)

    async def _run_wrap(self, job: dict[str, Any], paths: JobPaths,
                        manifest: dict[str, Any]) -> Path | None:
        """`style=wrap` —— ラップ方式（docs/09）を丸ごと1本のスクリプトで回す。

        **既存の10工程には載せない。** ラップ方式は「候補を複数作って評価で選ぶ」
        構造ではなく、テンプレート1体を変形する決定的な手順なので、
        `registry.for_style` も evaluation も通らない。ここでは独立サブプロセスとして
        `scripts/gen_wrap_avatar.py` を呼び、成果物を `output/` へ移すだけにする
        （リポジトリの「各モデルは独立サブプロセス」の約束にも沿う）。

        入力の対応:
          job.input_path        … 全身写真（体型を寄せるのに使う）
          params.face_image     … 顔写真。無ければ入力を顔にも使う
          params.height_cm      … 身長（任意）
          params.template       … テンプレート識別子
        """
        params = dict(job.get("params") or {})
        body = Path(job["input_path"])
        face = Path(params.get("face_image") or body)
        out_dir = paths.root / "wrap"
        argv = [sys.executable, str(config.ROOT / "scripts/gen_wrap_avatar.py"),
                "--face", str(face), "--body", str(body),
                "--template", str(params.get("template", "rocketbox_female_01")),
                "--out-dir", str(out_dir),
                "--vrm-name", str(params.get("vrm_name", "Genesis3D Character"))]
        if params.get("height_cm"):
            argv += ["--height-cm", str(params["height_cm"])]
        t0 = time.time()
        rc, text = await _run(argv, cwd=config.ROOT, timeout=3600.0,
                              log_path=paths.logs / "wrap.log")
        if rc != 0:
            raise RuntimeError(f"gen_wrap_avatar.py が失敗（rc={rc}）: "
                               f"{text[-800:]}")
        summary = json.loads((out_dir / "output/summary.json").read_text())
        final = None
        for name in ("avatar.vrm", "avatar.glb"):
            src = out_dir / "output" / name
            if src.exists():
                dst = paths.output / name
                shutil.copy2(src, dst)
                final = final or dst          # VRM を優先する（髪が揺れる）
        manifest["stages"]["wrap"] = {
            "seconds": round(time.time() - t0, 2),
            "summary": summary, "output": str(final) if final else None}
        return final

    async def run(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        paths = job_paths(job_id)
        paths.ensure()
        style = job["style"]
        preset = config.quality_preset(job["quality"])
        t_start = time.time()

        manifest: dict[str, Any] = {
            "job_id": job_id,
            "style": style,
            "quality": job["quality"],
            "input_sha256": job.get("input_sha256"),
            "started_at": t_start,
            "stages": {},
        }

        try:
            await events.emit(job_id, "job.created", {"style": style,
                                                      "quality": job["quality"]})

            # ---- style=wrap は別系統（docs/09）----------------------
            if style == "wrap":
                await self._set(job_id, "wrap", 10, None)
                final = await self._run_wrap(job, paths, manifest)
                manifest["completed_at"] = time.time()
                manifest["duration_sec"] = round(time.time() - t_start, 2)
                manifest["output"] = str(final) if final else None
                (paths.output / "job.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False))
                store.update_job(job_id, stage="done", progress=100)
                store.finish_job(job_id, "completed")
                await events.emit(job_id, "job.completed",
                                  {"output": str(final) if final else None,
                                   "duration_sec": manifest["duration_sec"]})
                return

            # ---- 工程1 前処理 --------------------------------------
            await self._set(job_id, "preprocess", 5)
            await events.emit(job_id, "job.preprocessing")
            ref_image = await self._preprocess(job, paths, manifest)

            # ---- 工程2 参照ビュー生成（全身Tポーズ）-----------------
            await self._set(job_id, "multiview", 12)
            ref_image = await self._multiview(job, paths, ref_image,
                                              preset, manifest)

            # ---- 工程3 ジオメトリ候補 -------------------------------
            await self._set(job_id, "geometry", 20)
            await events.emit(job_id, "job.generating")
            cands = await self._geometry(job, paths, ref_image, preset, manifest)
            ok_cands = [c for c in cands if c["status"] == "ok"]
            if not ok_cands:
                raise RuntimeError("すべてのジオメトリ候補が失敗しました")

            # ---- 工程4 レンダリング ---------------------------------
            await self._set(job_id, "render", 45, 0)
            await self._render(job, paths, ok_cands, manifest)

            # ---- 工程5 評価・順位付け -------------------------------
            await self._set(job_id, "evaluate", 55, 0)
            await events.emit(job_id, "job.evaluating")
            best = await self._evaluate(job, paths, ok_cands, ref_image, manifest)

            # ---- 工程7 mesh修復 -------------------------------------
            # **工程6より先に走らせる。** TRELLIS.2 の生の候補は Blender で
            # 読むと顔が白く潰れて描画され、ArcFace のランドマーク検出が
            # 落ちる（実測: 整列が exit=1 で失敗）。repair_mesh.py を通した
            # あとのメッシュは正常に描画される。工程6の顔投影は「レンダに
            # 実写を重ねる」ので、描画が壊れていると成立しない。
            await self._set(job_id, "mesh_repair", 70, None)
            best = await self._mesh_repair(job, paths, best, preset, manifest)

            # ---- 顔メッシュの接合（face_image 指定時のみ）-----------
            await self._set(job_id, "head_merge", 74)
            best = await self._head_merge(job, paths, best, manifest)

            # ---- 工程6 再テクスチャリング（顔テクスチャ投影）-------
            await self._set(job_id, "retexture", 78)
            best = await self._retexture(job, paths, best, ref_image, preset, manifest)

            # ---- 工程8 リギング -------------------------------------
            await self._set(job_id, "rigging", 85, None)
            await events.emit(job_id, "job.rigging")
            rigged = await self._rigging(job, paths, best, manifest)

            # ---- 工程9-10 VRM書き出し・検証 -------------------------
            await self._set(job_id, "vrm_export", 92, None)
            await events.emit(job_id, "job.exporting")
            final = await self._finalize(job, paths, best, rigged, manifest)

            manifest["completed_at"] = time.time()
            manifest["duration_sec"] = round(time.time() - t_start, 2)
            manifest["output"] = str(final) if final else None
            (paths.output / "job.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False))

            store.update_job(job_id, stage="done", progress=100)
            store.finish_job(job_id, "completed")
            await events.emit(job_id, "job.completed",
                              {"output": str(final) if final else None,
                               "duration_sec": manifest["duration_sec"]})
        except asyncio.CancelledError:
            store.finish_job(job_id, "canceled", "キャンセルされました")
            await events.emit(job_id, "job.failed", {"error": "canceled"})
            raise
        except Exception as e:
            log.exception("job %s failed", job_id)
            manifest["error"] = str(e)
            (paths.output / "job.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False))
            store.finish_job(job_id, "failed", str(e))
            await events.emit(job_id, "job.failed", {"error": str(e)})

    # -- 各工程 ---------------------------------------------------------
    @staticmethod
    def _skipped(reason: str) -> dict[str, Any]:
        return {"status": "skipped", "reason": reason}

    async def _set(self, job_id: str, stage: str, progress: int,
                   gpu: int | None = None) -> None:
        store.update_job(job_id, stage=stage, progress=progress, gpu=gpu)

    async def _preprocess(self, job, paths: JobPaths, manifest) -> Path:
        """RMBG-2.0 で背景を除去し、正方形・中央配置に正規化する。

        背景を残すと、模様やエフェクトがそのまま浮遊ジオメトリになる。
        実測では未処理の入力で連結成分が6個に増え、キャラクター周囲が
        ノイズだらけになった。ここは品質に直結する。
        """
        src = Path(job["input_path"])
        dst = paths.references / "rgba.png"
        meta_path = paths.references / "meta.json"

        argv = [
            _conda_bin(), "run", "-n", config.env("ENV_EVAL", "genesis-eval"),
            "--no-capture-output", "python", "-m", "preprocessing.rmbg",
            "--input", str(src), "--output", str(dst),
            "--mask", str(paths.references / "mask.png"),
            "--meta", str(meta_path),
        ]
        env = {"PYTHONPATH": str(config.ROOT), "CUDA_VISIBLE_DEVICES": "0"}
        try:
            async with self.lease.acquire(0, 3.0, what="rmbg2/preprocess"):
                code, tail = await _run(argv, env=env, cwd=config.ROOT,
                                        timeout=600,
                                        log_path=paths.logs / "preprocess.log")
        except Exception as e:
            code, tail = 1, str(e)

        if code == 0 and dst.exists():
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            manifest["stages"]["preprocess"] = {
                "status": "ok", "model": meta.get("model"),
                "output": str(dst),
                "foreground_coverage": meta.get("foreground_coverage"),
                "significant_specks": meta.get("significant_specks"),
                "warnings": meta.get("warnings", []),
            }
            for w in meta.get("warnings", []):
                log.warning("前処理の警告 [%s]: %s", job["job_id"], w)
            return dst

        # 前処理に失敗してもジョブは止めない。原本のまま続行し、その旨を記録する。
        log.warning("前処理に失敗 (%s)。原本のまま続行します: %s", code, tail[-300:])
        if not dst.exists():
            shutil.copy2(src, dst)
        manifest["stages"]["preprocess"] = {
            "status": "degraded",
            "reason": f"RMBG-2.0 が失敗したため原本を使用 (exit={code})",
            "detail": tail[-500:],
            "output": str(dst),
        }
        return dst

    async def _geometry(self, job, paths: JobPaths, ref_image: Path,
                        preset: dict, manifest) -> list[dict[str, Any]]:
        """導入済みモデル × seed で候補を作る。1つ失敗しても続行する（§30）。"""
        job_id = job["job_id"]
        style = job["style"]
        model_names = registry.for_style(style, purpose="geometry")
        if not model_names:
            raise RuntimeError(f"style={style} に使える導入済みジオメトリモデルがありません")

        n = int(preset.get("candidates_per_model", 1))
        seeds = list(preset.get("seeds", [0]))[:n]
        # 解像度は候補ごとに変える。品質に最も効くレバーだが VRAM リスクもあるため、
        # 高解像度と低解像度を混ぜて「1つ失敗しても候補が残る」形にする。
        resolutions = list(preset.get("resolutions", [])) or [None]
        while len(resolutions) < len(seeds):
            resolutions.append(resolutions[-1])
        combos = list(zip(seeds, resolutions[: len(seeds)]))
        results: list[dict[str, Any]] = []
        stage_log: list[dict[str, Any]] = []

        for model_name in model_names:
            adapter = registry.get(model_name)
            try:
                await adapter.load()
            except Exception as e:
                log.warning("%s のロード失敗: %s", model_name, e)
                stage_log.append({"model": model_name, "status": "failed",
                                  "error": f"load: {e}"})
                continue

            out_dir = paths.candidates / model_name
            for seed, resolution in combos:
                settings = dict(job.get("params", {}).get("model_settings", {}))
                if resolution:
                    settings.setdefault("resolution", int(resolution))
                req = GenerateRequest(
                    job_id=job_id, image_path=ref_image, out_dir=out_dir,
                    style=style, seed=int(seed), settings=settings,
                )
                gpu = adapter.preferred_gpu
                try:
                    async with self.lease.acquire(
                        gpu, adapter.required_vram_gb,
                        what=f"{model_name}/seed{seed}/res{resolution}",
                    ):
                        await self._set(job_id, f"{model_name}_geometry", 20, gpu)
                        res = await adapter.generate(req)
                except Exception as e:
                    res = type("R", (), {"ok": False, "error": str(e),
                                         "duration_sec": 0.0, "output_path": None,
                                         "model_version": None, "vram_peak_gb": None,
                                         "extra": {}})()

                cid = store.add_candidate(
                    job_id=job_id, model=model_name,
                    model_version=getattr(res, "model_version", None),
                    seed=int(seed),
                    parameters=getattr(res, "extra", {}) or {},
                    gpu=gpu,
                    status="ok" if res.ok else "failed",
                    error=getattr(res, "error", None),
                    output_path=str(res.output_path) if getattr(res, "output_path", None) else None,
                    duration_sec=round(getattr(res, "duration_sec", 0.0), 2),
                    vram_peak_gb=getattr(res, "vram_peak_gb", None),
                )
                entry = {
                    "candidate_id": cid, "model": model_name, "seed": int(seed),
                    "resolution": resolution,
                    "status": "ok" if res.ok else "failed",
                    "output_path": str(res.output_path) if getattr(res, "output_path", None) else None,
                    "error": getattr(res, "error", None),
                    "duration_sec": round(getattr(res, "duration_sec", 0.0), 2),
                }
                results.append(entry)
                stage_log.append(entry)
                await events.emit(job_id, "candidate.completed", entry)
                await adapter.cleanup()

        manifest["stages"]["geometry"] = {"status": "ok", "candidates": stage_log}
        return results

    async def _render(self, job, paths: JobPaths, cands: list[dict],
                      manifest) -> None:
        """全候補を同一のカメラ・照明・露出でレンダリングし、静的解析もかける。

        条件を候補ごとに変えると、評価が「モデルの差」ではなく
        「描画条件の差」を測ってしまう。
        """
        done, failed = [], []
        for c in cands:
            src = c.get("output_path")
            if not src:
                continue
            cid = Path(src).stem
            code, tail = await self._blender(
                "render_views.py",
                ["--input", src, "--outdir", str(paths.renders / cid),
                 "--res", "512", "--samples", "24",
                 "--views", "front,left,right,back,three_quarter"],
                gpu=0, vram_gb=2.0, timeout=900,
                log_path=paths.logs / f"render_{cid}.log",
                what=f"blender/render/{cid}")
            if code != 0:
                failed.append({"candidate": cid, "error": tail[-200:]})
                continue
            mcode, mtail = await self._blender(
                "mesh_report.py",
                ["--input", src, "--out", str(paths.scores / f"{cid}_mesh.json")],
                gpu=None, vram_gb=0, timeout=900,
                log_path=paths.logs / f"mesh_{cid}.log")
            done.append({"candidate": cid, "mesh_report": mcode == 0})
            c["candidate_dir"] = cid
        manifest["stages"]["render"] = {
            "status": "ok" if done else "failed",
            "rendered": done, "failed": failed,
        }

    async def _evaluate(self, job, paths: JobPaths, cands: list[dict],
                        ref_image: Path, manifest) -> dict[str, Any]:
        """DINOv3 でレンダと参照画像を比較し、メッシュ静的解析と合わせて順位付ける。

        **基準はジオメトリの実際の入力**にする。工程2が参照ビューを生成した場合、
        候補はその生成画像から起こしたものなので、元の実写と比べても
        「3Dがちゃんと入力を再現できたか」を測れない。

        後段で顔を差し替える構成（head_merge が走る）では identity と face を
        重みから外す。候補はすべて同じ生成画像由来なので顔で区別がつかず、
        実測でも全候補が identity=0.00 になって選択に寄与していなかった。
        外した分は残りの指標へ比例配分する。
        """
        out = paths.scores / "scores.json"

        will_replace_face = bool((job.get("params") or {}).get("face_image"))
        exclude = "identity,face" if will_replace_face else ""

        args = ["evaluation.score", "--job-dir", str(paths.root),
                "--style", job["style"],
                "--styles-yaml", str(config.CONFIG_DIR / "styles.yaml"),
                "--reference", str(ref_image),
                "--out", str(out)]
        if exclude:
            args += ["--exclude-metrics", exclude]

        code, tail = await self._eval_env(
            args, timeout=1200, vram_gb=4.0,
            log_path=paths.logs / "evaluate.log", what="dinov3/evaluate")

        if code == 0 and out.exists():
            data = json.loads(out.read_text())
            best_id = data.get("best_overall")
            chosen = next((c for c in cands
                           if Path(c.get("output_path", "")).stem == best_id), cands[0])
            manifest["stages"]["evaluate"] = {
                "status": "ok",
                "encoder": data.get("encoder"),
                "best_overall": best_id,
                "best_geometry": data.get("best_geometry"),
                "best_texture": data.get("best_texture"),
                "ranking": [(c["candidate"], c["overall"])
                            for c in data.get("candidates", [])],
                # 何を基準に、どの指標を外して選んだかを残す。
                # 後から「なぜこの候補が選ばれたか」を追えるようにするため。
                "reference": data.get("reference"),
                "excluded_metrics": data.get("excluded_metrics"),
                "weights": data.get("weights"),
                "not_measured": (data.get("candidates") or [{}])[0].get("not_measured"),
            }
            return chosen

        # 評価に失敗しても止めない。選択根拠が無いことを明示して先頭候補を使う。
        log.warning("評価に失敗 (%s): %s", code, tail[-300:])
        manifest["stages"]["evaluate"] = {
            "status": "degraded",
            "reason": f"評価器が失敗したため先頭候補を採用 (exit={code})",
            "detail": tail[-400:],
        }
        return cands[0]

    async def _project_face(self, paths: JobPaths, src: str, photo: str,
                            out: Path, tag: str,
                            frames: tuple[str, ...] = ("head", "full"),
                            bake_size: int | None = None):
        """実写の顔をメッシュのテクスチャへ焼く。(出力パス|None, 情報) を返す。

        整列レンダは**投影と同じスクリプト・同じカメラ**で撮る。別スクリプトで
        撮ると画角がずれて顔が二重像になる（docs/05-measurements.md §16）。

        画角は head を第一候補にする。頭部に寄せるほど実写のテクセル密度が
        上がる（バスト形状なら顔 311px、全身のままだと 61px）。ただし顔の造形が
        乏しいメッシュでは head だと顔検出が通らないので full に落ちる。
        """
        proj_dir = paths.renders / "faceproj"
        warped = paths.references / f"{tag}_warped.png"
        wmeta = paths.references / f"{tag}_warped.json"
        last = ""
        for frame in frames:
            front = proj_dir / f"align_{tag}_{frame}.png"
            code, tail = await self._blender(
                "project_face.py",
                ["--input", src, "--emit-align-render", str(front),
                 "--frame", frame, "--align-res", str(FACEPROJ_RENDER_RES)],
                gpu=0, vram_gb=2.0, timeout=900,
                log_path=paths.logs / f"faceproj_{tag}_render_{frame}.log",
                what=f"blender/faceproj/{tag}/{frame}")
            if code != 0 or not front.exists():
                last = f"整列用レンダに失敗 (frame={frame}, exit={code})"
                continue

            code, tail = await self._eval_env(
                ["preprocessing.face_projection", "--photo", str(photo),
                 "--render", str(front), "--out", str(warped),
                 "--meta", str(wmeta)],
                timeout=900, vram_gb=0.0,
                log_path=paths.logs / f"faceproj_{tag}_align_{frame}.log",
                what="arcface/align")
            if code != 0 or not warped.exists():
                last = f"顔の整列に失敗 (frame={frame}, exit={code})"
                log.warning("%s。次の画角を試します", last)
                continue

            code, tail = await self._blender(
                "project_face.py",
                ["--input", src, "--photo", str(warped), "--out", str(out),
                 "--frame", frame, "--align-res", str(FACEPROJ_RENDER_RES),
                 "--bake-size", str(bake_size or FACEPROJ_BAKE_SIZE),
                 "--report", str(paths.scores / f"faceproj_{tag}.json")],
                gpu=None, vram_gb=0, timeout=1800,
                log_path=paths.logs / f"faceproj_{tag}_bake_{frame}.log")
            if code != 0 or not out.exists():
                last = f"投影・焼き込みに失敗 (frame={frame}, exit={code})"
                continue

            align = json.loads(wmeta.read_text()) if wmeta.exists() else {}
            return out, {"frame": frame, "photo": str(photo),
                         "bake_size": bake_size or FACEPROJ_BAKE_SIZE,
                         "alignment": {k: align.get(k) for k in
                                       ("similarity_scale", "rotation_deg",
                                        "inliers", "photo_face_width_px",
                                        "render_face_width_px")}}
        return None, {"error": last or "顔投影に失敗"}

    async def _head_merge(self, job, paths: JobPaths, best: dict,
                          manifest) -> dict[str, Any]:
        """顔と体を別メッシュで作って接合する（二トラック構成）。

        なぜ分けるか（docs/05-measurements.md §15〜§18）:
        生成した全身Tポーズ画像は顔が 83px しかなく、TRELLIS.2 はそこから
        顔の造形を作れない。一方で実写のポートレートを単体で3D化すると顔は
        よく出るが（ArcFace 0.68）、脚も足も無いのでリギングが破綻する。
        そこで顔は実写から、体は生成Tポーズ画像から作って首で繋ぐ。

        `face_image` が指定されたときだけ走る。指定が無ければ単一メッシュの
        まま進む（従来の挙動）。失敗してもジョブは止めず、体メッシュで続行する。
        """
        face_src = (job.get("params") or {}).get("face_image")
        src = best.get("output_path")
        if not face_src or not Path(face_src).exists():
            manifest["stages"]["head_merge"] = self._skipped(
                "face_image の指定が無いため単一メッシュで続行")
            return best
        if not src:
            manifest["stages"]["head_merge"] = self._skipped("体メッシュがありません")
            return best

        def degraded(reason: str) -> dict[str, Any]:
            log.warning("顔メッシュの接合: %s", reason)
            manifest["stages"]["head_merge"] = {
                "status": "degraded",
                "reason": f"{reason}。体メッシュのまま続行します",
            }
            return best

        jid = job["job_id"]
        await self._set(jid, "head_merge", 74, 0)

        # 1) 実写から頭部＋肩を切り出す（顔検出基準。CPU なので VRAM 不要）
        head_in = paths.input / "head_crop.png"
        code, tail = await self._eval_env(
            ["preprocessing.head_crop", "--input", str(face_src),
             "--out", str(head_in),
             "--meta", str(paths.references / "head_crop.json")],
            timeout=600, vram_gb=0.0,
            log_path=paths.logs / "head_crop.log", what="arcface/head_crop")
        if code != 0 or not head_in.exists():
            return degraded(f"頭部の切り出しに失敗 (exit={code}): {tail[-160:]}")

        # 2) 背景除去。工程1と同じ RMBG-2.0 を使う
        head_rgba = paths.references / "head_rgba.png"
        code, tail = await self._eval_env(
            ["preprocessing.rmbg", "--input", str(head_in),
             "--output", str(head_rgba),
             "--meta", str(paths.references / "head_rgba.json")],
            timeout=900, vram_gb=3.0,
            log_path=paths.logs / "head_preprocess.log", what="rmbg/head")
        if code != 0 or not head_rgba.exists():
            return degraded(f"頭部の背景除去に失敗 (exit={code}): {tail[-160:]}")

        # 3) 顔メッシュを作る。顔だけなので res1024 が通る（全身と違い密度が低い）
        adapter = registry.get("trellis2")
        req = GenerateRequest(
            job_id=jid, image_path=head_rgba, out_dir=paths.candidates / "head",
            style=job["style"], seed=0, settings={"resolution": 1024},
        )
        gpu = adapter.preferred_gpu
        try:
            async with self.lease.acquire(gpu, adapter.required_vram_gb,
                                          what="trellis2/head"):
                res = await adapter.generate(req)
        except Exception as e:  # noqa: BLE001
            res = type("R", (), {"ok": False, "error": str(e),
                                 "output_path": None})()
        if not getattr(res, "ok", False) or not getattr(res, "output_path", None):
            return degraded(f"顔メッシュの生成に失敗: {getattr(res, 'error', '不明')}")

        # 4) 顔メッシュも浮遊片を落とす。生の候補は Blender 上で描画が破綻する
        head_clean = paths.selected / "head_clean.glb"
        code, tail = await self._blender(
            "repair_mesh.py",
            ["--input", str(res.output_path), "--out", str(head_clean),
             "--report", str(paths.scores / "head_repair.json")],
            gpu=None, vram_gb=0, timeout=900,
            log_path=paths.logs / "head_repair.log")
        head_mesh = head_clean if (code == 0 and head_clean.exists()) \
            else Path(res.output_path)

        # 5) **接合の前に**顔メッシュへ実写を投影する。ここはバスト形状なので
        #    画角 head が通り、テクセル密度が最大になる（顔 300px 前後）。
        #    全身に載せてから投影すると顔は 65px しか取れず、実測で
        #    ArcFace 0.59 → 0.34 まで落ちた。
        projected, pinfo = await self._project_face(
            paths, str(head_mesh), str(face_src),
            paths.selected / "head_projected.glb", tag="head",
            bake_size=FACEPROJ_HEAD_BAKE_SIZE)
        if projected is not None:
            head_mesh = projected
        else:
            log.warning("顔メッシュへの投影に失敗: %s。素の顔メッシュで接合します",
                        pinfo.get("error"))

        # 6) 接合
        merged = paths.selected / "merged.glb"
        rep = paths.scores / "head_merge.json"
        code, tail = await self._blender(
            "merge_head.py",
            ["--body", src, "--head", str(head_mesh), "--out", str(merged),
             "--report", str(rep)],
            gpu=None, vram_gb=0, timeout=1800,
            log_path=paths.logs / "head_merge.log")
        if code != 0 or not merged.exists():
            return degraded(f"接合に失敗 (exit={code}): {tail[-160:]}")

        # 7) 接合後にもう一度浮遊片を落とす。
        #    工程7の修復は接合**前**に各メッシュ単体へ掛かるので、そこでは
        #    自分の本体面積が基準になり、体側に残った小片が閾値を下回らない。
        #    接合すると本体がはっきりするので、同じ判定でゴミだけが浮き上がる
        #    （実測: 6片・面の2.03%。閾値 0.0025〜0.006 のどこでも同じ6片が消え、
        #    その間に隙間があるのでピアス等の小物は巻き込まない）。
        cleaned = paths.selected / "merged_clean.glb"
        code, tail = await self._blender(
            "repair_mesh.py",
            ["--input", str(merged), "--out", str(cleaned),
             "--report", str(paths.scores / "merged_repair.json"),
             "--speck-ratio", "0.003"],
            gpu=None, vram_gb=0, timeout=900,
            log_path=paths.logs / "merged_repair.log")
        merged_final = cleaned if (code == 0 and cleaned.exists()) else merged
        if merged_final is merged:
            log.warning("接合後の修復に失敗 (exit=%s)。修復前のメッシュを使います", code)

        data = json.loads(rep.read_text()) if rep.exists() else {}
        mrep = paths.scores / "merged_repair.json"
        manifest["stages"]["head_merge"] = {
            "status": "ok",
            "post_merge_repair": (json.loads(mrep.read_text()).get("totals")
                                  if mrep.exists() else None),
            "face_image": str(face_src),
            "head_mesh": str(head_mesh),
            "face_projection": pinfo,
            "output": str(merged),
            "applied_scale": data.get("applied_scale"),
            "scale_by_span": data.get("scale_by_span"),
            "seam_z": data.get("seam_z"),
            "body_landmarks": data.get("body_landmarks"),
            "head_landmarks": data.get("head_landmarks"),
            "merged": data.get("merged"),
        }
        return {**best, "output_path": str(merged_final)}


    async def _retexture(self, job, paths: JobPaths, best: dict,
                         ref_image: Path, preset: dict, manifest) -> dict[str, Any]:
        """工程6: テクスチャの作り直し。実体は「実写の顔テクスチャ投影」。

        設計時は「別モデルでテクスチャだけ貼り直す」ことを想定していたが、
        trellis-service に /texture が無く実現していない。代わりにここで行うのは
        実写を正面から UV 投影して焼き込むことで、本人性への効果はこちらが
        圧倒的に大きい（docs/05-measurements.md §16）:

          res512 → res1024            ArcFace +0.08
          sampler steps 12 → 30       ArcFace +0.01
          元画像を2.5倍の解像度に      ArcFace +0.003
          実写の顔テクスチャ投影       ArcFace +0.286   ← これ

        二トラック構成（head_merge）が走った場合、顔メッシュ側で既に投影済み
        なのでここは何もしない。あちらは画角 head が通ってテクセル密度が高く、
        全身画角から上書きすると質が落ちる（実測 0.59 → 0.34）。

        失敗してもジョブは止めない。投影なしの候補で続行する。
        """
        src = best.get("output_path")
        if not preset.get("retexture", False) or not src:
            manifest["stages"]["retexture"] = self._skipped(
                f"quality={job['quality']} では再テクスチャを行わない")
            return best

        if manifest["stages"].get("head_merge", {}).get("status") == "ok":
            manifest["stages"]["retexture"] = self._skipped(
                "head_merge が顔メッシュ側で実写を投影済み（二度貼らない）")
            return best

        # 顔に貼るのは params.face_image があればそれ、無ければ入力の背景除去済み。
        face_src = (job.get("params") or {}).get("face_image")
        photo = Path(face_src) if face_src and Path(face_src).exists() else ref_image

        await self._set(job["job_id"], "retexture", 78, 0)
        out = paths.selected / "face_projected.glb"
        projected, pinfo = await self._project_face(
            paths, src, str(photo), out, tag="body")
        if projected is None:
            log.warning("顔テクスチャ投影: %s", pinfo.get("error"))
            manifest["stages"]["retexture"] = {
                "status": "degraded", "method": "face_projection",
                "reason": f"{pinfo.get('error')}。投影なしのメッシュで続行します",
            }
            return best

        manifest["stages"]["retexture"] = {
            "status": "ok",
            "method": "face_projection",
            "output": str(out),
            "photo_source": "params.face_image" if photo != ref_image else "入力画像",
            **pinfo,
            "limitations": [
                "正面からの単一投影のため側面には実写が乗らない。"
                "真横は投影なしより本人性が下がる（3/4 までは実用範囲）",
                "横顔の実写素材があれば多視点投影に拡張できる",
            ],
        }
        return {**best, "output_path": str(out)}


    async def _mesh_repair(self, job, paths: JobPaths, best: dict,
                           preset: dict, manifest) -> dict[str, Any]:
        """浮遊片の除去のみ。Decimation・Remesh・穴埋め・頂点マージは行わない。

        実測で、頂点マージは UV シームを壊してテクスチャに黒斑を出し、
        除去閾値を上げすぎると表面に穴が開いた（docs/05-measurements.md）。
        """
        src = best.get("output_path")
        if not src or preset.get("mesh_repair") == "basic":
            manifest["stages"]["mesh_repair"] = self._skipped(
                f"quality={job['quality']} では詳細な修復を行わない")
            return best

        out = paths.selected / "clean.glb"
        rep = paths.scores / "repair.json"
        code, tail = await self._blender(
            "repair_mesh.py",
            ["--input", src, "--out", str(out), "--report", str(rep)],
            gpu=None, vram_gb=0, timeout=900,
            log_path=paths.logs / "repair.log")

        if code == 0 and out.exists():
            data = json.loads(rep.read_text()) if rep.exists() else {}
            manifest["stages"]["mesh_repair"] = {
                "status": data.get("status", "ok"),
                "face_loss_ratio": data.get("face_loss_ratio"),
                "totals": data.get("totals"),
                "warning": data.get("warning"),
                "note": data.get("note"),
            }
            # 削りすぎと判定された場合は修復前を使う（穴を作るより無修復が良い）
            if data.get("status") == "excessive_removal":
                log.warning("修復が過剰なため修復前のメッシュを使用します")
                return best
            return {**best, "output_path": str(out)}

        log.warning("mesh修復に失敗 (%s): %s", code, tail[-200:])
        manifest["stages"]["mesh_repair"] = {
            "status": "degraded", "reason": f"失敗のため修復前を使用 (exit={code})"}
        return best

    async def _rigging(self, job, paths: JobPaths, best: dict,
                       manifest) -> Path | None:
        """テンプレート転写による自動リギング。

        本命は UniRig だが 13GB 必要で未導入。image-to-3D の出力は T ポーズに
        なるため、既知の VRM Humanoid アーマチュアをスケールして被せ、
        Blender の自動ウェイトで結合する。UniRig 導入時にここを差し替える。
        """
        src = best.get("output_path")
        template = config.env("RIG_TEMPLATE_VRM", "")
        if not src:
            manifest["stages"]["rigging"] = self._skipped("入力メッシュがありません")
            return None
        if not template or not Path(template).exists():
            manifest["stages"]["rigging"] = self._skipped(
                "リグ用テンプレート VRM が未設定です（.env の RIG_TEMPLATE_VRM）。"
                "UniRig 導入後はテンプレート不要になります")
            return None

        out = paths.rigged / "rigged.vrm"
        rep = paths.scores / "rig.json"
        code, tail = await self._blender(
            "autorig.py",
            ["--input", src, "--template", template,
             "--out", str(out), "--report", str(rep), "--height", "1.6"],
            gpu=None, vram_gb=0, timeout=1800,
            log_path=paths.logs / "rigging.log")

        if code == 0 and out.exists():
            data = json.loads(rep.read_text()) if rep.exists() else {}
            manifest["stages"]["rigging"] = {
                "status": "ok", "method": "template_transfer",
                "template": template,
                "bones": data.get("template_bones"),
                "bind": (data.get("bind") or {}).get("bind_method"),
                "normalize": data.get("normalize"),
                "limitations": [
                    "骨の位置は身長基準の相似変形で決めるため体型差でずれる",
                    "指の骨はテンプレートのまま",
                    "spring bone（髪・スカートの揺れもの）は未設定",
                ],
            }
            return out

        log.warning("リギングに失敗 (%s): %s", code, tail[-300:])
        manifest["stages"]["rigging"] = {
            "status": "failed", "reason": f"exit={code}", "detail": tail[-400:]}
        return None

    async def _qa(self, paths: JobPaths, vrm: Path, manifest) -> None:
        """成果物そのものを点検する。工程10の検証では捕まらない欠陥を拾う。

        `export_vrm.py` の検証は「必須ボーンが割り当たっているか」「スキンが
        存在するか」までしか見ない。**骨を回したときにどう動くか**と
        **浮遊片が残っていないか**は別の話で、実際そこを見ていなかったために:

          - 1頂点もスキンされていない VRM を出し続けた（§28）
          - 首を振るとジャケットが裂けた（§29）
          - 体側の浮遊片が残った（§33）

        いずれも検証は通っていた。ここで毎回自動的に確かめる。
        失敗しても成果物は出す（点検であって門番ではない）。
        """
        qa: dict[str, Any] = {"status": "ok", "checks": {}}

        # 1) ポーズを当てて実際に変形するか
        prep = paths.scores / "pose_check.json"
        code, tail = await self._blender(
            "pose_check.py",
            ["--input", str(vrm), "--outdir", str(paths.renders / "pose"),
             "--poses", "arms,neck,all", "--res", "600",
             "--report", str(prep)],
            gpu=None, vram_gb=0, timeout=1800,
            log_path=paths.logs / "qa_pose.log")
        if code == 0 and prep.exists():
            d = json.loads(prep.read_text())
            qa["checks"]["pose"] = {
                "all_deformed": d.get("all_deformed"),
                "poses": [{"pose": r["pose"], "deformed": r["deformed"],
                           "max_delta": r["max_delta"]} for r in d.get("poses", [])],
            }
            if not d.get("all_deformed"):
                qa["status"] = "warn"
                qa.setdefault("warnings", []).append(
                    "骨を回してもメッシュが動かないポーズがある。"
                    "スキニングが効いていない疑い（docs/05-measurements.md §28）")
        else:
            qa["checks"]["pose"] = {"error": f"exit={code}: {tail[-160:]}"}
            qa["status"] = "warn"

        # 2) 最終メッシュの静的解析。浮遊片が残っていないか
        mrep = paths.scores / "final_mesh.json"
        code, tail = await self._blender(
            "mesh_report.py", ["--input", str(vrm), "--out", str(mrep)],
            gpu=None, vram_gb=0, timeout=900,
            log_path=paths.logs / "qa_mesh.log")
        if code == 0 and mrep.exists():
            t = json.loads(mrep.read_text()).get("total", {})
            qa["checks"]["mesh"] = {k: t.get(k) for k in
                                    ("triangles", "vertices", "disconnected_shells",
                                     "non_manifold_edges", "watertight")}

        # 3) 顔投影の整列が確からしかったか。5点中4点以下だと相似変換が
        #    不安定になり、投影が滲む（実測: inliers 4 で鼻と口が滲んだ）。
        for tag in ("head", "body"):
            wm = paths.references / f"{tag}_warped.json"
            if not wm.exists():
                continue
            a = json.loads(wm.read_text())
            qa["checks"][f"face_align_{tag}"] = {
                "inliers": a.get("inliers"),
                "similarity_scale": a.get("similarity_scale"),
                "photo_face_width_px": a.get("photo_face_width_px"),
            }
            if (a.get("inliers") or 0) < 5:
                qa["status"] = "warn"
                qa.setdefault("warnings", []).append(
                    f"顔の整列が {a.get('inliers')}/5 点でしか合っていない（{tag}）。"
                    "投影が滲む可能性がある")

        manifest["stages"]["qa"] = qa
        if qa.get("warnings"):
            for w in qa["warnings"]:
                log.warning("QA: %s", w)

    async def _finalize(self, job, paths: JobPaths, best: dict,
                        rigged: Path | None, manifest) -> Path | None:
        """VRM 化は Phase 4。現時点では選択された GLB を output に置く。"""
        want_vrm = job.get("output_format", "vrm") == "vrm"

        if rigged and rigged.exists() and want_vrm:
            dst = paths.output / "character.vrm"
            shutil.copy2(rigged, dst)
            manifest["stages"]["vrm_export"] = {"status": "ok", "output": str(dst)}

            # リグテンプレート（VRoid サンプル）から同乗してくる残骸を落とす。
            # autorig はテンプレートのアーマチュアを取り込んでそのまま書き出すので、
            # meta（作者・ライセンス条項・サムネイル）・眼球ボーン・spring bone が
            # テンプレートのまま残る。詳細は blender/vrm_finalize.py の冒頭。
            # 検証より先に走らせて、validation.json が実際に出荷される中身を写すようにする。
            frep = paths.output / "finalize.json"
            fcode, ftail = await _run(
                [sys.executable, str(config.ROOT / "blender" / "vrm_finalize.py"),
                 "--input", str(dst), "--out", str(dst), "--report", str(frep),
                 "--name", job.get("name") or "Genesis3D Character",
                 "--author", os.environ.get("VRM_AUTHOR", "Genesis3D")],
                timeout=300, log_path=paths.logs / "vrm_finalize.log")
            if fcode == 0 and frep.exists():
                f = json.loads(frep.read_text())
                manifest["stages"]["vrm_export"]["finalize"] = {
                    "meta_name": f["meta"]["after"]["name"],
                    "eye_bones_removed": f.get("eye_bones", {}).get("removed"),
                    "spring_bones_removed": f.get("spring_bones", {}).get("removed"),
                    "stub_morph": f.get("stub_morph", {}).get("added"),
                    "bytes": f"{f['bytes_before']} → {f['bytes_after']}",
                }
            else:
                # 仕上げに失敗しても成果物は出す。何が残っているかは記録する。
                manifest["stages"]["vrm_export"]["finalize"] = {
                    "status": "failed", "reason": f"exit={fcode}",
                    "detail": ftail[-300:]}
                log.warning("vrm_finalize に失敗（テンプレート由来の meta / "
                            "眼球ボーン / spring bone が残っている）: %s", ftail[-200:])

            # 工程10: Humanoid 必須ボーン・スケール・原点を検証する
            vrep = paths.output / "validation.json"
            code, tail = await self._blender(
                "export_vrm.py",
                # --fix-double-sided: 髪のような薄いシェルは裏面カリングで
                # 透ける。glTF の doubleSided は省略時 false なので、検証と同時に
                # 成果物へ立てる（docs/05-measurements.md §31）。
                ["--input", str(dst), "--out", "/dev/null",
                 "--report", str(vrep), "--dry-run", "--fix-double-sided"],
                gpu=None, vram_gb=0, timeout=600,
                log_path=paths.logs / "validate.log")
            if code == 0 and vrep.exists():
                v = json.loads(vrep.read_text())
                manifest["stages"]["validate"] = {
                    "status": "ok" if not v.get("missing_required_bones") else "failed",
                    "missing_required_bones": v.get("missing_required_bones"),
                    "optional_present": v.get("optional_present"),
                    "height_m": v.get("height_m"),
                    "height_plausible": v.get("height_plausible"),
                    "origin_at_feet": v.get("origin_at_feet"),
                    "bone_count": v.get("bone_count"),
                    "skinned_meshes": v.get("skinned_meshes"),
                    "double_sided": v.get("double_sided"),
                }
            else:
                manifest["stages"]["validate"] = {
                    "status": "failed", "reason": f"exit={code}",
                    "detail": tail[-300:]}

            # 成果物そのものを点検する。門番ではないので結果に関わらず出す。
            try:
                await self._qa(paths, dst, manifest)
            except Exception as e:  # noqa: BLE001
                log.warning("QA に失敗: %s", e)
                manifest["stages"]["qa"] = {"status": "failed", "error": str(e)}
            return dst

        # リギングまで到達できなかった場合。中身が VRM でないものに .vrm の
        # 拡張子を付けると後段の検証やクライアントを誤らせるので、GLB のまま出す。
        src = best.get("output_path")
        if not src:
            manifest["stages"]["vrm_export"] = self._skipped("出力候補がありません")
            return None
        dst = paths.output / "character.glb"
        shutil.copy2(Path(src), dst)
        manifest["stages"]["vrm_export"] = {
            "status": "degraded" if want_vrm else "ok",
            "reason": ("リギングに到達できなかったため GLB を出力している"
                       if want_vrm else None),
            "output": str(dst),
        }
        manifest["stages"]["validate"] = self._skipped("VRM が生成されていません")
        return dst
