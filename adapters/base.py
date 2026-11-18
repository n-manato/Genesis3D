"""ModelAdapter — 3Dモデル実装を Orchestrator から分離する境界（要件§28）。

Orchestrator は具体的なモデルを一切知らない。新しいモデルを足すときは
ここを継承したクラスを1つ書き、configs/models.yaml に登録するだけでよい。

2つの実行形態がある:

  http_service  既に別プロセスで動いているサービスを叩く（TRELLIS.2 = :8811）。
                Genesis3D は VRAM を消費しない。

  subprocess    `conda run -n <env> python -m ...` で独立プロセスとして起動する。
                CUDA OOM や segfault が Orchestrator を巻き込まず（§30）、
                プロセス終了で VRAM が完全に解放される（§31）。
"""
from __future__ import annotations

import abc
import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("genesis.adapter")


@dataclass
class GenerateRequest:
    """1候補ぶんの生成要求。"""
    job_id: str
    image_path: Path
    out_dir: Path
    style: str
    seed: int = 0
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerateResult:
    ok: bool
    output_path: Path | None = None
    duration_sec: float = 0.0
    model_version: str | None = None
    vram_peak_gb: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(abc.ABC):
    """すべてのモデルアダプタの基底。"""

    name: str = "base"

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.display_name = spec.get("display_name", self.name)
        self.kind = spec.get("kind", "subprocess")
        self.preferred_gpu = int(spec.get("preferred_gpu", 0))
        self.required_vram_gb = float(spec.get("required_vram_gb", 0.0))
        self.multi_gpu_supported = bool(spec.get("multi_gpu_supported", False))
        self.installed = bool(spec.get("installed", False))
        self._loaded = False

    # -- ライフサイクル ------------------------------------------------
    async def load(self) -> None:
        """必要なら事前準備。http_service 型は疎通確認のみ。"""
        self._loaded = True

    @abc.abstractmethod
    async def generate(self, req: GenerateRequest) -> GenerateResult:
        ...

    async def get_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "kind": self.kind,
            "installed": self.installed,
            "loaded": self._loaded,
            "preferred_gpu": self.preferred_gpu,
            "required_vram_gb": self.required_vram_gb,
        }

    async def cleanup(self) -> None:
        """subprocess 型はプロセス終了で解放済み。http_service 型は何もしない。"""
        self._loaded = False

    # -- サブプロセス実行の共通処理 ------------------------------------
    async def run_subprocess(
        self, argv: list[str], *, cwd: Path | None = None,
        env: dict[str, str] | None = None, gpu: int | None = None,
        timeout: float = 3600.0, log_path: Path | None = None,
    ) -> tuple[int, str]:
        """独立プロセスとして実行し、(returncode, 末尾ログ) を返す。

        CUDA_VISIBLE_DEVICES で GPU を1枚に固定する。呼び出し側が OOM で
        落ちても Orchestrator は生き残る。
        """
        proc_env = dict(os.environ)
        if env:
            proc_env.update(env)
        if gpu is not None:
            proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # 重みDL後は完全オフライン（§32）
        proc_env.setdefault("HF_HUB_OFFLINE", "1")

        log.info("subprocess: %s (gpu=%s)", " ".join(argv[:6]), gpu)
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd) if cwd else None, env=proc_env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"サブプロセスが {timeout:.0f}s でタイムアウト")

        text = (out or b"").decode("utf-8", "replace")
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(text)
        return proc.returncode or 0, text[-4000:]

    def conda_argv(self, module_args: list[str]) -> list[str]:
        """`conda run -n <env> python -m ...` を組み立てる。"""
        env_name = self.spec.get("conda_env") or self.name
        conda = shutil.which("conda") or os.path.expanduser("~/miniconda3/bin/conda")
        return [conda, "run", "-n", env_name, "--no-capture-output",
                "python", *module_args]


class NotInstalledAdapter(ModelAdapter):
    """未導入モデルのプレースホルダ。ジョブを止めず、この候補だけ失敗させる。"""

    def __init__(self, name: str, spec: dict[str, Any]) -> None:
        super().__init__(spec)
        self.name = name

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            ok=False,
            error=f"{self.display_name} は未導入です（configs/models.yaml: installed=false）",
        )
