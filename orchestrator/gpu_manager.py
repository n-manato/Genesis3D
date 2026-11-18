"""NVML による GPU 監視と、VRAM 不足時にステージを開始させないリース機構。

方針（configs/gpu.yaml と対応）:
  GPU0 = Genesis3D 専有。既存サイドカーと共有するため reserve を必ず残す。
  GPU1 = TRELLIS.2 が pinned 常駐。新規モデルはロードしない（:8811 経由のみ）。

VRAM は他プロセス(Gateway配下のサイドカー等)も食うので、確保予約ではなく
「開始直前に実測して足りなければ待つ」方式にしている。これが 1台共有環境で
最も OOM を起こしにくい。
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from . import config

log = logging.getLogger("genesis.gpu")

try:
    import pynvml
    _NVML_OK = True
except Exception:  # pragma: no cover - NVML が無い環境
    pynvml = None  # type: ignore
    _NVML_OK = False

_init_done = False


def _ensure_nvml() -> bool:
    global _init_done
    if not _NVML_OK:
        return False
    if not _init_done:
        try:
            pynvml.nvmlInit()
            _init_done = True
        except Exception as e:
            log.warning("NVML init failed: %s", e)
            return False
    return True


@dataclass
class GpuInfo:
    index: int
    name: str
    total_gb: float
    used_gb: float
    free_gb: float
    utilization_pct: int
    temperature_c: int
    role: str
    reserve_gb: float
    load_new_models: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_gb": round(self.total_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "usable_gb": round(max(0.0, self.free_gb - self.reserve_gb), 2),
            "utilization_pct": self.utilization_pct,
            "temperature_c": self.temperature_c,
            "role": self.role,
            "reserve_gb": self.reserve_gb,
            "load_new_models": self.load_new_models,
        }


def _device_policy(index: int) -> dict[str, Any]:
    for d in config.gpu_config().get("devices", []):
        if d.get("index") == index:
            return d
    return {}


def snapshot() -> list[GpuInfo]:
    """nvidia-smi / NVML の実測値。取得できない場合は空リスト。"""
    if not _ensure_nvml():
        return []
    out: list[GpuInfo] = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                util = 0
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            pol = _device_policy(i)
            gb = 1024 ** 3
            out.append(GpuInfo(
                index=i, name=name,
                total_gb=mem.total / gb, used_gb=mem.used / gb, free_gb=mem.free / gb,
                utilization_pct=int(util), temperature_c=int(temp),
                role=pol.get("role", "unknown"),
                reserve_gb=float(pol.get("reserve_gb", 2.0)),
                load_new_models=bool(pol.get("load_new_models", True)),
            ))
    except Exception as e:
        log.warning("NVML snapshot failed: %s", e)
    return out


def report() -> dict[str, Any]:
    gpus = snapshot()
    return {
        "source": "nvml" if gpus else "unavailable",
        "count": len(gpus),
        "gpus": [g.as_dict() for g in gpus],
    }


def usable_gb(index: int) -> float | None:
    """予約分を差し引いた、いま実際に使える VRAM。"""
    for g in snapshot():
        if g.index == index:
            return max(0.0, g.free_gb - g.reserve_gb)
    return None


def can_load(index: int, required_gb: float) -> tuple[bool, str]:
    """そのGPUに required_gb をロードしてよいか。"""
    pol = _device_policy(index)
    if required_gb > 0 and not pol.get("load_new_models", True):
        return False, (f"GPU{index} は load_new_models=false "
                       f"(role={pol.get('role')})。HTTP経由のサービスのみ利用可")
    avail = usable_gb(index)
    if avail is None:
        # NVML が使えない場合は素通しにする（開発機での動作を止めない）
        return True, "NVML 利用不可のため事前チェックをスキップ"
    if avail + 1e-6 < required_gb:
        return False, (f"GPU{index} の空き {avail:.1f}GiB < 必要 {required_gb:.1f}GiB"
                       f"（予約 {pol.get('reserve_gb', 2.0)}GiB を除く）")
    return True, f"GPU{index} 空き {avail:.1f}GiB / 必要 {required_gb:.1f}GiB"


class GpuLease:
    """GPUステージの逐次化 + VRAM 事前チェック。

    実際の VRAM 確保はモデルのサブプロセスが行うため、ここでやるのは
    (1) 同一GPUで複数の重いステージを同時に走らせない
    (2) 空きが足りないうちは起動させない
    の2点。プロセス終了で VRAM は完全に解放される（§31）。
    """

    def __init__(self) -> None:
        pol = config.gpu_config().get("policy", {})
        self._preflight = bool(pol.get("preflight_vram_check", True))
        self._timeout = float(pol.get("wait_timeout_sec", 900))
        self._poll = float(pol.get("poll_interval_sec", 5))
        limit = int(pol.get("max_concurrent_gpu_stages", 1))
        self._sem = asyncio.Semaphore(limit)

    @contextlib.asynccontextmanager
    async def acquire(self, index: int, required_gb: float, *, what: str = ""):
        async with self._sem:
            if self._preflight and required_gb > 0:
                deadline = asyncio.get_event_loop().time() + self._timeout
                while True:
                    ok, why = can_load(index, required_gb)
                    if ok:
                        log.info("GPU%d 確保 (%s): %s", index, what, why)
                        break
                    if asyncio.get_event_loop().time() >= deadline:
                        raise RuntimeError(
                            f"VRAM 待ちタイムアウト ({self._timeout:.0f}s): {why}")
                    log.info("GPU%d 待機 (%s): %s", index, what, why)
                    await asyncio.sleep(self._poll)
            yield index
