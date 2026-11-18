"""configs/models.yaml から ModelAdapter を組み立てる。"""
from __future__ import annotations

import logging
from typing import Any

from orchestrator import config

from .base import ModelAdapter, NotInstalledAdapter

log = logging.getLogger("genesis.registry")

_CACHE: dict[str, ModelAdapter] = {}


def _build(name: str, spec: dict[str, Any]) -> ModelAdapter:
    adapter_key = spec.get("adapter", name)
    if not spec.get("installed", False):
        return NotInstalledAdapter(name, spec)
    if adapter_key == "trellis":
        from .trellis.adapter import TrellisAdapter
        return TrellisAdapter(spec)
    # Phase 4/5 で charactergen / unirig / pshuman を追加する
    log.warning("adapter '%s' の実装がまだありません", adapter_key)
    return NotInstalledAdapter(name, spec)


def get(name: str) -> ModelAdapter:
    if name not in _CACHE:
        spec = config.models_config().get(name, {})
        _CACHE[name] = _build(name, spec)
    return _CACHE[name]


def all_names() -> list[str]:
    return list(config.models_config().keys())


def for_style(style: str, *, purpose: str = "geometry") -> list[str]:
    """そのスタイル向けの、導入済みモデル名を優先順で返す。"""
    styles = config.styles_config().get("styles", {})
    preferred = styles.get(style, {}).get("preferred_models", [])
    models = config.models_config()

    out = [n for n in preferred
           if models.get(n, {}).get("installed")
           and purpose in (models.get(n, {}).get("purpose") or [])]
    # preferred に無いが条件を満たすものを後ろに足す
    for n, spec in models.items():
        if n in out or not spec.get("installed"):
            continue
        if style in (spec.get("styles") or []) and purpose in (spec.get("purpose") or []):
            out.append(n)
    return out
