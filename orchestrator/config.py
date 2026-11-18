"""設定の読み込み。configs/*.yaml と .env を単一の入口にまとめる。"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def _load_env() -> None:
    """.env を環境変数へ読み込む（既存の環境変数を上書きしない）。"""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # 行末コメントを除去する。bash は `VAR=v  # c` を v と解釈するので
        # Python 側もそれに合わせないと値がずれる。引用符内の # は残す。
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()
        # ~ をホームに展開しておく（BLENDER_BIN など）
        if value.startswith("~"):
            value = os.path.expanduser(value)
        os.environ.setdefault(key, value)


_load_env()


@lru_cache(maxsize=None)
def _yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def gpu_config() -> dict[str, Any]:
    return _yaml("gpu.yaml")


def models_config() -> dict[str, Any]:
    return _yaml("models.yaml").get("models", {})


def styles_config() -> dict[str, Any]:
    return _yaml("styles.yaml")


def gateway_config() -> dict[str, Any]:
    return _yaml("gateway.yaml")


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def data_dir() -> Path:
    d = Path(env("GENESIS_DATA_DIR", str(ROOT / "jobs")) or (ROOT / "jobs"))
    if not d.is_absolute():
        d = ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir() -> Path:
    d = Path(env("GENESIS_LOG_DIR", str(ROOT / "logs")) or (ROOT / "logs"))
    if not d.is_absolute():
        d = ROOT / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def blender_bin() -> str:
    return os.path.expanduser(env("BLENDER_BIN", "~/opt/blender/blender") or "")


def style_names() -> list[str]:
    return list(styles_config().get("styles", {}).keys())


def quality_preset(quality: str) -> dict[str, Any]:
    presets = styles_config().get("quality_presets", {})
    return presets.get(quality, presets.get("high", {}))
