#!/usr/bin/env python3
"""TRELLIS.2 を**常駐させずに**1回だけ回す。画像1枚 → テクスチャ付き GLB。

既存の `adapters/trellis/adapter.py` は `127.0.0.1:8811` の常駐サービスを叩く。
それは Gateway の Model Runtime から起動する運用で、17GB を占有し続ける。

こちらはジョブごとにプロセスを起こして終わらせる。**プロセス終了で VRAM が
完全解放される**ので常駐が要らない。リポジトリの設計上の約束
（「各モデルは独立サブプロセス。... プロセス終了で VRAM が完全解放される」）に沿う。

代償は重みの読み込み時間（16GB、30〜60秒）。ジョブ1本あたりならこれで足りる。

**このスクリプトは `trellis2` conda env で動かすこと。** TRELLIS.2 は
flash-attn / nvdiffrast / o-voxel / flexgemm をコンパイル済みで持つ専用 env が要り、
Genesis3D の env からは import できない（one model = one conda env）。

    ~/miniconda3/envs/trellis2/bin/python adapters/trellis/oneshot.py \\
        --image head_rgba.png --out head.glb --resolution 1024 --gpu 1

解像度は 1024 まで。**1536 は 24GB カードで CuMesh の remesh 中に OOM し、
失敗した CUDA 命令がプロセスを汚染する**（sidecar 側の実測コメント）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 既定のサイドカー位置。ここに microsoft/TRELLIS.2 のクローンと runner がある。
SIDECAR = Path(os.environ.get(
    "TRELLIS_SERVICE_DIR",
    "/home/ai-ccore/Documents/DEV/AICCORE-Nexus/trellis-service"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="RGBA 推奨（背景除去済み）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolution", default="1024", choices=("512", "1024"),
                    help="1536 は 24GB では OOM する。既定 1024")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--decimation-target", type=int, default=1_000_000)
    ap.add_argument("--gpu", default=None,
                    help="使う物理GPU。既定は環境の CUDA_VISIBLE_DEVICES、無ければ 1")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    # CUDA を触る前に決めること。import 後では効かない。
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        # GPU1 が既定。GPU0 は Nexus のサイドカーが 9GB 使っている。
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("TRELLIS_REPO", str(SIDECAR / "TRELLIS.2"))
    os.environ.setdefault("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")

    if not (SIDECAR / "trellis_runner.py").exists():
        raise SystemExit(f"trellis_runner.py が無い: {SIDECAR}"
                         f"（TRELLIS_SERVICE_DIR で指定できる）")
    sys.path.insert(0, str(SIDECAR))

    from trellis_runner import TrellisRunner  # noqa: E402

    t0 = time.time()
    runner = TrellisRunner()
    runner.load()
    t_load = time.time() - t0
    if not runner.ready:
        # スタブモードは「重みが無い」の意味。ここで落としておかないと
        # generate() が NotImplementedError を投げるだけで原因が伝わらない。
        raise SystemExit("TRELLIS.2 の重みが入っていない（runner が stub モード）")

    t1 = time.time()
    glb = runner.generate(
        args.image.read_bytes(),
        resolution=args.resolution,
        seed=args.seed,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
    )
    t_gen = time.time() - t1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(glb)

    rep = {"image": str(args.image), "out": str(args.out),
           "resolution": args.resolution, "seed": args.seed,
           "texture_size": args.texture_size,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "load_sec": round(t_load, 1), "generate_sec": round(t_gen, 1),
           "size_mb": round(len(glb) / 1048576, 1),
           "gpu_info": runner.gpu_info() if hasattr(runner, "gpu_info") else None}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print("TRELLIS_ONESHOT " + json.dumps(rep, ensure_ascii=False))
    print(f"読み込み {t_load:.1f}s + 生成 {t_gen:.1f}s = {t_load + t_gen:.1f}s  "
          f"{rep['size_mb']}MB → {args.out}")


if __name__ == "__main__":
    main()
