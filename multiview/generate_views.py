"""Tポーズ全身の参照ビューを生成する（工程2）。

バストアップ写真からは脚も足も推定できず、リギングが破綻する
（docs/05-measurements.md §4）。ここで全身Tポーズの参照画像を作り、
そこから3D化する。

**顔の本人性はここでは追わない。** 実写の顔テクスチャ投影（工程6）が
ArcFace を 0.38 → 0.68 まで持ち上げる一方、全身像では顔が画面の 5% 程度しか
占めず、生成された顔は最終的に上書きされる（同 §16）。ここで欲しいのは
**正しい人体比率・Tポーズ・脚と足の存在**と衣装。

TRELLIS.2 は画像を1枚しか受け取らない（`run(image: Image.Image)`）ので、
複数シードを振るのは「融合するため」ではなく「最良の1枚を選ぶため」。

  python -m multiview.generate_views --pose refs/pose_front.png \
      --outdir refs/ --seeds 4 --prompt-file prompts.txt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BASE_MODEL = "SG161222/RealVisXL_V5.0"
CONTROLNET = "xinsir/controlnet-openpose-sdxl-1.0"

# 背景は無地の明るいグレー。RMBG-2.0 は暗背景＋暗い服の分離が苦手なので
# （実測: 被写体Aのポートレートがまさにこれ）、生成側で分離しやすくしておく。
DEFAULT_PROMPT = (
    # CLIP は 77 トークンで打ち切る。長く書くと末尾が黙って捨てられるので、
    # 効かせたい語（T-pose / head to feet / 背景）を必ず前半に置く。
    "full body studio photo, T-pose, arms stretched straight out sideways, "
    "standing upright, feet flat on floor, head to feet fully visible, "
    "plain light grey seamless background, "
    "East Asian woman in her late fifties, short dark brown bob, "
    "red blazer, black trousers, black flat shoes, "
    "soft even studio lighting, photorealistic, sharp focus"
)
DEFAULT_NEGATIVE = (
    "cropped, cut off, out of frame, close-up, portrait, bust, headshot, "
    "sitting, walking, arms down, hands on hips, crossed arms, "
    "deformed hands, extra limbs, extra fingers, missing legs, missing feet, "
    "low resolution, blurry, jpeg artifacts, watermark, text, "
    "cartoon, anime, illustration, 3d render, cgi"
)


def build_pipeline(device: str, offload: bool):
    import torch
    from diffusers import (ControlNetModel, StableDiffusionXLControlNetPipeline,
                           EulerAncestralDiscreteScheduler)

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if offload:
        # GPU0 は他サービスが常時 9GB 使っている（docs/05 §12）。
        # 足りないときはステージ単位で退避させる。
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    # VAE スライシングの API は diffusers 0.4x で pipe から vae へ移った。
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    elif hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True, help="OpenPose 骨格画像")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--controlnet-scale", type=float, default=0.85,
                    help="上げると姿勢に忠実だが人物が硬くなる")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--offload", action="store_true",
                    help="VRAM が足りないとき CPU へ退避しながら回す")
    ap.add_argument("--tag", default="front")
    args = ap.parse_args()

    import torch
    from PIL import Image

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pose = Image.open(args.pose).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = build_pipeline(device, args.offload)

    written = []
    for i in range(args.seeds):
        seed = args.seed_start + i
        gen = torch.Generator(device="cpu").manual_seed(seed)
        img = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative,
            image=pose,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.controlnet_scale,
            generator=gen,
            width=pose.width, height=pose.height,
        ).images[0]
        p = outdir / f"view_{args.tag}_seed{seed}.png"
        img.save(p)
        written.append(str(p))
        print(f"  seed {seed} -> {p.name}", flush=True)

    meta = {
        "base_model": BASE_MODEL,
        "controlnet": CONTROLNET,
        "pose": args.pose,
        "tag": args.tag,
        "steps": args.steps,
        "guidance": args.guidance,
        "controlnet_conditioning_scale": args.controlnet_scale,
        "seeds": [args.seed_start + i for i in range(args.seeds)],
        "prompt": args.prompt,
        "negative": args.negative,
        "size": [pose.width, pose.height],
        "written": written,
    }
    (outdir / f"view_meta_{args.tag}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    print("GENVIEWS_OK " + json.dumps(
        {"tag": args.tag, "count": len(written)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
