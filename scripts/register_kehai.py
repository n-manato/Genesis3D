#!/usr/bin/env python3
"""できあがったアバターを KEHAI に登録する。

KEHAI 側で必要なのは三つ:

  1. `Live2D Characters/vrm/<name>.glb`（または `.vrm`）にモデルを置く
  2. `model_dict.json` にエントリを足す（ビューアのモデル一覧）
  3. `characters/<name>.yaml` を書く（人格・声・履歴の設定）

**`scripts/check_kehai_model.py` を先に通し、落ちたら登録しない。** 通らないものを
入れると、コンソールに `no mouth-open morph found` のような警告が出るだけで
静かに動かない状態になる。

`emotionMap` に注意。`vrm-expression.ts` は語彙が arkit のとき ARKIT_PRESETS で
表情を作るが、**`emotionMap` に無い感情は呼ばれない**。ARKit 52 を持つモデルに
`{"neutral": 0}` だけ書くと、モーフはあるのに表情が動かない
（実際に `subject_a_v2` がその状態だった）。語彙が arkit なら全14感情を入れる。

使い方:
    python3 scripts/register_kehai.py --model <final.glb> --name subject_a_game \\
        --display "Subject A（ゲーム体）" --conf-uid subject_a_game_001 \\
        --persona prompts/personas/subject_a_en.txt \\
        --voice VQ5VvKYXlRGiLCzb4owE [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parent.parent

# subject_a（ARKit 52 を持つ実績のあるモデル）と同じ14感情。
EMOTION_MAP_ARKIT = {
    "neutral": 0, "joy": 1, "joy_smile": 2, "joy_big": 3,
    "sadness": 4, "sadness_worry": 5, "anger": 6, "surprise": 7,
    "surprise_shock": 8, "fear": 9, "disgust": 10, "thinking": 11,
    "shy": 12, "smug": 13,
}


def check(model: Path) -> dict:
    r = subprocess.run([sys.executable, str(REPO / "scripts/check_kehai_model.py"),
                        str(model)], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("KEHAI_CHECK ")), None)
    if line is None:
        raise SystemExit(f"検査器が結果を返さなかった:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return json.loads(line[len("KEHAI_CHECK "):])


def upsert_model_dict(path: Path, entry: dict) -> str:
    """name をキーに冪等に差し替える。並び順は変えない。"""
    data = json.loads(path.read_text())
    for i, e in enumerate(data):
        if e.get("name") == entry["name"]:
            data[i] = entry
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            return "更新"
    data.append(entry)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return "追加"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--name", required=True, help="KEHAI 内での識別子（ファイル名にもなる）")
    ap.add_argument("--display", required=True)
    ap.add_argument("--conf-uid", required=True,
                    help="**他と重複させないこと。** 重複すると chat_history が混ざる")
    ap.add_argument("--character-name", default=None)
    ap.add_argument("--persona", default=None,
                    help="省略すると conf.yaml のミクのペルソナを継承してしまう")
    ap.add_argument("--voice", default=None, help="ElevenLabs voice_id")
    ap.add_argument("--note", default="", help="description に足す一行")
    ap.add_argument("--kehai", type=Path,
                    default=Path("/home/ai-ccore/Documents/DEV/KEHAI"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="検査に落ちても登録する")
    args = ap.parse_args()

    res = check(args.model)
    verdict = "合格" if res["ok"] else "不合格"
    print(f"検査: {verdict}  語彙={res['profile']} モーフ={res['morphs']} "
          f"まばたき={res['blink']} 口={res['mouth']} {res['mb']}MB")
    if not res["ok"] and not args.force:
        raise SystemExit("検査に落ちたので登録しない（--force で強行できる）")

    dst_dir = args.kehai / "Live2D Characters" / "vrm"
    dst = dst_dir / f"{args.name}{args.model.suffix}"
    url = "/" + quote(f"Live2D Characters/vrm/{dst.name}")

    entry = {
        "name": args.name,
        "url": url,
        "kScale": 1.0,
        "initialXshift": 0,
        "initialYshift": 0,
        "kXOffset": 0,
        "idleMotionGroupName": "",
        # 語彙が arkit なら全感情を入れる。ここが不足すると表情が呼ばれない。
        "emotionMap": (EMOTION_MAP_ARKIT if res["profile"] == "arkit"
                       else {"neutral": 0}),
        "tapMotions": {},
        "description": (
            f"{args.display} — Genesis3D のラップ方式（docs/09）が写真から起こしたモデル。"
            f"**写真から生成した頭（TRELLIS）は測るためだけに使い、Microsoft Rocketbox"
            f"（MIT）のテンプレートの頭を非剛体 ICP でその形に沿わせ、実写を投影**した。"
            f"出荷するのはテンプレートのトポロジそのもの（接合も削減もしない）ので、"
            f"手塗りウェイトと ARKit 命名のモーフ {res['morphs']}個が無傷で残る。"
            f"髪は生成メッシュから切り出した別メッシュで、テンプレートの髪は外してある。"
            + ("VRM 1.0 なので spring bone で髪が揺れる。" if str(args.model).endswith(".vrm")
               else "glb なので spring bone は効かない（髪は揺れない）。")
            + (f" {args.note}" if args.note else "")),
    }

    springs = str(args.model).endswith(".vrm")
    yaml_text = f'''# {args.display} — Genesis3D のラップ方式が写真1枚から起こした体。
#
# 方式は docs/09-wrap-avatar.md。**生成した頭は測るためだけに使い、テンプレートの
# 頭をその形に沿わせる。** 出荷するのはテンプレートのトポロジそのもの。
#
#   頭の形 = TRELLIS の生成結果に非剛体 ICP で沿わせた Rocketbox の頭 ＋ 実写の投影
#   体     = Microsoft Rocketbox（MIT）の素体を身長と体型に合わせて変形（骨も動かす）
#   髪     = 生成メッシュから切り出した別メッシュ（テンプレートの髪は外してある）
#
# 接合と削減をやめたので、手塗りウェイトとモーフが毎回そのまま残る。
#
# ── この体でできること ──
#   会話に伴う体の動き（GestureLSM）、アイドル、視線追従。
#   **口パク・表情・まばたきが動く** — ICT-FaceKit 由来の ARKit 命名のモーフ
#   {res['morphs']}個（変位を実測して確認済み）。
#   指も動く（素体の指ボーン30本と手塗りウェイトがそのまま入っている）。
#   身長は写真と指定値に合わせてある。
#   {"髪が揺れる（VRM 1.0 の spring bone）。" if springs else "髪の揺れは無し（glb なので spring bone が効かない）。"}
#
# ── この体でできないこと ──
#   衣装が本人のものではない。素体の服に写真の色と胴の構造を焼いてある。
#   Oculus のビセーム（aa/ih/ou）。ICT は ARKit 命名しか出さないので、
#   リップシンクは `jawOpen` で動く。
character_config:
  conf_name: "{args.display}"
  # 他と重複させないこと。重複すると chat_history が混ざる。
  conf_uid: "{args.conf_uid}"
  live2d_model_name: "{args.name}"
  character_name: "{args.character_name or args.display}"
'''
    if args.persona:
        yaml_text += f'''  # 省略すると conf.yaml のミクのペルソナを継承してしまう。
  persona_prompt_file: "{args.persona}"
'''
    yaml_text += '''  # アニメ的なトークループは実写寄りの体では別人に見えるので GestureLSM だけにする。
  gesture_lsm_only: true
'''
    if args.voice:
        yaml_text += f'''  tts_config:
    elevenlabs_stream:
      voice_id: "{args.voice}"
      model_id: "eleven_flash_v2_5"
      language_code: "en"
'''

    ychar = args.kehai / "characters" / f"{args.name}.yaml"
    if args.dry_run:
        print(f"[dry-run] コピー   {args.model} → {dst}")
        print(f"[dry-run] model_dict に '{args.name}'（emotionMap "
              f"{len(entry['emotionMap'])}個）")
        print(f"[dry-run] 書き出し {ychar}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model, dst)
    how = upsert_model_dict(args.kehai / "model_dict.json", entry)
    ychar.parent.mkdir(parents=True, exist_ok=True)
    ychar.write_text(yaml_text)

    print(f"モデル      {dst}")
    print(f"model_dict  '{args.name}' を{how}（emotionMap {len(entry['emotionMap'])}個）")
    print(f"設定        {ychar}")


if __name__ == "__main__":
    main()
