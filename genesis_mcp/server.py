"""Genesis3D MCP サーバー。

既存の AICCORE-Nexus/mcp-tools/server.py と同じ形（FastMCP 系 + add_tool + stdio）。
公開するのは「ジョブ制御」と「工程単体」の2系統。

設計上の約束:
  - すべて 127.0.0.1:8820 への HTTP か、Blender / conda のサブプロセス実行に閉じる
  - ツールは**パスを返す**。巨大なバイナリを応答に載せない（PNG は Read で開ける）
  - 長時間かかるものは timeout 引数を必ず持つ
  - configs/*.yaml を書き換えるツールは作らない

stdio でプロトコルを喋るので、**標準出力を汚してはいけない**。
サブプロセスの出力はすべて捕捉し、戻り値の文字列に畳み込む。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

# mcp SDK は 2.0 で FastMCP -> MCPServer に改名された。どちらでも動くようにする。
try:  # v2.x
    from mcp.server.mcpserver import MCPServer as _Server
except Exception:  # v1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
def _genesis_url() -> str:
    """orchestrator のベースURL。

    .env の GENESIS_HOST は uvicorn の**バインドアドレス**（"127.0.0.1"）で、
    Nexus 側 genesis.ts の GENESIS_HOST は**完全なURL**という二つの流儀がある。
    どちらで渡されても動くよう、スキームが無ければポートを足して組み立てる。
    """
    explicit = os.environ.get("GENESIS_URL")
    if explicit:
        return explicit.rstrip("/")
    host = os.environ.get("GENESIS_HOST", "127.0.0.1").strip()
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    port = os.environ.get("GENESIS_PORT", "8820")
    return f"http://{host}:{port}"


GENESIS = _genesis_url()
BLENDER = os.path.expanduser(os.environ.get("BLENDER_BIN", "~/opt/blender/blender"))
CONDA = os.path.expanduser(os.environ.get("CONDA_BASE", "~/miniconda3")) + "/bin/conda"
ENV_EVAL = os.environ.get("ENV_EVAL", "genesis-eval")
RIG_TEMPLATE = os.environ.get(
    "RIG_TEMPLATE_VRM",
    "/home/ai-ccore/Documents/DEV/KEHAI/Live2D Characters/vrm/sample.vrm",
)

mcp = _Server("genesis3d")

MAX_OUT = 12000


def _cap(text: str, limit: int = MAX_OUT) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [{len(text) - limit} 文字省略]"


def _j(obj) -> str:
    return _cap(json.dumps(obj, indent=2, ensure_ascii=False))


def _get(path: str, timeout: float = 20.0):
    r = httpx.get(f"{GENESIS}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _resolve(p: str) -> Path:
    """相対パスは Genesis3D ルート基準で解決する。"""
    q = Path(os.path.expanduser(p))
    return q if q.is_absolute() else (ROOT / q)


def _run(argv: list[str], timeout: float, cwd: Path | None = None,
         env_extra: dict[str, str] | None = None) -> tuple[int, str]:
    """サブプロセスを実行し stdout+stderr を捕捉する。stdout は絶対に汚さない。"""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("HF_HUB_OFFLINE", "1")
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(argv, cwd=str(cwd or ROOT), env=env, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, f"タイムアウト ({timeout:.0f}s)"


def _blender(script: str, args: list[str], timeout: float) -> tuple[int, str]:
    return _run([BLENDER, "--background", "--factory-startup",
                 "--python", str(ROOT / "blender" / script), "--", *args], timeout)


def _marker(out: str, marker: str) -> str:
    """スクリプトが出す `MARKER {json}` 行だけを拾う。無ければ末尾を返す。"""
    for line in out.splitlines():
        if line.startswith(marker):
            return line[len(marker):].strip()
    return _cap(out[-2000:])


# ======================================================================
# ジョブ制御
# ======================================================================

def genesis_health() -> str:
    """Genesis3D オーケストレータの状態。TRELLIS.2 と Blender への到達性も含む。"""
    try:
        return _j(_get("/health"))
    except Exception as e:
        return (f"到達不可: {e}\n"
                f"起動するには: cd {ROOT} && ./scripts/run.sh")


def genesis_gpus() -> str:
    """NVML 実測の GPU 状態（空きVRAM・役割・予約・新規ロード可否）。

    GPU0 は Genesis3D 専有。GPU1 は TRELLIS.2 が常駐しており、Genesis3D は
    新規モデルを載せず :8811 経由でのみ使う。
    """
    return _j(_get("/gpus"))


def genesis_models() -> str:
    """導入済みモデルと状態。未導入モデルもその旨が分かる。"""
    return _j(_get("/models"))


def genesis_create_job(image_path: str, style: str = "stylized",
                       quality: str = "high", output_format: str = "vrm",
                       multi_candidate: bool = True,
                       face_image_path: str = "") -> str:
    """画像1枚から VRM 生成ジョブを作成し、即座に job_id を返す。

    style   : stylized（3Dアニメ映画的デフォルメ）/ realistic_jrpg（リアル寄りJRPG）
              / wrap（ラップ方式・docs/09。テンプレのトポロジを出荷。VRM で
                髪が揺れる。params に face_image / height_cm / template を渡せる）
    quality : fast（1候補・res512）/ high（2候補・res1024+512）/ maximum（4候補）

    入力は**全身像**が望ましい。バストアップだと脚部が推定されずリギングが破綻する。
    ただし quality=high/maximum なら、見切れを検出した時点で工程2が
    全身Tポーズの参照ビューを生成して代用する。

    face_image_path : 顔に貼る実写ポートレート（任意）。
        生成したTポーズ画像から体を作る場合、その顔は本人ではない
        （実測 ArcFace 0.20）。ここに実写を渡すと工程 head_merge が
        顔だけ実写から作って接合する（実測 ArcFace 0.66）。
        省略時、工程2が走ったなら入力画像が自動的に使われる。
    """
    src = _resolve(image_path)
    if not src.is_file():
        return f"画像が見つかりません: {src}"
    face = _resolve(face_image_path) if face_image_path else None
    if face is not None and not face.is_file():
        return f"顔画像が見つかりません: {face}"

    data = {"style": style, "quality": quality,
            "output_format": output_format,
            "models": "auto",
            "multi_candidate": str(multi_candidate).lower()}
    with open(src, "rb") as fh:
        files = {"image": (src.name, fh, "image/png")}
        if face is None:
            r = httpx.post(f"{GENESIS}/jobs", data=data, files=files, timeout=120)
        else:
            with open(face, "rb") as ffh:
                files["face_image"] = (face.name, ffh, "image/png")
                r = httpx.post(f"{GENESIS}/jobs", data=data, files=files,
                               timeout=120)
    if r.status_code != 200:
        return f"ジョブ作成失敗 {r.status_code}: {_cap(r.text, 500)}"
    return _j(r.json())


def genesis_job_status(job_id: str) -> str:
    """ジョブの現在の工程・進捗・割当GPU。"""
    try:
        return _j(_get(f"/jobs/{job_id}"))
    except Exception as e:
        return f"取得失敗: {e}"


def genesis_wait_job(job_id: str, timeout_sec: int = 900, poll_sec: int = 10) -> str:
    """ジョブが完了/失敗するまで待つ。反復作業ではこれを使うのが一番速い。

    完了したら工程サマリ（どの工程が実行され、何がスキップされたか）と
    VRM 検証結果まで含めて返す。
    """
    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        try:
            st = _get(f"/jobs/{job_id}", timeout=15)
        except Exception as e:
            return f"取得失敗: {e}"
        last = st
        if st["status"] in ("completed", "failed", "canceled"):
            break
        time.sleep(poll_sec)
    else:
        return _j({"timeout": True, "waited_sec": timeout_sec, "last": last})

    out = {"job_id": job_id, "status": last["status"],
           "duration_sec": round((last.get("completed_at") or 0)
                                 - (last.get("started_at") or 0), 1),
           "error": last.get("error")}

    manifest = ROOT / "jobs" / job_id / "output" / "job.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        out["stages"] = {k: {"status": v.get("status"),
                             "reason": v.get("reason"),
                             **({"ranking": v["ranking"]} if "ranking" in v else {}),
                             **({"missing_required_bones": v["missing_required_bones"],
                                 "height_m": v.get("height_m"),
                                 "origin_at_feet": v.get("origin_at_feet")}
                                if "missing_required_bones" in v else {})}
                         for k, v in m.get("stages", {}).items()}
        out["output"] = m.get("output")
    return _j(out)


def genesis_job_results(job_id: str) -> str:
    """候補一覧・評価スコア・生成された成果物の名前。"""
    try:
        res = _get(f"/jobs/{job_id}/results", timeout=30)
    except Exception as e:
        return f"取得失敗: {e}"
    scores = ROOT / "jobs" / job_id / "scores" / "scores.json"
    if scores.exists():
        s = json.loads(scores.read_text())
        res["evaluation"] = {
            "encoder": s.get("encoder"),
            "best_overall": s.get("best_overall"),
            "best_geometry": s.get("best_geometry"),
            "best_texture": s.get("best_texture"),
            "ranking": [(c["candidate"], c["overall"]) for c in s.get("candidates", [])],
            "not_measured": (s.get("candidates") or [{}])[0].get("not_measured"),
        }
    return _j(res)


def genesis_list_jobs(limit: int = 20) -> str:
    """過去のジョブ一覧（新しい順）。"""
    try:
        d = _get(f"/jobs?limit={limit}", timeout=20)
    except Exception as e:
        return f"取得失敗: {e}"
    rows = [{"job_id": j["job_id"], "status": j["status"], "style": j["style"],
             "quality": j["quality"], "stage": j["stage"],
             "duration_sec": round((j.get("completed_at") or 0)
                                   - (j.get("started_at") or 0), 1)
             if j.get("completed_at") else None}
            for j in d.get("jobs", [])]
    return _j(rows)


def genesis_cancel_job(job_id: str) -> str:
    """実行中のジョブをキャンセルする。"""
    r = httpx.post(f"{GENESIS}/jobs/{job_id}/cancel", timeout=20)
    return _j(r.json()) if r.status_code == 200 else f"失敗 {r.status_code}: {r.text[:200]}"


def genesis_artifact_path(job_id: str, filename: str = "") -> str:
    """成果物の絶対パスを返す。filename 省略時は output/ の一覧。

    PNG はこのパスを Read ツールに渡せばそのまま見られる。
    """
    outdir = ROOT / "jobs" / job_id / "output"
    if not outdir.exists():
        return f"見つかりません: {outdir}"
    if not filename:
        listing = {"output_dir": str(outdir),
                   "files": sorted(p.name for p in outdir.iterdir() if p.is_file())}
        rend = ROOT / "jobs" / job_id / "renders"
        if rend.exists():
            listing["renders"] = {d.name: sorted(p.name for p in d.iterdir())
                                  for d in sorted(rend.iterdir()) if d.is_dir()}
        return _j(listing)
    target = (outdir / filename).resolve()
    if not str(target).startswith(str(outdir.resolve())) or not target.is_file():
        return f"見つかりません: {filename}"
    return str(target)


# ======================================================================
# 工程単体（任意のメッシュ／画像に対して個別実行）
# ======================================================================

def genesis_preprocess_image(image_path: str, out_path: str = "",
                             timeout_sec: int = 600) -> str:
    """工程1 — RMBG-2.0 で背景除去し、正方形・中央配置に正規化する。

    見切れ検出つき。バストアップ像は「脚部が推定されずリギングが破綻する」と警告する。
    背景を残したまま 3D 化すると、模様やエフェクトがそのまま浮遊ジオメトリになる。
    """
    src = _resolve(image_path)
    if not src.is_file():
        return f"画像が見つかりません: {src}"
    out = _resolve(out_path) if out_path else src.with_name(src.stem + "_rgba.png")
    meta = out.with_suffix(".json")
    code, log = _run([CONDA, "run", "-n", ENV_EVAL, "--no-capture-output",
                      "python", "-m", "preprocessing.rmbg",
                      "--input", str(src), "--output", str(out), "--meta", str(meta)],
                     timeout_sec, env_extra={"CUDA_VISIBLE_DEVICES": "0"})
    if code != 0:
        return f"失敗 (exit={code}):\n{_cap(log[-1500:])}"
    return _j({"output": str(out), "meta": str(meta),
               "result": json.loads(_marker(log, "PREPROCESS_OK"))})


def genesis_render_mesh(mesh_path: str, out_dir: str = "",
                        views: str = "front,left,right,back,three_quarter",
                        resolution: int = 512, samples: int = 24,
                        timeout_sec: int = 900) -> str:
    """工程4 — 統一カメラ・照明でメッシュをレンダリングし、統計も出す。

    候補間で描画条件を厳密に揃えるので、見比べや評価の入力に使える。
    glb / gltf / obj / fbx / ply / vrm を受け付ける。返すのは PNG のパス。
    """
    src = _resolve(mesh_path)
    if not src.is_file():
        return f"メッシュが見つかりません: {src}"
    out = _resolve(out_dir) if out_dir else src.parent / f"{src.stem}_renders"
    code, log = _blender("render_views.py",
                         ["--input", str(src), "--outdir", str(out),
                          "--res", str(resolution), "--samples", str(samples),
                          "--views", views], timeout_sec)
    if code != 0:
        return f"失敗 (exit={code}):\n{_cap(log[-1500:])}"
    info = json.loads(_marker(log, "RENDER_OK"))
    return _j({"outdir": str(out),
               "files": [str(out / v) for v in info.get("views", [])],
               "triangles": info.get("tris"), "vertices": info.get("verts")})


def genesis_mesh_report(mesh_path: str, out_path: str = "",
                        timeout_sec: int = 900) -> str:
    """工程5の一部 — 穴・非多様体・浮遊ジオメトリ・重複頂点を数える。

    注意: 実用アバターは watertight ではない（衣装・髪・目が別オブジェクト、
    UVシームで頂点分割）。非watertight や連結成分の多さを単純に欠陥と見なさないこと。
    """
    src = _resolve(mesh_path)
    if not src.is_file():
        return f"メッシュが見つかりません: {src}"
    out = _resolve(out_path) if out_path else src.with_name(src.stem + "_mesh.json")
    code, log = _blender("mesh_report.py",
                         ["--input", str(src), "--out", str(out)], timeout_sec)
    if code != 0:
        return f"失敗 (exit={code}):\n{_cap(log[-1500:])}"
    return _j({"report": str(out), "total": json.loads(_marker(log, "MESH_REPORT_OK"))})


def genesis_repair_mesh(mesh_path: str, out_path: str = "",
                        speck_ratio: float = 0.0002,
                        timeout_sec: int = 900) -> str:
    """工程7 — 浮遊片の除去のみ。Decimation / Remesh / 穴埋め / 頂点マージはしない。

    頂点マージは UV シームを壊してテクスチャに黒斑を出し、除去閾値を上げすぎると
    表面に穴が開く（実測）。面の10%超を消した場合は excessive_removal を返すので、
    そのときは修復前のメッシュを使うこと。
    """
    src = _resolve(mesh_path)
    if not src.is_file():
        return f"メッシュが見つかりません: {src}"
    out = _resolve(out_path) if out_path else src.with_name(src.stem + "_clean.glb")
    rep = out.with_suffix(".repair.json")
    code, log = _blender("repair_mesh.py",
                         ["--input", str(src), "--out", str(out),
                          "--report", str(rep), "--speck-ratio", str(speck_ratio)],
                         timeout_sec)
    if code != 0:
        return f"失敗 (exit={code}):\n{_cap(log[-1500:])}"
    return _j({"output": str(out), "report": str(rep),
               "result": json.loads(_marker(log, "REPAIR_OK"))})


def genesis_autorig(mesh_path: str, out_path: str = "", height_m: float = 1.6,
                    template_vrm: str = "", timeout_sec: int = 1800) -> str:
    """工程8 — テンプレート転写による自動リギング（UniRig 導入までの経路）。

    既知の VRM Humanoid アーマチュアをスケールして被せ、Blender の自動ウェイトで
    結合する。image-to-3D の出力が T ポーズになるため成立する。

    限界: 骨の位置は身長基準の相似変形なので体型差でずれる／指はテンプレートのまま／
    spring bone（髪・スカートの揺れもの）は未設定。
    """
    src = _resolve(mesh_path)
    if not src.is_file():
        return f"メッシュが見つかりません: {src}"
    tmpl = Path(template_vrm) if template_vrm else Path(RIG_TEMPLATE)
    if not tmpl.is_file():
        return f"テンプレート VRM が見つかりません: {tmpl}"
    out = _resolve(out_path) if out_path else src.with_name(src.stem + "_rigged.vrm")
    rep = out.with_suffix(".rig.json")
    code, log = _blender("autorig.py",
                         ["--input", str(src), "--template", str(tmpl),
                          "--out", str(out), "--report", str(rep),
                          "--height", str(height_m)], timeout_sec)
    if code != 0:
        return f"失敗 (exit={code}):\n{_cap(log[-1500:])}"
    return _j({"output": str(out), "report": str(rep),
               "result": json.loads(_marker(log, "AUTORIG_OK"))})


def genesis_export_vrm(rigged_path: str, out_path: str = "",
                       name: str = "Genesis3D Character",
                       timeout_sec: int = 900) -> str:
    """工程9 — リグ済みメッシュを VRM 1.0 として書き出す。

    VRM Humanoid の必須ボーンが1本でも割り当たらなければ、書き出さずに失敗させる
    （見た目は成功して中身が壊れた VRM を作らないため）。
    """
    src = _resolve(rigged_path)
    if not src.is_file():
        return f"入力が見つかりません: {src}"
    out = _resolve(out_path) if out_path else src.with_name(src.stem + ".vrm")
    rep = out.with_suffix(".validation.json")
    code, log = _blender("export_vrm.py",
                         ["--input", str(src), "--out", str(out),
                          "--report", str(rep), "--name", name], timeout_sec)
    if code != 0:
        detail = json.loads(rep.read_text()) if rep.exists() else {}
        return _j({"status": "failed", "exit": code,
                   "error": detail.get("error"),
                   "missing_required_bones": detail.get("missing_required_bones"),
                   "log": _cap(log[-800:])})
    return _j({"output": str(out), "report": str(rep),
               "result": json.loads(_marker(log, "VRM_EXPORT_OK"))})


def genesis_validate_vrm(vrm_path: str, timeout_sec: int = 600) -> str:
    """工程10 — VRM Humanoid の必須ボーン・スケール・原点・マテリアルを検査する。

    必須17本（hips/spine/chest/neck/head, 左右の upperArm/lowerArm/hand,
    左右の upperLeg/lowerLeg/foot）が揃っているかを見る。
    """
    src = _resolve(vrm_path)
    if not src.is_file():
        return f"VRM が見つかりません: {src}"
    rep = src.with_suffix(".validation.json")
    code, log = _blender("export_vrm.py",
                         ["--input", str(src), "--out", "/dev/null",
                          "--report", str(rep), "--dry-run"], timeout_sec)
    if not rep.exists():
        return f"検証失敗 (exit={code}):\n{_cap(log[-1200:])}"
    d = json.loads(rep.read_text())
    return _j({
        "ok": not d.get("missing_required_bones") and code == 0,
        "missing_required_bones": d.get("missing_required_bones"),
        "optional_present": d.get("optional_present"),
        "bone_count": d.get("bone_count"),
        "mesh_count": d.get("mesh_count"),
        "height_m": d.get("height_m"),
        "height_plausible": d.get("height_plausible"),
        "origin_at_feet": d.get("origin_at_feet"),
        "materials": d.get("materials"),
        "report": str(rep),
    })


_TOOLS = [
    # ジョブ制御
    genesis_health, genesis_gpus, genesis_models,
    genesis_create_job, genesis_job_status, genesis_wait_job,
    genesis_job_results, genesis_list_jobs, genesis_cancel_job,
    genesis_artifact_path,
    # 工程単体
    genesis_preprocess_image, genesis_render_mesh, genesis_mesh_report,
    genesis_repair_mesh, genesis_autorig, genesis_export_vrm, genesis_validate_vrm,
]
for _fn in _TOOLS:
    mcp.add_tool(_fn)

if __name__ == "__main__":
    mcp.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))
