"""メッシュから人体のランドマークを幾何的に見つける。

Blender の各エントリスクリプトから共用する。bpy には依存しない — 呼び出し側が
ワールド座標の頂点を渡す。

肩の検出だけが要件だが、これは見た目より難しい。素直に「上から走査して横幅が
跳ねた点」を肩とすると、**髪の広がりを肩と誤認する**（docs/05-measurements.md §34）。
実測: 髪の豊かな写真から作った頭部メッシュで、頭部帯が 0.563 → 0.218 に縮み、
接合スケールが 1.5 倍狂って頭が胴に沈んだ。

上から下へ辿ると幅は「頭頂 → 髪で最大 → **首でくびれ** → 肩で急拡大」と動く。
そこで **先に首のくびれを見つけ、その下で肩を探す**。髪の膨らみはくびれより
上にあるので巻き込まれない。
"""
from __future__ import annotations

import statistics

NBINS = 64
# 首の幅の何倍で「肩」とみなすか。首→肩は実測で 2 倍前後に開く。
SHOULDER_RATIO = 1.6
# くびれを探す範囲（上端からの割合）。これより下は胴なので見ない。
NECK_SEARCH = 0.55


def _profile(points, top_z: float, bottom_z: float) -> list[float]:
    """高さビンごとの横幅。"""
    height = top_z - bottom_z
    spans: list[tuple[float, float] | None] = [None] * NBINS
    for x, z in points:
        b = min(NBINS - 1, max(0, int((top_z - z) / height * NBINS)))
        cur = spans[b]
        spans[b] = (min(cur[0], x), max(cur[1], x)) if cur else (x, x)
    return [(s[1] - s[0]) if s else 0.0 for s in spans]


def shoulder_split(points, top_z: float, bottom_z: float,
                   fallback_fraction: float = 0.25) -> tuple[float, float, float]:
    """(shoulder_z, head_width, confidence) を返す。

    points: (x, z) のイテラブル（ワールド座標）。
    shoulder_z: 頭部と胴の境目の Z。
    head_width: 頭部帯の代表的な横幅。スケール合わせに使う。
    confidence: 1.0 = くびれと肩の両方を検出 / 0.5 = 単純な跳ね上がりで代用
                / 0.0 = fallback に落ちた。
    """
    height = top_z - bottom_z
    if height <= 0:
        return top_z - 1e-6, 1e-6, 0.0

    ext = _profile(points, top_z, bottom_z)
    head_bins = max(2, NBINS // 12)
    head_ref = [e for e in ext[:head_bins] if e > 0]
    if not head_ref:
        return top_z - height * fallback_fraction, 1e-6, 0.0
    base = statistics.median(head_ref)

    def z_of(b: int) -> float:
        return top_z - (b / NBINS) * height

    # 1) 首のくびれ: 上端〜NECK_SEARCH の範囲で幅が最小になるビン。
    #    髪の最大幅より下、肩より上にある。
    lo_b = head_bins
    hi_b = max(lo_b + 2, int(NBINS * NECK_SEARCH))
    window = [(b, ext[b]) for b in range(lo_b, min(hi_b, NBINS)) if ext[b] > 0]
    if window:
        neck_b, neck_w = min(window, key=lambda t: t[1])
        # 2) くびれより下で、首の SHOULDER_RATIO 倍を超えた最初のビン＝肩。
        for b in range(neck_b + 1, NBINS):
            if ext[b] > neck_w * SHOULDER_RATIO:
                return z_of(b), base, 1.0

    # くびれが見つからない（頭部だけのメッシュなど）。素直な跳ね上がりで代用する。
    for b in range(head_bins, NBINS):
        if ext[b] > base * 1.9:
            return z_of(b), base, 0.5

    return top_z - height * fallback_fraction, base, 0.0
