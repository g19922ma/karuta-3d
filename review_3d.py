"""
録画データをブラウザでインタラクティブに3D表示するツール

動作を自動でセグメント分割して、払い動作ごとに色を変えて表示する。

使い方:
    .venv/bin/python review_3d.py output/realtime/rec_YYYYMMDD_HHMMSS/3d_log.json

    # 最新の録画を自動選択
    .venv/bin/python review_3d.py

    # セグメント感度を調整
    .venv/bin/python review_3d.py --speed-thresh 0.05 --rest-frames 8
"""

import json
import os
import sys
import argparse
import glob
import webbrowser
import numpy as np
from pathlib import Path


# ---------- データ読み込み ----------

def load_log(log_path: str) -> list[dict]:
    with open(log_path) as f:
        return json.load(f)["frames"]


def load_cards(card_path: str = "calibration/card_positions.json") -> list:
    if not os.path.exists(card_path):
        return []
    with open(card_path) as f:
        return json.load(f).get("cards", [])


# ---------- 自動セグメント分割 ----------

def segment_by_velocity(
    frames: list[dict],
    landmark: str = "wrist",
    speed_thresh: float = 0.03,   # この速度[m/s相当]以下を「静止」とみなす
    rest_frames:  int   = 8,      # N フレーム以上静止したらセグメント境界
    min_seg_len:  int   = 10,     # この長さ未満のセグメントは除外
) -> list[list[dict]]:
    """
    手首の速度から払い動作ごとにフレームを分割する。

    Returns:
        segments: [[frame, ...], [frame, ...], ...]  動作ごとのフレームリスト
    """
    # 手首座標を抽出
    wrist = []
    for f in frames:
        lm = f.get("landmarks", {}).get(landmark)
        wrist.append(np.array(lm) if lm else None)

    # 各フレームの速度を計算
    speeds = [0.0]
    for i in range(1, len(wrist)):
        if wrist[i] is not None and wrist[i-1] is not None:
            speeds.append(float(np.linalg.norm(wrist[i] - wrist[i-1])))
        else:
            speeds.append(0.0)

    # 静止フラグ
    is_rest = [s < speed_thresh for s in speeds]

    # 連続N フレームの静止 → セグメント境界
    boundaries = set()
    rest_count = 0
    for i, r in enumerate(is_rest):
        if r:
            rest_count += 1
            if rest_count == rest_frames:
                boundaries.add(i - rest_frames + 1)
        else:
            rest_count = 0

    # フレームをセグメントに分割
    boundaries = sorted(boundaries)
    seg_starts = [0] + boundaries
    seg_ends   = boundaries + [len(frames)]

    segments = []
    for s, e in zip(seg_starts, seg_ends):
        seg = frames[s:e]
        # 手が検出されているフレームが min_seg_len 以上あるものだけ残す
        active = [f for f in seg if f.get("landmarks", {}).get(landmark)]
        if len(active) >= min_seg_len:
            segments.append(seg)

    print(f"セグメント数: {len(segments)}")
    for i, seg in enumerate(segments):
        t0 = seg[0].get("t", 0)
        t1 = seg[-1].get("t", 0)
        active = len([f for f in seg if f.get("landmarks", {}).get(landmark)])
        print(f"  #{i+1}: {t0:.2f}s〜{t1:.2f}s ({active}フレーム検出)")

    return segments


# ---------- HTML生成 ----------

# 動作ごとの色セット
SEG_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    "#BB8FCE", "#85C1E9",
]

CONNECTIONS = [
    ("wrist", "index_finger_tip"),
    ("wrist", "middle_finger_tip"),
    ("index_finger_tip", "middle_finger_tip"),
]


def seg_to_traces(seg: list[dict], color: str, seg_id: int) -> list[str]:
    """1つのセグメントのtraceリストを生成する。"""
    traces = []
    label = f"動作#{seg_id+1}"

    lm_keys = ["wrist", "index_finger_tip", "middle_finger_tip"]
    lm_labels = {"wrist": "手首", "index_finger_tip": "人差し指", "middle_finger_tip": "中指"}

    # ランドマークごとの軌跡
    for lm in lm_keys:
        xs, ys, zs, ts = [], [], [], []
        for f in seg:
            pt = f.get("landmarks", {}).get(lm)
            if pt:
                xs.append(pt[0]); ys.append(pt[1]); zs.append(pt[2])
                ts.append(f.get("t", 0))
        if not xs:
            continue

        # 軌跡ライン
        traces.append(f"""{{
            type: 'scatter3d', mode: 'lines',
            x: {xs}, y: {ys}, z: {zs},
            name: '{label} {lm_labels[lm]}',
            legendgroup: '{label}',
            line: {{color: '{color}', width: 4}},
            opacity: 0.8
        }}""")

        # 始点・終点マーカー
        if xs:
            traces.append(f"""{{
                type: 'scatter3d', mode: 'markers+text',
                x: [{xs[0]}, {xs[-1]}], y: [{ys[0]}, {ys[-1]}], z: [{zs[0]}, {zs[-1]}],
                name: '{label}',
                legendgroup: '{label}',
                showlegend: false,
                text: ['▶', '■'],
                textposition: 'top center',
                textfont: {{color: '{color}', size: 14}},
                marker: {{size: [8, 6], color: ['{color}', '#888'], symbol: ['circle', 'square']}},
                hovertext: {[f'{label} t={t:.2f}s' for t in [ts[0], ts[-1]]]}
            }}""")

    # スケルトン（間引き）
    step = max(1, len(seg) // 20)
    for f in seg[::step]:
        lms = f.get("landmarks", {})
        for a, b in CONNECTIONS:
            if a in lms and b in lms:
                traces.append(f"""{{
                    type: 'scatter3d', mode: 'lines',
                    x: [{lms[a][0]}, {lms[b][0]}],
                    y: [{lms[a][1]}, {lms[b][1]}],
                    z: [{lms[a][2]}, {lms[b][2]}],
                    line: {{color: '{color}', width: 1}},
                    opacity: 0.25,
                    showlegend: false, hoverinfo: 'skip',
                    legendgroup: '{label}'
                }}""")

    return traces


def card_traces(cards: list) -> list[str]:
    """札のtraceリストを生成する。"""
    traces = []
    for i, card in enumerate(cards):
        c = card["corners"]
        xs = [p[0] for p in c] + [c[0][0]]
        ys = [p[1] for p in c] + [c[0][1]]
        zs = [p[2] for p in c] + [c[0][2]]
        traces.append(f"""{{
            type: 'scatter3d', mode: 'lines+text',
            x: {xs}, y: {ys}, z: {zs},
            name: '札#{i+1}',
            legendgroup: '札',
            line: {{color: '#FFD700', width: 3}},
            text: ['','','#{i+1}','',''],
            textposition: 'top center',
            textfont: {{color: '#FFD700', size: 13}}
        }}""")
        # 塗りつぶし面
        fx = [p[0] for p in c]
        fy = [p[1] for p in c]
        fz = [p[2] for p in c]
        traces.append(f"""{{
            type: 'mesh3d',
            x: {fx}, y: {fy}, z: {fz},
            i: [0,0], j: [1,2], k: [2,3],
            color: '#8B6914', opacity: 0.25,
            showlegend: false, hoverinfo: 'skip'
        }}""")
    return traces


def build_html(segments: list[list[dict]], cards: list, title: str,
               all_frames: list[dict]) -> str:
    n_seg = len(segments)
    n_frames = len(all_frames)

    traces = []

    # 札
    traces += card_traces(cards)

    # 各セグメント
    for i, seg in enumerate(segments):
        color = SEG_COLORS[i % len(SEG_COLORS)]
        traces += seg_to_traces(seg, color, i)

    traces_js = ",\n".join(traces)

    # セグメントのサマリ情報
    seg_info = ""
    for i, seg in enumerate(segments):
        t0 = seg[0].get("t", 0)
        t1 = seg[-1].get("t", 0)
        color = SEG_COLORS[i % len(SEG_COLORS)]
        seg_info += f'<span style="color:{color}">●動作#{i+1} ({t0:.1f}s〜{t1:.1f}s)</span>　'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
  #plot {{ width: 100vw; height: 88vh; }}
  #info {{ padding: 8px 16px; font-size: 13px; line-height: 1.8; }}
</style>
</head>
<body>
<div id="info">
  <b>{title}</b> &nbsp;|&nbsp; 総フレーム: {n_frames} &nbsp;|&nbsp; 検出動作: {n_seg}件<br>
  {seg_info}<br>
  <span style="color:#888">ドラッグ=回転 &nbsp; スクロール=ズーム &nbsp; 凡例クリック=表示切替</span>
</div>
<div id="plot"></div>
<script>
Plotly.newPlot('plot', [
  {traces_js}
], {{
  paper_bgcolor: '#111',
  plot_bgcolor:  '#111',
  scene: {{
    bgcolor: '#111',
    xaxis: {{title: 'X', color: '#888', gridcolor: '#333'}},
    yaxis: {{title: 'Y', color: '#888', gridcolor: '#333'}},
    zaxis: {{title: 'Z', color: '#888', gridcolor: '#333'}},
    aspectmode: 'data'
  }},
  legend: {{bgcolor: '#222', font: {{color: '#ccc'}}, itemclick: 'toggleothers'}},
  margin: {{l:0, r:0, t:0, b:0}}
}}, {{responsive: true}});
</script>
</body>
</html>"""


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser(description="払い動作ごとに色分けして3D表示")
    parser.add_argument("log",           nargs="?", help="3d_log.json のパス（省略で最新）")
    parser.add_argument("--speed-thresh", type=float, default=0.03,
                        help="静止とみなす速度閾値（小さくすると細かく分割）")
    parser.add_argument("--rest-frames",  type=int,   default=8,
                        help="静止と判定するフレーム数（小さくすると細かく分割）")
    args = parser.parse_args()

    if args.log:
        log_path = args.log
    else:
        logs = sorted(glob.glob("output/realtime/rec_*/3d_log.json"))
        if not logs:
            print("録画データが見つかりません")
            sys.exit(1)
        log_path = logs[-1]
        print(f"最新の録画を使用: {log_path}")

    frames = load_log(log_path)
    cards  = load_cards()
    title  = Path(log_path).parent.name

    print(f"総フレーム数: {len(frames)}")

    segments = segment_by_velocity(
        frames,
        speed_thresh=args.speed_thresh,
        rest_frames=args.rest_frames,
    )

    if not segments:
        print("動作が検出できませんでした。--speed-thresh を大きくしてみてください。")
        sys.exit(1)

    html = build_html(segments, cards, title, frames)

    out_path = str(Path(log_path).parent / "review_3d.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"保存: {out_path}")
    webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
