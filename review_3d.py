"""
録画データをブラウザでインタラクティブに3D表示するツール

3d_log.json を読み込み、plotly で回転・ズームできるHTMLを生成してブラウザで開く。

使い方:
    .venv/bin/python review_3d.py output/realtime/rec_YYYYMMDD_HHMMSS/3d_log.json

    # 最新の録画を自動選択
    .venv/bin/python review_3d.py
"""

import json
import os
import sys
import argparse
import glob
import webbrowser
from pathlib import Path


def load_log(log_path: str) -> list[dict]:
    with open(log_path) as f:
        return json.load(f)["frames"]


def load_cards(card_path: str = "calibration/card_positions.json") -> list:
    if not os.path.exists(card_path):
        return []
    with open(card_path) as f:
        return json.load(f).get("cards", [])


def build_html(frames: list[dict], cards: list, title: str) -> str:
    """plotly を使ったインタラクティブ3D HTMLを生成する。"""

    landmark_colors = {
        "wrist":             "#FF5555",
        "index_finger_tip":  "#55AAFF",
        "middle_finger_tip": "#55FF88",
    }
    connections = [
        ("wrist", "index_finger_tip"),
        ("wrist", "middle_finger_tip"),
        ("index_finger_tip", "middle_finger_tip"),
    ]

    traces = []
    n = len(frames)

    # --- 手のトラジェクトリ ---
    for lm_name, color in landmark_colors.items():
        xs, ys, zs, ts = [], [], [], []
        for frame in frames:
            lm = frame.get("landmarks", {}).get(lm_name)
            if lm:
                xs.append(lm[0]); ys.append(lm[1]); zs.append(lm[2])
                ts.append(frame.get("t", 0))

        if not xs:
            continue

        label = {"wrist": "手首", "index_finger_tip": "人差し指",
                 "middle_finger_tip": "中指"}.get(lm_name, lm_name)

        # 軌跡ライン
        traces.append(f"""{{
            type: 'scatter3d', mode: 'lines',
            x: {xs}, y: {ys}, z: {zs},
            name: '{label}軌跡',
            line: {{color: '{color}', width: 3}},
            opacity: 0.5,
            hoverinfo: 'skip'
        }}""")

        # 点（時刻で色変化）
        traces.append(f"""{{
            type: 'scatter3d', mode: 'markers',
            x: {xs}, y: {ys}, z: {zs},
            name: '{label}',
            text: {[f't={t:.2f}s' for t in ts]},
            marker: {{
                size: 5,
                color: {ts},
                colorscale: 'Plasma',
                opacity: 0.8,
                colorbar: {{title: '時刻[s]', len: 0.5}}
            }}
        }}""")

    # --- 骨格ライン（各フレーム） ---
    # 全フレームだと重いのでフレームを間引く
    step = max(1, n // 60)
    for frame in frames[::step]:
        lms = frame.get("landmarks", {})
        for a, b in connections:
            if a in lms and b in lms:
                xs = [lms[a][0], lms[b][0]]
                ys = [lms[a][1], lms[b][1]]
                zs = [lms[a][2], lms[b][2]]
                t  = frame.get("t", 0)
                traces.append(f"""{{
                    type: 'scatter3d', mode: 'lines',
                    x: {xs}, y: {ys}, z: {zs},
                    line: {{color: 'rgba(200,200,200,0.15)', width: 1}},
                    showlegend: false, hoverinfo: 'skip'
                }}""")

    # --- 札の矩形 ---
    for i, card in enumerate(cards):
        c = card["corners"]
        # 矩形の4辺 + 閉じる
        xs = [p[0] for p in c] + [c[0][0]]
        ys = [p[1] for p in c] + [c[0][1]]
        zs = [p[2] for p in c] + [c[0][2]]
        traces.append(f"""{{
            type: 'scatter3d', mode: 'lines+text',
            x: {xs}, y: {ys}, z: {zs},
            name: '札#{i+1}',
            line: {{color: '#B8860B', width: 3}},
            text: ['','','','#{i+1}',''],
            textposition: 'top center',
            textfont: {{color: '#FFD700', size: 12}}
        }}""")
        # 面（塗りつぶし）
        cx = sum(p[0] for p in c) / 4
        cy = sum(p[1] for p in c) / 4
        cz = sum(p[2] for p in c) / 4
        for j in range(4):
            tri_x = [c[j][0], c[(j+1)%4][0], cx]
            tri_y = [c[j][1], c[(j+1)%4][1], cy]
            tri_z = [c[j][2], c[(j+1)%4][2], cz]
            traces.append(f"""{{
                type: 'mesh3d',
                x: {tri_x}, y: {tri_y}, z: {tri_z},
                i: [0], j: [1], k: [2],
                color: '#8B6914', opacity: 0.3,
                showlegend: false, hoverinfo: 'skip'
            }}""")

    traces_js = ",\n".join(traces)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
  #plot {{ width: 100vw; height: 90vh; }}
  #info {{ padding: 8px 16px; font-size: 13px; color: #aaa; }}
</style>
</head>
<body>
<div id="info">
  {title} &nbsp;|&nbsp; {n} フレーム &nbsp;|&nbsp;
  ドラッグ=回転 &nbsp; スクロール=ズーム &nbsp; ダブルクリック=リセット
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
    zaxis: {{title: 'Z（奥行き）', color: '#888', gridcolor: '#333'}},
    aspectmode: 'data'
  }},
  legend: {{bgcolor: '#222', font: {{color: '#ccc'}}}},
  margin: {{l:0, r:0, t:0, b:0}}
}}, {{responsive: true}});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="録画データをブラウザで3D表示")
    parser.add_argument("log", nargs="?", help="3d_log.json のパス（省略で最新録画）")
    args = parser.parse_args()

    if args.log:
        log_path = args.log
    else:
        # 最新の録画を自動選択
        logs = sorted(glob.glob("output/realtime/rec_*/3d_log.json"))
        if not logs:
            print("録画データが見つかりません: output/realtime/rec_*/3d_log.json")
            sys.exit(1)
        log_path = logs[-1]
        print(f"最新の録画を使用: {log_path}")

    frames = load_log(log_path)
    cards  = load_cards()
    title  = Path(log_path).parent.name

    print(f"フレーム数: {len(frames)}, 札: {len(cards)}枚")

    html = build_html(frames, cards, title)

    out_path = str(Path(log_path).parent / "review_3d.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"保存: {out_path}")
    webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
