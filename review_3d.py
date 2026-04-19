"""
録画データをブラウザでインタラクティブに3D表示するツール

払い動作ごとに色分け・凡例は動作番号1行にまとめて表示。

使い方:
    .venv/bin/python review_3d.py
    .venv/bin/python review_3d.py output/realtime/rec_YYYYMMDD/3d_log.json
    .venv/bin/python review_3d.py --speed-thresh 0.05 --rest-frames 5
"""

import json, os, sys, argparse, glob, webbrowser
import numpy as np
from pathlib import Path


def load_log(path):
    with open(path) as f:
        return json.load(f)["frames"]

def load_cards(path="calibration/card_positions.json"):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("cards", [])


# ---------- セグメント分割 ----------

def segment(frames, speed_thresh=0.03, rest_frames=8, min_len=10):
    wrist = [np.array(f["landmarks"]["wrist"]) if f.get("landmarks", {}).get("wrist") else None
             for f in frames]
    speeds = [0.0] + [float(np.linalg.norm(wrist[i] - wrist[i-1]))
                      if wrist[i] is not None and wrist[i-1] is not None else 0.0
                      for i in range(1, len(wrist))]
    is_rest = [s < speed_thresh for s in speeds]

    boundaries, cnt = set(), 0
    for i, r in enumerate(is_rest):
        cnt = cnt + 1 if r else 0
        if cnt == rest_frames:
            boundaries.add(i - rest_frames + 1)

    cuts = sorted(boundaries)
    segs = []
    for s, e in zip([0] + cuts, cuts + [len(frames)]):
        seg = frames[s:e]
        active = sum(1 for f in seg if f.get("landmarks", {}).get("wrist"))
        if active >= min_len:
            segs.append(seg)

    print(f"検出動作数: {len(segs)}")
    for i, seg in enumerate(segs):
        t0, t1 = seg[0].get("t", 0), seg[-1].get("t", 0)
        print(f"  #{i+1}: {t0:.2f}s〜{t1:.2f}s")
    return segs


# ---------- trace生成 ----------

SEG_COLORS = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4","#FFEAA7",
              "#DDA0DD","#98D8C8","#F7DC6F","#BB8FCE","#85C1E9"]

CONNS = [("wrist","index_finger_tip"),("wrist","middle_finger_tip"),
         ("index_finger_tip","middle_finger_tip")]

def seg_traces(seg, color, idx):
    """1セグメント分のtraces。凡例は手首1行のみ、他はshowlegend:false。"""
    traces = []
    label = f"動作#{idx+1}"
    t0 = seg[0].get("t",0); t1 = seg[-1].get("t",0)
    hover_label = f"{label} ({t0:.1f}s〜{t1:.1f}s)"

    # --- 手首軌跡（凡例あり・デフォルト表示） ---
    xs,ys,zs,ts = [],[],[],[]
    for f in seg:
        w = f.get("landmarks",{}).get("wrist")
        if w: xs.append(w[0]); ys.append(w[1]); zs.append(w[2]); ts.append(f.get("t",0))
    if xs:
        traces.append(f"""{{
            type:'scatter3d', mode:'lines+markers',
            x:{xs}, y:{ys}, z:{zs},
            name:'{hover_label}',
            legendgroup:'{label}',
            line:{{color:'{color}',width:5}},
            marker:{{size:3,color:'{color}'}},
            hovertemplate:'%{{text}}<extra>{label}</extra>',
            text:{[f't={t:.2f}s' for t in ts]}
        }}""")
        # 始点▶終点■
        traces.append(f"""{{
            type:'scatter3d', mode:'markers+text',
            x:[{xs[0]},{xs[-1]}], y:[{ys[0]},{ys[-1]}], z:[{zs[0]},{zs[-1]}],
            legendgroup:'{label}', showlegend:false,
            text:['▶','■'], textposition:'top center',
            textfont:{{color:'{color}',size:16}},
            marker:{{size:[10,7], color:['{color}','#666']}},
            hoverinfo:'skip'
        }}""")

    # --- 指先軌跡（デフォルト非表示） ---
    for lm, lm_label in [("index_finger_tip","人差し指"),("middle_finger_tip","中指")]:
        xs2,ys2,zs2 = [],[],[]
        for f in seg:
            p = f.get("landmarks",{}).get(lm)
            if p: xs2.append(p[0]); ys2.append(p[1]); zs2.append(p[2])
        if xs2:
            traces.append(f"""{{
                type:'scatter3d', mode:'lines',
                x:{xs2}, y:{ys2}, z:{zs2},
                name:'{label} {lm_label}',
                legendgroup:'{label}', showlegend:false,
                visible:'legendonly',
                line:{{color:'{color}',width:2}}, opacity:0.5
            }}""")

    # --- スケルトン（デフォルト非表示・間引き） ---
    step = max(1, len(seg)//15)
    for f in seg[::step]:
        lms = f.get("landmarks",{})
        for a,b in CONNS:
            if a in lms and b in lms:
                traces.append(f"""{{
                    type:'scatter3d', mode:'lines',
                    x:[{lms[a][0]},{lms[b][0]}],
                    y:[{lms[a][1]},{lms[b][1]}],
                    z:[{lms[a][2]},{lms[b][2]}],
                    legendgroup:'{label}', showlegend:false,
                    visible:'legendonly',
                    line:{{color:'{color}',width:1}}, opacity:0.3,
                    hoverinfo:'skip'
                }}""")
    return traces


def card_traces(cards):
    traces = []
    for i, card in enumerate(cards):
        c = card["corners"]
        xs = [p[0] for p in c]+[c[0][0]]
        ys = [p[1] for p in c]+[c[0][1]]
        zs = [p[2] for p in c]+[c[0][2]]
        traces.append(f"""{{
            type:'scatter3d', mode:'lines+text',
            x:{xs}, y:{ys}, z:{zs},
            name:'札#{i+1}', legendgroup:'cards',
            line:{{color:'#FFD700',width:3}},
            text:['','','#{i+1}','',''],
            textposition:'top center',
            textfont:{{color:'#FFD700',size:13}}
        }}""")
        fx=[p[0] for p in c]; fy=[p[1] for p in c]; fz=[p[2] for p in c]
        traces.append(f"""{{
            type:'mesh3d', x:{fx}, y:{fy}, z:{fz},
            i:[0,0], j:[1,2], k:[2,3],
            color:'#8B6914', opacity:0.25,
            legendgroup:'cards', showlegend:false, hoverinfo:'skip'
        }}""")
    return traces


def build_html(segs, cards, title, n_total):
    traces = card_traces(cards)
    for i, seg in enumerate(segs):
        traces += seg_traces(seg, SEG_COLORS[i % len(SEG_COLORS)], i)

    traces_js = ",\n".join(traces)
    n_seg = len(segs)

    seg_badges = "".join(
        f'<span style="background:{SEG_COLORS[i%len(SEG_COLORS)]};color:#111;'
        f'border-radius:4px;padding:2px 8px;margin:2px;font-size:12px;cursor:pointer"'
        f' onclick="isolate({i})">'
        f'#{i+1} {segs[i][0].get("t",0):.1f}s〜{segs[i][-1].get("t",0):.1f}s</span>'
        for i in range(n_seg)
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body{{margin:0;background:#111;color:#eee;font-family:sans-serif}}
  #plot{{width:100vw;height:82vh}}
  #ctrl{{padding:6px 12px;font-size:12px;line-height:2}}
  button{{background:#333;color:#eee;border:1px solid #555;border-radius:4px;
          padding:3px 10px;margin:2px;cursor:pointer}}
  button:hover{{background:#555}}
</style>
</head>
<body>
<div id="ctrl">
  <b>{title}</b> &nbsp; 総フレーム:{n_total} &nbsp; 検出動作:{n_seg}件 &nbsp;
  <button onclick="showAll()">全表示</button>
  <button onclick="hideAll()">全非表示</button>
  &nbsp; 動作を選択:&nbsp;
  {seg_badges}
  <br>
  <span style="color:#888;font-size:11px">
    ドラッグ=回転 &nbsp; スクロール=ズーム &nbsp;
    バッジクリック=その動作だけ表示 &nbsp; 凡例クリック=個別切替
  </span>
</div>
<div id="plot"></div>
<script>
var data = [{traces_js}];
Plotly.newPlot('plot', data, {{
  paper_bgcolor:'#111', plot_bgcolor:'#111',
  scene:{{
    bgcolor:'#111',
    xaxis:{{title:'X',color:'#666',gridcolor:'#333'}},
    yaxis:{{title:'Y',color:'#666',gridcolor:'#333'}},
    zaxis:{{title:'Z',color:'#666',gridcolor:'#333'}},
    aspectmode:'data'
  }},
  legend:{{bgcolor:'#1e1e1e',font:{{color:'#ccc',size:12}},
           tracegroupgap:4, itemclick:'toggle'}},
  margin:{{l:0,r:160,t:0,b:0}}
}}, {{responsive:true}});

function showAll(){{
  Plotly.restyle('plot', {{visible:true}}, [...Array(data.length).keys()]);
}}
function hideAll(){{
  Plotly.restyle('plot', {{visible:'legendonly'}}, [...Array(data.length).keys()]);
}}
function isolate(segIdx){{
  // 全トレースを非表示にしてから指定セグメントだけ表示
  var updates = data.map(function(t,i){{
    var grp = t.legendgroup || '';
    var target = '動作#'+(segIdx+1);
    return grp === target || grp === 'cards' ? true : 'legendonly';
  }});
  Plotly.restyle('plot', 'visible', updates);
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?")
    parser.add_argument("--speed-thresh", type=float, default=0.03)
    parser.add_argument("--rest-frames",  type=int,   default=8)
    args = parser.parse_args()

    if args.log:
        log_path = args.log
    else:
        logs = sorted(glob.glob("output/realtime/rec_*/3d_log.json"))
        if not logs:
            print("録画データが見つかりません"); sys.exit(1)
        log_path = logs[-1]
        print(f"使用: {log_path}")

    frames = load_log(log_path)
    cards  = load_cards()
    title  = Path(log_path).parent.name

    segs = segment(frames, args.speed_thresh, args.rest_frames)
    if not segs:
        print("動作が検出できません。--speed-thresh を大きくしてみてください")
        sys.exit(1)

    html = build_html(segs, cards, title, len(frames))
    out  = str(Path(log_path).parent / "review_3d.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"保存: {out}")
    webbrowser.open(f"file://{os.path.abspath(out)}")

if __name__ == "__main__":
    main()
