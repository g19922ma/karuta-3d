"""
配信用ハイライト動画生成の前処理

録画データ（3d_log.json）を配信品質向けに整形して
Three.js ベースのビューア（highlight/index.html）が読める形式で出力する。

処理:
    1. フレーム取得（既存の3d_log.jsonを読み込み）
    2. 欠損補間（検出失敗フレームを線形補間）
    3. 滑らか平滑化（一次IIRローパスフィルタ）
    4. 札位置の読み込み
    5. Take瞬間の自動検出（速度ピーク後の急減速）
    6. JSON出力

使い方:
    .venv/bin/python render_highlight.py output/realtime/rec_YYYYMMDD_HHMMSS/3d_log.json

    # 最新の録画を自動選択
    .venv/bin/python render_highlight.py
"""

import json
import os
import sys
import glob
import argparse
import webbrowser
import http.server
import socketserver
import threading
import numpy as np
from pathlib import Path


LANDMARKS = ["wrist", "index_finger_tip", "middle_finger_tip"]


# ---------- データ読み込み ----------

def load_log(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["frames"]


def load_cards(path: str = "calibration/card_positions.json") -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("cards", [])


# ---------- 補間と平滑化 ----------

def fill_missing_frames(frames: list[dict]) -> list[dict]:
    """
    検出失敗フレームを直近有効フレームから線形補間する。
    """
    out = []
    for lm in LANDMARKS:
        coords = []
        for f in frames:
            p = f.get("landmarks", {}).get(lm)
            coords.append(np.array(p) if p else None)

        # 前後の有効フレームから線形補間
        n = len(coords)
        for i in range(n):
            if coords[i] is None:
                # 前後を探す
                prev_idx = next((j for j in range(i-1, -1, -1) if coords[j] is not None), None)
                next_idx = next((j for j in range(i+1, n)    if coords[j] is not None), None)
                if prev_idx is not None and next_idx is not None:
                    t = (i - prev_idx) / (next_idx - prev_idx)
                    coords[i] = coords[prev_idx] * (1 - t) + coords[next_idx] * t
                elif prev_idx is not None:
                    coords[i] = coords[prev_idx]
                elif next_idx is not None:
                    coords[i] = coords[next_idx]
                else:
                    coords[i] = np.zeros(3)

        for i, c in enumerate(coords):
            if i >= len(out):
                out.append({"t": frames[i].get("t", i/30), "landmarks": {}})
            out[i]["landmarks"][lm] = c.tolist()

    return out


def smooth_trajectories(frames: list[dict], alpha: float = 0.35) -> list[dict]:
    """
    一次IIRローパスで平滑化する。alpha 大きいほど滑らか。
    """
    if len(frames) < 2:
        return frames

    smoothed = {lm: None for lm in LANDMARKS}
    out = []
    for f in frames:
        new_f = {"t": f["t"], "landmarks": {}}
        for lm in LANDMARKS:
            cur = np.array(f["landmarks"][lm])
            if smoothed[lm] is None:
                smoothed[lm] = cur
            else:
                smoothed[lm] = alpha * smoothed[lm] + (1 - alpha) * cur
            new_f["landmarks"][lm] = smoothed[lm].tolist()
        out.append(new_f)
    return out


# ---------- Take瞬間の検出 ----------

def detect_take_moments(frames: list[dict],
                         speed_peak: float = 0.8,
                         decel_thresh: float = 0.5) -> list[float]:
    """
    手首の速度プロファイルから Take の瞬間を検出する。

    シンプルなヒューリスティック:
      1. 速度が speed_peak 以上になった後
      2. 加速度が -decel_thresh 以下（急減速）になるフレーム
    """
    n = len(frames)
    if n < 3:
        return []

    # 時系列
    times = np.array([f["t"] for f in frames])
    wrist = np.array([f["landmarks"]["wrist"] for f in frames])

    # 速度
    dt = np.diff(times)
    dt[dt < 1e-6] = 1e-6
    velocity = np.linalg.norm(np.diff(wrist, axis=0), axis=1) / dt
    # 加速度（速度の微分）
    dv = np.diff(velocity)

    takes = []
    armed = False
    for i in range(len(dv)):
        if velocity[i] > speed_peak:
            armed = True
        if armed and dv[i] < -decel_thresh:
            takes.append(float(times[i + 1]))
            armed = False

    return takes


# ---------- 札平面に座標系を合わせる ----------

def align_to_card_plane(frames: list[dict], cards: list) -> tuple[list[dict], list]:
    """
    札が置かれている平面を Y=0（地面）にして、平面の法線を Y+（上）にする。
    Three.js は Y-up なので、これで「札が地面、手が上から」という自然な向きになる。
    """
    if not cards:
        print("  札がないので座標変換はスキップ")
        return frames, cards

    # 札の全コーナーから平面を推定
    all_corners = []
    for card in cards:
        for c in card["corners"]:
            all_corners.append(c)
    pts = np.array(all_corners, dtype=np.float64)

    # 重心
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # SVDで平面の法線を求める
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]   # 最小特異値に対応
    # 上向きに向ける（zが正の方向）
    if normal[2] < 0:
        normal = -normal

    # 法線を Y+ に揃える回転を作る
    up_axis = np.array([0, 1, 0], dtype=np.float64)
    # normal と up_axis の外積から回転軸、内積から角度
    v = np.cross(normal, up_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, up_axis)

    if s < 1e-8:
        # すでに平行
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([
            [    0, -v[2],  v[1]],
            [ v[2],     0, -v[0]],
            [-v[1],  v[0],     0],
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    def transform(p):
        p_arr = np.asarray(p, dtype=np.float64)
        return (R @ (p_arr - centroid)).tolist()

    # フレームを変換
    new_frames = []
    for f in frames:
        nf = {"t": f["t"], "landmarks": {}}
        for lm in LANDMARKS:
            nf["landmarks"][lm] = transform(f["landmarks"][lm])
        new_frames.append(nf)

    # 札を変換
    new_cards = []
    for card in cards:
        new_cards.append({
            **card,
            "corners": [transform(c) for c in card["corners"]],
        })

    print(f"  平面合わせ: 法線 {normal.round(3)} -> Y+ に回転")
    return new_frames, new_cards


# ---------- 出力 ----------

def build_output(frames: list[dict], cards: list, takes: list,
                  input_path: str) -> dict:
    """
    Three.js ビューアが読む形式に変換する。
    """
    return {
        "meta": {
            "source":      input_path,
            "n_frames":    len(frames),
            "duration":    frames[-1]["t"] - frames[0]["t"] if len(frames) > 1 else 0,
            "landmarks":   LANDMARKS,
            "connections": [
                ["wrist", "index_finger_tip"],
                ["wrist", "middle_finger_tip"],
                ["index_finger_tip", "middle_finger_tip"],
            ],
        },
        "frames": [
            {"t": round(f["t"], 4),
             "points": {lm: [round(v, 5) for v in f["landmarks"][lm]] for lm in LANDMARKS}}
            for f in frames
        ],
        "cards": cards,
        "take_moments": [round(t, 3) for t in takes],
    }


# ---------- ローカルサーバ起動（fetchがfile://だと動かないので） ----------

class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def start_server(port: int = 8011, directory: str = None) -> tuple[socketserver.TCPServer, int]:
    """
    Three.js ビューア用の静的ファイルサーバを立てる。
    指定ポートが使用中なら +1 ずつずらして空きを探す。
    """
    if directory:
        os.chdir(directory)
    handler = http.server.SimpleHTTPRequestHandler
    for tried in range(20):
        try:
            httpd = _ReusableTCPServer(("127.0.0.1", port + tried), handler)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            return httpd, port + tried
        except OSError:
            continue
    raise RuntimeError("空きポートが見つかりません")


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser(description="配信用ハイライトデータ生成")
    parser.add_argument("log", nargs="?", help="3d_log.json のパス")
    parser.add_argument("--smooth", type=float, default=0.35,
                        help="平滑化 α (0=平滑化なし, 0.5=中, 0.8=強い)")
    parser.add_argument("--no-open", action="store_true", help="ブラウザを開かない")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    if args.log:
        log_path = args.log
    else:
        logs = sorted(glob.glob("output/realtime/rec_*/3d_log.json"))
        if not logs:
            print("録画が見つかりません")
            sys.exit(1)
        log_path = logs[-1]
        print(f"最新の録画を使用: {log_path}")

    # 処理
    frames_raw = load_log(log_path)
    print(f"読み込み: {len(frames_raw)} フレーム")

    frames = fill_missing_frames(frames_raw)
    print(f"欠損補間後: {len(frames)} フレーム")

    if args.smooth > 0:
        frames = smooth_trajectories(frames, alpha=args.smooth)
        print(f"平滑化 α={args.smooth}")

    cards = load_cards()
    print(f"札: {len(cards)} 個")

    # 札を地面として座標系を揃える
    frames, cards = align_to_card_plane(frames, cards)

    takes = detect_take_moments(frames)
    print(f"Take検出: {len(takes)} 件 at {takes}")

    output = build_output(frames, cards, takes, log_path)

    # 書き出し（highlight/data.json）
    out_path = "highlight/data.json"
    os.makedirs("highlight", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n出力: {out_path}")

    # ビューア起動
    if not args.no_open:
        httpd, used_port = start_server(args.port)
        url = f"http://127.0.0.1:{used_port}/highlight/"
        print(f"ビューア起動: {url}")
        webbrowser.open(url)
        print("Ctrl+C で終了")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n終了")
            httpd.shutdown()


if __name__ == "__main__":
    main()
