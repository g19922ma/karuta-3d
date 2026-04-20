"""
録画動画群から同期用QRコードを読み取り、各カメラのフレーム時刻を揃える。

使い方:
    # 複数動画を一括処理（multi_cam_record.py の出力）
    .venv/bin/python sync_decode.py output/multi_rec_20260420_120000/

    # 個別動画を処理
    .venv/bin/python sync_decode.py --videos video1.mp4 video2.mp4

    # スキャン間隔を指定（高速化）
    .venv/bin/python sync_decode.py output/multi_rec_.../ --step 5

出力:
    <入力ディレクトリ>/sync_offsets.json
    {
        "video_0.mp4": {
            "frame_to_time_us": {...},    # frame_idx → unix_time_us
            "offset_frames":     0,        # 基準カメラからのフレームオフセット
            "quality":           0.95     # QR検出成功率
        },
        ...
    }
"""

import cv2
import numpy as np
import json
import argparse
import glob
import os
import sys
from pathlib import Path


# ---------- QR検出 ----------

def decode_qr_from_frame(frame: np.ndarray, detector) -> int | None:
    """
    フレームから karutas: 形式のQRコードを読み、タイムスタンプ[μs]を返す。
    検出失敗なら None。
    """
    try:
        data, bbox, _ = detector.detectAndDecode(frame)
    except cv2.error:
        return None
    if not data or not data.startswith("karutas:"):
        return None
    try:
        return int(data.split(":")[1])
    except (ValueError, IndexError):
        return None


# ---------- 動画スキャン ----------

def scan_video(video_path: str, step: int = 3,
               max_frames: int | None = None) -> dict:
    """
    動画を走査して各フレームのタイムスタンプを抽出する。

    Args:
        video_path : 動画パス
        step       : N フレームに1回QR検出（高速化）
        max_frames : 最大処理フレーム数（デバッグ用）

    Returns:
        {
          "video":             path,
          "total_frames":      N,
          "fps":               fps,
          "frame_to_time_us":  {frame_idx: timestamp_us},
          "success_rate":      成功率
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画を開けません: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    detector = cv2.QRCodeDetector()

    frame_to_time = {}
    checked = 0
    found   = 0

    print(f"[{Path(video_path).name}] スキャン中... ({total} frames)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if frame_idx % step == 0:
            ts = decode_qr_from_frame(frame, detector)
            checked += 1
            if ts is not None:
                frame_to_time[frame_idx] = ts
                found += 1

        frame_idx += 1
        if frame_idx % 100 == 0:
            rate = found / max(checked, 1) * 100
            print(f"  frame {frame_idx}/{total}  QR検出率 {rate:.1f}%  "
                  f"(成功 {found}/{checked})")

    cap.release()

    rate = found / max(checked, 1)
    print(f"[{Path(video_path).name}] 完了: QR検出率 {rate*100:.1f}%")

    return {
        "video":            video_path,
        "total_frames":     total,
        "fps":              fps,
        "frame_to_time_us": frame_to_time,
        "success_rate":     rate,
        "n_checked":        checked,
        "n_found":          found,
    }


# ---------- 時刻マッピング ----------

def fit_linear_model(frame_to_time: dict) -> dict:
    """
    frame_idx → timestamp_us の関係を線形近似する（t = a * f + b）。
    検出できなかったフレームの時刻を補間で求めるのに使う。
    """
    if len(frame_to_time) < 2:
        return None

    frames = np.array(sorted(frame_to_time.keys()), dtype=float)
    times  = np.array([frame_to_time[int(f)] for f in frames], dtype=float)

    # 最小二乗
    a, b = np.polyfit(frames, times, 1)

    # 残差
    residuals = times - (a * frames + b)
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "a":        float(a),          # μs per frame
        "b":        float(b),          # offset μs
        "rms_us":   rms,
        "n_points": len(frame_to_time),
    }


# ---------- 複数カメラの整合 ----------

def align_cameras(scan_results: list[dict]) -> dict:
    """
    スキャン結果を整合し、各カメラのフレームオフセットを計算する。

    基準: 最も検出成功率が高いカメラ
    """
    # 線形近似
    for r in scan_results:
        r["linear"] = fit_linear_model(r["frame_to_time_us"])

    # 基準カメラを選ぶ
    valid = [r for r in scan_results if r.get("linear")]
    if not valid:
        return {"error": "どのカメラからもQRを検出できませんでした"}

    ref = max(valid, key=lambda r: r["success_rate"])
    print(f"\n基準カメラ: {Path(ref['video']).name} (検出率 {ref['success_rate']*100:.1f}%)")

    # 各カメラのフレーム0における時刻 = b
    # 基準カメラのフレームXに相当する他カメラのフレームY は:
    # t_ref(X) = a_ref * X + b_ref
    # t_other(Y) = a_other * Y + b_other
    # 同じ時刻なら Y = (a_ref * X + b_ref - b_other) / a_other

    alignments = {}
    for r in scan_results:
        if r is ref:
            alignments[r["video"]] = {
                "is_reference":  True,
                "offset_frames": 0,
                "success_rate":  r["success_rate"],
                "rms_us":        r["linear"]["rms_us"] if r["linear"] else None,
            }
            continue

        if not r["linear"]:
            alignments[r["video"]] = {"error": "QR検出不足", "success_rate": r["success_rate"]}
            continue

        # ref のフレーム 0 に対応する other のフレーム
        # t_ref_0 = b_ref
        # → other で t_ref_0 になるのは (b_ref - b_other) / a_other
        a_ref, b_ref = ref["linear"]["a"], ref["linear"]["b"]
        a_oth, b_oth = r["linear"]["a"],   r["linear"]["b"]

        other_frame_at_ref_start = (b_ref - b_oth) / a_oth

        alignments[r["video"]] = {
            "is_reference":        False,
            "offset_frames":       float(other_frame_at_ref_start),
            "fps_ratio":           a_oth / a_ref,   # 1.0が理想、FPSが違えば1から外れる
            "success_rate":        r["success_rate"],
            "rms_us":              r["linear"]["rms_us"],
        }

    return {
        "reference_video": ref["video"],
        "alignments":       alignments,
    }


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser(description="QR時刻パターンから動画群を同期")
    parser.add_argument("input", nargs="?", help="録画ディレクトリ（multi_cam_record.pyの出力）")
    parser.add_argument("--videos", nargs="+", help="個別に動画ファイルを指定する場合")
    parser.add_argument("--step",   type=int, default=3, help="QR検出間隔（大きくすると高速化）")
    parser.add_argument("--max-frames", type=int, default=None, help="最大処理フレーム数（デバッグ用）")
    args = parser.parse_args()

    # 入力動画の決定
    if args.videos:
        videos = args.videos
        output_dir = Path(args.videos[0]).parent
    elif args.input:
        input_dir = Path(args.input)
        videos = sorted(glob.glob(str(input_dir / "cam_*.mp4")))
        if not videos:
            print(f"動画が見つかりません: {input_dir}/cam_*.mp4")
            sys.exit(1)
        output_dir = input_dir
    else:
        parser.error("入力ディレクトリまたは --videos を指定してください")

    print(f"対象動画: {len(videos)}本")
    for v in videos:
        print(f"  {v}")

    # 各動画をスキャン
    scan_results = []
    for v in videos:
        result = scan_video(v, step=args.step, max_frames=args.max_frames)
        scan_results.append(result)

    # アライメント
    alignment = align_cameras(scan_results)

    # 保存
    out_path = output_dir / "sync_offsets.json"
    with open(out_path, "w") as f:
        # frame_to_time_us は多いので要約のみ保存
        slim_scans = []
        for r in scan_results:
            slim_scans.append({
                "video":        r["video"],
                "total_frames": r["total_frames"],
                "fps":          r["fps"],
                "n_found":      r["n_found"],
                "n_checked":    r["n_checked"],
                "success_rate": r["success_rate"],
                "linear":       r.get("linear"),
            })
        json.dump({
            "alignment": alignment,
            "scans":     slim_scans,
        }, f, indent=2)

    # 結果表示
    print(f"\n=== 結果 ===")
    print(f"保存: {out_path}")
    print(f"\n基準カメラ: {Path(alignment['reference_video']).name}")
    print(f"\nフレームオフセット:")
    for video, align in alignment["alignments"].items():
        name = Path(video).name
        if "error" in align:
            print(f"  {name}: エラー ({align['error']})")
        elif align.get("is_reference"):
            print(f"  {name}: 基準 (RMS {align['rms_us']:.1f}μs)")
        else:
            off = align["offset_frames"]
            rms = align["rms_us"]
            ratio = align.get("fps_ratio", 1.0)
            print(f"  {name}: offset={off:+.2f}frames  "
                  f"FPS比={ratio:.4f}  RMS={rms:.1f}μs")


if __name__ == "__main__":
    main()
