"""
競技かるた 複数視点3D手姿勢復元 - メインスクリプト

使い方:
    # 基本（フレーム120〜160 を処理）
    python main.py --pc video_pc.mp4 --phone video_phone.mp4 --start 120 --end 160

    # シンクオフセット指定
    python main.py --pc video_pc.mp4 --phone video_phone.mp4 --start 120 --end 160 --sync sync_offset.json

    # 本格キャリブレーション使用
    python main.py --pc video_pc.mp4 --phone video_phone.mp4 --start 120 --end 160 --calib-mode full

    # 自動検出失敗時の手動アノテーションを無効化
    python main.py --pc video_pc.mp4 --phone video_phone.mp4 --start 120 --end 160 --no-manual

事前準備:
    pip install -r requirements.txt
"""

import argparse
import json
import csv
import sys
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detect_hand import HandDetector, manual_click_fallback, TARGET_LANDMARKS
from triangulate import triangulate_landmarks
from visualize_3d import plot_3d_trajectory


# ================================================================
# ユーティリティ
# ================================================================

def load_config(config_path: str | None) -> dict:
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        print(f"[main] 設定ファイル読み込み: {config_path}")
        return cfg
    return {}


def load_sync_offset(sync_path: str | None) -> int:
    """sync_tool.py が出力した JSON からオフセットを取得する。"""
    if sync_path and os.path.exists(sync_path):
        with open(sync_path) as f:
            data = json.load(f)
        offset = data.get("frame_offset", 0)
        print(f"[main] シンクオフセット: {offset} (phone = pc + {offset})")
        return offset
    print("[main] シンクオフセット未指定 → 0 を使用")
    return 0


def get_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """指定フレームを取得する。"""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def make_output_dir(base: str) -> Path:
    """タイムスタンプ付きの出力ディレクトリを作成する。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


# ================================================================
# 保存
# ================================================================

def save_results(frames_3d: list[dict], output_dir: Path):
    """3D点列を CSV と JSON で保存する。"""

    # JSON
    json_path = output_dir / "3d_points.json"
    with open(json_path, "w") as f:
        json.dump({"frames": frames_3d}, f, indent=2)
    print(f"[main] JSON 保存: {json_path}")

    # CSV
    csv_path = output_dir / "3d_points.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "landmark", "x", "y", "z"])
        for item in frames_3d:
            for lm_name, coords in item["landmarks"].items():
                writer.writerow([item["frame_idx"], lm_name] + coords)
    print(f"[main] CSV 保存: {csv_path}")


# ================================================================
# フレームペア処理
# ================================================================

def process_frame_pair(
    frame_pc: np.ndarray,
    frame_phone: np.ndarray,
    detector: HandDetector,
    P1: np.ndarray,
    P2: np.ndarray,
    landmark_names: list[str],
    frame_idx: int,
    allow_manual: bool,
) -> dict | None:
    """
    1フレームペアを処理して {landmark_name: [x,y,z]} を返す。
    失敗した場合は None。
    """
    pts_pc    = detector.detect(frame_pc,    landmark_names)
    pts_phone = detector.detect(frame_phone, landmark_names)

    # フォールバック: 手動クリック
    if pts_pc is None:
        print(f"    PC frame {frame_idx}: 手の検出失敗")
        if allow_manual:
            print("    -> 手動アノテーションを開始します（PC）...")
            pts_pc = manual_click_fallback(frame_pc, landmark_names)

    if pts_phone is None:
        print(f"    Phone frame {frame_idx + _phone_offset_placeholder}: 手の検出失敗")
        if allow_manual:
            print("    -> 手動アノテーションを開始します（Phone）...")
            pts_phone = manual_click_fallback(frame_phone, landmark_names)

    if pts_pc is None or pts_phone is None:
        print(f"    フレーム {frame_idx}: スキップ（検出 or アノテーション失敗）")
        return None

    return triangulate_landmarks(pts_pc, pts_phone, P1, P2, landmark_names)

# グローバル変数（ログ表示用の簡易対応）
_phone_offset_placeholder = 0


# ================================================================
# メイン
# ================================================================

def main():
    global _phone_offset_placeholder

    parser = argparse.ArgumentParser(
        description="競技かるた 複数視点3D手姿勢復元",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pc",       required=True, help="PCカメラ動画パス")
    parser.add_argument("--phone",    required=True, help="スマホ動画パス")
    parser.add_argument("--start",    type=int, required=True, help="処理開始フレーム（PCフレーム番号）")
    parser.add_argument("--end",      type=int, required=True, help="処理終了フレーム（PCフレーム番号）")
    parser.add_argument("--sync",     default=None,   help="シンクオフセットJSON (sync_tool.py 出力)")
    parser.add_argument("--config",   default=None,   help="カメラ設定JSONパス")
    parser.add_argument("--output",   default="output", help="出力ディレクトリ")
    parser.add_argument("--step",     type=int, default=1, help="フレームステップ（n枚に1枚処理）")
    parser.add_argument("--no-manual", action="store_true", help="手動アノテーションを無効化")
    parser.add_argument("--calib-mode", choices=["simple", "full"], default="simple",
                        help="キャリブレーションモード（simple=近似, full=チェッカーボード結果）")
    parser.add_argument("--no-show", action="store_true", help="3D可視化のウィンドウ表示をスキップ")
    args = parser.parse_args()

    # -------------------- キャリブレーション読み込み --------------------
    config = load_config(args.config)

    if args.calib_mode == "simple":
        from calibration.simple_calib import get_camera_matrices
        K1, K2, R1, R2, t1, t2, P1, P2 = get_camera_matrices(config)
    else:
        from calibration.full_calib import load_calibration
        K1, K2, R1, R2, t1, t2, P1, P2 = load_calibration(config)

    # -------------------- シンクオフセット --------------------
    offset = load_sync_offset(args.sync)
    _phone_offset_placeholder = offset

    # -------------------- ビデオを開く --------------------
    cap_pc    = cv2.VideoCapture(args.pc)
    cap_phone = cv2.VideoCapture(args.phone)

    if not cap_pc.isOpened():
        print(f"[main] ERROR: PCビデオを開けません: {args.pc}", file=sys.stderr)
        sys.exit(1)
    if not cap_phone.isOpened():
        print(f"[main] ERROR: スマホビデオを開けません: {args.phone}", file=sys.stderr)
        sys.exit(1)

    pc_total    = int(cap_pc.get(cv2.CAP_PROP_FRAME_COUNT))
    phone_total = int(cap_phone.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[main] PC: {pc_total} frames, Phone: {phone_total} frames")
    print(f"[main] 処理範囲: frame {args.start} 〜 {args.end}, step={args.step}")

    # -------------------- 処理ループ --------------------
    landmark_names = TARGET_LANDMARKS
    frames_3d: list[dict] = []
    frame_range = range(args.start, args.end + 1, args.step)
    total = len(frame_range)

    with HandDetector() as detector:
        for i, pc_idx in enumerate(frame_range):
            phone_idx = pc_idx + offset
            print(f"  [{i+1:3d}/{total}] PC:{pc_idx}  Phone:{phone_idx}")

            frame_pc = get_frame(cap_pc, pc_idx)
            if frame_pc is None:
                print(f"    PC frame {pc_idx} 読み取り失敗")
                continue

            frame_phone = get_frame(cap_phone, phone_idx)
            if frame_phone is None:
                print(f"    Phone frame {phone_idx} 読み取り失敗（範囲外の可能性）")
                continue

            result = process_frame_pair(
                frame_pc, frame_phone,
                detector, P1, P2,
                landmark_names, pc_idx,
                allow_manual=not args.no_manual,
            )

            if result is not None:
                frames_3d.append({"frame_idx": pc_idx, "landmarks": result})

    cap_pc.release()
    cap_phone.release()

    # -------------------- 結果確認 --------------------
    print(f"\n[main] 復元成功: {len(frames_3d)} / {total} フレーム")

    if not frames_3d:
        print("[main] 3D点が1つも取れませんでした。")
        print("  ヒント: --no-manual を外して手動アノテーションを試すか、")
        print("          カメラパラメータ (config.json) を確認してください。")
        sys.exit(1)

    # -------------------- 保存 --------------------
    out_dir = make_output_dir(args.output)
    save_results(frames_3d, out_dir)

    # -------------------- 可視化 --------------------
    landmark_dicts = [item["landmarks"] for item in frames_3d]
    vis_path = str(out_dir / "3d_visualization.png")
    plot_3d_trajectory(
        landmark_dicts,
        landmark_names,
        output_path=vis_path,
        show=not args.no_show,
        title=f"Hand 3D  (frames {args.start}–{args.end})",
    )

    print(f"\n[main] 完了! 結果: {out_dir}/")


if __name__ == "__main__":
    main()
