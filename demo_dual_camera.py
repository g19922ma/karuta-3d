"""
デモ: PCカメラ + スマホ（Continuity Camera）を同時に使って3D復元

使い方:
    .venv/bin/python demo_dual_camera.py

    # カメラIDが違う場合
    .venv/bin/python demo_dual_camera.py --cam-pc 1 --cam-phone 0

操作:
    SPACE : 両カメラから同時にフレームを取得 → 検出 → 3D復元
    q     : 終了
"""

import cv2
import numpy as np
import sys
import os
import argparse
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # GUI不要・ファイル保存専用（macOS Tk非互換のため）
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

sys.path.insert(0, os.path.dirname(__file__))

from detect_hand import HandDetector, manual_click_fallback, TARGET_LANDMARKS, SKELETON_CONNECTIONS
from triangulate import triangulate_landmarks
from calibration.simple_calib import get_camera_matrices

# ランドマークの日本語ラベル
LANDMARK_LABELS = {
    "wrist":             "手首",
    "index_finger_tip":  "人差し指",
    "middle_finger_tip": "中指",
}

POINT_COLORS = {
    "wrist":             "#FF4444",
    "index_finger_tip":  "#44AAFF",
    "middle_finger_tip": "#44FF88",
}


def show_dual_preview(cap_pc: cv2.VideoCapture, cap_phone: cv2.VideoCapture) -> tuple[np.ndarray, np.ndarray] | None:
    """
    両カメラのプレビューを横並びで表示し、SPACEで同時キャプチャする。
    """
    print("両カメラのプレビューを表示中...")
    print("  SPACE : 撮影（両カメラ同時）")
    print("  q     : 終了")

    while True:
        ret1, f1 = cap_pc.read()
        ret2, f2 = cap_phone.read()

        if not ret1 or not ret2:
            print("カメラの読み取りに失敗しました")
            return None

        # 表示サイズを揃える（高さ480に統一）
        target_h = 480
        s1 = target_h / f1.shape[0]
        s2 = target_h / f2.shape[0]
        d1 = cv2.resize(f1, (int(f1.shape[1] * s1), target_h))
        d2 = cv2.resize(f2, (int(f2.shape[1] * s2), target_h))

        # ラベル描画
        cv2.putText(d1, "PC (Mac内蔵)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(d2, "Phone (iPhone)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
        cv2.putText(d1, "SPACE=撮影  q=終了", (10, target_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # 横並びに結合
        combined = np.hstack([d1, d2])
        cv2.imshow("Dual Camera Preview", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return None
        elif key == ord(" "):
            # 撮影確認表示
            flash = combined.copy()
            cv2.putText(flash, "CAPTURED!", (combined.shape[1]//2 - 100, combined.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            cv2.imshow("Dual Camera Preview", flash)
            cv2.waitKey(300)
            cv2.destroyAllWindows()
            return f1, f2


def draw_2d_detections(frame: np.ndarray, pts: dict, label: str) -> np.ndarray:
    """2Dフレームに検出点・ラベル・スケルトン線を描画する。"""
    vis = frame.copy()
    for name, (x, y) in pts.items():
        color_hex = POINT_COLORS.get(name, "#FFFFFF")
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        color_bgr = (b, g, r)
        cv2.circle(vis, (int(x), int(y)), 10, color_bgr, -1)
        jp = LANDMARK_LABELS.get(name, name)
        cv2.putText(vis, jp, (int(x) + 12, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
    for a, b_name in SKELETON_CONNECTIONS:
        if a in pts and b_name in pts:
            ca = POINT_COLORS.get(a, "#FFFFFF")
            bgr = (int(ca[5:7], 16), int(ca[3:5], 16), int(ca[1:3], 16))
            cv2.line(vis, (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b_name][0]), int(pts[b_name][1])), bgr, 2)
    cv2.putText(vis, label, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return vis


def plot_summary(
    frame_pc: np.ndarray,
    frame_phone: np.ndarray,
    pts_pc: dict,
    pts_phone: dict,
    result_3d: dict,
    output_path: str,
):
    """
    左: PC フレーム (2D検出点)
    中: Phoneフレーム (2D検出点)
    右: 3D復元結果
    の3列サマリ図を表示・保存する。
    """
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    # --- 左: PC フレーム ---
    ax1 = fig.add_subplot(1, 3, 1)
    vis_pc = draw_2d_detections(frame_pc, pts_pc, "Mac カメラ")
    ax1.imshow(cv2.cvtColor(vis_pc, cv2.COLOR_BGR2RGB))
    ax1.set_title("① Mac カメラ（2D検出点）", color="white", fontsize=12)
    ax1.axis("off")

    # --- 中: Phone フレーム ---
    ax2 = fig.add_subplot(1, 3, 2)
    vis_ph = draw_2d_detections(frame_phone, pts_phone, "iPhone")
    ax2.imshow(cv2.cvtColor(vis_ph, cv2.COLOR_BGR2RGB))
    ax2.set_title("② iPhone（2D検出点）", color="white", fontsize=12)
    ax2.axis("off")

    # --- 右: 3D復元 ---
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.set_facecolor("#0d0d1a")

    for name, coords in result_3d.items():
        x, y, z = coords
        color = POINT_COLORS.get(name, "#FFFFFF")
        label = LANDMARK_LABELS.get(name, name)
        ax3.scatter(x, y, z, color=color, s=120, zorder=5)
        ax3.text(x, y, z + 0.005, label, color=color, fontsize=9)

    for a, b_name in SKELETON_CONNECTIONS:
        if a in result_3d and b_name in result_3d:
            xs = [result_3d[a][0], result_3d[b_name][0]]
            ys = [result_3d[a][1], result_3d[b_name][1]]
            zs = [result_3d[a][2], result_3d[b_name][2]]
            color = POINT_COLORS.get(a, "#AAAAAA")
            ax3.plot(xs, ys, zs, color=color, linewidth=2.5, alpha=0.8)

    ax3.set_xlabel("X", color="white")
    ax3.set_ylabel("Y", color="white")
    ax3.set_zlabel("Z (奥行き)", color="white")
    ax3.tick_params(colors="white")
    ax3.set_title("③ 3D復元（2視点から計算）", color="white", fontsize=12)

    # 矢印で「流れ」を示す注釈
    fig.text(0.34, 0.5, "→", fontsize=28, color="#AAAAAA",
             ha="center", va="center")
    fig.text(0.355, 0.42, "2D対応点\nから三角測量", fontsize=8,
             color="#AAAAAA", ha="center", va="center")
    fig.text(0.67, 0.5, "→", fontsize=28, color="#AAAAAA",
             ha="center", va="center")

    plt.suptitle("2視点 → 3D手姿勢復元", color="white", fontsize=14, y=1.01)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"サマリ画像保存: {output_path}")
    # macOS のプレビューで自動的に開く
    os.system(f"open '{output_path}'")


def run_dual_demo(cam_pc: int, cam_phone: int, config: dict):
    """2カメラデモのメイン処理。"""

    # カメラを開く
    cap_pc    = cv2.VideoCapture(cam_pc)
    cap_phone = cv2.VideoCapture(cam_phone)

    if not cap_pc.isOpened():
        print(f"cam_id={cam_pc} を開けませんでした")
        return
    if not cap_phone.isOpened():
        print(f"cam_id={cam_phone} を開けませんでした")
        return

    print(f"cam_pc    = {cam_pc}  ({int(cap_pc.get(3))}x{int(cap_pc.get(4))})")
    print(f"cam_phone = {cam_phone}  ({int(cap_phone.get(3))}x{int(cap_phone.get(4))})")

    # 同時プレビュー → SPACE で撮影
    result = show_dual_preview(cap_pc, cap_phone)
    cap_pc.release()
    cap_phone.release()

    if result is None:
        return

    frame_pc, frame_phone = result

    # 撮影したフレームを保存
    out_dir = f"output/dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(f"{out_dir}/frame_pc.jpg",    frame_pc)
    cv2.imwrite(f"{out_dir}/frame_phone.jpg", frame_phone)
    print(f"\nフレーム保存: {out_dir}/")

    # 手の検出
    print("\n手の検出中...")
    with HandDetector() as detector:
        pts_pc    = detector.detect(frame_pc)
        pts_phone = detector.detect(frame_phone)

    if pts_pc is None:
        print("PC視点: 手が検出されませんでした → 手動クリックモード")
        pts_pc = manual_click_fallback(frame_pc, TARGET_LANDMARKS)
    else:
        print(f"PC視点: 検出OK  {list(pts_pc.keys())}")

    if pts_phone is None:
        print("Phone視点: 手が検出されませんでした → 手動クリックモード")
        pts_phone = manual_click_fallback(frame_phone, TARGET_LANDMARKS)
    else:
        print(f"Phone視点: 検出OK  {list(pts_phone.keys())}")

    if pts_pc is None or pts_phone is None:
        print("点の取得に失敗しました。終了します。")
        return

    # 三角測量
    print("\n三角測量...")
    _, _, _, _, _, _, P1, P2 = get_camera_matrices(config)
    result_3d = triangulate_landmarks(pts_pc, pts_phone, P1, P2, TARGET_LANDMARKS)

    if result_3d is None:
        print("3D復元に失敗しました")
        return

    # 結果表示
    print("\n=== 復元された3D座標（単位: 近似m, カメラ1原点） ===")
    for name, coords in result_3d.items():
        label = LANDMARK_LABELS.get(name, name)
        print(f"  {label:12s}: X={coords[0]:.3f}  Y={coords[1]:.3f}  Z={coords[2]:.3f}")

    # JSON 保存
    json_path = f"{out_dir}/3d_result.json"
    with open(json_path, "w") as f:
        json.dump({"landmarks": result_3d, "cam_pc": cam_pc, "cam_phone": cam_phone}, f, indent=2)
    print(f"JSON 保存: {json_path}")

    # サマリ可視化（2Dフレーム + 3D）
    print("可視化中...")
    summary_path = f"{out_dir}/summary.png"
    plot_summary(frame_pc, frame_phone, pts_pc, pts_phone, result_3d, summary_path)
    print(f"\n完了！結果: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2カメラ同時撮影 3D復元デモ")
    parser.add_argument("--cam-pc",    type=int, default=1, help="MacのカメラID (default: 1)")
    parser.add_argument("--cam-phone", type=int, default=0, help="iPhoneのカメラID (default: 0)")
    parser.add_argument("--config",    default=None, help="カメラ設定JSONパス")
    args = parser.parse_args()

    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)

    run_dual_demo(args.cam_pc, args.cam_phone, config)
