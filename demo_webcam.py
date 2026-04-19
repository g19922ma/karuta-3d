"""
デモ: PCのWebカメラ1台だけで動作確認する最小スクリプト

「2台の動画」が手元にない場合でも、
  - カメラを1台使って左・右から2回撮影したフレームを「2視点」とみなす
  - または同一フレームに視差を擬似的に加えて三角測量を確認する

使い方:
    .venv/bin/python demo_webcam.py

操作:
    SPACE  : フレームを取得（2回押して2視点分を取得）
    q      : 終了

2枚取得後、自動で手の検出 → 3D復元 → 可視化します。
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from detect_hand import HandDetector, manual_click_fallback, TARGET_LANDMARKS
from triangulate import triangulate_landmarks
from visualize_3d import plot_single_frame
from calibration.simple_calib import get_camera_matrices


def capture_frames_from_webcam(cam_id: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Webカメラを開いてSPACEで2フレームをキャプチャする。
    1枚目=PC視点、2枚目=スマホ視点として扱う（位置を少し変えて撮影する）。
    """
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return None

    frames = []
    print("=== デモモード: Webカメラキャプチャ ===")
    print(f"SPACE を {2 - len(frames)} 回押してフレームを取得してください")
    print("  1枚目 → カメラを正面に向けて撮影（PC視点）")
    print("  2枚目 → カメラを少し右にずらして撮影（スマホ視点の代替）")
    print("q = 終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        n = len(frames)
        labels = ["[1枚目] PC視点 (SPACE=撮影)", "[2枚目] スマホ視点 (SPACE=撮影)"]
        label = labels[n] if n < 2 else "取得完了"
        cv2.putText(display, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(display, f"取得済み: {n}/2", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.imshow("Demo Webcam", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == ord(" ") and n < 2:
            frames.append(frame.copy())
            print(f"  フレーム {n+1} 取得完了")
            if len(frames) == 2:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) < 2:
        print("フレームが足りません")
        return None

    return frames[0], frames[1]


def fake_stereo_from_single(frame: np.ndarray, shift_px: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """
    1枚の画像を横にシフトして疑似ステレオペアを作る。
    視差を人工的に与えるのでカメラ内でのデモ用。
    """
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    shifted = cv2.warpAffine(frame, M, (w, h))
    return frame, shifted


def run_demo(use_fake_stereo: bool = False):
    """デモを実行する。"""

    # ---- フレーム取得 ----
    if use_fake_stereo:
        print("=== 疑似ステレオモード（1枚をシフト） ===")
        cap = cv2.VideoCapture(0)
        print("SPACE でフレームを1枚取得します...")
        frame1 = None
        while True:
            ret, f = cap.read()
            if not ret:
                break
            cv2.putText(f, "SPACE=撮影  q=終了", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Demo", f)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(" "):
                frame1 = f.copy()
                break
        cap.release()
        cv2.destroyAllWindows()
        if frame1 is None:
            return
        frame_pc, frame_phone = fake_stereo_from_single(frame1, shift_px=60)
    else:
        result = capture_frames_from_webcam()
        if result is None:
            return
        frame_pc, frame_phone = result

    # ---- 手の検出 ----
    print("\n手の検出中...")
    with HandDetector() as detector:
        pts_pc    = detector.detect(frame_pc)
        pts_phone = detector.detect(frame_phone)

    if pts_pc is None:
        print("PC視点: 手が検出できませんでした → 手動入力モードへ")
        pts_pc = manual_click_fallback(frame_pc, TARGET_LANDMARKS)
    else:
        print(f"PC視点: 検出OK  {pts_pc}")

    if pts_phone is None:
        print("スマホ視点: 手が検出できませんでした → 手動入力モードへ")
        pts_phone = manual_click_fallback(frame_phone, TARGET_LANDMARKS)
    else:
        print(f"スマホ視点: 検出OK  {pts_phone}")

    if pts_pc is None or pts_phone is None:
        print("点の取得に失敗しました。終了します。")
        return

    # ---- 三角測量 ----
    # 疑似ステレオの場合はシフト量に合わせて baseline を小さく設定
    config = {"baseline": 0.05} if use_fake_stereo else {}
    _, _, _, _, _, _, P1, P2 = get_camera_matrices(config)

    print("\n三角測量中...")
    result_3d = triangulate_landmarks(pts_pc, pts_phone, P1, P2, TARGET_LANDMARKS)

    if result_3d is None:
        print("3D復元に失敗しました")
        return

    print("\n=== 復元された3D座標 ===")
    for name, coords in result_3d.items():
        print(f"  {name:25s}: {[f'{c:.4f}' for c in coords]}")

    # ---- 保存 ----
    os.makedirs("output/demo", exist_ok=True)
    cv2.imwrite("output/demo/frame_pc.jpg",    frame_pc)
    cv2.imwrite("output/demo/frame_phone.jpg", frame_phone)
    print("\nフレーム保存: output/demo/frame_pc.jpg, frame_phone.jpg")

    # ---- 3D可視化 ----
    print("3D可視化...")
    plot_single_frame(
        result_3d,
        TARGET_LANDMARKS,
        output_path="output/demo/3d_result.png",
        show=True,
        title="Demo: Single Frame 3D Hand",
    )
    print("保存: output/demo/3d_result.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="karuta-3d デモ（Webカメラ1台で確認）")
    parser.add_argument("--fake-stereo", action="store_true",
                        help="1枚をシフトした疑似ステレオで動作確認（カメラを動かさなくてOK）")
    args = parser.parse_args()

    run_demo(use_fake_stereo=args.fake_stereo)
