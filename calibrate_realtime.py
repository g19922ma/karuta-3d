"""
リアルタイム・インタラクティブ ステレオキャリブレーション

チェッカーボードを両カメラに見せながら対話的にキャリブレーションデータを収集し、
自動でステレオキャリブレーションを実行して stereo_calib.json に保存する。

チェッカーボードの準備:
    - 印刷する場合: 9x6 内側コーナーのチェッカーボードを A4 に印刷
    - 画面で代用:   generate_checkerboard.py で画像を生成してフルスクリーン表示

使い方:
    .venv/bin/python calibrate_realtime.py

操作:
    SPACE      : 両カメラで検出できていたらペアを取得
    c          : 収集済みデータでキャリブレーション実行（5ペア以上必要）
    d          : 最後に追加したペアを削除
    q          : 終了

推奨手順:
    1. チェッカーボードをいろいろな角度・距離で両カメラに見せる
    2. 緑のコーナーが両側に表示されたら SPACE
    3. 15〜20 ペア集めたら c でキャリブレーション
"""

import cv2
import numpy as np
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
from demo_realtime import CameraThread, CAM_PC, CAM_PHONE, DISPLAY_W, DISPLAY_H

# ---------- 設定 ----------

BOARD_COLS   = 9      # 内側コーナー列数
BOARD_ROWS   = 6      # 内側コーナー行数
SQUARE_SIZE  = 0.025  # 正方形の一辺 [m]（印刷時のサイズに合わせる）
MIN_PAIRS    = 5      # キャリブレーションに必要な最低ペア数
GOOD_PAIRS   = 15     # 推奨ペア数
OUTPUT_PATH  = "calibration/stereo_calib.json"


# ---------- キャリブレーション ----------

def run_calibration(obj_pts, img_pts1, img_pts2, img_size):
    """収集したデータでステレオキャリブレーションを実行する。"""
    print(f"\nキャリブレーション実行中... ({len(obj_pts)} ペア)")

    # 個別キャリブレーション
    _, K1, d1, _, _ = cv2.calibrateCamera(obj_pts, img_pts1, img_size, None, None)
    _, K2, d2, _, _ = cv2.calibrateCamera(obj_pts, img_pts2, img_size, None, None)

    # ステレオキャリブレーション
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts1, img_pts2,
        K1, d1, K2, d2,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    result = {
        "K1":         K1.tolist(),
        "K2":         K2.tolist(),
        "dist1":      d1.tolist(),
        "dist2":      d2.tolist(),
        "R":          R.tolist(),
        "T":          T.tolist(),
        "image_size": img_size,
        "n_images":   len(obj_pts),
        "rms_error":  round(rms, 4),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"保存完了: {OUTPUT_PATH}")
    print(f"  RMS再投影誤差: {rms:.4f} px  （1.0以下が目安）")
    return result


# ---------- メインUI ----------

def main():
    board_size = (BOARD_COLS, BOARD_ROWS)
    criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D物体点（Z=0 の平面）
    objp = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    obj_pts, img_pts1, img_pts2 = [], [], []
    img_size = None

    # カメラ起動
    cam_pc    = CameraThread(CAM_PC,    "Mac")
    cam_phone = CameraThread(CAM_PHONE, "Phone")
    cam_pc.start()
    cam_phone.start()
    time.sleep(1.0)

    print("=== ステレオキャリブレーション ===")
    print(f"チェッカーボード: {BOARD_COLS}x{BOARD_ROWS} 内側コーナー")
    print(f"SPACE=ペア取得  c=キャリブ実行  d=最後を削除  q=終了")
    print(f"目標: {GOOD_PAIRS} ペア\n")

    detect_interval = 3   # N フレームに1回検出（重いので間引く）
    frame_count = 0
    corners1_ok = None
    corners2_ok = None
    both_detected = False

    while True:
        f1 = cam_pc.get_frame()
        f2 = cam_phone.get_frame()

        if f1 is None or f2 is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # チェッカーボード検出（間引き）
        if frame_count % detect_interval == 0:
            g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = (g1.shape[1], g1.shape[0])

            ret1, c1 = cv2.findChessboardCorners(g1, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            ret2, c2 = cv2.findChessboardCorners(g2, board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret1:
                corners1_ok = cv2.cornerSubPix(g1, c1, (11,11), (-1,-1), criteria)
            else:
                corners1_ok = None
            if ret2:
                corners2_ok = cv2.cornerSubPix(g2, c2, (11,11), (-1,-1), criteria)
            else:
                corners2_ok = None

            both_detected = (corners1_ok is not None) and (corners2_ok is not None)

        # 表示用に描画
        d1 = cv2.resize(f1.copy(), (DISPLAY_W, DISPLAY_H))
        d2 = cv2.resize(f2.copy(), (DISPLAY_W, DISPLAY_H))

        # スケール係数（検出はフル解像度、表示はリサイズ後）
        sx1 = DISPLAY_W / f1.shape[1]
        sy1 = DISPLAY_H / f1.shape[0]
        sx2 = DISPLAY_W / f2.shape[1]
        sy2 = DISPLAY_H / f2.shape[0]

        if corners1_ok is not None:
            c1_disp = corners1_ok.copy()
            c1_disp[:, 0, 0] *= sx1
            c1_disp[:, 0, 1] *= sy1
            cv2.drawChessboardCorners(d1, board_size, c1_disp, True)

        if corners2_ok is not None:
            c2_disp = corners2_ok.copy()
            c2_disp[:, 0, 0] *= sx2
            c2_disp[:, 0, 1] *= sy2
            cv2.drawChessboardCorners(d2, board_size, c2_disp, True)

        # ステータスバー
        n = len(obj_pts)
        bar_color = (0, 255, 0) if both_detected else (0, 100, 255)
        status = "両カメラ検出OK！ SPACE で取得" if both_detected else "チェッカーボードを両カメラに見せてください"
        cv2.putText(d1, f"Mac  [{n}/{GOOD_PAIRS}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)
        cv2.putText(d2, f"iPhone  [{n}/{GOOD_PAIRS}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)

        # 横並び
        canvas = np.hstack([d1, d2])

        # 下部ステータスバー
        bar = np.zeros((50, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)
        hint = f"SPACE=取得  c=キャリブ実行({n}ペア)  d=削除  q=終了"
        cv2.putText(bar, hint, (canvas.shape[1] - 600, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # プログレスバー
        if GOOD_PAIRS > 0:
            prog_w = int(canvas.shape[1] * min(n, GOOD_PAIRS) / GOOD_PAIRS)
            cv2.rectangle(bar, (0, 44), (prog_w, 50), (0, 200, 100), -1)

        canvas = np.vstack([canvas, bar])
        cv2.imshow("Stereo Calibration", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            if both_detected:
                obj_pts.append(objp)
                img_pts1.append(corners1_ok)
                img_pts2.append(corners2_ok)
                print(f"  ペア {len(obj_pts)} 取得")
                # 取得フラッシュ
                flash = canvas.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]),
                              (0, 255, 0), 8)
                cv2.imshow("Stereo Calibration", flash)
                cv2.waitKey(200)
            else:
                print("  両カメラで検出できていません")

        elif key == ord("d"):
            if obj_pts:
                obj_pts.pop(); img_pts1.pop(); img_pts2.pop()
                print(f"  最後のペアを削除 (残り {len(obj_pts)} ペア)")

        elif key == ord("c"):
            if len(obj_pts) < MIN_PAIRS:
                print(f"  ペアが足りません: {len(obj_pts)}/{MIN_PAIRS}")
            else:
                cv2.destroyAllWindows()
                result = run_calibration(obj_pts, img_pts1, img_pts2, img_size)
                print("\nキャリブレーション完了！")
                print("demo_realtime.py を再起動すると自動的に適用されます。")
                break

    cam_pc.stop()
    cam_phone.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
