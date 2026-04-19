"""
かるたの札を使ったクリックベース・ステレオキャリブレーション

チェッカーボードなしで、手元にある札を使ってキャリブレーションできる。

準備:
    - 机の上に札を 3〜4 枚並べる（両カメラに見える位置）
    - 札は等間隔でなくてよいが、同一平面（机の上）に置くこと

使い方:
    .venv/bin/python calibrate_click.py

手順:
    1. 起動したら「撮影」ウィンドウが開く
    2. SPACE で両カメラのフレームを同時取得
    3. 「カメラ1クリック」ウィンドウで、各札のコーナーを順番にクリック
    4. 「カメラ2クリック」ウィンドウで、同じ順序でコーナーをクリック
    5. ポイント数・実寸を入力して自動計算
    6. calibration/stereo_calib.json に保存

推奨配置（3枚の場合）:
    [札A] [札B] [札C]  ← 机の上に横並び

    クリック順: 各札の左上 → 右上 → 右下 → 左下
    （= 12点 × 2カメラ）
"""

import cv2
import numpy as np
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from demo_realtime import CameraThread, CAM_PC, CAM_PHONE

OUTPUT_PATH  = "calibration/stereo_calib.json"

# 札のデフォルトサイズ [m]（競技かるた 取り札）
CARD_W = 0.073
CARD_H = 0.052


# ---------- クリックUIで2D点を収集 ----------

def collect_clicks(frame: np.ndarray, title: str, n_points: int) -> list[tuple]:
    """
    フレームを表示してユーザーに n_points 個クリックさせる。
    クリックした2D座標のリストを返す。
    """
    points = []
    display = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((float(x), float(y)))
            idx = len(points)
            cv2.circle(display, (x, y), 6, (0, 255, 80), -1)
            cv2.putText(display, str(idx), (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
            cv2.imshow(title, display)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)
    cv2.setMouseCallback(title, on_click)

    while len(points) < n_points:
        overlay = display.copy()
        n = len(points)
        cv2.putText(overlay,
                    f"クリック: {n+1}/{n_points}点目  (z=1つ戻す  q=中断)",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)
        cv2.imshow(title, overlay)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(title)
            return []
        elif key == ord("z") and points:
            points.pop()
            # 再描画
            display = frame.copy()
            for i, (px, py) in enumerate(points):
                cv2.circle(display, (int(px), int(py)), 6, (0, 255, 80), -1)
                cv2.putText(display, str(i+1), (int(px)+8, int(py)-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

    cv2.waitKey(400)
    cv2.destroyWindow(title)
    return points


# ---------- 3D点を対話的に生成 ----------

def build_3d_points(n_cards: int, n_corners_per_card: int,
                    card_w: float, card_h: float,
                    spacing: float) -> np.ndarray:
    """
    横一列に並んだ n_cards 枚の札のコーナー3D座標を生成する。
    机の平面を Z=0 とする。

    クリック順: 左上 → 右上 → 右下 → 左下 (各札)

    Args:
        n_cards            : 札の枚数
        n_corners_per_card : 1枚あたりのコーナー数（通常4）
        card_w, card_h     : 札のサイズ [m]
        spacing            : 札の間隔 [m]

    Returns:
        Nx3 の3D座標配列（Z=0 の平面）
    """
    pts = []
    for i in range(n_cards):
        x0 = i * (card_w + spacing)
        # 左上, 右上, 右下, 左下
        corners = [
            (x0,          0,      0),
            (x0 + card_w, 0,      0),
            (x0 + card_w, card_h, 0),
            (x0,          card_h, 0),
        ]
        pts.extend(corners[:n_corners_per_card])
    return np.array(pts, dtype=np.float32)


# ---------- キャリブレーション ----------

def calibrate_from_clicks(pts_2d_1, pts_2d_2, pts_3d, img_size_1, img_size_2):
    """
    2カメラの2D点対応と既知3D点からステレオキャリブレーションを実行する。
    """
    pts_2d_1 = np.array(pts_2d_1, dtype=np.float32).reshape(-1, 1, 2)
    pts_2d_2 = np.array(pts_2d_2, dtype=np.float32).reshape(-1, 1, 2)
    pts_3d   = pts_3d.reshape(-1, 1, 3)

    obj_pts = [pts_3d]
    img_pts1 = [pts_2d_1]
    img_pts2 = [pts_2d_2]

    # 個別キャリブレーション（画像1枚なので精度は限定的）
    _, K1, d1, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts1, img_size_1, None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO
    )
    _, K2, d2, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts2, img_size_2, None, None,
        flags=cv2.CALIB_FIX_ASPECT_RATIO
    )

    # ステレオキャリブレーション
    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts1, img_pts2,
        K1, d1, K2, d2,
        img_size_1,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    return K1, K2, d1, d2, R, T, rms


# ---------- メイン ----------

def main():
    # カメラ起動
    cam_pc    = CameraThread(CAM_PC,    "Mac")
    cam_phone = CameraThread(CAM_PHONE, "Phone")
    cam_pc.start()
    cam_phone.start()
    time.sleep(1.0)

    print("=== 札クリックキャリブレーション ===")
    print("机の上に札を 3〜4 枚横並びにしてください。")
    print("SPACE でフレームを取得します。")

    # フレーム取得
    frame1 = frame2 = None
    while True:
        f1 = cam_pc.get_frame()
        f2 = cam_phone.get_frame()
        if f1 is None or f2 is None:
            time.sleep(0.01)
            continue

        disp = np.hstack([
            cv2.resize(f1, (640, 360)),
            cv2.resize(f2, (640, 360)),
        ])
        cv2.putText(disp, "SPACE=撮影  q=終了", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Capture", disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cam_pc.stop(); cam_phone.stop()
            cv2.destroyAllWindows(); return
        elif key == ord(" "):
            frame1, frame2 = f1.copy(), f2.copy()
            break

    cam_pc.stop()
    cam_phone.stop()
    cv2.destroyAllWindows()

    # パラメータ入力
    print("\n札の枚数を入力してください（例: 3）: ", end="")
    try:
        n_cards = int(input().strip())
    except ValueError:
        n_cards = 3

    print(f"1枚あたり4コーナーをクリックします（合計 {n_cards*4} 点 × 2カメラ）")
    print(f"札サイズ: {CARD_W*100:.1f}cm × {CARD_H*100:.1f}cm")
    print("札の間隔 [cm] を入力（密着なら 0）: ", end="")
    try:
        spacing = float(input().strip()) / 100
    except ValueError:
        spacing = 0.01

    n_total = n_cards * 4
    pts_3d  = build_3d_points(n_cards, 4, CARD_W, CARD_H, spacing)

    print(f"\n--- カメラ1（Mac）でクリック ---")
    print("各札の: 左上 → 右上 → 右下 → 左下 の順にクリック")
    pts1 = collect_clicks(frame1, "Camera1 - Mac (click corners)", n_total)
    if not pts1:
        print("中断しました"); return

    print(f"\n--- カメラ2（iPhone）でクリック ---")
    print("同じ順序でクリックしてください")
    pts2 = collect_clicks(frame2, "Camera2 - iPhone (click corners)", n_total)
    if not pts2:
        print("中断しました"); return

    # キャリブレーション
    print("\nキャリブレーション計算中...")
    img_size_1 = (frame1.shape[1], frame1.shape[0])
    img_size_2 = (frame2.shape[1], frame2.shape[0])

    try:
        K1, K2, d1, d2, R, T, rms = calibrate_from_clicks(
            pts1, pts2, pts_3d, img_size_1, img_size_2
        )
    except cv2.error as e:
        print(f"キャリブレーション失敗: {e}")
        print("点の数を増やすか、クリック位置を確認してください")
        return

    result = {
        "K1":         K1.tolist(),
        "K2":         K2.tolist(),
        "dist1":      d1.tolist(),
        "dist2":      d2.tolist(),
        "R":          R.tolist(),
        "T":          T.tolist(),
        "image_size": img_size_1,
        "n_images":   1,
        "rms_error":  round(rms, 4),
        "method":     "click-based (karuta cards)",
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # 札の3D座標を保存（3Dビューで表示するため）
    cards = []
    for i in range(n_cards):
        x0 = i * (CARD_W + spacing)
        cards.append({
            "id": i,
            "corners": [
                [x0,          0,      0],
                [x0 + CARD_W, 0,      0],
                [x0 + CARD_W, CARD_H, 0],
                [x0,          CARD_H, 0],
            ]
        })
    card_path = "calibration/card_positions.json"
    with open(card_path, "w") as f:
        json.dump({"cards": cards, "card_w": CARD_W, "card_h": CARD_H}, f, indent=2)

    print(f"\n保存: {OUTPUT_PATH}")
    print(f"保存: {card_path}")
    print(f"RMS再投影誤差: {rms:.4f} px")
    if rms > 5.0:
        print("  ※ 誤差が大きいです。クリック精度を上げるか点数を増やすと改善します")
    else:
        print("  ✓ 良好です")
    print("\ndemo_realtime.py を再起動すると自動で適用されます。")


if __name__ == "__main__":
    main()
