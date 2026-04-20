"""
複数カメラ同期用の時刻パターン表示ツール

iPad やノートPCでフルスクリーン表示する。
全カメラがこの画面を撮影することで、後処理で時刻が揃えられる。

表示内容:
    - 中央: QRコード（現在時刻を μs 単位で埋め込む）
    - 上部: 水平移動バー（サブフレーム精度用）
    - 下部: 人間が読める時刻テキスト（バックアップ）
    - 四隅: 位置合わせ用マーカー（カメラから見ても歪み補正可能）

使い方:
    .venv/bin/python sync_display.py
    → ウィンドウが開く。F または f キーでフルスクリーン切替。
    → 撮影中はカメラの画角内に画面を収めておく。

操作:
    f    : フルスクリーン切替
    q    : 終了
"""

import cv2
import numpy as np
import qrcode
import time
import argparse
from datetime import datetime


# ---------- QRコード生成 ----------

def make_qr_image(data: str, size: int) -> np.ndarray:
    """QRコードを指定サイズの白黒画像として生成する。"""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # 30%破損まで復元可
        box_size=10,
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    arr = np.array(img)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)


# ---------- 表示フレーム生成 ----------

def make_sync_frame(width: int, height: int, start_time: float,
                     qr_cache: dict) -> tuple[np.ndarray, int]:
    """
    同期フレームを1枚生成する。

    Returns:
        frame: BGR画像
        timestamp_us: 埋め込んだタイムスタンプ [μs]
    """
    now = time.time()
    elapsed_us = int((now - start_time) * 1_000_000)

    # 白背景
    frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # --- QRコード（中央）---
    qr_size = min(width, height) // 2
    # QRは毎フレーム生成すると重いので、ms単位でキャッシュ（同じms内は再利用）
    ms_key = elapsed_us // 1000
    if ms_key not in qr_cache:
        # データフォーマット: "karutas:<unix_time_us>"
        data = f"karutas:{int(now * 1_000_000)}"
        qr_cache[ms_key] = make_qr_image(data, qr_size)
        # キャッシュが大きくなりすぎないよう古いのを削除
        if len(qr_cache) > 300:
            oldest = min(qr_cache.keys())
            del qr_cache[oldest]

    qr = qr_cache[ms_key]
    qx = (width - qr_size) // 2
    qy = (height - qr_size) // 2
    frame[qy:qy+qr_size, qx:qx+qr_size] = qr

    # --- 四隅マーカー（黒枠＋中央点）---
    marker_size = 60
    for (cx, cy) in [(marker_size, marker_size),
                      (width - marker_size, marker_size),
                      (marker_size, height - marker_size),
                      (width - marker_size, height - marker_size)]:
        cv2.rectangle(frame, (cx - 30, cy - 30), (cx + 30, cy + 30), (0, 0, 0), 6)
        cv2.circle(frame, (cx, cy), 8, (0, 0, 0), -1)

    # --- 上部: 水平移動バー（サブフレーム精度用）---
    # 黒いバーが左から右へ一定速度で移動（1秒周期）
    bar_y1 = marker_size + 40
    bar_y2 = bar_y1 + 20
    bar_period_us = 1_000_000   # 1秒で1周
    bar_pos = int((elapsed_us % bar_period_us) / bar_period_us * width)
    cv2.rectangle(frame, (bar_pos - 15, bar_y1), (bar_pos + 15, bar_y2), (0, 0, 0), -1)
    # バーのトラック（参考線）
    cv2.line(frame, (0, (bar_y1 + bar_y2) // 2), (width, (bar_y1 + bar_y2) // 2),
             (180, 180, 180), 1)

    # --- 下部: テキスト時刻（人間用バックアップ）---
    dt_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    text_y = height - marker_size - 40
    cv2.putText(frame, dt_str, (width // 2 - 200, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f"karutas SYNC", (width // 2 - 90, text_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)

    return frame, elapsed_us


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser(description="カメラ同期用の時刻パターン表示")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    win_name = "karutas SYNC  (f=fullscreen, q=quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    start_time = time.time()
    qr_cache = {}
    is_fs = args.fullscreen

    print("=== 同期表示開始 ===")
    print("カメラの画角内に画面が収まるよう設置してください。")
    print("f = フルスクリーン切替, q = 終了")

    try:
        while True:
            frame, ts_us = make_sync_frame(
                args.width, args.height, start_time, qr_cache
            )
            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                is_fs = not is_fs
                if is_fs:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)

    finally:
        cv2.destroyAllWindows()
        print("終了")


if __name__ == "__main__":
    main()
