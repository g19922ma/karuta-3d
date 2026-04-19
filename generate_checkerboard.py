"""
キャリブレーション用チェッカーボード画像を生成する。

スマホやタブレットの画面に表示して使う（印刷不要）。

使い方:
    .venv/bin/python generate_checkerboard.py
    → checkerboard.png が生成される
    → この画像をフルスクリーンで別デバイスに表示して両カメラに見せる
"""

import cv2
import numpy as np

COLS        = 10   # マス目の列数（内側コーナー = COLS-1 = 9）
ROWS        = 7    # マス目の行数（内側コーナー = ROWS-1 = 6）
SQUARE_PX   = 80   # 1マスのピクセルサイズ
OUTPUT_PATH = "checkerboard.png"

h = ROWS * SQUARE_PX
w = COLS * SQUARE_PX
img = np.zeros((h, w), dtype=np.uint8)

for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:
            y1, y2 = r * SQUARE_PX, (r + 1) * SQUARE_PX
            x1, x2 = c * SQUARE_PX, (c + 1) * SQUARE_PX
            img[y1:y2, x1:x2] = 255

# 白枠を追加（端のコーナーが見やすくなる）
bordered = cv2.copyMakeBorder(img, SQUARE_PX, SQUARE_PX, SQUARE_PX, SQUARE_PX,
                               cv2.BORDER_CONSTANT, value=255)
cv2.imwrite(OUTPUT_PATH, bordered)
print(f"生成: {OUTPUT_PATH}  ({bordered.shape[1]}x{bordered.shape[0]}px)")
print(f"内側コーナー: {COLS-1} x {ROWS-1}")
print("この画像をフルスクリーンで表示して、calibrate_realtime.py に見せてください。")
