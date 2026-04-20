"""
キャリブレーション用 ArUco マーカーシートを生成する

A4 サイズに4個の ArUco マーカーを既知の位置で配置する。
印刷してフラットな面に置くだけで自動キャリブレーション可能。

使い方:
    .venv/bin/python generate_calib_sheet.py
    → calib_sheet.png が生成される
    → A4で印刷（拡大縮小なし・実寸100%で）
    → 印刷物を硬い板に貼るとさらに精度が上がる

出力:
    calib_sheet.png       - 印刷用画像
    calib_sheet_meta.json - マーカーIDと3D座標（m単位）
"""

import cv2
import numpy as np
import json
import os


# A4 サイズ (mm)
A4_W_MM = 210
A4_H_MM = 297

# 印刷DPI (高精細)
DPI = 300

# マーカーサイズと配置 (mm)
MARKER_SIZE_MM = 50    # 各マーカーの一辺
MARGIN_MM = 30         # シート端からマーカーまでの余白

# ArUco 辞書
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def generate():
    # キャンバス作成（白）
    w_px = mm_to_px(A4_W_MM)
    h_px = mm_to_px(A4_H_MM)
    sheet = np.full((h_px, w_px), 255, dtype=np.uint8)

    marker_px = mm_to_px(MARKER_SIZE_MM)
    margin_px = mm_to_px(MARGIN_MM)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    # 4個のマーカー位置（左上、右上、左下、右下）
    positions_mm = [
        (MARGIN_MM,                        MARGIN_MM),                         # 0: 左上
        (A4_W_MM - MARGIN_MM - MARKER_SIZE_MM, MARGIN_MM),                    # 1: 右上
        (MARGIN_MM,                        A4_H_MM - MARGIN_MM - MARKER_SIZE_MM),  # 2: 左下
        (A4_W_MM - MARGIN_MM - MARKER_SIZE_MM, A4_H_MM - MARGIN_MM - MARKER_SIZE_MM),  # 3: 右下
    ]

    markers_3d = {}   # id → 4角の3D座標 [m]（Z=0 の平面）
    for idx, (x_mm, y_mm) in enumerate(positions_mm):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, idx, marker_px)
        x_px, y_px = mm_to_px(x_mm), mm_to_px(y_mm)
        sheet[y_px:y_px + marker_px, x_px:x_px + marker_px] = marker_img

        # 3D座標（左上, 右上, 右下, 左下 の順）
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        size_m = MARKER_SIZE_MM / 1000.0
        markers_3d[idx] = [
            [x_m,          y_m,          0],   # 左上
            [x_m + size_m, y_m,          0],   # 右上
            [x_m + size_m, y_m + size_m, 0],   # 右下
            [x_m,          y_m + size_m, 0],   # 左下
        ]

        # 画面にIDを印字
        label = f"#{idx}"
        cv2.putText(sheet, label,
                    (x_px + marker_px // 2 - 20, y_px + marker_px + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, 80, 3)

    # 取扱説明をシート下部に印字
    footer_y = mm_to_px(A4_H_MM - 15)
    cv2.putText(sheet, "karuta-3d CALIBRATION SHEET  -  print at 100% scale (A4)",
                (mm_to_px(15), footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 100, 2)

    # 出力
    png_path = "calib_sheet.png"
    cv2.imwrite(png_path, sheet)

    meta = {
        "aruco_dict":     "DICT_4X4_50",
        "marker_size_mm": MARKER_SIZE_MM,
        "a4_size_mm":     [A4_W_MM, A4_H_MM],
        "dpi":            DPI,
        "markers_3d":     markers_3d,
        "corner_order":   ["top-left", "top-right", "bottom-right", "bottom-left"],
        "note":           "3D coords are on the Z=0 plane. Units: meters.",
    }
    meta_path = "calib_sheet_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"生成完了:")
    print(f"  画像   : {png_path}  ({sheet.shape[1]} x {sheet.shape[0]} px @ {DPI}dpi)")
    print(f"  メタ   : {meta_path}")
    print(f"\n使い方:")
    print(f"  1. {png_path} を A4 で100%印刷")
    print(f"  2. 硬い板などに貼って平面を保つ")
    print(f"  3. 両カメラに全マーカーが写るよう置く")
    print(f"  4. calibrate_auto.py を実行")


if __name__ == "__main__":
    generate()
