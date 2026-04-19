"""
本格ステレオキャリブレーション（チェッカーボードまたは ArUco 使用）

【使い方】
1. 2台のカメラで同じチェッカーボードを様々な角度から撮影する（10〜20枚推奨）
2. PC側の画像を calibration_images/calib_pc_*.jpg に保存
   スマホ側の画像を calibration_images/calib_phone_*.jpg に保存
3. python calibration/full_calib.py --images-dir calibration_images/ を実行
4. calibration/stereo_calib.json が生成される

main.py で --calib-mode full を指定すれば自動的にこのファイルを使う。
"""

import json
import os
import glob
import numpy as np
import cv2
import argparse


# --- キャリブレーション結果のロード ---

def load_calibration(config: dict) -> tuple:
    """
    stereo_calib.json からキャリブレーション結果を読み込んで投影行列を返す。

    Returns:
        K1, K2, R1, R2, t1, t2, P1, P2
    """
    calib_path = config.get("calibration_file", "calibration/stereo_calib.json")

    if not os.path.exists(calib_path):
        raise FileNotFoundError(
            f"キャリブレーションファイルが見つかりません: {calib_path}\n"
            f"先に実行してください: python calibration/full_calib.py"
        )

    with open(calib_path) as f:
        data = json.load(f)

    K1 = np.array(data["K1"], dtype=np.float64)
    K2 = np.array(data["K2"], dtype=np.float64)
    R  = np.array(data["R"],  dtype=np.float64)   # カメラ1→2 の回転
    T  = np.array(data["T"],  dtype=np.float64)   # カメラ1→2 の並進

    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)
    R2 = R
    t2 = T.reshape(3, 1)

    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([R2, t2])

    print(f"[full_calib] {calib_path} からキャリブレーション読み込み完了")
    return K1, K2, R1, R2, t1, t2, P1, P2


# --- チェッカーボードによるステレオキャリブレーション ---

def calibrate_stereo_checkerboard(
    images_dir: str,
    output_path: str = "calibration/stereo_calib.json",
    board_size: tuple = (9, 6),   # 内側コーナー数 (cols, rows)
    square_size: float = 0.025,   # 正方形の一辺 [m]
):
    """
    チェッカーボード画像ペアからステレオキャリブレーションを実行する。

    画像命名規則:
        calibration_images/calib_pc_001.jpg
        calibration_images/calib_phone_001.jpg
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3D物体点の生成（Z=0 の平面）
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    pc_images    = sorted(glob.glob(f"{images_dir}/calib_pc_*.jpg"))
    phone_images = sorted(glob.glob(f"{images_dir}/calib_phone_*.jpg"))

    if len(pc_images) == 0:
        raise FileNotFoundError(f"PC画像が見つかりません: {images_dir}/calib_pc_*.jpg")
    if len(pc_images) != len(phone_images):
        raise ValueError(f"PC={len(pc_images)}枚, Phone={len(phone_images)}枚 → 数が合いません")

    obj_points, img_pts1, img_pts2 = [], [], []
    img_size = None

    for p1, p2 in zip(pc_images, phone_images):
        g1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        g2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        if g1 is None or g2 is None:
            print(f"  スキップ: {p1} or {p2}")
            continue

        if img_size is None:
            img_size = (g1.shape[1], g1.shape[0])  # (width, height)

        ret1, c1 = cv2.findChessboardCorners(g1, board_size)
        ret2, c2 = cv2.findChessboardCorners(g2, board_size)

        if ret1 and ret2:
            c1 = cv2.cornerSubPix(g1, c1, (11, 11), (-1, -1), criteria)
            c2 = cv2.cornerSubPix(g2, c2, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_pts1.append(c1)
            img_pts2.append(c2)
            print(f"  OK: {os.path.basename(p1)}")
        else:
            print(f"  コーナー検出失敗: {os.path.basename(p1)} (ret1={ret1}, ret2={ret2})")

    n_valid = len(obj_points)
    if n_valid < 5:
        raise RuntimeError(f"有効なペアが足りません: {n_valid}枚（最低5枚必要）")

    print(f"\n有効ペア数: {n_valid}")

    # 個別キャリブレーション
    _, K1, d1, _, _ = cv2.calibrateCamera(obj_points, img_pts1, img_size, None, None)
    _, K2, d2, _, _ = cv2.calibrateCamera(obj_points, img_pts2, img_size, None, None)

    # ステレオキャリブレーション（内部パラメータは固定）
    _, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_pts1, img_pts2,
        K1, d1, K2, d2,
        img_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    result = {
        "K1":      K1.tolist(),
        "K2":      K2.tolist(),
        "dist1":   d1.tolist(),
        "dist2":   d2.tolist(),
        "R":       R.tolist(),
        "T":       T.tolist(),
        "image_size": img_size,
        "n_images": n_valid,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"キャリブレーション結果を保存: {output_path}")
    return result


# --- ArUco (TODO) ---

def calibrate_stereo_aruco(images_dir: str, output_path: str):
    """ArUco マーカーによるキャリブレーション（未実装）。"""
    raise NotImplementedError("ArUco キャリブレーションは未実装です。checkerboard を使用してください。")


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ステレオキャリブレーション実行")
    parser.add_argument("--images-dir", default="calibration_images", help="キャリブレーション画像ディレクトリ")
    parser.add_argument("--pattern", choices=["checkerboard", "aruco"], default="checkerboard")
    parser.add_argument("--output", default="calibration/stereo_calib.json")
    parser.add_argument("--board-cols", type=int, default=9, help="チェッカーボード内側コーナー列数")
    parser.add_argument("--board-rows", type=int, default=6, help="チェッカーボード内側コーナー行数")
    parser.add_argument("--square-size", type=float, default=0.025, help="正方形の一辺 [m]")
    args = parser.parse_args()

    if args.pattern == "checkerboard":
        calibrate_stereo_checkerboard(
            args.images_dir,
            output_path=args.output,
            board_size=(args.board_cols, args.board_rows),
            square_size=args.square_size,
        )
    else:
        calibrate_stereo_aruco(args.images_dir, args.output)
