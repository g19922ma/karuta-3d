"""
2視点の2D対応点から三角測量で3D座標を復元する

cv2.triangulatePoints を使用。
入力はピクセル座標、出力はカメラ座標系での3D座標（単位はconfigのbaselineに依存）。
"""

import numpy as np
import cv2


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """
    2カメラの2D対応点から3D座標を復元する。

    Args:
        pts1 : Nx2 の2D点配列（カメラ1 = PC側）
        pts2 : Nx2 の2D点配列（カメラ2 = スマホ側）
        P1   : カメラ1の投影行列 (3x4)
        P2   : カメラ2の投影行列 (3x4)

    Returns:
        Nx3 の3D座標配列
    """
    pts1 = np.array(pts1, dtype=np.float64)
    pts2 = np.array(pts2, dtype=np.float64)

    # cv2.triangulatePoints は 2xN を期待する
    pts1_T = pts1.T  # shape: (2, N)
    pts2_T = pts2.T

    # 同次座標 (4xN) で返る
    points_4d = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)

    # 非同次座標に変換 (Nx3)
    w = points_4d[3]
    points_3d = (points_4d[:3] / w).T

    return points_3d


def triangulate_landmarks(
    pts_cam1: dict,
    pts_cam2: dict,
    P1: np.ndarray,
    P2: np.ndarray,
    landmark_names: list[str],
) -> dict | None:
    """
    ランドマーク名付き辞書形式で三角測量を実行する。

    Args:
        pts_cam1       : {landmark_name: (x, y)} from camera 1
        pts_cam2       : {landmark_name: (x, y)} from camera 2
        P1, P2         : 投影行列
        landmark_names : 処理対象のランドマーク名リスト

    Returns:
        {landmark_name: [x3d, y3d, z3d]} / 1点も対応がなければ None
    """
    coords1, coords2, valid_names = [], [], []

    for name in landmark_names:
        if name in pts_cam1 and name in pts_cam2:
            coords1.append(pts_cam1[name])
            coords2.append(pts_cam2[name])
            valid_names.append(name)

    if not valid_names:
        return None

    pts_3d = triangulate_points(
        np.array(coords1),
        np.array(coords2),
        P1, P2,
    )

    return {name: pts_3d[i].tolist() for i, name in enumerate(valid_names)}
