"""
Nカメラ対応 三角測量モジュール

- 2視点: cv2.triangulatePoints を使用（高速）
- 3視点以上: Direct Linear Transform (DLT) による最小二乗解
- 動的ペア選択: 検出できたカメラだけで最適復元
- 片方欠け耐性: あるカメラで見えない点も他のカメラで補完可能
"""

import numpy as np
import cv2


# ============================================================
# 基本三角測量
# ============================================================

def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """
    2カメラの2D対応点から3D座標を復元する（旧API互換）。

    Args:
        pts1, pts2 : Nx2 の2D点配列
        P1, P2     : 3x4 投影行列

    Returns:
        Nx3 の3D座標配列
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    w = points_4d[3]
    return (points_4d[:3] / w).T


# ============================================================
# DLT (Direct Linear Transform) による N視点三角測量
# ============================================================

def triangulate_dlt(
    points_2d: list[tuple[float, float]],
    projection_matrices: list[np.ndarray],
) -> np.ndarray:
    """
    DLT による N 視点三角測量（1点のみ）。

    各カメラから x = P X という投影式を線形方程式に並べて最小二乗で解く。
    X = [X, Y, Z, 1]^T（同次座標）

    Args:
        points_2d           : [(x1, y1), (x2, y2), ...] 各カメラでの2D座標
        projection_matrices : [P1, P2, ...] 対応する投影行列

    Returns:
        [X, Y, Z] の3D座標（numpy array）
    """
    if len(points_2d) != len(projection_matrices):
        raise ValueError("points_2d と projection_matrices の数が一致しません")
    if len(points_2d) < 2:
        raise ValueError("三角測量には最低2視点が必要です")

    A = []
    for (x, y), P in zip(points_2d, projection_matrices):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.asarray(A, dtype=np.float64)

    # SVDで最小二乗解を求める（Ax=0 の最小特異値に対応する右特異ベクトル）
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]   # 同次座標の正規化
    return X[:3]


def triangulate_dlt_batch(
    points_2d_list: list[list[tuple]],
    projection_matrices: list[np.ndarray],
) -> np.ndarray:
    """
    複数点を一括でDLT三角測量する。

    Args:
        points_2d_list      : [[pt_cam1, pt_cam2, ...], [...], ...] 外側が点、内側がカメラ
        projection_matrices : 全カメラの投影行列

    Returns:
        Nx3 の3D座標配列
    """
    return np.array([
        triangulate_dlt(pts, projection_matrices)
        for pts in points_2d_list
    ])


# ============================================================
# 再投影誤差
# ============================================================

def reprojection_error(
    pt3d: np.ndarray,
    pt2d: tuple[float, float],
    P: np.ndarray,
) -> float:
    """1カメラでの再投影誤差（ピクセル）を返す。"""
    X = np.append(pt3d, 1.0)
    x_proj = P @ X
    x_proj = x_proj[:2] / x_proj[2]
    return float(np.linalg.norm(x_proj - np.array(pt2d)))


def mean_reprojection_error(
    pt3d: np.ndarray,
    points_2d: list[tuple[float, float]],
    projection_matrices: list[np.ndarray],
) -> float:
    """全カメラの平均再投影誤差を返す。"""
    errors = [
        reprojection_error(pt3d, p, P)
        for p, P in zip(points_2d, projection_matrices)
    ]
    return float(np.mean(errors))


# ============================================================
# ランドマーク辞書ベースの三角測量（Nカメラ対応・動的ペア選択）
# ============================================================

def triangulate_landmarks(
    pts_cam1: dict,
    pts_cam2: dict,
    P1: np.ndarray,
    P2: np.ndarray,
    landmark_names: list[str],
) -> dict | None:
    """
    （旧API互換）2カメラでランドマーク辞書を三角測量する。

    Args:
        pts_cam1, pts_cam2 : {landmark_name: (x, y)}
        P1, P2             : 投影行列
        landmark_names     : 処理対象

    Returns:
        {landmark_name: [x, y, z]} / 1点も対応なければ None
    """
    result = triangulate_landmarks_nview(
        [pts_cam1, pts_cam2], [P1, P2], landmark_names
    )
    return result["landmarks"] if result else None


def triangulate_landmarks_nview(
    detections: list[dict | None],
    projection_matrices: list[np.ndarray],
    landmark_names: list[str],
    min_cameras: int = 2,
    reprojection_threshold: float = 20.0,
) -> dict | None:
    """
    Nカメラからの検出結果を統合して3D復元する。

    動作:
    - 各ランドマークについて、それを検出できたカメラを抽出
    - 2視点以上で見えている点のみDLTで復元
    - 再投影誤差が閾値を超えた点は信頼度低とマーク

    Args:
        detections             : 各カメラの検出結果 [{name: (x,y)}, None, ...]
                                 Noneは検出失敗を意味する
        projection_matrices    : 各カメラの投影行列（同じ長さ）
        landmark_names         : 処理対象のランドマーク名
        min_cameras            : 3D復元に必要な最低カメラ数
        reprojection_threshold : この誤差を超えた点は low_confidence とする

    Returns:
        {
            "landmarks":  {name: [x, y, z]},
            "confidence": {name: float (誤差px)},
            "used_cams":  {name: [cam_idx, ...]},
        }
        1点も復元できなければ None
    """
    if len(detections) != len(projection_matrices):
        raise ValueError("detections と projection_matrices の数が一致しません")

    result_landmarks = {}
    result_confidence = {}
    result_used_cams = {}

    for name in landmark_names:
        # この点が見えているカメラを収集
        seen_pts = []
        seen_P = []
        seen_cam_idx = []
        for cam_idx, (det, P) in enumerate(zip(detections, projection_matrices)):
            if det is not None and name in det:
                seen_pts.append(det[name])
                seen_P.append(P)
                seen_cam_idx.append(cam_idx)

        if len(seen_pts) < min_cameras:
            continue

        # DLT 三角測量
        pt3d = triangulate_dlt(seen_pts, seen_P)
        err  = mean_reprojection_error(pt3d, seen_pts, seen_P)

        result_landmarks[name] = pt3d.tolist()
        result_confidence[name] = err
        result_used_cams[name] = seen_cam_idx

    if not result_landmarks:
        return None

    return {
        "landmarks":  result_landmarks,
        "confidence": result_confidence,
        "used_cams":  result_used_cams,
    }


# ============================================================
# ベストペア選択（Layer A用：2視点に絞り込む）
# ============================================================

def select_best_pair(
    detections: list[dict | None],
    landmark_name: str = "wrist",
) -> tuple[int, int] | None:
    """
    指定ランドマークが最も安定して見えているカメラペアを選ぶ。

    いまは「両方で検出できている最初のペア」を返す簡易版。
    将来的には視差角・再投影誤差などで最適化できる。

    Returns:
        (cam_idx_1, cam_idx_2) / 該当なしは None
    """
    visible = [
        i for i, d in enumerate(detections)
        if d is not None and landmark_name in d
    ]
    if len(visible) < 2:
        return None
    return visible[0], visible[1]
