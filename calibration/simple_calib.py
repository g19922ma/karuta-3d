"""
簡易キャリブレーション（仮パラメータによる動作確認用）

実際の計測値がない場合でも「とりあえず動く」状態にするための近似値。
精度より動作確認を優先している。本格運用は full_calib.py に差し替える。

【仮実装の限界】
- 焦点距離・歪みは機種依存。実機で計測すると精度が格段に上がる。
- 2カメラ間の相対姿勢（R, T）は手で測った数値。ステレオキャリブで置き換えること。
"""

import numpy as np


def get_camera_matrices(config: dict | None = None):
    """
    PC・スマホカメラの投影行列を近似値で返す。

    Args:
        config: JSON設定ファイルから読み込んだパラメータ（任意）

    Returns:
        K1, K2    : 内部パラメータ行列 (3x3)
        R1, R2    : 回転行列 (3x3)
        t1, t2    : 並進ベクトル (3x1)
        P1, P2    : 投影行列 P = K[R|t] (3x4)
    """
    if config is None:
        config = {}

    # --- PC カメラ (Webカメラ想定) ---
    w1 = config.get("pc_width", 1920)
    h1 = config.get("pc_height", 1080)
    # 焦点距離の近似: 35mm換算50mm相当 → fx ≈ width * 0.8 が経験則
    fx1 = config.get("pc_fx", w1 * 0.8)
    fy1 = config.get("pc_fy", w1 * 0.8)
    K1 = np.array(
        [[fx1, 0, w1 / 2],
         [0, fy1, h1 / 2],
         [0,   0,      1]], dtype=np.float64
    )

    # --- スマホカメラ ---
    w2 = config.get("phone_width", 1920)
    h2 = config.get("phone_height", 1080)
    # スマホは広角気味に設定
    fx2 = config.get("phone_fx", w2 * 0.9)
    fy2 = config.get("phone_fy", w2 * 0.9)
    K2 = np.array(
        [[fx2, 0, w2 / 2],
         [0, fy2, h2 / 2],
         [0,   0,      1]], dtype=np.float64
    )

    # --- カメラ1 (PC): 原点、正面向き ---
    R1 = np.eye(3, dtype=np.float64)
    t1 = np.zeros((3, 1), dtype=np.float64)

    # --- カメラ2 (スマホ): X方向にbaseline離れた位置から中心方向を向く ---
    baseline = config.get("baseline", 0.5)       # カメラ間距離[m]
    angle_deg = config.get("phone_angle_deg", -20.0)  # Y軸まわりの回転角[deg]（負=内向き）

    ang = np.radians(angle_deg)
    R2 = np.array(
        [[ np.cos(ang), 0, np.sin(ang)],
         [           0, 1,           0],
         [-np.sin(ang), 0, np.cos(ang)]], dtype=np.float64
    )
    t2 = np.array([[baseline], [0.0], [0.0]], dtype=np.float64)

    # 投影行列 P = K[R|t]
    P1 = K1 @ np.hstack([R1, t1])
    P2 = K2 @ np.hstack([R2, t2])

    print("[simple_calib] 近似パラメータを使用中。精度が低い場合は full_calib に切り替えてください。")
    print(f"  baseline={baseline}m, phone_angle={angle_deg}deg")

    return K1, K2, R1, R2, t1, t2, P1, P2
