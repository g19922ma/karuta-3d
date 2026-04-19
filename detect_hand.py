"""
MediaPipe Hands による手のキーポイント検出 + 手動クリックフォールバック

mediapipe 0.10+ の Tasks API を使用。
モデルファイル hand_landmarker.task が必要（初回のみダウンロード）。

検出対象のランドマーク:
    wrist            : 手首 (idx=0)
    index_finger_tip : 人差し指先端 (idx=8)
    middle_finger_tip: 中指先端 (idx=12)
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- 設定 ----------

# MediaPipe ランドマーク番号
LANDMARK_INDEX = {
    "wrist":             0,
    "thumb_tip":         4,
    "index_finger_tip":  8,
    "middle_finger_tip": 12,
    "ring_finger_tip":   16,
    "pinky_tip":         20,
}

# デフォルトで取得するランドマーク名
TARGET_LANDMARKS = ["wrist", "index_finger_tip", "middle_finger_tip"]

# 可視化用スケルトン接続
SKELETON_CONNECTIONS = [
    ("wrist", "index_finger_tip"),
    ("wrist", "middle_finger_tip"),
    ("index_finger_tip", "middle_finger_tip"),
]

# モデルファイルのデフォルトパス
_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


# ---------- 検出器クラス ----------

class HandDetector:
    """
    MediaPipe Hand Landmarker (Tasks API) のラッパー。
    インスタンスを使い回してモデルの再初期化コストを避ける。
    """

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.5,
        model_path: str = _DEFAULT_MODEL,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {model_path}\n"
                "ダウンロード:\n"
                "  curl -L -o hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, target_landmarks: list[str] | None = None) -> dict | None:
        """
        フレームから指定ランドマークの2D座標を返す。

        Args:
            frame           : BGR 画像 (numpy array)
            target_landmarks: 取得するランドマーク名のリスト

        Returns:
            {landmark_name: (x_pixel, y_pixel)} または None（検出失敗）
        """
        if target_landmarks is None:
            target_landmarks = TARGET_LANDMARKS

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            return None

        h, w = frame.shape[:2]
        hand = result.hand_landmarks[0]  # 最初の手のみ使用

        points = {}
        for name in target_landmarks:
            idx = LANDMARK_INDEX.get(name)
            if idx is None:
                continue
            lm = hand[idx]
            points[name] = (lm.x * w, lm.y * h)

        return points

    def draw_landmarks(self, frame: np.ndarray, points: dict) -> np.ndarray:
        """検出結果をフレームに描画して返す（デバッグ用）。"""
        vis = frame.copy()
        for name, (x, y) in points.items():
            cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(vis, name[:3], (int(x) + 8, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for a, b in SKELETON_CONNECTIONS:
            if a in points and b in points:
                pt1 = (int(points[a][0]), int(points[a][1]))
                pt2 = (int(points[b][0]), int(points[b][1]))
                cv2.line(vis, pt1, pt2, (0, 200, 255), 2)
        return vis

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------- 手動クリックフォールバック ----------

def manual_click_fallback(frame: np.ndarray, landmark_names: list[str]) -> dict | None:
    """
    自動検出が失敗したとき、ユーザーがクリックでランドマークを指定する。

    操作:
        左クリック: 現在のランドマークを指定
        q キー    : スキップ（このフレームをスキップ）

    Returns:
        {landmark_name: (x, y)} / 全点指定できた場合のみ返す。途中でqを押すとNone。
    """
    points: dict = {}
    cursor = [0]
    display = frame.copy()
    win_name = "Manual Annotation (q=skip)"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and cursor[0] < len(landmark_names):
            name = landmark_names[cursor[0]]
            points[name] = (float(x), float(y))
            cv2.circle(display, (x, y), 6, (0, 255, 80), -1)
            cv2.putText(display, name, (x + 8, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1)
            cursor[0] += 1
            cv2.imshow(win_name, display)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)
    cv2.setMouseCallback(win_name, on_click)

    while cursor[0] < len(landmark_names):
        overlay = display.copy()
        remaining = landmark_names[cursor[0]]
        cv2.putText(overlay, f"Click: {remaining}  ({cursor[0]+1}/{len(landmark_names)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
        cv2.putText(overlay, "q = skip this frame",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow(win_name, overlay)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(win_name)
            return None  # スキップ

    cv2.waitKey(300)
    cv2.destroyWindow(win_name)
    return points
