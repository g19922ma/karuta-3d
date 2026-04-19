"""
リアルタイム2視点3D手姿勢復元

2台のカメラから同時にフレームを取得し、MediaPipeで手を検出、
三角測量で3D座標を計算してリアルタイム表示する。

表示レイアウト:
    左上: Mac カメラ (2D骨格重ね)
    右上: iPhone  (2D骨格重ね)
    左下: 上から見た3D投影 (X-Z平面)
    右下: 正面から見た3D投影 (X-Y平面) + 座標数値

使い方:
    .venv/bin/python demo_realtime.py

操作:
    q     : 終了
    s     : 現在フレームをスナップショット保存
    t     : トレイル（軌跡）のクリア
    SPACE : 一時停止 / 再開
"""

import cv2
import numpy as np
import threading
import time
import os
import json
from collections import deque
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(__file__))

from detect_hand import HandDetector, TARGET_LANDMARKS, SKELETON_CONNECTIONS, POINT_COLORS_BGR
from triangulate import triangulate_landmarks
from calibration.simple_calib import get_camera_matrices

# ---------- 設定 ----------

CAM_PC    = 1   # Mac 内蔵カメラ
CAM_PHONE = 0   # iPhone (Continuity Camera)

DISPLAY_W = 640   # 各カメラ表示幅
DISPLAY_H = 360   # 各カメラ表示高さ
MINI_SIZE = 320   # 3D投影ビューのサイズ

TRAIL_LEN = 60    # 軌跡として残すフレーム数
DETECT_INTERVAL = 2  # N フレームに1回検出（1=毎フレーム, 2=半分）


# ---------- カメラスレッド ----------

class CameraThread(threading.Thread):
    """バックグラウンドでカメラを読み続け、最新フレームを保持するスレッド。"""

    def __init__(self, cam_id: int, name: str):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.name_ = name
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self._cap = None

    def run(self):
        self._cap = cv2.VideoCapture(self.cam_id)
        if not self._cap.isOpened():
            print(f"[{self.name_}] カメラ {self.cam_id} を開けませんでした")
            return
        while self.running:
            ret, frame = self._cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()


# ---------- 3D投影ビュー ----------

def make_projection_view(
    trail: deque,
    current: dict | None,
    view: str,       # "top" (X-Z) or "front" (X-Y)
    size: int = MINI_SIZE,
) -> np.ndarray:
    """
    3D点をOpenCV画像上に2D投影して描画する。
      top   : X軸=水平, Z軸=垂直（上から見た図）
      front : X軸=水平, Y軸=垂直（正面から見た図）
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    # 座標範囲（近似キャリブなので固定スケールで見やすく）
    x_range = (-0.5, 0.5)
    z_range = (0.2, 1.2)
    y_range = (-0.4, 0.4)

    def to_px(val, lo, hi):
        return int((val - lo) / (hi - lo) * (size - 40) + 20)

    def project(pt3d):
        x, y, z = pt3d
        if view == "top":
            px = to_px(x, *x_range)
            py = size - to_px(z, *z_range)   # Z は奥=上
        else:  # front
            px = to_px(x, *x_range)
            py = to_px(y, *y_range)
        return px, py

    # グリッド
    for i in range(0, size, size // 4):
        cv2.line(canvas, (i, 0), (i, size), (30, 30, 30), 1)
        cv2.line(canvas, (0, i), (size, i), (30, 30, 30), 1)

    # 軸ラベル
    label = "TOP (X-Z)" if view == "top" else "FRONT (X-Y)"
    cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    # トレイル（薄い軌跡）
    wrist_trail = [f["wrist"] for f in trail if "wrist" in f]
    for i in range(1, len(wrist_trail)):
        alpha = i / len(wrist_trail)
        color = (int(80 * alpha), int(80 * alpha), int(200 * alpha))
        p1 = project(wrist_trail[i - 1])
        p2 = project(wrist_trail[i])
        cv2.line(canvas, p1, p2, color, 1)

    # 現在フレームの骨格
    if current:
        for a, b in SKELETON_CONNECTIONS:
            if a in current and b in current:
                pa = project(current[a])
                pb = project(current[b])
                cv2.line(canvas, pa, pb, (180, 180, 180), 2)

        for name, pt3d in current.items():
            px, py = project(pt3d)
            color = POINT_COLORS_BGR.get(name, (200, 200, 200))
            cv2.circle(canvas, (px, py), 7, color, -1)
            short = {"wrist": "W", "index_finger_tip": "I", "middle_finger_tip": "M"}.get(name, "?")
            cv2.putText(canvas, short, (px + 8, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return canvas


# ---------- 2D 骨格描画 ----------

def draw_skeleton_2d(frame: np.ndarray, pts: dict, cam_label: str) -> np.ndarray:
    """フレームに2D骨格・ラベルを描画する。"""
    vis = frame.copy()
    for a, b in SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            color = POINT_COLORS_BGR.get(a, (200, 200, 200))
            cv2.line(vis,
                     (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])),
                     color, 2)
    for name, (x, y) in pts.items():
        color = POINT_COLORS_BGR.get(name, (200, 200, 200))
        cv2.circle(vis, (int(x), int(y)), 8, color, -1)
        short = {"wrist": "W", "index_finger_tip": "I", "middle_finger_tip": "M"}.get(name, "?")
        cv2.putText(vis, short, (int(x) + 10, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(vis, cam_label, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return vis


# ---------- 3D座標テキスト描画 ----------

def draw_3d_text(canvas: np.ndarray, pts3d: dict | None, fps: float):
    """座標数値と FPS を描画する。"""
    cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if pts3d is None:
        cv2.putText(canvas, "No hand detected", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
        return

    y = 55
    for name, (x, coord_y, z) in pts3d.items():
        short = {"wrist": "wrist  ", "index_finger_tip": "index  ",
                 "middle_finger_tip": "middle "}.get(name, name)
        color = POINT_COLORS_BGR.get(name, (200, 200, 200))
        text = f"{short}  X:{x:+.3f}  Y:{coord_y:+.3f}  Z:{z:.3f}"
        cv2.putText(canvas, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 26

    cv2.putText(canvas, "s=save  t=clear trail  q=quit",
                (10, canvas.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)


# ---------- メインループ ----------

def main():
    config = {}
    _, _, _, _, _, _, P1, P2 = get_camera_matrices(config)

    # カメラスレッド起動
    cam_pc    = CameraThread(CAM_PC,    "Mac")
    cam_phone = CameraThread(CAM_PHONE, "Phone")
    cam_pc.start()
    cam_phone.start()
    time.sleep(1.0)   # カメラ起動待ち

    print("リアルタイム3D復元を開始します")
    print("  q=終了  s=スナップショット  t=トレイルクリア  SPACE=一時停止")

    os.makedirs("output/realtime", exist_ok=True)

    with HandDetector() as detector:
        trail   = deque(maxlen=TRAIL_LEN)
        pts3d   = None
        pts_pc_last   = {}
        pts_phone_last = {}
        frame_count = 0
        paused  = False
        fps     = 0.0
        t_prev  = time.time()

        while True:
            if not paused:
                f_pc    = cam_pc.get_frame()
                f_phone = cam_phone.get_frame()

                if f_pc is None or f_phone is None:
                    time.sleep(0.01)
                    continue

                # FPS計算
                frame_count += 1
                now = time.time()
                if now - t_prev >= 0.5:
                    fps = frame_count / (now - t_prev)
                    frame_count = 0
                    t_prev = now

                # DETECT_INTERVAL フレームに1回検出
                if frame_count % DETECT_INTERVAL == 0:
                    p_pc    = detector.detect(f_pc)
                    p_phone = detector.detect(f_phone)
                    if p_pc:
                        pts_pc_last = p_pc
                    if p_phone:
                        pts_phone_last = p_phone

                    if pts_pc_last and pts_phone_last:
                        r = triangulate_landmarks(
                            pts_pc_last, pts_phone_last, P1, P2, TARGET_LANDMARKS
                        )
                        if r:
                            pts3d = {k: tuple(v) for k, v in r.items()}
                            trail.append(pts3d)
                    else:
                        pts3d = None

                # 2D骨格描画
                d_pc    = draw_skeleton_2d(f_pc,    pts_pc_last,    "Mac")
                d_phone = draw_skeleton_2d(f_phone, pts_phone_last, "iPhone")

                # リサイズして横並び
                d_pc    = cv2.resize(d_pc,    (DISPLAY_W, DISPLAY_H))
                d_phone = cv2.resize(d_phone, (DISPLAY_W, DISPLAY_H))
                top_row = np.hstack([d_pc, d_phone])

                # 3D投影ビュー
                top_view   = make_projection_view(trail, pts3d, view="top")
                front_view = make_projection_view(trail, pts3d, view="front")

                # 座標テキストパネル
                text_panel = np.zeros((MINI_SIZE, DISPLAY_W * 2 - MINI_SIZE * 2, 3), dtype=np.uint8)
                draw_3d_text(text_panel, pts3d, fps)

                bottom_row = np.hstack([top_view, front_view, text_panel])

                # 全体
                canvas = np.vstack([top_row, bottom_row])

                # 一時停止インジケータ
                if paused:
                    cv2.putText(canvas, "PAUSED", (canvas.shape[1]//2 - 60, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

                cv2.imshow("karuta-3d  realtime", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("t"):
                trail.clear()
                print("トレイルをクリアしました")
            elif key == ord("s"):
                ts = datetime.now().strftime("%H%M%S")
                path = f"output/realtime/snap_{ts}.png"
                cv2.imwrite(path, canvas)
                if pts3d:
                    json_path = f"output/realtime/snap_{ts}.json"
                    with open(json_path, "w") as f:
                        json.dump({"landmarks": pts3d}, f, indent=2)
                print(f"スナップショット保存: {path}")

    cam_pc.stop()
    cam_phone.stop()
    cv2.destroyAllWindows()
    print("終了")


if __name__ == "__main__":
    main()
