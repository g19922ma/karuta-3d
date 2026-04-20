"""
高FPSリアルタイム2視点3D手姿勢復元（Rhizomatiks風ビジュアル）

アーキテクチャ: フルパイプライン並列化
    [Camera Thread #1] ──┐
                         ├──> [Fusion Thread] → [Display Main]
    [Camera Thread #2] ──┤
         │               │
         └─> [Detector #1] ─┐
             [Detector #2] ─┘

各ステージは独立スレッドで動き、常に「最新の値」を読む。
検出が遅くても表示は止まらない → 実FPSを最大化。

使い方:
    .venv/bin/python demo_live.py

操作:
    ドラッグ=回転  ホイール=ズーム
    r=録画開始/停止  t=軌跡クリア  SPACE=一時停止  q=終了
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
from viewer_3d import Viewer3D, load_cards

# ---------- 設定 ----------

CAM_PC, CAM_PHONE = 1, 0
DISPLAY_W, DISPLAY_H = 640, 360
VIEW3D_SIZE = 720
TRAIL_LEN = 120
STEREO_CALIB_PATH = "calibration/stereo_calib.json"


# ============================================================
# スレッド: カメラキャプチャ
# ============================================================

class CameraThread(threading.Thread):
    """カメラから最新フレームを常に保持する。"""
    def __init__(self, cam_id, name, target_fps=60):
        super().__init__(daemon=True)
        self.cam_id, self.name = cam_id, name
        self.target_fps = target_fps
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self._cap = None
        self.actual_fps = 0.0

    def run(self):
        self._cap = cv2.VideoCapture(self.cam_id)
        # 高FPS設定を試みる
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # バッファを最小化してレイテンシ削減
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            print(f"[{self.name}] カメラ {self.cam_id} を開けませんでした")
            return

        actual = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[{self.name}] cam={self.cam_id}, request={self.target_fps}fps, actual={actual:.0f}fps")

        t_prev, count = time.time(), 0
        while self.running:
            ret, frame = self._cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                count += 1
                now = time.time()
                if now - t_prev >= 1.0:
                    self.actual_fps = count / (now - t_prev)
                    count, t_prev = 0, now
            else:
                time.sleep(0.001)

    def get(self):
        with self.lock:
            return (self.frame.copy(), self.frame_id) if self.frame is not None else (None, -1)

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()


# ============================================================
# スレッド: 手検出（カメラごとに1つ）
# ============================================================

class DetectorThread(threading.Thread):
    """
    CameraThread から最新フレームを継続的に取得して MediaPipe で検出する。
    結果は latest_pts/latest_frame_id に保持。
    """
    def __init__(self, cam_thread: CameraThread, name: str,
                 target_hand: str = "Right"):
        super().__init__(daemon=True)
        self.cam = cam_thread
        self.name = name
        self.target_hand = target_hand
        self.running = True
        self.lock = threading.Lock()
        self.latest_pts = None
        self.latest_frame_id = -1
        self.last_detected_pts = None  # フォールバック用
        self.last_processed_id = -1
        self.actual_fps = 0.0

    def run(self):
        detector = HandDetector(max_hands=2, target_hand=self.target_hand)
        t_prev, count = time.time(), 0
        try:
            while self.running:
                frame, frame_id = self.cam.get()
                if frame is None or frame_id == self.last_processed_id:
                    time.sleep(0.001)
                    continue
                self.last_processed_id = frame_id

                pts = detector.detect(frame)
                with self.lock:
                    self.latest_pts = pts
                    self.latest_frame_id = frame_id
                    if pts:
                        self.last_detected_pts = pts

                count += 1
                now = time.time()
                if now - t_prev >= 1.0:
                    self.actual_fps = count / (now - t_prev)
                    count, t_prev = 0, now
        finally:
            detector.close()

    def get(self):
        with self.lock:
            return self.latest_pts, self.latest_frame_id, self.last_detected_pts

    def stop(self):
        self.running = False


# ============================================================
# キャリブ読み込み
# ============================================================

def load_projection_matrices():
    if os.path.exists(STEREO_CALIB_PATH):
        from calibration.full_calib import load_calibration
        print(f"[live] キャリブ読み込み: {STEREO_CALIB_PATH}")
        _,_,_,_,_,_, P1, P2 = load_calibration({"calibration_file": STEREO_CALIB_PATH})
    else:
        print("[live] 近似キャリブを使用")
        _,_,_,_,_,_, P1, P2 = get_camera_matrices({})
    return P1, P2


# ============================================================
# ビジュアル: ネオングロー効果
# ============================================================

def apply_glow(img: np.ndarray, intensity: float = 0.8) -> np.ndarray:
    """
    画像にネオングロー効果を追加する（明るい領域をぼかして加算）。
    """
    # 明るい領域を抽出（閾値より明るいピクセルを残す）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    bright_bgr = cv2.bitwise_and(img, img, mask=bright)

    # 多段階ぼかしで光の広がりを作る
    glow = cv2.GaussianBlur(bright_bgr, (0, 0), 8)
    glow = cv2.GaussianBlur(glow,        (0, 0), 16)

    # 加算合成
    out = cv2.addWeighted(img, 1.0, glow, intensity, 0)
    return out


# ============================================================
# スピードベース色
# ============================================================

def speed_to_color(speed_mps: float) -> tuple[int, int, int]:
    """
    速度（m/s）を BGR 色に変換。
    遅い=青、中=緑〜黄、速い=赤〜マゼンタ。
    """
    s = min(speed_mps * 2.0, 1.0)   # 正規化（0.5m/s で max）
    if s < 0.33:
        r = 0
        g = int(255 * (s / 0.33))
        b = 255
    elif s < 0.66:
        r = int(255 * ((s - 0.33) / 0.33))
        g = 255
        b = int(255 * (1 - (s - 0.33) / 0.33))
    else:
        r = 255
        g = int(255 * (1 - (s - 0.66) / 0.34))
        b = 0
    return (b, g, r)


# ============================================================
# Rhizomatiks風 3Dビューア（グロー・速度色対応）
# ============================================================

class NeonViewer3D(Viewer3D):
    """通常のViewer3Dにグロー描画を追加したもの。"""

    def draw_speed_trail(self, canvas, trail_with_speed: list):
        """
        速度情報付きの軌跡を描画する（速度で色が変わる）。
        trail_with_speed : [(pt3d, speed), ...]
        """
        if len(trail_with_speed) < 2:
            return
        n = len(trail_with_speed)
        for i in range(1, n):
            pt_a, sp_a = trail_with_speed[i-1]
            pt_b, sp_b = trail_with_speed[i]
            pa = self.pt2px(np.array(pt_a))
            pb = self.pt2px(np.array(pt_b))
            color = speed_to_color((sp_a + sp_b) / 2)
            # 新しいほど太い
            thickness = max(1, int(1 + 3 * (i / n)))
            cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)

    def render_neon(self, pts3d, wrist_trail_speed, cards, connections, colors):
        """Rhizomatiks風の黒背景＋グロー描画。"""
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        # 控えめな床グリッド
        self.draw_grid(canvas)

        # 座標軸（薄く）
        self.draw_axes(canvas, length=0.08)

        # 札（金色の縁取り）
        for i, card in enumerate(cards):
            self.draw_card(canvas, card["corners"],
                           color=(40, 180, 240), label=f"#{i+1}")

        # 速度ベース軌跡
        self.draw_speed_trail(canvas, list(wrist_trail_speed))

        # 現在の手
        if pts3d:
            self.draw_hand(canvas, pts3d, connections, colors)
            # 手首にリング
            w_px = self.pt2px(np.array(pts3d["wrist"]))
            cv2.circle(canvas, w_px, 14, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, w_px, 20, (80, 80, 255), 1, cv2.LINE_AA)

        # グロー効果
        canvas = apply_glow(canvas, intensity=0.7)
        return canvas


# ============================================================
# メイン
# ============================================================

def main():
    P1, P2 = load_projection_matrices()

    # カメラスレッド起動
    cam_pc    = CameraThread(CAM_PC,    "Mac",    target_fps=60)
    cam_phone = CameraThread(CAM_PHONE, "iPhone", target_fps=60)
    cam_pc.start(); cam_phone.start()
    time.sleep(1.2)  # ウォームアップ

    # 検出スレッド起動
    det_pc    = DetectorThread(cam_pc,    "DetPC",    target_hand="Right")
    det_phone = DetectorThread(cam_phone, "DetPhone", target_hand="Right")
    det_pc.start(); det_phone.start()

    # ビューア
    cards = load_cards()
    viewer = NeonViewer3D(size=VIEW3D_SIZE)
    win_name = "karuta-3d  LIVE"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, viewer.on_mouse)

    wrist_trail = deque(maxlen=TRAIL_LEN)
    display_count, t_prev = 0, time.time()
    display_fps, fusion_fps = 0.0, 0.0
    fusion_count, t_fusion = 0, time.time()

    # 録画
    writer, rec_log, rec_start_t, rec_dir = None, [], None, None
    last_fusion_id = (-1, -1)  # (pc_id, phone_id)
    paused = False

    print("\n=== 起動 ===")
    print("操作: ドラッグ=回転  r=録画  t=クリア  SPACE=一時停止  q=終了\n")

    try:
        while True:
            # --- フレーム取得 ---
            f_pc,    id_pc    = cam_pc.get()
            f_phone, id_phone = cam_phone.get()
            if f_pc is None or f_phone is None:
                time.sleep(0.01); continue

            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "): paused = False
                elif key == ord("q"): break
                continue

            # --- 検出結果取得 ---
            pts_pc,    did_pc,    last_pc    = det_pc.get()
            pts_phone, did_phone, last_phone = det_phone.get()

            # --- 3D復元 ---
            pts3d = None
            if (did_pc, did_phone) != last_fusion_id:
                last_fusion_id = (did_pc, did_phone)
                if pts_pc and pts_phone:
                    result = triangulate_landmarks(
                        pts_pc, pts_phone, P1, P2, TARGET_LANDMARKS
                    )
                    if result:
                        pts3d = {k: tuple(v) for k, v in result.items()}
                        # 速度計算＆軌跡追加
                        if wrist_trail:
                            prev_pt, _ = wrist_trail[-1]
                            cur_pt = np.array(pts3d["wrist"])
                            dt = max(1/60.0, 1/max(display_fps, 1))
                            sp = float(np.linalg.norm(cur_pt - np.array(prev_pt)) / dt)
                        else:
                            sp = 0.0
                        wrist_trail.append((pts3d["wrist"], sp))
                        fusion_count += 1

            now = time.time()
            if now - t_fusion >= 1.0:
                fusion_fps = fusion_count / (now - t_fusion)
                fusion_count, t_fusion = 0, now

            # --- 左右カメラビュー（2D描画） ---
            def draw_cam(frame, pts, label, color_hi):
                vis = frame.copy()
                if pts:
                    for a, b in SKELETON_CONNECTIONS:
                        if a in pts and b in pts:
                            cv2.line(vis,
                                     (int(pts[a][0]), int(pts[a][1])),
                                     (int(pts[b][0]), int(pts[b][1])),
                                     POINT_COLORS_BGR.get(a, (200,200,200)), 2, cv2.LINE_AA)
                    for name, (x, y) in pts.items():
                        cv2.circle(vis, (int(x), int(y)), 7,
                                   POINT_COLORS_BGR.get(name, (200,200,200)), -1, cv2.LINE_AA)
                cv2.putText(vis, label, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_hi, 2, cv2.LINE_AA)
                return cv2.resize(vis, (DISPLAY_W, DISPLAY_H))

            cam_pc_vis    = draw_cam(f_pc,    pts_pc,    "Mac",    (0, 255, 255))
            cam_phone_vis = draw_cam(f_phone, pts_phone, "iPhone", (100, 255, 100))

            # --- 3Dビュー ---
            view3d = viewer.render_neon(
                pts3d, wrist_trail, cards,
                SKELETON_CONNECTIONS, POINT_COLORS_BGR
            )

            # --- レイアウト ---
            left_col = np.vstack([cam_pc_vis, cam_phone_vis])    # 640 x 720
            canvas = np.hstack([left_col, view3d])                # 1360 x 720

            # --- 情報表示 ---
            display_count += 1
            if time.time() - t_prev >= 1.0:
                display_fps = display_count / (time.time() - t_prev)
                display_count, t_prev = 0, time.time()

            info_lines = [
                f"Display {display_fps:4.1f}fps",
                f"Cam PC  {cam_pc.actual_fps:4.1f}fps   Det {det_pc.actual_fps:4.1f}fps",
                f"Cam iPh {cam_phone.actual_fps:4.1f}fps  Det {det_phone.actual_fps:4.1f}fps",
                f"3D fuse {fusion_fps:4.1f}fps",
            ]
            for i, line in enumerate(info_lines):
                cv2.putText(canvas, line, (DISPLAY_W + 12, 24 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)

            if writer is not None:
                elapsed = time.time() - rec_start_t
                cv2.circle(canvas, (canvas.shape[1] - 24, 24), 10, (0, 0, 255), -1)
                cv2.putText(canvas, f"REC {elapsed:.1f}s",
                            (canvas.shape[1] - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                writer.write(canvas)
                if pts3d:
                    rec_log.append({"t": round(elapsed, 3),
                                    "landmarks": {k: list(v) for k, v in pts3d.items()}})

            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = True
            elif key == ord("t"):
                wrist_trail.clear()
            elif key == ord("r"):
                if writer is None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    rec_dir = f"output/live/rec_{ts}"
                    os.makedirs(rec_dir, exist_ok=True)
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(f"{rec_dir}/video.mp4",
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             30, (w, h))
                    rec_log = []
                    rec_start_t = time.time()
                    print(f"録画開始: {rec_dir}/")
                else:
                    writer.release(); writer = None
                    with open(f"{rec_dir}/3d_log.json", "w") as f:
                        json.dump({"frames": rec_log}, f, indent=2)
                    print(f"録画停止 → {rec_dir}/")

    finally:
        if writer is not None:
            writer.release()
            with open(f"{rec_dir}/3d_log.json", "w") as f:
                json.dump({"frames": rec_log}, f, indent=2)
        det_pc.stop(); det_phone.stop()
        cam_pc.stop(); cam_phone.stop()
        cv2.destroyAllWindows()
        print("終了")


if __name__ == "__main__":
    main()
