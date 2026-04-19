"""
OpenCV ベースのリアルタイム3Dビューア

matplotlib なしで、OpenCVキャンバスに3D点・線・矩形を描画する。
マウスドラッグで視点を回転できる。

使い方（単体テスト）:
    .venv/bin/python viewer_3d.py
"""

import cv2
import numpy as np
import json
import os


class Viewer3D:
    """
    シンプルなOpenCVベース3Dビューア。
    方位角（azimuth）と仰角（elevation）で視点を制御する。
    マウスドラッグで回転、ホイールでズーム。
    """

    def __init__(self, size: int = 480, fov_deg: float = 60.0):
        self.size = size
        self.fov  = np.radians(fov_deg)
        self.az   = np.radians(-30)   # 方位角（初期値）
        self.el   = np.radians(30)    # 仰角（初期値）
        self.zoom = 1.0
        self.center = np.array([0.0, 0.0, 0.8])  # 注視点

        # マウス操作用
        self._drag = False
        self._last_mouse = (0, 0)

    # ---------- 視点操作 ----------

    def rotate(self, daz: float, del_: float):
        self.az += daz
        self.el = np.clip(self.el + del_, np.radians(-89), np.radians(89))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = True
            self._last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = False
        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self.rotate(np.radians(dx * 0.5), np.radians(-dy * 0.5))
            self._last_mouse = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom *= 1.1 if flags > 0 else 0.9
            self.zoom = np.clip(self.zoom, 0.2, 5.0)

    # ---------- 3D→2D 投影 ----------

    def _view_matrix(self) -> np.ndarray:
        """カメラ視点からのビュー行列を返す（3x3回転行列）。"""
        Raz = np.array([
            [ np.cos(self.az), 0, np.sin(self.az)],
            [ 0,               1, 0              ],
            [-np.sin(self.az), 0, np.cos(self.az)],
        ])
        Rel = np.array([
            [1, 0,               0              ],
            [0, np.cos(self.el), -np.sin(self.el)],
            [0, np.sin(self.el),  np.cos(self.el)],
        ])
        return Rel @ Raz

    def project(self, pts_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Nx3 の3D点を2D画素座標に投影する。

        Returns:
            pts_2d  : Nx2 の画素座標
            depths  : N の深度値（遠いほど大きい）
        """
        R = self._view_matrix()
        dist = np.linalg.norm(pts_3d - self.center, axis=1).mean() + 1.0
        cam_dist = dist / self.zoom

        pts = (R @ (pts_3d - self.center).T).T
        pts[:, 2] += cam_dist

        f = self.size / (2 * np.tan(self.fov / 2))
        cx = cy = self.size / 2

        depths = pts[:, 2].copy()
        eps = 1e-6
        x = pts[:, 0] / (pts[:, 2] + eps) * f + cx
        y = pts[:, 1] / (pts[:, 2] + eps) * f + cy

        return np.stack([x, y], axis=1), depths

    def pt2px(self, pt: np.ndarray) -> tuple[int, int]:
        """単一の3D点を画素座標に変換する。"""
        pts_2d, _ = self.project(pt.reshape(1, 3))
        return int(pts_2d[0, 0]), int(pts_2d[0, 1])

    # ---------- 描画プリミティブ ----------

    def draw_axes(self, canvas: np.ndarray, length: float = 0.1):
        """原点に座標軸を描画する。"""
        origin = self.center.copy()
        axes = {
            "X": (origin + [length, 0, 0], (80, 80, 255)),
            "Y": (origin + [0, length, 0], (80, 255, 80)),
            "Z": (origin + [0, 0, length], (255, 80, 80)),
        }
        o_px = self.pt2px(origin)
        for label, (tip, color) in axes.items():
            t_px = self.pt2px(tip)
            cv2.arrowedLine(canvas, o_px, t_px, color, 2, tipLength=0.3)
            cv2.putText(canvas, label, t_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_grid(self, canvas: np.ndarray, y: float = 0.0,
                  extent: float = 0.4, step: float = 0.1):
        """水平グリッドを描画する（Y=y の平面）。"""
        vals = np.arange(-extent, extent + step, step)
        for v in vals:
            p1 = self.pt2px(np.array([-extent, y, v]))
            p2 = self.pt2px(np.array([ extent, y, v]))
            cv2.line(canvas, p1, p2, (40, 40, 40), 1)
            p1 = self.pt2px(np.array([v, y, -extent]))
            p2 = self.pt2px(np.array([v, y,  extent]))
            cv2.line(canvas, p1, p2, (40, 40, 40), 1)

    def draw_card(self, canvas: np.ndarray, corners: list, color=(180, 140, 60),
                  label: str = ""):
        """札の矩形を描画する（corners: 4点リスト [[x,y,z], ...]）。"""
        corners = np.array(corners, dtype=np.float64)
        px = [self.pt2px(c) for c in corners]
        for i in range(4):
            cv2.line(canvas, px[i], px[(i+1) % 4], color, 2)
        # 塗りつぶし（半透明風）
        poly = np.array(px, dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], (int(color[0]*0.3), int(color[1]*0.3), int(color[2]*0.3)))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
        if label:
            center_px = self.pt2px(corners.mean(axis=0))
            cv2.putText(canvas, label, center_px,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_hand(self, canvas: np.ndarray, pts3d: dict,
                  connections: list, colors: dict):
        """手のキーポイントとスケルトンを描画する。"""
        for a, b in connections:
            if a in pts3d and b in pts3d:
                pa = self.pt2px(np.array(pts3d[a]))
                pb = self.pt2px(np.array(pts3d[b]))
                color = colors.get(a, (200, 200, 200))
                cv2.line(canvas, pa, pb, color, 2)
        for name, pt in pts3d.items():
            px = self.pt2px(np.array(pt))
            color = colors.get(name, (200, 200, 200))
            cv2.circle(canvas, px, 7, color, -1)

    def draw_trail(self, canvas: np.ndarray, trail: list,
                   key: str = "wrist", color=(100, 100, 220)):
        """手首などの軌跡を描画する。"""
        pts = [f[key] for f in trail if key in f]
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            c = tuple(int(v * alpha) for v in color)
            pa = self.pt2px(np.array(pts[i-1]))
            pb = self.pt2px(np.array(pts[i]))
            cv2.line(canvas, pa, pb, c, 1)

    # ---------- フレーム生成 ----------

    def render(self, pts3d: dict | None, trail, cards: list,
               connections: list, colors: dict) -> np.ndarray:
        """1フレーム分のキャンバスを描画して返す。"""
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        self.draw_grid(canvas)
        self.draw_axes(canvas)

        # 札
        for i, card in enumerate(cards):
            self.draw_card(canvas, card["corners"],
                           label=f"#{i+1}")

        # 軌跡
        self.draw_trail(canvas, list(trail))

        # 手
        if pts3d:
            self.draw_hand(canvas, pts3d, connections, colors)

        # 操作ヒント
        cv2.putText(canvas, "drag=rotate  wheel=zoom",
                    (6, self.size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (80, 80, 80), 1)
        return canvas


def load_cards(path: str = "calibration/card_positions.json") -> list:
    """card_positions.json から札リストを読み込む。"""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f).get("cards", [])
