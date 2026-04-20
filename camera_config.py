"""
カメラ設定ファイル（cameras_config.json）の読み込みと、
CameraThread 生成ユーティリティ。

全ツールはここを経由してカメラを扱うことで、
将来 3台・4台・6台 に増やしても同じコードが使える。
"""

import json
import os
import sys
import threading
import time
import cv2
from dataclasses import dataclass
from typing import Optional


CONFIG_PATH_DEFAULT = "cameras_config.json"


@dataclass
class CameraSpec:
    id:           int
    role:         str
    name:         str
    resolution:   Optional[tuple]   # (w, h) or None = native
    fps:          Optional[int]     # int or None = native
    is_reference: bool


def load_config(path: str = CONFIG_PATH_DEFAULT) -> list[CameraSpec]:
    """
    cameras_config.json を読み込んで CameraSpec のリストを返す。
    _future_cameras_example などアンダースコアで始まるキーは無視。
    """
    if not os.path.exists(path):
        print(f"[camera_config] {path} が見つかりません。デフォルト 2 カメラで動作します。")
        # フォールバック: 旧ハードコード互換
        return [
            CameraSpec(id=1, role="main_left",  name="Mac",    resolution=None, fps=None, is_reference=True),
            CameraSpec(id=0, role="main_right", name="iPhone", resolution=None, fps=None, is_reference=False),
        ]

    with open(path) as f:
        data = json.load(f)

    specs = []
    for c in data.get("cameras", []):
        res = c.get("resolution")
        specs.append(CameraSpec(
            id=int(c["id"]),
            role=str(c["role"]),
            name=str(c.get("name", c["role"])),
            resolution=tuple(res) if res else None,
            fps=c.get("fps"),
            is_reference=bool(c.get("is_reference", False)),
        ))

    # 参照カメラが1つもなければ最初を参照にする
    if not any(s.is_reference for s in specs) and specs:
        specs[0].is_reference = True

    return specs


def get_reference(specs: list[CameraSpec]) -> CameraSpec:
    """参照カメラを返す（外部パラメータの基準）。"""
    for s in specs:
        if s.is_reference:
            return s
    return specs[0]


# ============================================================
# CameraThread（設定駆動版）
# ============================================================

class CameraThread(threading.Thread):
    """設定駆動のカメラキャプチャスレッド。"""

    def __init__(self, spec: CameraSpec):
        super().__init__(daemon=True)
        self.spec = spec
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self._cap = None
        self.actual_fps = 0.0

    def run(self):
        self._cap = cv2.VideoCapture(self.spec.id)
        if self.spec.resolution is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.spec.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.resolution[1])
        if self.spec.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self.spec.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            print(f"[{self.spec.role}] カメラ {self.spec.id} を開けませんでした", file=sys.stderr)
            return

        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[{self.spec.role}] {self.spec.name} cam_id={self.spec.id}: {aw}x{ah} @ {af:.0f}fps")

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

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get(self):
        with self.lock:
            return (self.frame.copy(), self.frame_id) if self.frame is not None else (None, -1)

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()


# ============================================================
# CLI: 現在の設定とカメラを確認
# ============================================================

if __name__ == "__main__":
    specs = load_config()
    print(f"=== cameras_config.json から {len(specs)} 台のカメラを検出 ===")
    for s in specs:
        ref = " [REF]" if s.is_reference else ""
        res = f"{s.resolution[0]}x{s.resolution[1]}" if s.resolution else "native"
        fps = f"{s.fps}fps" if s.fps else "native"
        print(f"  {s.role}: id={s.id}  {s.name}  {res}  {fps}{ref}")
