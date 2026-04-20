"""
複数カメラ同時録画ツール（N=2〜6台対応）

各カメラを独立スレッドで録画し、フレームタイムスタンプと音声を同時記録する。
後から音声波形で同期できるように、全カメラ共通の音声ファイルを保存する。

使い方:
    # 使用可能なカメラを確認
    .venv/bin/python multi_cam_record.py --list

    # 2台で録画（Mac + iPhone）
    .venv/bin/python multi_cam_record.py --cams 0 1

    # 4台で録画
    .venv/bin/python multi_cam_record.py --cams 0 1 2 3

    # 条件ラベル付き録画（あとで整理しやすい）
    .venv/bin/python multi_cam_record.py --cams 0 1 --label "fast_sweep"

操作:
    r     : 録画開始/停止
    c     : シンク用手拍子マーカー（現在時刻をログに追記）
    q     : 終了

出力:
    output/multi_rec_YYYYMMDD_HHMMSS_<label>/
      ├── cam_0.mp4
      ├── cam_1.mp4
      ├── ...
      ├── audio.wav
      ├── timestamps.json   # 各カメラの各フレームのタイムスタンプ
      └── metadata.json     # 録画設定とシンクマーカー時刻
"""

import cv2
import numpy as np
import threading
import time
import argparse
import json
import os
import sys
import wave
from datetime import datetime
from pathlib import Path


# ============================================================
# カメラキャプチャ＆録画スレッド
# ============================================================

class RecordingCamera(threading.Thread):
    """
    カメラから継続的にフレームを取得し、録画中はVideoWriterに書き込む。
    タイムスタンプも各フレームで記録する。
    """

    def __init__(self, cam_id: int, name: str,
                 width: int = 1280, height: int = 720, fps: int = 30):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.name = name
        self.target_fps = fps
        self.target_size = (width, height)
        self.running = True
        self.recording = False

        self._cap = None
        self._writer = None
        self._video_path = None
        self._timestamps = []   # 録画中のタイムスタンプ [sec from start]
        self._rec_start = None

        self.latest_frame = None
        self._lock = threading.Lock()
        self.actual_fps = 0.0
        self.frame_count = 0

    def run(self):
        self._cap = cv2.VideoCapture(self.cam_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.target_size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            print(f"[{self.name}] カメラ {self.cam_id} を開けませんでした")
            return

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[{self.name}] cam={self.cam_id}: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

        t_prev, count = time.time(), 0
        while self.running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.001)
                continue

            with self._lock:
                self.latest_frame = frame

            # 録画中なら書き込み
            if self.recording and self._writer is not None:
                self._writer.write(frame)
                ts = time.time() - self._rec_start
                self._timestamps.append(ts)
                self.frame_count += 1

            count += 1
            now = time.time()
            if now - t_prev >= 1.0:
                self.actual_fps = count / (now - t_prev)
                count, t_prev = 0, now

        if self._writer:
            self._writer.release()
        if self._cap:
            self._cap.release()

    # --- 制御 ---

    def start_recording(self, output_dir: str):
        """録画開始。"""
        video_path = f"{output_dir}/cam_{self.cam_id}.mp4"
        w, h = self.target_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 書き込みFPSは実測値（target_fps固定だとズレる）
        write_fps = int(self.actual_fps) if self.actual_fps > 0 else self.target_fps
        self._writer = cv2.VideoWriter(video_path, fourcc, write_fps, (w, h))
        self._video_path = video_path
        self._timestamps = []
        self._rec_start = time.time()
        self.frame_count = 0
        self.recording = True

    def stop_recording(self) -> dict:
        """録画停止してメタデータを返す。"""
        self.recording = False
        # writerを flush してから release
        time.sleep(0.1)
        if self._writer:
            self._writer.release()
            self._writer = None
        return {
            "cam_id":       self.cam_id,
            "name":         self.name,
            "video_path":   self._video_path,
            "n_frames":     self.frame_count,
            "duration_sec": self._timestamps[-1] if self._timestamps else 0,
            "actual_fps":   self.actual_fps,
            "timestamps":   self._timestamps,
        }

    def get_preview(self):
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False


# ============================================================
# 音声録音スレッド
# ============================================================

class AudioRecorder(threading.Thread):
    """
    sounddevice を使って録画中に音声を記録する。
    手拍子での同期用。
    """

    def __init__(self, sample_rate: int = 44100, channels: int = 1):
        super().__init__(daemon=True)
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.frames = []
        self._start_t = None
        self.running = True

    def run(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("[audio] sounddevice 未導入。音声録音はスキップされます。")
            return

        def callback(indata, frames, time_info, status):
            if self.recording:
                self.frames.append(indata.copy())

        try:
            with sd.InputStream(samplerate=self.sample_rate,
                                channels=self.channels,
                                callback=callback):
                while self.running:
                    time.sleep(0.05)
        except Exception as e:
            print(f"[audio] 録音エラー: {e}")

    def start_recording(self):
        self.frames = []
        self._start_t = time.time()
        self.recording = True

    def stop_recording(self, output_path: str) -> dict:
        self.recording = False
        time.sleep(0.1)
        if not self.frames:
            return {"path": None, "n_samples": 0, "duration_sec": 0}

        audio = np.concatenate(self.frames, axis=0)

        # int16 wav として保存
        audio_int = (audio * 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int.tobytes())

        return {
            "path":         output_path,
            "n_samples":    len(audio),
            "duration_sec": len(audio) / self.sample_rate,
            "sample_rate":  self.sample_rate,
        }

    def stop(self):
        self.running = False


# ============================================================
# カメラ列挙
# ============================================================

def list_available_cameras(max_check: int = 10) -> list[dict]:
    """使用可能なカメラを列挙する。"""
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            available.append({"id": i, "width": w, "height": h, "fps": fps})
            cap.release()
    return available


# ============================================================
# プレビューグリッド生成
# ============================================================

def make_preview_grid(cameras: list[RecordingCamera],
                       cell_w: int = 480, cell_h: int = 270) -> np.ndarray | None:
    """全カメラのプレビューをグリッドに並べて返す。"""
    frames = []
    for c in cameras:
        f = c.get_preview()
        if f is None:
            f = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(f, f"{c.name}: NO SIGNAL", (20, cell_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 200), 2)
        else:
            f = cv2.resize(f, (cell_w, cell_h))

        # ラベル
        cv2.putText(f, f"{c.name} cam{c.cam_id} {c.actual_fps:.0f}fps",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if c.recording:
            cv2.circle(f, (cell_w - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(f, "REC", (cell_w - 60, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frames.append(f)

    # グリッド配置（自動列数）
    n = len(frames)
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols

    grid_rows = []
    for r in range(rows):
        row_frames = frames[r * cols:(r + 1) * cols]
        while len(row_frames) < cols:
            row_frames.append(np.zeros_like(row_frames[0]))
        grid_rows.append(np.hstack(row_frames))
    return np.vstack(grid_rows)


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="複数カメラ同時録画ツール")
    parser.add_argument("--list",  action="store_true", help="使用可能なカメラを一覧表示")
    parser.add_argument("--cams",  type=int, nargs="+", help="録画するカメラID（例: --cams 0 1 2）")
    parser.add_argument("--label", default="", help="録画ラベル（例: fast_sweep, grip_test）")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps",    type=int, default=30)
    parser.add_argument("--no-audio", action="store_true", help="音声録音を無効化")
    args = parser.parse_args()

    # カメラ列挙モード
    if args.list:
        cams = list_available_cameras()
        print("=== 使用可能なカメラ ===")
        for c in cams:
            print(f"  cam {c['id']}: {c['width']}x{c['height']} @ {c['fps']:.0f}fps")
        return

    if not args.cams:
        parser.error("--cams でカメラIDを指定してください（例: --cams 0 1）")

    # カメラスレッド起動
    cameras = []
    for i, cam_id in enumerate(args.cams):
        cam = RecordingCamera(cam_id, f"Cam{i}",
                              width=args.width, height=args.height, fps=args.fps)
        cam.start()
        cameras.append(cam)

    # 音声スレッド
    audio = None
    if not args.no_audio:
        audio = AudioRecorder()
        audio.start()

    print("\nカメラ起動中...")
    time.sleep(2.0)

    print("\n=== 操作 ===")
    print("  r : 録画開始/停止")
    print("  c : 手拍子マーカー追記")
    print("  q : 終了")

    out_dir = None
    is_recording = False
    clap_markers = []
    rec_start = None

    win_name = "Multi-Camera Recorder"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            grid = make_preview_grid(cameras)
            if grid is None:
                time.sleep(0.05)
                continue

            # ステータスバー
            h = 40
            bar = np.zeros((h, grid.shape[1], 3), dtype=np.uint8)
            if is_recording:
                elapsed = time.time() - rec_start
                txt = f"● RECORDING  {elapsed:6.2f}s   claps: {len(clap_markers)}"
                color = (0, 0, 255)
            else:
                txt = "READY   r=録画開始   c=手拍子マーカー   q=終了"
                color = (180, 180, 180)
            cv2.putText(bar, txt, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            canvas = np.vstack([grid, bar])
            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r"):
                if not is_recording:
                    # 録画開始
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    label_suffix = f"_{args.label}" if args.label else ""
                    out_dir = f"output/multi_rec_{ts}{label_suffix}"
                    os.makedirs(out_dir, exist_ok=True)
                    rec_start = time.time()
                    for c in cameras:
                        c.start_recording(out_dir)
                    if audio:
                        audio.start_recording()
                    is_recording = True
                    clap_markers = []
                    print(f"\n録画開始: {out_dir}")
                else:
                    # 録画停止・保存
                    cam_meta = [c.stop_recording() for c in cameras]
                    audio_meta = None
                    if audio:
                        audio_meta = audio.stop_recording(f"{out_dir}/audio.wav")

                    # タイムスタンプ保存
                    ts_data = {
                        "cameras": [
                            {"cam_id": m["cam_id"], "name": m["name"],
                             "timestamps": m["timestamps"]}
                            for m in cam_meta
                        ]
                    }
                    with open(f"{out_dir}/timestamps.json", "w") as f:
                        json.dump(ts_data, f, indent=2)

                    # メタデータ保存
                    metadata = {
                        "label": args.label,
                        "rec_start": rec_start,
                        "rec_end":   time.time(),
                        "cameras":   [
                            {k: v for k, v in m.items() if k != "timestamps"}
                            for m in cam_meta
                        ],
                        "audio":         audio_meta,
                        "clap_markers":  clap_markers,
                        "resolution":    [args.width, args.height],
                        "target_fps":    args.fps,
                    }
                    with open(f"{out_dir}/metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)

                    print(f"\n録画停止: {out_dir}")
                    for m in cam_meta:
                        print(f"  {m['name']}: {m['n_frames']}フレーム, "
                              f"{m['duration_sec']:.2f}s, {m['actual_fps']:.1f}fps")
                    if audio_meta and audio_meta["path"]:
                        print(f"  音声: {audio_meta['duration_sec']:.2f}s")
                    if clap_markers:
                        print(f"  手拍子マーカー: {clap_markers}")

                    is_recording = False

            elif key == ord("c"):
                if is_recording:
                    t = time.time() - rec_start
                    clap_markers.append(round(t, 3))
                    print(f"  手拍子マーカー追記: t={t:.3f}s")

    finally:
        if is_recording:
            for c in cameras:
                c.stop_recording()
            if audio:
                audio.stop_recording(f"{out_dir}/audio.wav")
        for c in cameras:
            c.stop()
        if audio:
            audio.stop()
        cv2.destroyAllWindows()
        print("終了")


if __name__ == "__main__":
    main()
