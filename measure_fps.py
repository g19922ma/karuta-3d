"""各カメラの実測FPSを測る。解像度・FPSを段階的に上げてテスト。"""
import cv2
import time
import sys

# 試す組み合わせ
CONFIGS = [
    (1280, 720,  30),
    (1280, 720,  60),
    (1280, 720, 120),
    (1920, 1080, 30),
    (1920, 1080, 60),
    (1920, 1080, 120),
    (3840, 2160, 30),
]


def measure_one(cam_id: int, w: int, h: int, fps: int, duration: float = 3.0):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    af = cap.get(cv2.CAP_PROP_FPS)

    # ウォームアップ
    for _ in range(5):
        cap.read()

    t0 = time.time()
    n = 0
    while time.time() - t0 < duration:
        ret, _ = cap.read()
        if ret:
            n += 1
    cap.release()
    measured = n / (time.time() - t0)
    return {"req": (w, h, fps), "actual_size": (aw, ah), "reported_fps": af, "measured_fps": measured}


def main():
    cam_ids = [0, 1, 2]
    if len(sys.argv) > 1:
        cam_ids = [int(x) for x in sys.argv[1:]]

    for cid in cam_ids:
        print(f"\n=== cam {cid} ===")
        for w, h, fps in CONFIGS:
            r = measure_one(cid, w, h, fps)
            if r is None:
                print(f"  {w}x{h}@{fps}fps: 開けず")
                continue
            ok = "✓" if r["measured_fps"] > fps * 0.8 else ""
            print(f"  要求 {w}x{h}@{fps}fps → 実測 {r['actual_size'][0]}x{r['actual_size'][1]}@{r['measured_fps']:.1f}fps  (CAP報告 {r['reported_fps']:.1f}) {ok}")


if __name__ == "__main__":
    main()
