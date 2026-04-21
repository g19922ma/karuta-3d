"""カメラIDごとに1枚撮影して、どれがどのカメラか目で確認するための簡易ツール。"""
import cv2
import os

N = 4
for i in range(N):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        print(f"cam {i}: 開けず")
        continue
    ret, frame = cap.read()
    if ret:
        path = f"/tmp/cam_{i}.jpg"
        cv2.imwrite(path, frame)
        print(f"cam {i} → {path}")
    cap.release()

os.system("open /tmp/cam_*.jpg")
