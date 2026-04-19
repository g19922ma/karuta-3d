"""
2動画の同期オフセットを対話的に決定するツール

使い方:
    python sync_tool.py --pc video_pc.mp4 --phone video_phone.mp4

操作:
    a / ←  : 1フレーム戻る
    d / →  : 1フレーム進む
    j / k  : 10フレーム戻る / 進む
    SPACE  : この位置をシンク点としてマーク
    q      : 確定して終了

出力:
    sync_offset.json  --  {pc_sync_frame, phone_sync_frame, frame_offset}
    frame_offset は「スマホフレーム番号 = PCフレーム番号 + frame_offset」を意味する
"""

import cv2
import json
import argparse
import os
import sys


def _nav_window(cam_name: str, cap: cv2.VideoCapture) -> int:
    """
    1つのカメラ動画を操作してシンクフレームを選択する。
    選択したフレーム番号を返す。
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    cur   = [0]
    sync  = [None]
    win   = f"SyncTool [{cam_name}]  (a/d=±1, j/k=±10, SPACE=mark, q=done)"

    def show():
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur[0])
        ret, frame = cap.read()
        if not ret:
            return
        h, w = frame.shape[:2]
        scale = min(960 / w, 540 / h, 1.0)
        disp = cv2.resize(frame, (int(w * scale), int(h * scale)))

        info = f"[{cam_name}]  Frame {cur[0]} / {total - 1}  ({fps:.1f} fps)"
        if sync[0] is not None:
            info += f"  | SYNC MARKED @ {sync[0]}"
        cv2.putText(disp, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(disp, "SPACE=mark sync  |  a/d or </> arrows=step  |  j/k=±10  |  q=done",
                    (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow(win, disp)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 560)
    show()

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key in (ord("a"), 81, 2):   # a, left-arrow (Linux/macOS)
            cur[0] = max(0, cur[0] - 1)
            show()
        elif key in (ord("d"), 83, 3):   # d, right-arrow
            cur[0] = min(total - 1, cur[0] + 1)
            show()
        elif key == ord("j"):
            cur[0] = max(0, cur[0] - 10)
            show()
        elif key in (ord("k"), ord("l")):
            cur[0] = min(total - 1, cur[0] + 10)
            show()
        elif key == ord(" "):
            sync[0] = cur[0]
            print(f"  [{cam_name}] シンクフレーム: {sync[0]}")
            show()

    cv2.destroyWindow(win)

    if sync[0] is None:
        print(f"  [{cam_name}] シンク未指定 → フレーム0 を使用")
        sync[0] = 0

    return sync[0]


def find_sync_offset(
    pc_path: str,
    phone_path: str,
    output_path: str = "sync_offset.json",
) -> int:
    """
    PC動画とスマホ動画のシンクオフセットを対話的に決定する。

    Returns:
        frame_offset: int  (phone_frame = pc_frame + frame_offset)
    """
    cap_pc    = cv2.VideoCapture(pc_path)
    cap_phone = cv2.VideoCapture(phone_path)

    if not cap_pc.isOpened():
        print(f"[sync_tool] ERROR: PCビデオを開けません: {pc_path}", file=sys.stderr)
        return 0
    if not cap_phone.isOpened():
        print(f"[sync_tool] ERROR: Phoneビデオを開けません: {phone_path}", file=sys.stderr)
        return 0

    print("\n=== PC カメラのシンクフレームを選択 ===")
    print("手拍子など、同時刻のイベントが写っているフレームを選んで SPACE を押してください。")
    pc_sync = _nav_window("PC", cap_pc)

    print("\n=== スマホカメラのシンクフレームを選択 ===")
    print("PCと同じイベントが写っているフレームを選んで SPACE を押してください。")
    phone_sync = _nav_window("Phone", cap_phone)

    cap_pc.release()
    cap_phone.release()

    offset = phone_sync - pc_sync
    result = {
        "pc_sync_frame":    pc_sync,
        "phone_sync_frame": phone_sync,
        "frame_offset":     offset,
        "note":             "phone_frame_idx = pc_frame_idx + frame_offset",
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nシンクオフセットを保存: {output_path}")
    print(f"  PC sync @ {pc_sync}, Phone sync @ {phone_sync}, offset = {offset}")
    return offset


# ---------- CLI ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2動画のシンクオフセット決定ツール")
    parser.add_argument("--pc",     required=True, help="PC カメラ動画パス")
    parser.add_argument("--phone",  required=True, help="スマホ動画パス")
    parser.add_argument("--output", default="sync_offset.json", help="出力JSONパス")
    args = parser.parse_args()

    find_sync_offset(args.pc, args.phone, args.output)
