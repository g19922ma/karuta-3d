"""
ArUcoシート自動検出によるステレオキャリブレーション

手順:
    1. generate_calib_sheet.py で calib_sheet.png を生成
    2. A4で100%印刷、硬い板に貼って平面を保つ
    3. 両カメラに全マーカーが写るよう置く
    4. このスクリプトを起動し、シートを様々な角度・位置で見せて SPACE を押す
    5. 5枚以上集めたら c でキャリブレーション実行

使い方:
    .venv/bin/python calibrate_auto.py

操作:
    SPACE : 両カメラで4マーカー全検出ができていたらペアを取得
    c     : 収集したペアでキャリブレーション実行
    d     : 最後に追加したペアを削除
    q     : 終了

推奨:
    - 10〜15枚のペアで十分な精度
    - シートの角度・位置・距離を変えながら取得
    - 極端な斜めより、少しずつ変える方が精度が安定
"""

import cv2
import numpy as np
import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from demo_realtime import CameraThread, CAM_PC, CAM_PHONE


META_PATH      = "calib_sheet_meta.json"
OUTPUT_PATH    = "calibration/stereo_calib.json"
RAW_DATA_PATH  = "calibration/raw_captures.npz"   # 生データ（再計算用）
MIN_PAIRS      = 5
RECOMMEND      = 10


# ---------- ArUco検出 ----------

def setup_detector(meta: dict):
    """メタから検出器を構築。"""
    dict_name = meta["aruco_dict"]
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return detector


def detect_markers(frame: np.ndarray, detector) -> dict:
    """
    フレームからArUcoを検出する。

    Returns:
        {marker_id: corners_np(4, 2)}  全マーカーが検出されなくても見つかった分だけ返す
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    result = {}
    if ids is None:
        return result
    for i, mid in enumerate(ids.flatten()):
        # corners[i]: shape (1, 4, 2), 時計回り（左上→右上→右下→左下）
        result[int(mid)] = corners[i].reshape(4, 2).astype(np.float64)
    return result


def extract_correspondences(
    markers: dict,
    markers_3d: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    検出マーカーと既知3D座標から対応点ペアを作る。

    Returns:
        img_pts_2d: (N, 2)
        obj_pts_3d: (N, 3)
    """
    img_pts, obj_pts = [], []
    for mid, corners_2d in markers.items():
        if str(mid) not in markers_3d:
            continue
        corners_3d = np.array(markers_3d[str(mid)], dtype=np.float64)
        for c2, c3 in zip(corners_2d, corners_3d):
            img_pts.append(c2)
            obj_pts.append(c3)
    return np.array(img_pts, dtype=np.float32), np.array(obj_pts, dtype=np.float32)


# ---------- キャリブレーション ----------

def run_calibration(obj_pts, img_pts1, img_pts2, img_size1, img_size2):
    """個別キャリブ→ステレオキャリブ。"""
    print(f"\nキャリブレーション実行中... ({len(obj_pts)}ペア)")
    print(f"  カメラ1画像サイズ: {img_size1}")
    print(f"  カメラ2画像サイズ: {img_size2}")

    # 歪みモデルを単純化（平面ターゲット + 少数キャプチャでの発散を防止）
    flags = (cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K2 |
             cv2.CALIB_FIX_K3)

    rms1, K1, d1, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts1, img_size1, None, None, flags=flags
    )
    rms2, K2, d2, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts2, img_size2, None, None, flags=flags
    )
    print(f"  個別RMS  Cam1={rms1:.3f}px  Cam2={rms2:.3f}px")
    print(f"  Cam1 焦点 fx={K1[0,0]:.0f} fy={K1[1,1]:.0f}  中心 cx={K1[0,2]:.0f} cy={K1[1,2]:.0f}")
    print(f"  Cam2 焦点 fx={K2[0,0]:.0f} fy={K2[1,1]:.0f}  中心 cx={K2[0,2]:.0f} cy={K2[1,2]:.0f}")

    # ステレオキャリブレーションは imageSize を使わないが、API上必須
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts1, img_pts2,
        K1, d1, K2, d2,
        img_size1,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    print(f"  ステレオRMS = {rms:.3f}px")

    return {
        "K1": K1.tolist(),
        "K2": K2.tolist(),
        "dist1": d1.tolist(),
        "dist2": d2.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "image_size":  img_size1,    # 旧API互換
        "image_size1": img_size1,
        "image_size2": img_size2,
        "n_images": len(obj_pts),
        "rms_error": round(rms, 4),
        "rms_cam1":  round(rms1, 4),
        "rms_cam2":  round(rms2, 4),
        "method": "aruco_sheet",
    }


# ---------- 生データの保存・読み込み ----------

def save_raw_captures(obj_pts_all, img_pts1_all, img_pts2_all,
                       img_size1, img_size2, path: str = RAW_DATA_PATH):
    """キャプチャの生データを npz に保存する（再計算用）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        obj_pts=np.array([p.reshape(-1, 3) for p in obj_pts_all], dtype=object),
        img_pts1=np.array([p.reshape(-1, 2) for p in img_pts1_all], dtype=object),
        img_pts2=np.array([p.reshape(-1, 2) for p in img_pts2_all], dtype=object),
        img_size1=np.array(img_size1),
        img_size2=np.array(img_size2),
    )


def load_raw_captures(path: str = RAW_DATA_PATH):
    """保存された生データを読み込む。"""
    data = np.load(path, allow_pickle=True)
    obj_pts_all  = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 3)
                    for p in data["obj_pts"]]
    img_pts1_all = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2)
                    for p in data["img_pts1"]]
    img_pts2_all = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 2)
                    for p in data["img_pts2"]]
    img_size1 = tuple(int(x) for x in data["img_size1"])
    img_size2 = tuple(int(x) for x in data["img_size2"])
    return obj_pts_all, img_pts1_all, img_pts2_all, img_size1, img_size2


# ---------- メイン ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="前回保存した生データから再計算（撮影しない）")
    args = parser.parse_args()

    # 再計算モード
    if args.recompute:
        if not os.path.exists(RAW_DATA_PATH):
            print(f"{RAW_DATA_PATH} がありません。一度通常モードで取得してください。")
            sys.exit(1)
        print(f"=== 再計算モード ===")
        print(f"生データ読み込み: {RAW_DATA_PATH}")
        obj_pts_all, img_pts1_all, img_pts2_all, img_size1, img_size2 = load_raw_captures()
        print(f"  {len(obj_pts_all)} ペア")

        result = run_calibration(obj_pts_all, img_pts1_all, img_pts2_all,
                                  img_size1, img_size2)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(result, f, indent=2)

        # シート位置も保存
        with open(META_PATH) as f:
            meta = json.load(f)
        sheet_corners = []
        for mid, corners_3d in sorted(meta["markers_3d"].items(), key=lambda x: int(x[0])):
            sheet_corners.append({"id": int(mid), "corners": corners_3d})
        with open("calibration/card_positions.json", "w") as f:
            json.dump({"cards": sheet_corners,
                        "note": "ArUco markers from calib sheet"}, f, indent=2)

        print(f"\n保存: {OUTPUT_PATH}")
        print(f"ステレオRMS: {result['rms_error']:.4f} px")
        return

    if not os.path.exists(META_PATH):
        print(f"{META_PATH} がありません。先に generate_calib_sheet.py を実行してください。")
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)

    detector = setup_detector(meta)
    markers_3d = meta["markers_3d"]
    n_required = len(markers_3d)

    cam_pc    = CameraThread(CAM_PC,    "Mac")
    cam_phone = CameraThread(CAM_PHONE, "Phone")
    cam_pc.start(); cam_phone.start()
    time.sleep(1.0)

    print(f"=== ArUcoシート自動キャリブレーション ===")
    print(f"必要マーカー: {n_required}個（全部見えた時だけペアを取れます）")
    print(f"モード: AUTO（停止→自動取得）  a=手動/自動切替  c=キャリブ実行  d=削除  q=終了")
    print(f"目標: {RECOMMEND}ペア\n")

    obj_pts_all, img_pts1_all, img_pts2_all = [], [], []
    img_size1 = None
    img_size2 = None

    # 自動取得モード（デフォルトON）
    auto_mode = True
    # 安定判定: 全マーカーが検出できて、平均位置が前回取得時から十分動いた状態で
    # STABLE_FRAMES フレーム連続静止したら取得
    STABLE_FRAMES      = 12    # 約0.4秒（30fps想定）
    STILL_THRESHOLD_PX = 3.0   # このピクセル以下の動きを「静止」とみなす
    DIFF_THRESHOLD_PX  = 30.0  # 前回取得との最小変化量

    stable_count = 0
    last_m1_center = None
    last_captured_m1_center = None

    try:
        while True:
            f1 = cam_pc.get_frame()
            f2 = cam_phone.get_frame()
            if f1 is None or f2 is None:
                time.sleep(0.01); continue

            if img_size1 is None:
                img_size1 = (f1.shape[1], f1.shape[0])
                img_size2 = (f2.shape[1], f2.shape[0])
                if img_size1 != img_size2:
                    print(f"  注意: 解像度が異なります Cam1={img_size1} Cam2={img_size2}")

            m1 = detect_markers(f1, detector)
            m2 = detect_markers(f2, detector)

            # 両カメラで検出された共通のマーカーID
            common_ids = set(m1.keys()) & set(m2.keys()) & set(int(k) for k in markers_3d.keys())
            all_detected = len(common_ids) == n_required

            # 自動取得判定: 全マーカー検出 + 静止 + 前回から動いた
            auto_capture_now = False
            if all_detected:
                # 現在のマーカー中心（全マーカー平均）
                cur_center = np.mean([m1[i].mean(axis=0) for i in common_ids], axis=0)

                if last_m1_center is not None:
                    movement = float(np.linalg.norm(cur_center - last_m1_center))
                else:
                    movement = 999.0

                # 前回取得からの変化量
                if last_captured_m1_center is not None:
                    diff_from_last = float(np.linalg.norm(cur_center - last_captured_m1_center))
                else:
                    diff_from_last = 999.0

                last_m1_center = cur_center

                # 静止 + 十分変化 → カウントアップ
                if movement < STILL_THRESHOLD_PX and diff_from_last > DIFF_THRESHOLD_PX:
                    stable_count += 1
                else:
                    stable_count = 0

                if auto_mode and stable_count >= STABLE_FRAMES:
                    auto_capture_now = True
                    stable_count = 0
            else:
                stable_count = 0
                last_m1_center = None

            # 表示
            d1 = f1.copy()
            d2 = f2.copy()
            for mid, corners in m1.items():
                color = (0, 255, 0) if mid in common_ids else (0, 200, 255)
                cv2.polylines(d1, [corners.astype(np.int32)], True, color, 2)
                c = corners.mean(axis=0).astype(int)
                cv2.putText(d1, f"#{mid}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            for mid, corners in m2.items():
                color = (0, 255, 0) if mid in common_ids else (0, 200, 255)
                cv2.polylines(d2, [corners.astype(np.int32)], True, color, 2)
                c = corners.mean(axis=0).astype(int)
                cv2.putText(d2, f"#{mid}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            d1 = cv2.resize(d1, (640, 360))
            d2 = cv2.resize(d2, (640, 360))

            status_color = (0, 255, 0) if all_detected else (0, 140, 255)
            status = f"{len(common_ids)}/{n_required} common markers"
            if all_detected and auto_mode:
                progress = stable_count / STABLE_FRAMES
                status += f"  -> 静止中 [{int(progress*100)}%]"
            elif all_detected:
                status += "  -> SPACE で取得"
            cv2.putText(d1, f"Mac    {len(m1)} detected",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(d2, f"iPhone {len(m2)} detected",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            # 自動取得ゲージ
            if all_detected and auto_mode:
                bar_w = int(640 * stable_count / STABLE_FRAMES)
                cv2.rectangle(d1, (0, 352), (bar_w, 360), (0, 255, 0), -1)

            canvas = np.hstack([d1, d2])
            bar = np.zeros((44, canvas.shape[1], 3), dtype=np.uint8)
            mode_label = "AUTO" if auto_mode else "MANUAL"
            cv2.putText(bar,
                        f"Pairs: {len(obj_pts_all)}/{RECOMMEND}  [{mode_label}]  {status}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            canvas = np.vstack([canvas, bar])
            cv2.imshow("ArUco Calibration", canvas)

            def capture_pair():
                nonlocal last_captured_m1_center
                m1_common = {k: m1[k] for k in common_ids}
                m2_common = {k: m2[k] for k in common_ids}
                img_pts1, obj_pts = extract_correspondences(m1_common, markers_3d)
                img_pts2, _       = extract_correspondences(m2_common, markers_3d)
                obj_pts_all.append(obj_pts.reshape(-1, 1, 3))
                img_pts1_all.append(img_pts1.reshape(-1, 1, 2))
                img_pts2_all.append(img_pts2.reshape(-1, 1, 2))
                last_captured_m1_center = np.mean([m1[i].mean(axis=0) for i in common_ids], axis=0)
                # 生データを毎回保存（落ちても大丈夫なように）
                save_raw_captures(obj_pts_all, img_pts1_all, img_pts2_all,
                                   img_size1, img_size2)
                print(f"  ペア {len(obj_pts_all)} 取得 ({len(obj_pts)}点)")
                # 取得フラッシュ
                flash = canvas.copy()
                cv2.rectangle(flash, (0, 0), (canvas.shape[1], canvas.shape[0]),
                              (0, 255, 0), 10)
                cv2.putText(flash, f"CAPTURED {len(obj_pts_all)}",
                            (flash.shape[1]//2 - 120, flash.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                cv2.imshow("ArUco Calibration", flash)
                cv2.waitKey(300)

            if auto_capture_now:
                capture_pair()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:   # q or ESC
                break
            elif key == ord("a"):
                auto_mode = not auto_mode
                stable_count = 0
                print(f"  モード切替: {'AUTO' if auto_mode else 'MANUAL'}")
            elif key == ord(" ") and all_detected:
                capture_pair()

            elif key == ord("d") and obj_pts_all:
                obj_pts_all.pop(); img_pts1_all.pop(); img_pts2_all.pop()
                print(f"  最後のペアを削除 (残り {len(obj_pts_all)})")

            elif key == ord("c"):
                if len(obj_pts_all) < MIN_PAIRS:
                    print(f"  ペア不足: {len(obj_pts_all)}/{MIN_PAIRS}")
                    continue
                cv2.destroyAllWindows()

                result = run_calibration(
                    obj_pts_all, img_pts1_all, img_pts2_all,
                    img_size1, img_size2
                )

                os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
                with open(OUTPUT_PATH, "w") as f:
                    json.dump(result, f, indent=2)

                # シート位置を card_positions.json 相当として保存（3Dビューで見られる）
                sheet_corners = []
                for mid, corners_3d in sorted(markers_3d.items(), key=lambda x: int(x[0])):
                    sheet_corners.append({
                        "id": int(mid),
                        "corners": corners_3d,
                    })
                with open("calibration/card_positions.json", "w") as f:
                    json.dump({
                        "cards": sheet_corners,
                        "note": "ArUco markers from calib sheet (not actual karuta cards)"
                    }, f, indent=2)

                print(f"\n保存: {OUTPUT_PATH}")
                print(f"RMS再投影誤差: {result['rms_error']:.4f} px")
                if result["rms_error"] < 1.0:
                    print("  ✓ 非常に良好")
                elif result["rms_error"] < 3.0:
                    print("  ✓ 良好")
                else:
                    print("  ⚠ やや大きい。シートを平らに保つ・枚数を増やすと改善")
                print("\ndemo_realtime/demo_live 再起動で自動適用されます。")
                break

    finally:
        cam_pc.stop(); cam_phone.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
