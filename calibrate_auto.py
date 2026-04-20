"""
ArUcoシート自動検出による N カメラ ステレオキャリブレーション

cameras_config.json で定義されたカメラ全台に対して:
  1. 各カメラの内部パラメータを推定
  2. 参照カメラからの外部パラメータ（R, T）を推定

手順:
  1. generate_calib_sheet.py で calib_sheet.png を生成、A4 で 100% 印刷
  2. 硬い板に貼って平面を保つ
  3. このスクリプトを起動、シートを各カメラに見せながら動かす
  4. 全カメラが一斉に検出したら自動キャプチャ
  5. 10〜15 ペア集まったら c でキャリブ実行

使い方:
  .venv/bin/python calibrate_auto.py            # 撮影→キャリブ
  .venv/bin/python calibrate_auto.py --recompute  # 保存済み生データから再計算
"""

import cv2
import numpy as np
import json
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from camera_config import load_config, get_reference, CameraThread, CameraSpec


META_PATH      = "calib_sheet_meta.json"
OUTPUT_PATH    = "calibration/cameras_calib.json"
LEGACY_PATH    = "calibration/stereo_calib.json"   # 旧形式との互換用
RAW_DATA_PATH  = "calibration/raw_captures.npz"
MIN_PAIRS      = 5
RECOMMEND      = 10


# ============================================================
# ArUco 検出
# ============================================================

def setup_detector(meta: dict):
    dict_name = meta["aruco_dict"]
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_markers(frame: np.ndarray, detector) -> dict:
    """frame から全ArUcoを検出。{id: (4, 2) corners}"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {
        int(mid): corners[i].reshape(4, 2).astype(np.float64)
        for i, mid in enumerate(ids.flatten())
    }


def extract_correspondences(markers: dict, markers_3d: dict):
    """検出結果と既知3D座標からペアを作る。"""
    img_pts, obj_pts = [], []
    for mid, corners_2d in markers.items():
        if str(mid) not in markers_3d:
            continue
        corners_3d = np.array(markers_3d[str(mid)], dtype=np.float64)
        for c2, c3 in zip(corners_2d, corners_3d):
            img_pts.append(c2)
            obj_pts.append(c3)
    return np.array(img_pts, dtype=np.float32), np.array(obj_pts, dtype=np.float32)


# ============================================================
# キャリブレーション計算
# ============================================================

def _init_intrinsic_matrix(img_size: tuple) -> np.ndarray:
    """画像サイズから初期内部パラメータ行列を作る。"""
    w, h = img_size
    return np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1.0]], dtype=np.float64)


def calibrate_intrinsics(obj_pts, img_pts, img_size):
    """単一カメラの内部パラメータを推定。"""
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_FIX_PRINCIPAL_POINT |   # 主点は画像中心に固定（広角補正済みカメラで妥当）
             cv2.CALIB_FIX_ASPECT_RATIO |      # fx = fy
             cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K2 |
             cv2.CALIB_FIX_K3)
    K_init = _init_intrinsic_matrix(img_size)
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, K_init, None, flags=flags
    )
    return rms, K, dist


def calibrate_stereo(obj_pts, img_pts_a, img_pts_b, K_a, d_a, K_b, d_b, img_size_a):
    """2カメラ間の外部パラメータ（R, T）を推定。"""
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_a, img_pts_b,
        K_a, d_a, K_b, d_b,
        img_size_a,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    return rms, R, T


# ============================================================
# 生データ保存/読み込み
# ============================================================

def save_raw_captures(captures: dict, roles: list[str],
                       img_sizes: dict, obj_pts_all: list, path: str = RAW_DATA_PATH):
    """
    captures: {role: [img_pts_array, ...]}  各キャプチャでのカメラ別2D点
    obj_pts_all: [obj_pts, ...]             共通の3D点（各キャプチャ）
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {
        "roles": np.array(roles),
        "obj_pts": np.array([p.reshape(-1, 3) for p in obj_pts_all], dtype=object),
    }
    for role in roles:
        save_dict[f"img_pts_{role}"] = np.array(
            [p.reshape(-1, 2) for p in captures[role]], dtype=object
        )
        save_dict[f"size_{role}"] = np.array(img_sizes[role])
    np.savez(path, **save_dict)


def load_raw_captures(path: str = RAW_DATA_PATH):
    data = np.load(path, allow_pickle=True)
    roles = [str(r) for r in data["roles"]]
    obj_pts_all = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 3) for p in data["obj_pts"]]
    captures = {}
    img_sizes = {}
    for role in roles:
        captures[role] = [
            np.asarray(p, dtype=np.float32).reshape(-1, 1, 2)
            for p in data[f"img_pts_{role}"]
        ]
        img_sizes[role] = tuple(int(x) for x in data[f"size_{role}"])
    return roles, obj_pts_all, captures, img_sizes


# ============================================================
# 全処理
# ============================================================

def compute_all(roles: list[str], obj_pts_all: list,
                 captures: dict, img_sizes: dict,
                 reference_role: str) -> dict:
    """
    各カメラの内部パラメータ + 参照カメラからの外部パラメータを計算。
    """
    print(f"\n=== キャリブレーション計算 ({len(obj_pts_all)}ペア) ===\n")

    intrinsics = {}
    for role in roles:
        img_pts = captures[role]
        size = img_sizes[role]
        rms, K, dist = calibrate_intrinsics(obj_pts_all, img_pts, size)
        intrinsics[role] = {
            "K":          K.tolist(),
            "dist":       dist.tolist(),
            "image_size": list(size),
            "rms":        round(float(rms), 4),
        }
        print(f"[{role}] {size}  RMS={rms:.3f}px  "
              f"fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f}")

    print()

    # 外部パラメータ: 参照カメラを原点（R=I, T=0）、他は参照からの相対
    extrinsics = {reference_role: {"R": np.eye(3).tolist(), "T": [0, 0, 0], "rms": 0.0}}
    for role in roles:
        if role == reference_role:
            continue
        K_ref  = np.array(intrinsics[reference_role]["K"])
        d_ref  = np.array(intrinsics[reference_role]["dist"])
        K_this = np.array(intrinsics[role]["K"])
        d_this = np.array(intrinsics[role]["dist"])
        rms, R, T = calibrate_stereo(
            obj_pts_all,
            captures[reference_role], captures[role],
            K_ref, d_ref, K_this, d_this,
            img_sizes[reference_role],
        )
        extrinsics[role] = {
            "R":   R.tolist(),
            "T":   T.tolist(),
            "rms": round(float(rms), 4),
        }
        print(f"[{reference_role} -> {role}] ステレオRMS={rms:.3f}px")

    return {
        "version":         2,
        "reference":       reference_role,
        "camera_roles":    roles,
        "intrinsics":      intrinsics,
        "extrinsics":      extrinsics,
    }


def save_result(result: dict):
    """新形式（cameras_calib.json）と旧形式（stereo_calib.json）の両方を出力。"""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n保存: {OUTPUT_PATH}")

    # 旧形式互換（2カメラの場合のみ）
    roles = result["camera_roles"]
    if len(roles) == 2:
        ref = result["reference"]
        other = [r for r in roles if r != ref][0]
        legacy = {
            "K1":          result["intrinsics"][ref]["K"],
            "K2":          result["intrinsics"][other]["K"],
            "dist1":       result["intrinsics"][ref]["dist"],
            "dist2":       result["intrinsics"][other]["dist"],
            "R":           result["extrinsics"][other]["R"],
            "T":           result["extrinsics"][other]["T"],
            "image_size":  result["intrinsics"][ref]["image_size"],
            "image_size1": result["intrinsics"][ref]["image_size"],
            "image_size2": result["intrinsics"][other]["image_size"],
            "n_images":    0,   # 新形式に記録
            "rms_error":   result["extrinsics"][other]["rms"],
            "rms_cam1":    result["intrinsics"][ref]["rms"],
            "rms_cam2":    result["intrinsics"][other]["rms"],
            "method":      "aruco_sheet_v2",
        }
        with open(LEGACY_PATH, "w") as f:
            json.dump(legacy, f, indent=2)
        print(f"保存（互換）: {LEGACY_PATH}")


def save_sheet_positions(meta: dict):
    """シート位置を card_positions.json として保存（3Dビュー用）。"""
    sheet_corners = []
    for mid, corners_3d in sorted(meta["markers_3d"].items(), key=lambda x: int(x[0])):
        sheet_corners.append({"id": int(mid), "corners": corners_3d})
    with open("calibration/card_positions.json", "w") as f:
        json.dump({"cards": sheet_corners,
                   "note": "ArUco markers from calib sheet"}, f, indent=2)


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="N カメラ自動ステレオキャリブレーション（ArUcoシート使用）"
    )
    parser.add_argument("--recompute", action="store_true",
                        help="保存済み生データから再計算（撮影しない）")
    parser.add_argument("--config",    default="cameras_config.json",
                        help="カメラ設定ファイルパス")
    args = parser.parse_args()

    # === 再計算モード ===
    if args.recompute:
        if not os.path.exists(RAW_DATA_PATH):
            print(f"{RAW_DATA_PATH} がありません。先に撮影してください。")
            sys.exit(1)
        print(f"=== 再計算モード ===  {RAW_DATA_PATH}")
        roles, obj_pts_all, captures, img_sizes = load_raw_captures()

        # 参照カメラ: config から決定（無ければ最初）
        specs = load_config(args.config)
        ref_spec = get_reference(specs) if specs else None
        reference_role = ref_spec.role if ref_spec and ref_spec.role in roles else roles[0]

        print(f"  ペア数: {len(obj_pts_all)}")
        print(f"  カメラ: {roles}")
        print(f"  参照  : {reference_role}")

        result = compute_all(roles, obj_pts_all, captures, img_sizes, reference_role)
        save_result(result)
        with open(META_PATH) as f:
            meta = json.load(f)
        save_sheet_positions(meta)
        return

    # === 撮影 → キャリブ モード ===
    if not os.path.exists(META_PATH):
        print(f"{META_PATH} がありません。先に generate_calib_sheet.py を実行してください。")
        sys.exit(1)

    with open(META_PATH) as f:
        meta = json.load(f)
    detector = setup_detector(meta)
    markers_3d = meta["markers_3d"]
    n_required = len(markers_3d)

    # カメラ読み込み
    specs = load_config(args.config)
    print(f"=== ArUco 自動キャリブレーション ===")
    print(f"カメラ数: {len(specs)}")
    for s in specs:
        ref = " [REF]" if s.is_reference else ""
        print(f"  {s.role}: id={s.id} {s.name}{ref}")

    threads = {s.role: CameraThread(s) for s in specs}
    for t in threads.values():
        t.start()
    time.sleep(1.5)
    roles = [s.role for s in specs]
    reference_role = get_reference(specs).role

    # 取得状態
    obj_pts_all = []
    captures = {role: [] for role in roles}
    img_sizes = {role: None for role in roles}

    auto_mode = True
    STABLE_FRAMES = 12
    STILL_THRESHOLD_PX = 3.0
    DIFF_THRESHOLD_PX = 30.0
    stable_count = 0
    last_center = None
    last_captured_center = None

    print(f"\n必要マーカー: {n_required}個 × 全 {len(roles)} カメラで検出")
    print(f"AUTO=静止で自動取得  a=手動/自動切替  c=キャリブ実行  d=削除  q/ESC=終了")
    print(f"目標: {RECOMMEND} ペア\n")

    win = "N-Camera Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = {role: threads[role].get_frame() for role in roles}
            if any(f is None for f in frames.values()):
                time.sleep(0.01); continue

            # 解像度を記録
            for role in roles:
                if img_sizes[role] is None:
                    img_sizes[role] = (frames[role].shape[1], frames[role].shape[0])

            # 各カメラでArUco検出
            detected = {role: detect_markers(frames[role], detector) for role in roles}

            # 全カメラで検出された共通ID
            common = set(detected[roles[0]].keys())
            for role in roles[1:]:
                common &= set(detected[role].keys())
            common &= set(int(k) for k in markers_3d.keys())
            all_detected = len(common) == n_required

            # 自動取得判定（参照カメラの中心を基準）
            auto_capture_now = False
            if all_detected:
                ref_marker_centers = np.array([detected[reference_role][i].mean(axis=0)
                                                for i in common])
                cur_center = ref_marker_centers.mean(axis=0)
                movement = float(np.linalg.norm(cur_center - last_center)) if last_center is not None else 999.0
                diff_from_last = float(np.linalg.norm(cur_center - last_captured_center)) if last_captured_center is not None else 999.0
                last_center = cur_center
                if movement < STILL_THRESHOLD_PX and diff_from_last > DIFF_THRESHOLD_PX:
                    stable_count += 1
                else:
                    stable_count = 0
                if auto_mode and stable_count >= STABLE_FRAMES:
                    auto_capture_now = True
                    stable_count = 0
            else:
                stable_count = 0
                last_center = None

            # 表示
            tiles = []
            for role in roles:
                f = frames[role].copy()
                for mid, corners in detected[role].items():
                    color = (0, 255, 0) if mid in common else (0, 160, 255)
                    cv2.polylines(f, [corners.astype(np.int32)], True, color, 3)
                    c = corners.mean(axis=0).astype(int)
                    cv2.putText(f, f"#{mid}", tuple(c),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                f_small = cv2.resize(f, (480, 270))
                ref_str = " [REF]" if role == reference_role else ""
                cv2.putText(f_small, f"{role}{ref_str}  det={len(detected[role])}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if auto_mode and all_detected:
                    bar_w = int(480 * stable_count / STABLE_FRAMES)
                    cv2.rectangle(f_small, (0, 265), (bar_w, 270), (0, 255, 0), -1)
                tiles.append(f_small)

            # グリッド配置
            cols = min(3, len(tiles))
            rows = (len(tiles) + cols - 1) // cols
            while len(tiles) < rows * cols:
                tiles.append(np.zeros_like(tiles[0]))
            grid_rows = []
            for r in range(rows):
                grid_rows.append(np.hstack(tiles[r * cols:(r + 1) * cols]))
            canvas = np.vstack(grid_rows)

            bar = np.zeros((40, canvas.shape[1], 3), dtype=np.uint8)
            mode = "AUTO" if auto_mode else "MANUAL"
            stat_c = (0, 255, 0) if all_detected else (0, 140, 255)
            cv2.putText(bar,
                        f"Pairs: {len(obj_pts_all)}/{RECOMMEND}  [{mode}]  "
                        f"common={len(common)}/{n_required}",
                        (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stat_c, 2)
            canvas = np.vstack([canvas, bar])
            cv2.imshow(win, canvas)

            def do_capture():
                nonlocal last_captured_center
                # 最初のキャプチャで3D点が確定する
                any_role = next(iter(common))
                # obj_pts は全カメラ共通。ここでは参照カメラの検出から抽出
                img_pts_ref, obj_pts = extract_correspondences(
                    {k: detected[reference_role][k] for k in common},
                    markers_3d
                )
                obj_pts_all.append(obj_pts.reshape(-1, 1, 3))

                for role in roles:
                    img_pts, _ = extract_correspondences(
                        {k: detected[role][k] for k in common},
                        markers_3d
                    )
                    captures[role].append(img_pts.reshape(-1, 1, 2))

                # 生データ保存
                save_raw_captures(captures, roles, img_sizes, obj_pts_all)

                last_captured_center = cur_center
                print(f"  ペア {len(obj_pts_all)} 取得 ({len(obj_pts)}点 × {len(roles)}カメラ)")
                flash = canvas.copy()
                cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]),
                              (0, 255, 0), 10)
                cv2.putText(flash, f"CAPTURED {len(obj_pts_all)}",
                            (flash.shape[1]//2 - 130, flash.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                cv2.imshow(win, flash)
                cv2.waitKey(300)

            if auto_capture_now:
                do_capture()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("a"):
                auto_mode = not auto_mode
                stable_count = 0
                print(f"  モード: {'AUTO' if auto_mode else 'MANUAL'}")
            elif key == ord(" ") and all_detected:
                do_capture()
            elif key == ord("d") and obj_pts_all:
                obj_pts_all.pop()
                for role in roles:
                    captures[role].pop()
                print(f"  最後のペアを削除 (残り {len(obj_pts_all)})")
            elif key == ord("c"):
                if len(obj_pts_all) < MIN_PAIRS:
                    print(f"  ペア不足: {len(obj_pts_all)}/{MIN_PAIRS}")
                    continue
                cv2.destroyAllWindows()
                result = compute_all(roles, obj_pts_all, captures, img_sizes, reference_role)
                save_result(result)
                save_sheet_positions(meta)
                break

    finally:
        for t in threads.values():
            t.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
