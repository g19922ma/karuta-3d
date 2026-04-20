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
    """単一カメラの内部パラメータを推定。
    収束しない/異常値の場合は初期値（fx = image_width）にフォールバックする。
    戻り値: (rms, K, dist, used_fallback)
    """
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_FIX_PRINCIPAL_POINT |   # 主点は画像中心に固定
             cv2.CALIB_FIX_ASPECT_RATIO |      # fx = fy
             cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K2 |
             cv2.CALIB_FIX_K3)
    expected_fx = img_size[0]

    try:
        # cv2.calibrateCamera は K を破壊的に書き換えるので、
        # 呼び出し用と「フォールバック用」の初期値を別々に保持する
        K_for_calib = _init_intrinsic_matrix(img_size)
        rms, K, dist, _, _ = cv2.calibrateCamera(
            obj_pts, img_pts, img_size, K_for_calib, None, flags=flags
        )
        fx = K[0, 0]
        if fx < expected_fx / 3 or fx > expected_fx * 3:
            print(f"  警告: fx={fx:.0f} が期待値{expected_fx}から大きく外れる → 初期値使用")
            return 0.0, _init_intrinsic_matrix(img_size), np.zeros(5, dtype=np.float64), True
        if rms > 30.0:
            print(f"  警告: RMS={rms:.1f}px が大きすぎる → 初期値使用")
            return 0.0, _init_intrinsic_matrix(img_size), np.zeros(5, dtype=np.float64), True
        return rms, K, dist, False
    except cv2.error as e:
        print(f"  キャリブ失敗 → 初期値使用: {e}")
        return 0.0, _init_intrinsic_matrix(img_size), np.zeros(5, dtype=np.float64), True


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
    captures[role][i] は (img_pts) または None（そのキャプチャでカメラが未検出）。
    obj_pts_all[i] は常に同じ3D点（共通の物理平面）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {
        "roles":   np.array(roles),
        "obj_pts": np.array([p.reshape(-1, 3) for p in obj_pts_all], dtype=object),
    }
    for role in roles:
        # None は shape (0, 2) の空配列にする（復元時に識別可）
        arr = []
        for p in captures[role]:
            if p is None:
                arr.append(np.zeros((0, 2), dtype=np.float32))
            else:
                arr.append(p.reshape(-1, 2))
        save_dict[f"img_pts_{role}"] = np.array(arr, dtype=object)
        save_dict[f"size_{role}"] = np.array(img_sizes[role])
    np.savez(path, **save_dict)


def load_raw_captures(path: str = RAW_DATA_PATH):
    data = np.load(path, allow_pickle=True)
    roles = [str(r) for r in data["roles"]]
    obj_pts_all = [np.asarray(p, dtype=np.float32).reshape(-1, 1, 3) for p in data["obj_pts"]]
    captures = {}
    img_sizes = {}
    for role in roles:
        raw = data[f"img_pts_{role}"]
        captures[role] = []
        for p in raw:
            arr = np.asarray(p, dtype=np.float32)
            if arr.shape[0] == 0:
                captures[role].append(None)   # 未検出
            else:
                captures[role].append(arr.reshape(-1, 1, 2))
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

    captures[role][i] は None 可能（そのキャプチャでこのカメラが未検出）。
    内部計算は各カメラが有効なキャプチャだけを使う。
    外部計算は両カメラが同時に有効なキャプチャだけを使う。
    """
    print(f"\n=== キャリブレーション計算 (total {len(obj_pts_all)} キャプチャ) ===\n")

    # --- 内部パラメータ: 各カメラが見えているキャプチャだけ使う ---
    intrinsics = {}
    for role in roles:
        valid_indices = [i for i, p in enumerate(captures[role]) if p is not None]
        if len(valid_indices) < 3:
            print(f"[{role}] 有効キャプチャ不足: {len(valid_indices)}（最低3）")
            raise RuntimeError(f"{role} の内部キャリブに十分なデータがありません")
        img_pts_valid = [captures[role][i] for i in valid_indices]
        obj_pts_valid = [obj_pts_all[i]    for i in valid_indices]

        size = img_sizes[role]
        rms, K, dist, fallback = calibrate_intrinsics(obj_pts_valid, img_pts_valid, size)
        intrinsics[role] = {
            "K":          K.tolist(),
            "dist":       dist.tolist(),
            "image_size": list(size),
            "rms":        round(float(rms), 4),
            "n_captures": len(valid_indices),
            "used_fallback": fallback,
        }
        tag = " [FALLBACK]" if fallback else ""
        print(f"[{role}]{tag} n={len(valid_indices)} 画像サイズ={size}  RMS={rms:.3f}px  "
              f"fx={K[0,0]:.0f} fy={K[1,1]:.0f} cx={K[0,2]:.0f} cy={K[1,2]:.0f}")

    print()

    # --- 外部パラメータ: 参照との同時検出キャプチャだけ使う ---
    extrinsics = {reference_role: {"R": np.eye(3).tolist(), "T": [0, 0, 0], "rms": 0.0,
                                    "n_captures": 0}}
    for role in roles:
        if role == reference_role:
            continue

        # 両方が同時に見えたキャプチャだけ
        shared_indices = [
            i for i in range(len(obj_pts_all))
            if captures[reference_role][i] is not None and captures[role][i] is not None
        ]
        if len(shared_indices) < 2:
            print(f"[{reference_role} -> {role}] 共通キャプチャ不足: {len(shared_indices)}")
            extrinsics[role] = {"error": f"共通キャプチャ {len(shared_indices)} 件（最低2）",
                                 "n_captures": len(shared_indices)}
            continue

        img_pts_ref   = [captures[reference_role][i] for i in shared_indices]
        img_pts_this  = [captures[role][i]           for i in shared_indices]
        obj_pts_shared = [obj_pts_all[i]             for i in shared_indices]

        K_ref  = np.array(intrinsics[reference_role]["K"])
        d_ref  = np.array(intrinsics[reference_role]["dist"])
        K_this = np.array(intrinsics[role]["K"])
        d_this = np.array(intrinsics[role]["dist"])

        rms, R, T = calibrate_stereo(
            obj_pts_shared, img_pts_ref, img_pts_this,
            K_ref, d_ref, K_this, d_this,
            img_sizes[reference_role],
        )
        extrinsics[role] = {
            "R":          R.tolist(),
            "T":          T.tolist(),
            "rms":        round(float(rms), 4),
            "n_captures": len(shared_indices),
        }
        print(f"[{reference_role} -> {role}] n={len(shared_indices)} ステレオRMS={rms:.3f}px")

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

    # 旧形式互換（2カメラ + 外部キャリブ成功時のみ）
    roles = result["camera_roles"]
    if len(roles) == 2:
        ref = result["reference"]
        other = [r for r in roles if r != ref][0]
        ext = result["extrinsics"].get(other, {})
        if "R" not in ext or "T" not in ext:
            print(f"  警告: 外部キャリブが失敗しているため {LEGACY_PATH} は更新しません")
            print(f"       理由: {ext.get('error', '不明')}")
            return
        legacy = {
            "K1":          result["intrinsics"][ref]["K"],
            "K2":          result["intrinsics"][other]["K"],
            "dist1":       result["intrinsics"][ref]["dist"],
            "dist2":       result["intrinsics"][other]["dist"],
            "R":           ext["R"],
            "T":           ext["T"],
            "image_size":  result["intrinsics"][ref]["image_size"],
            "image_size1": result["intrinsics"][ref]["image_size"],
            "image_size2": result["intrinsics"][other]["image_size"],
            "n_images":    ext.get("n_captures", 0),
            "rms_error":   ext["rms"],
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

    # カバレッジ: 3x3 のどのゾーンでシートを撮ったか（カメラごと）
    GRID = 3
    coverage = {role: set() for role in roles}

    auto_mode = True
    STABLE_FRAMES = 12
    STILL_THRESHOLD_PX = 3.0
    DIFF_THRESHOLD_PX = 30.0
    stable_count = 0
    last_center = None
    last_captured_center = None

    def zone_of(x: float, y: float, w: int, h: int) -> tuple[int, int]:
        return (min(GRID-1, int(x / w * GRID)),
                min(GRID-1, int(y / h * GRID)))

    def current_zones(role: str) -> set:
        """今シートが跨っているゾーン集合（このカメラが全マーカー検出時）。"""
        if img_sizes[role] is None:
            return set()
        if role not in detected or not mono_ok.get(role, False):
            return set()
        w, h = img_sizes[role]
        zs = set()
        for mid in all_ids:
            if mid in detected[role]:
                for (x, y) in detected[role][mid]:
                    zs.add(zone_of(x, y, w, h))
        return zs

    def coverage_progress() -> tuple[int, int]:
        """全カメラを合わせたカバレッジ進捗 (covered, total)。"""
        total = GRID * GRID * len(roles)
        covered = sum(len(s) for s in coverage.values())
        return covered, total

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

            # 各カメラが全マーカーを検出したか（単独検出）
            all_ids = set(int(k) for k in markers_3d.keys())
            mono_ok = {
                role: len(set(detected[role].keys()) & all_ids) == n_required
                for role in roles
            }
            # 2カメラ以上で同時検出できているもの（ステレオ可能）
            multi_ok_roles = [r for r in roles if mono_ok[r]]
            common = all_ids if len(multi_ok_roles) >= 2 else set()
            all_detected = any(mono_ok.values())

            # 自動取得判定: どこか1台でも全マーカー検出 + 静止 + 未踏ゾーン
            auto_capture_now = False
            new_zone_found = False
            status_msg = ""
            cur_center = None

            if all_detected:
                # 未踏ゾーンを新しく含んでいるカメラがあるか
                cur_zones = {
                    role: (current_zones(role) if mono_ok[role] else set())
                    for role in roles
                }
                new_zones_per_cam = {
                    role: cur_zones[role] - coverage[role] for role in roles
                }
                new_zone_found = any(bool(v) for v in new_zones_per_cam.values())

                # 静止判定: 全有効カメラの中心を平均したもの
                centers = []
                for r in roles:
                    if mono_ok[r]:
                        marker_centers = [detected[r][i].mean(axis=0) for i in all_ids]
                        centers.append(np.mean(marker_centers, axis=0))
                cur_center = np.mean(centers, axis=0)

                movement = float(np.linalg.norm(cur_center - last_center)) if last_center is not None else 999.0
                last_center = cur_center

                if movement < STILL_THRESHOLD_PX and new_zone_found:
                    stable_count += 1
                    status_msg = f"静止確認中 {stable_count}/{STABLE_FRAMES}"
                else:
                    stable_count = 0
                    if movement >= STILL_THRESHOLD_PX:
                        status_msg = "動いてます"
                    elif not new_zone_found:
                        status_msg = "新しいゾーンへ移動してください"

                if auto_mode and stable_count >= STABLE_FRAMES:
                    auto_capture_now = True
                    stable_count = 0
            else:
                stable_count = 0
                last_center = None
                # なぜ検出できていないかを表示
                missing = [r for r in roles if not mono_ok[r]]
                status_msg = f"マーカー検出不足: {', '.join(missing)}"

            # 表示
            tiles = []
            for role in roles:
                f = frames[role].copy()
                h, w = f.shape[:2]

                # --- カバレッジグリッド（半透明オーバーレイ）---
                overlay = f.copy()
                cur_zs = current_zones(role) if all_detected else set()
                for zx in range(GRID):
                    for zy in range(GRID):
                        x1 = zx * w // GRID
                        y1 = zy * h // GRID
                        x2 = (zx + 1) * w // GRID
                        y2 = (zy + 1) * h // GRID
                        z = (zx, zy)
                        if z in cur_zs and z not in coverage[role]:
                            # 今シートが乗っている未踏ゾーン = 黄
                            color = (0, 255, 255)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        elif z in coverage[role]:
                            # 撮影済み = 緑
                            color = (0, 200, 0)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        else:
                            # 未撮影 = 赤薄塗り
                            color = (0, 0, 120)
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        cv2.rectangle(f, (x1, y1), (x2, y2), (60, 60, 60), 1)
                f = cv2.addWeighted(overlay, 0.25, f, 0.75, 0)

                # --- マーカー検出結果 ---
                for mid, corners in detected[role].items():
                    color = (0, 255, 0) if mid in common else (0, 160, 255)
                    cv2.polylines(f, [corners.astype(np.int32)], True, color, 3)
                    c = corners.mean(axis=0).astype(int)
                    cv2.putText(f, f"#{mid}", tuple(c),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # --- ラベルと進捗 ---
                f_small = cv2.resize(f, (480, 270))
                ref_str = " [REF]" if role == reference_role else ""
                cov = len(coverage[role])
                cv2.putText(f_small,
                            f"{role}{ref_str}  coverage {cov}/{GRID*GRID}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                if auto_mode and all_detected and new_zone_found:
                    bar_w = int(480 * stable_count / STABLE_FRAMES)
                    cv2.rectangle(f_small, (0, 265), (bar_w, 270), (0, 255, 0), -1)
                elif auto_mode and all_detected and not new_zone_found:
                    cv2.putText(f_small, "すでに撮影済みのゾーン - 移動してください",
                                (8, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 160, 255), 1)
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

            bar = np.zeros((56, canvas.shape[1], 3), dtype=np.uint8)
            mode = "AUTO" if auto_mode else "MANUAL"
            stat_c = (0, 255, 0) if auto_capture_now else ((0, 220, 220) if all_detected else (0, 120, 255))
            # 1行目: 進捗
            cov_totals = [f"{r}:{len(coverage[r])}/{GRID*GRID}" for r in roles]
            cv2.putText(bar,
                        f"[{mode}] caps={len(obj_pts_all)}  " +
                        "  ".join(cov_totals),
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 2)
            # 2行目: 状況メッセージ
            cv2.putText(bar, status_msg,
                        (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, stat_c, 2)
            canvas = np.vstack([canvas, bar])
            cv2.imshow(win, canvas)

            def do_capture():
                nonlocal last_captured_center
                # 3D点は any_roleの検出から抽出（マーカーIDごとに3D位置は固定）
                first_ok = next(r for r in roles if mono_ok[r])
                _, obj_pts = extract_correspondences(
                    {k: detected[first_ok][k] for k in all_ids if k in detected[first_ok]},
                    markers_3d
                )
                obj_pts_all.append(obj_pts.reshape(-1, 1, 3))

                # 各カメラごとに検出できてれば記録、なければ None
                for role in roles:
                    if mono_ok[role]:
                        img_pts, _ = extract_correspondences(
                            {k: detected[role][k] for k in all_ids},
                            markers_3d
                        )
                        captures[role].append(img_pts.reshape(-1, 1, 2))
                    else:
                        captures[role].append(None)

                # カバレッジ更新（検出できたカメラだけ）
                for role in roles:
                    if mono_ok[role]:
                        coverage[role].update(current_zones(role))

                # 生データ保存
                save_raw_captures(captures, roles, img_sizes, obj_pts_all)

                last_captured_center = cur_center
                cov_done, cov_total = coverage_progress()
                print(f"  ペア {len(obj_pts_all)} 取得 ({len(obj_pts)}点 × {len(roles)}カメラ) "
                      f"カバレッジ {cov_done}/{cov_total}")
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
                # カバレッジを再計算（残りのキャプチャから）
                for role in roles:
                    coverage[role] = set()
                    w, h = img_sizes[role]
                    for pts in captures[role]:
                        if pts is None:
                            continue
                        for (x, y) in pts.reshape(-1, 2):
                            coverage[role].add(zone_of(float(x), float(y), w, h))
                print(f"  最後のキャプチャを削除 (残り {len(obj_pts_all)})")
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
