"""
札の矩形を画像から自動検出する

アプローチ:
1. 適応的二値化 + Cannyエッジで輪郭を抽出
2. 4頂点の凸多角形を絞り込み
3. 面積・アスペクト比（約1.4）でフィルタ
4. 頂点を 左上→右上→右下→左下 の順に整列
"""

import cv2
import numpy as np


CARD_ASPECT_RATIO = 7.3 / 5.2   # 競技かるたの取り札のアスペクト比（長辺/短辺）


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    4点を 左上→右上→右下→左下 の順に並び替える。

    Args:
        corners: (4, 2) または (4, 1, 2) の配列
    Returns:
        (4, 2) の順序整列済み配列
    """
    pts = corners.reshape(4, 2).astype(np.float32)

    # 重心
    center = pts.mean(axis=0)

    # 各点の角度でソート（左上を起点に時計回り）
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    # 左上 = 左上方向（角度 ~-3π/4 あたり）から時計回り
    # まずは角度でソートして左上を探す
    sorted_idx = np.argsort(angles)
    pts_sorted = pts[sorted_idx]

    # 左上を見つける：左上の点は x+y が最小
    sums = pts_sorted.sum(axis=1)
    tl_idx = np.argmin(sums)

    # 左上から時計回りに並び直す
    ordered = np.roll(pts_sorted, -tl_idx, axis=0)

    # 時計回りかどうか確認：左上→右上 は x が増える方向
    if ordered[1][0] < ordered[0][0]:
        # 反時計回りになっていたら反転
        ordered = ordered[[0, 3, 2, 1]]

    return ordered


def detect_cards(
    frame: np.ndarray,
    min_area_ratio: float = 0.003,   # 画面面積に対する最小割合
    max_area_ratio: float = 0.15,    # 画面面積に対する最大割合
    aspect_tolerance: float = 0.25,  # アスペクト比の許容範囲
    debug: bool = False,
) -> list[np.ndarray]:
    """
    画像から札の矩形を検出する。

    Args:
        frame             : BGR 画像
        min/max_area_ratio: 画面サイズに対する札サイズの許容範囲
        aspect_tolerance  : アスペクト比（1.4）からのずれ許容

    Returns:
        [corners_1, corners_2, ...]
        各 corners は (4, 2) の 左上→右上→右下→左下 順
    """
    h, w = frame.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2系統のエッジ抽出を並列に試す
    edges1 = cv2.Canny(blur, 30, 100)
    edges2 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 6
    )
    edges = cv2.bitwise_or(edges1, edges2)

    # 近接エッジをつなげる
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_ratio or area > img_area * max_area_ratio:
            continue

        # 凸多角形に近似
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        # 頂点整列
        corners = order_corners(approx)

        # アスペクト比チェック
        edge_lens = [
            np.linalg.norm(corners[(i+1) % 4] - corners[i])
            for i in range(4)
        ]
        long_side  = max(edge_lens)
        short_side = min(edge_lens)
        if short_side < 10:
            continue
        ratio = long_side / short_side
        if abs(ratio - CARD_ASPECT_RATIO) > aspect_tolerance:
            continue

        cards.append(corners)

    # 左から右の順に並び替え
    cards.sort(key=lambda c: c[:, 0].mean())

    if debug:
        debug_img = frame.copy()
        for i, card in enumerate(cards):
            for j in range(4):
                p1 = tuple(card[j].astype(int))
                p2 = tuple(card[(j+1) % 4].astype(int))
                cv2.line(debug_img, p1, p2, (0, 255, 0), 2)
                cv2.circle(debug_img, p1, 5, (0, 0, 255), -1)
                cv2.putText(debug_img, str(j+1), p1,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            center = card.mean(axis=0).astype(int)
            cv2.putText(debug_img, f"#{i+1}", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return cards, debug_img

    return cards


def match_cards_between_views(
    cards_a: list[np.ndarray],
    cards_b: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    2視点で検出された札を位置で対応づける。

    いまは「左から右に並んだ順に1対1対応」という単純な前提。
    枚数が合わないときは共通する枚数だけ返す。

    Returns:
        [(corners_a, corners_b), ...]
    """
    n = min(len(cards_a), len(cards_b))
    return [(cards_a[i], cards_b[i]) for i in range(n)]


# ---------- 単体テスト ----------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python detect_cards.py <image>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    cards, debug = detect_cards(img, debug=True)
    print(f"検出: {len(cards)}枚")
    cv2.imwrite("/tmp/cards_debug.png", debug)
    print("デバッグ画像: /tmp/cards_debug.png")
