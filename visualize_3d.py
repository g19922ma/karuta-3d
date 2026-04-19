"""
3D手のキーポイントの可視化

- 単一フレームおよび複数フレームのトラジェクトリ表示
- PNG 保存対応
- matplotlib 使用（追加インストール不要、かつ静的画像として保存しやすい）
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401（3D axesの初期化に必要）
from pathlib import Path

# 可視化するスケルトン接続
SKELETON_CONNECTIONS = [
    ("wrist", "index_finger_tip"),
    ("wrist", "middle_finger_tip"),
    ("index_finger_tip", "middle_finger_tip"),
]


def plot_3d_trajectory(
    frames_data: list[dict],
    landmark_names: list[str],
    output_path: str | None = None,
    show: bool = True,
    title: str = "3D Hand Trajectory",
):
    """
    複数フレームの3Dキーポイントをトラジェクトリとして描画する。

    Args:
        frames_data    : [{landmark_name: [x, y, z], ...}, ...] のリスト（フレーム順）
        landmark_names : 表示するランドマーク名リスト
        output_path    : 保存先PNGパス（None なら保存しない）
        show           : インタラクティブ表示するか
        title          : グラフタイトル
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    n_frames = len(frames_data)
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, n_frames)) if n_frames > 1 else ["royalblue"]

    for i, (frame, color) in enumerate(zip(frames_data, cmap)):
        alpha = 0.3 + 0.7 * (i / max(n_frames - 1, 1))

        # 各ランドマーク点を描画
        for name in landmark_names:
            if name not in frame:
                continue
            x, y, z = frame[name]
            ax.scatter(x, y, z, color=[color], alpha=alpha, s=40, zorder=5)

        # スケルトン線を描画
        for a, b in SKELETON_CONNECTIONS:
            if a in frame and b in frame:
                xa, ya, za = frame[a]
                xb, yb, zb = frame[b]
                ax.plot([xa, xb], [ya, yb], [za, zb],
                        color=color, alpha=alpha, linewidth=1.8)

    # 手首のトラジェクトリを太線で強調
    wrist_traj = [f["wrist"] for f in frames_data if "wrist" in f]
    if len(wrist_traj) > 1:
        xs, ys, zs = zip(*wrist_traj)
        ax.plot(xs, ys, zs, "w--", alpha=0.5, linewidth=1, label="wrist trajectory")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title}\n({n_frames} frames)")
    ax.legend(fontsize=8)

    # カラーバー（フレーム進行）
    if n_frames > 1:
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, n_frames - 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label("Frame index")

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_3d] 3D plot 保存: {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_single_frame(
    frame_data: dict,
    landmark_names: list[str],
    output_path: str | None = None,
    show: bool = True,
    title: str = "3D Hand Keypoints",
):
    """単一フレームの3Dキーポイントを表示する（デバッグ用）。"""
    plot_3d_trajectory([frame_data], landmark_names, output_path, show, title)
