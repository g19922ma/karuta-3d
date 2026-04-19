# karuta-3d — 競技かるた 複数視点3D手姿勢復元 プロトタイプ

PCカメラとスマホの2視点動画から、手のキーポイントを3D座標として復元・可視化するプロトタイプです。
単一視点では遮蔽に弱い問題を、2視点化によって補う足場を作ることが目的です。

---

## ディレクトリ構成

```
karuta-3d/
├── main.py               # メインスクリプト（CLI）
├── sync_tool.py          # 2動画の同期オフセット決定ツール
├── detect_hand.py        # MediaPipe による手のキーポイント検出
├── triangulate.py        # 2視点→3D三角測量
├── visualize_3d.py       # 3D可視化・保存
├── calibration/
│   ├── simple_calib.py   # 簡易キャリブレーション（近似パラメータ）
│   └── full_calib.py     # 本格ステレオキャリブレーション（要実行）
├── config/
│   └── sample_config.json # カメラパラメータ設定サンプル
├── output/               # 結果出力ディレクトリ（自動作成）
├── requirements.txt
└── README.md
```

---

## セットアップ

```bash
cd karuta-3d
pip install -r requirements.txt
```

> **mediapipe に関する注意**: Apple Silicon Mac では mediapipe の wheel が提供されていない場合があります。
> その場合は `pip install mediapipe-silicon` を試してください。

---

## まず何をすればよいか（最短手順）

### ステップ 1: 動画を用意する

- `video_pc.mp4`  : PCカメラで撮影した映像
- `video_phone.mp4`: スマホで撮影した映像
- どちらも同じ手拍子などのシンクイベントを含めておくと良い

### ステップ 2: 同期オフセットを決定する（初回のみ）

```bash
python sync_tool.py --pc video_pc.mp4 --phone video_phone.mp4
```

- 両動画で同じ瞬間（手拍子など）のフレームを選んで `SPACE` を押す
- `sync_offset.json` が出力される

### ステップ 3: 3D復元を実行する

```bash
python main.py \
  --pc video_pc.mp4 \
  --phone video_phone.mp4 \
  --start 120 \
  --end 160 \
  --sync sync_offset.json \
  --config config/sample_config.json
```

- 自動検出が失敗したフレームは画面に点をクリックして手動指定できます
- `output/YYYYMMDD_HHMMSS/` に結果が保存されます

---

## 実行コマンド一覧

### 基本実行（近似キャリブレーション）

```bash
python main.py --pc video_pc.mp4 --phone video_phone.mp4 --start 120 --end 160
```

### シンクオフセット + 設定ファイルを指定

```bash
python main.py \
  --pc video_pc.mp4 \
  --phone video_phone.mp4 \
  --start 120 --end 160 \
  --sync sync_offset.json \
  --config config/sample_config.json
```

### フレームを間引いて高速処理（5フレームに1枚）

```bash
python main.py --pc video_pc.mp4 --phone video_phone.mp4 \
  --start 0 --end 300 --step 5
```

### 手動アノテーションを無効化

```bash
python main.py ... --no-manual
```

### 本格キャリブレーションを使用

```bash
# 先にキャリブレーション実行
python calibration/full_calib.py \
  --images-dir calibration_images/ \
  --board-cols 9 --board-rows 6 --square-size 0.025

# 本格キャリブを使って3D復元
python main.py ... --calib-mode full --config config/sample_config.json
```

### 同期ツール

```bash
python sync_tool.py --pc video_pc.mp4 --phone video_phone.mp4
```

---

## 出力

```
output/YYYYMMDD_HHMMSS/
├── 3d_points.json        # フレームごとの3D座標
├── 3d_points.csv         # 同上（CSV形式）
└── 3d_visualization.png  # 3Dトラジェクトリ図
```

### 3d_points.json の形式

```json
{
  "frames": [
    {
      "frame_idx": 120,
      "landmarks": {
        "wrist":             [0.12, -0.05, 0.83],
        "index_finger_tip":  [0.18, -0.12, 0.79],
        "middle_finger_tip": [0.17, -0.11, 0.80]
      }
    }
  ]
}
```

---

## キャリブレーションについて

### A. 簡易版（デフォルト）

`calibration/simple_calib.py` が近似パラメータを使用します。  
設定は `config/sample_config.json` で調整できます:

| パラメータ       | 意味                           | デフォルト |
|----------------|-------------------------------|---------|
| `baseline`     | カメラ間距離 [m]                | 0.5     |
| `phone_angle_deg` | スマホの内向き角度 [deg]       | -20.0   |
| `pc_fx` / `pc_fy` | PC カメラ焦点距離 [px]       | 1536    |
| `phone_fx` / `phone_fy` | スマホ焦点距離 [px]   | 1728    |

**この段階での精度の限界**: 絶対座標は不正確です。相対的な動きの確認には使えます。

### B. 本格版（チェッカーボード）

精密な計測が必要になったら以下を実施してください:

1. 9×6 のチェッカーボードを印刷（正方形 2.5cm 推奨）
2. 2台のカメラで同時にチェッカーボードを様々な角度から撮影（10〜20枚）
3. PC側を `calibration_images/calib_pc_001.jpg` 〜、スマホ側を `calib_phone_001.jpg` 〜 と保存
4. `python calibration/full_calib.py` を実行

---

## 検出ランドマーク

| 名前                | MediaPipe インデックス | 説明         |
|--------------------|----------------------|-------------|
| `wrist`            | 0                    | 手首         |
| `index_finger_tip` | 8                    | 人差し指先端  |
| `middle_finger_tip`| 12                   | 中指先端      |

追加したい場合は `detect_hand.py` の `TARGET_LANDMARKS` リストに追記してください。

---

## 手動アノテーション

MediaPipe の自動検出が失敗した場合、フレームが表示されてクリックで点を指定できます:

- 表示されたランドマーク名を見て、対応する位置をクリック
- `q` キーでそのフレームをスキップ

---

## 仮実装の箇所・今後の本格化ポイント

### 仮実装（現在）

| 箇所 | 内容 | 対処法 |
|------|------|-------|
| `simple_calib.py` | カメラパラメータが近似値 | `full_calib.py` + チェッカーボードで計測 |
| カメラ外部パラメータ (R, t) | 手動設定（baseline, angle）| ステレオキャリブレーションで自動計算 |
| 歪み補正なし | MediaPipe の入力が歪んだ画像のまま | 歪み補正後の画像で検出する |
| 単一ハンド | `max_num_hands=1` | 両手対応は `max_num_hands=2` に変更 |
| フレーム同期 | 手動でシンク点を指定 | 音声解析（手拍子ピーク検出）で自動化 |

### 次に札の認識を入れるなら

1. `detect_hand.py` と並列に `detect_card.py` を作成
2. MediaPipe の Hand Tracking に加えて、YOLO or テンプレートマッチングで札を検出
3. 手の3D座標と札の2D座標（または3D座標）を対応づけて「接触判定」を実装
4. `main.py` に `--detect-cards` フラグを追加して処理を分岐させる

### リアルタイム化するなら

1. `static_image_mode=True` → `False` に変更（MediaPipe の追跡モードを有効化）
2. フレームループをスレッドで分離（カメラ読み取り / 検出 / 表示 を非同期化）
3. `cv2.VideoCapture(0)` / `cv2.VideoCapture(1)` でリアルタイム入力に切り替え

---

## 必要ライブラリ

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## 想定する実験セットアップ

```
         [PC カメラ]
              |
              |  ~0.5m baseline
              |
         [スマホ]
              ↓
          [かるた盤面]（手が動く領域）
```

- PC カメラ: 正面から俯瞰気味
- スマホ: 横〜斜め方向から固定撮影（三脚推奨）
- 照明: 均一な室内照明（影が強いと検出精度低下）
