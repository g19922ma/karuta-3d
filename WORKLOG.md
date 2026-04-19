# 作業ログ — karuta-3d

## 最終更新: 2026-04-19

### 現在の状態

プロトタイプ初版を実装完了。

- [x] `main.py` — CLI エントリポイント。フレーム範囲指定・シンク・キャリブ切り替え対応
- [x] `sync_tool.py` — 2動画の同期オフセット対話的決定ツール
- [x] `detect_hand.py` — MediaPipe Hands + 手動クリックフォールバック
- [x] `triangulate.py` — cv2.triangulatePoints による3D復元
- [x] `visualize_3d.py` — matplotlib 3D トラジェクトリ表示・PNG保存
- [x] `calibration/simple_calib.py` — 近似パラメータによる即時動作確認用
- [x] `calibration/full_calib.py` — チェッカーボードによる本格ステレオキャリブ
- [x] `config/sample_config.json` — カメラパラメータ設定サンプル
- [x] `requirements.txt`
- [x] `README.md`

### 次のステップ

1. 実際の動画（video_pc.mp4, video_phone.mp4）で動作確認
2. `simple_calib` の近似パラメータを実機に合わせてチューニング
3. 検出精度が低い場合はチェッカーボードキャリブへ移行（full_calib）
4. 音声解析による自動シンク（sync_tool のアップグレード）
5. 両手対応（max_num_hands=2）
6. 札認識との統合（detect_card.py の追加）

### 制約・注意事項

- `simple_calib` の絶対座標精度は低い。相対的な動きの確認用と割り切る
- MediaPipe は `static_image_mode=True`（フレーム独立）なのでリアルタイム追跡はしない
- Apple Silicon Mac では mediapipe-silicon が必要な場合がある
- sync_tool の矢印キーはOS・OpenCVバージョンによって keycode が異なる場合がある（a/d を代替に使う）

### 未解決の問題

- 歪み補正（distortion correction）が未実装。精度向上には必要
- ArUco キャリブレーションは stub のまま

### メモ

- 北川ら（単一視点骨格可視化）の次の段階として2視点3D化を実装
- 成功条件: ①同じ点が2視点で取れる ②その点が3Dで表示できる
- 精密な絶対座標・接触判定・リアルタイム・札認識は今回スコープ外
