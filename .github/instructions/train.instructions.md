---
applyTo: "train.py"
---

# train.py 詳細説明書

このスクリプトは、OphNet手術フェーズ分類タスクのパイプラインを構築します。

## 主要な依存パッケージ

- torch: PyTorchモデルとデータ処理
- pytorch_lightning: 学習フレームワーク
- timm: 事前学習済みモデル
- pandas: データ操作
- numpy: 数値計算
- click: CLIインターフェース
- sklearn: データ分割
- cv2: 画像処理

## クラスとメソッドの詳細

### OphNetSurgicalDataset (torch.utils.data.Dataset)
手術フェーズ分類用のデータセットクラス

#### コンストラクタ引数
- df (pd.DataFrame): 動画メタデータのデータフレーム
- phase2idx (dict): フェーズ名からインデックスへのマッピング
- window_size (int, default=16): ウィンドウサイズ
- stride (int, default=8): ストライド幅
- max_frames (int, default=64): 読み込む最大フレーム数

#### メソッド

##### __len__()
データセットの長さを返す

##### __getitem__(idx)
指定インデックスのサンプルを取得

処理ステップ:
1. データフレームから動画情報取得
2. 動画フレーム読み込み
3. マスク生成（R/Gチャンネル）
4. ウィンドウ抽出とランダム選択
5. 画像処理（平均化、リサイズ、正規化）
6. one-hotラベル生成
7. エラー時はダミーデータ返却

##### _read_video(video_path)
動画ファイルからフレームを読み込む内部メソッド

### SlidingWindowExtractor
動画フレームからウィンドウを抽出するクラス

#### コンストラクタ引数
- window_size (int, default=16): ウィンドウサイズ
- stride (int, default=8): ストライド幅

#### メソッド

##### extract_windows(video_frames, masks=None)
ウィンドウ抽出処理

処理ステップ:
1. ウィンドウリスト初期化
2. フレーム分割とスタック
3. マスクウィンドウ生成（オプション）
4. numpy配列に変換して返却

### OphNetDataModule (pl.LightningDataModule)
データセットの管理とデータローダーの構築を行うクラス

#### コンストラクタ引数
- csv_path (str): CSVファイルパス
- phase2idx (dict): フェーズマッピング
- batch_size (int, default=8): バッチサイズ
- num_workers (int, default=4): 並列ワーカー数
- window_size (int, default=16): ウィンドウサイズ
- stride (int, default=8): ストライド幅
- max_frames (int, default=64): 最大フレーム数

#### メソッド

##### setup(stage: Optional[str] = None)
データセットの準備

処理ステップ:
1. CSVデータ読み込み
2. トレーニング/検証データ分割（8:2）
3. データセットインスタンス作成

##### train_dataloader()
トレーニング用DataLoader生成

##### val_dataloader()
検証用DataLoader生成

### SurgicalPhaseClassificationModel (nn.Module)
手術フェーズ分類モデル

#### コンストラクタ引数
- num_classes (int): 分類クラス数

#### コンポーネント
1. 画像エンコーダ: maxvit_large_tf_224.in21k（事前学習済み）
2. マスク特徴量変換: 線形層
3. 特徴量統合: Multi-Head Attention
4. 分類器: 全結合層（512次元中間層）

#### メソッド

##### forward(image, mask)
順伝播処理

処理ステップ:
1. 画像特徴量抽出
2. マスク特徴量変換
3. Attention処理
4. 分類器で予測

### SurgicalPhaseModule (pl.LightningModule)
学習プロセスを管理するLightningModule

#### コンストラクタ引数
- num_classes (int): 分類クラス数
- class_weights (Optional[torch.Tensor]): クラス重み
- lr (float, default=1e-4): 学習率

#### メソッド

##### training_step(batch, batch_idx)
1エポックの学習ステップ

処理ステップ:
1. バッチデータ取得
2. モデル予測
3. 損失計算
4. ログ出力

##### validation_step(batch, batch_idx)
1エポックの検証ステップ

処理ステップ:
1. バッチデータ取得
2. モデル予測
3. 損失と精度計算
4. ログ出力

##### configure_optimizers()
最適化器の設定（Adam）

## ユーティリティ関数

### calculate_class_weights(df, column)
クラス重みの計算

処理ステップ:
1. クラス頻度カウント
2. 逆頻度重み計算
3. 最大値で正規化

### main関数 (CLIコマンド)

#### オプション
- --csv_path (str, required): メタ情報CSV
- --output_pth (str, default='model.ckpt'): モデル保存パス
- --batch_size (int, default=8): バッチサイズ
- --num_workers (int, default=4): ワーカー数

#### 処理ステップ
1. CSVデータ読み込み
2. フェーズマッピング作成
3. クラス重み計算
4. DataModule構築
5. モデル初期化
6. Trainer設定と学習実行
7. モデル保存

## 実装上のポイント

1. モデルアーキテクチャ
   - 事前学習済みバックボーン
   - マスク情報の効果的な統合
   - Attention機構の活用

2. 学習プロセス
   - クラス不均衡への対応
   - 効率的なデータローディング
   - GPU活用の最適化

3. エラー処理
   - データロード時の例外処理
   - ダミーデータによる学習継続
   - エラーログ出力

4. モジュール性
   - PyTorch Lightningによる整理
   - 再利用可能なコンポーネント
   - 設定のカスタマイズ性

5. 監視と可視化
   - 損失と精度のログ記録
   - 進捗バーでの表示
   - チェックポイント保存
