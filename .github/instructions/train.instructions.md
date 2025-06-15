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
- wandb: 実験管理とログ記録

## クラスとメソッドの詳細

### NPZFeatureDataset (torch.utils.data.Dataset)
vdata.pyで生成されたnpzファイルから特徴量とラベルを読み込むデータセット

#### コンストラクタ引数
- npz_files (list): npzファイルのパスリスト

#### メソッド

##### __len__()
データセットの長さを返す

##### __getitem__(idx)
指定インデックスのサンプルを取得（特徴量とラベル）

### NPZDataModule (pl.LightningDataModule)
npzファイルを管理し、トレーニング/検証用のDataLoaderを提供するDataModule

#### コンストラクタ引数
- npz_dir (str): npzファイルが保存されているディレクトリ
- batch_size (int, default=32): バッチサイズ
- num_workers (int, default=4): DataLoaderのワーカー数
- val_split (float, default=0.2): 検証データの割合

#### メソッド

##### setup(stage: Optional[str] = None)
データセットの準備
1. npzファイルの一覧取得
2. train/val分割
3. データセットインスタンス生成

##### train_dataloader()
トレーニング用DataLoader生成

##### val_dataloader()
検証用DataLoader生成

### TransformerFeatureClassifier (nn.Module)
特徴量をTransformerで統合し、分類を行うモデル

#### コンストラクタ引数
- feature_dim (int): 入力特徴量の次元数
- num_classes (int): 分類クラス数
- nhead (int, default=8): Multi-head Attentionのヘッド数
- num_layers (int, default=3): Transformerレイヤー数
- dim_feedforward (int, default=2048): フィードフォワード層の次元数
- dropout (float, default=0.1): ドロップアウト率

#### メソッド

##### forward(features, mask_features)
順伝播処理
1. 特徴量の結合
2. Transformer Encoderでの特徴量統合
3. 分類層での予測

### OphNetDataModule (pl.LightningDataModule)
データセットの管理とデータローダーの構築を行うクラス

#### コンストラクタ引数
- npz_dir (str): npzファイルディレクトリ
- batch_size (int, default=32): バッチサイズ
- num_workers (int, default=4): 並列ワーカー数
- val_split (float, default=0.2): 検証データの割合

#### メソッド

##### setup(stage: Optional[str] = None)
データセットの準備
1. npzファイルの一覧取得
2. トレーニング/検証データ分割
3. データセットインスタンス作成
4. 特徴量モードの有効化

##### train_dataloader()
トレーニング用DataLoader生成

##### val_dataloader()
検証用DataLoader生成

### SurgicalPhaseModule (pl.LightningModule)
学習プロセスを管理するLightningModule

#### コンストラクタ引数
- feature_dim (int): 入力特徴量の次元数
- num_classes (int): 分類クラス数
- learning_rate (float, default=1e-4): 学習率

#### メソッド

##### forward(features, mask_features)
モデルの順伝播処理

##### training_step(batch, batch_idx)
1エポックの学習ステップ
1. バッチデータ取得
2. モデル予測
3. 損失計算
4. ログ出力

##### validation_step(batch, batch_idx)
1エポックの検証ステップ
1. バッチデータ取得
2. モデル予測
3. 損失と精度計算
4. ログ出力

##### configure_optimizers()
最適化器の設定（Adam）

## ユーティリティ関数

### calculate_class_weights(df, column)
クラス重みの計算
1. クラス頻度カウント
2. 逆頻度重み計算
3. 最大値で正規化

### main関数 (CLIコマンド)

#### オプション
- --npz_dir (str, required): npzファイルディレクトリ
- --output_dir (str, default='models'): モデル保存ディレクトリ
- --batch_size (int, default=32): バッチサイズ
- --num_workers (int, default=4): ワーカー数
- --max_epochs (int, default=100): 最大エポック数
- --feature_dim (int, default=1024): 特徴量の次元数
- --learning_rate (float, default=1e-4): 学習率

#### 処理ステップ
1. DataModuleの構築
2. モデルの初期化
3. WandbLoggerの設定
4. Trainerの設定と学習実行
5. モデル保存

## 実装上のポイント

1. モデルアーキテクチャ
   - Transformerベースのアーキテクチャ
   - 特徴量の効果的な統合
   - 柔軟な設定オプション

2. 学習プロセス
   - 効率的なデータローディング
   - Wandbによる実験管理
   - GPU活用の最適化

3. エラー処理
   - ファイル存在チェック
   - データロードエラーの処理
   - 適切なエラーメッセージ

4. モジュール性
   - PyTorch Lightningによる整理
   - 再利用可能なコンポーネント
   - 設定のカスタマイズ性

5. 監視と可視化
   - Wandbによる実験ログ
   - 学習進捗の可視化
   - チェックポイント管理
