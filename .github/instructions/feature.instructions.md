---
applyTo: "features.py"
---

# features.py 詳細説明書

このスクリプトは、vdata.pyで生成された中間ファイル（npz）から特徴量を抽出し、
新しいデータセット形式（特徴量ベース）に変換・保存するためのものです。

## 概要

- 入力: OphNetSurgicalDataset形式のnpzファイル（画像・マスク・ラベル）
- 出力: 画像特徴量・マスク特徴量・ラベルを含むnpzファイル
- 画像・マスクから事前学習済み画像エンコーダ（timm/maxvit_large_tf_224.in21k）で特徴量抽出
- 単一ファイル/ディレクトリ一括処理の両方に対応
- CLIサブコマンドで柔軟に利用可能

## 主要な依存パッケージ

- os: ファイルパス操作
- torch: PyTorchモデル・テンソル操作
- timm: 画像エンコーダ
- numpy: 数値計算と配列操作
- click: CLIインターフェース
- tqdm: 進捗表示
- logging: ログ出力

## クラスとメソッドの詳細

### OphNetFeatureDataset (torch.utils.data.Dataset)
特徴量ベースの手術フェーズ分類用データセット

#### コンストラクタ引数
- features (np.ndarray): 画像特徴量配列 (N, feature_dim)
- mask_features (np.ndarray): マスク特徴量配列 (N, feature_dim)
- labels (np.ndarray): one-hotラベル配列 (N, num_classes)

#### メソッド
- __len__(): サンプル数を返す
- __getitem__(idx): 指定インデックスの特徴量・マスク特徴量・ラベルを辞書で返す
- save_npz(file_path): データセットをnpz形式で保存
- load_npz(file_path): npzファイルからデータセットを読み込み

### extract_features(dataset, batch_size=8, num_workers=4, device='cuda')
OphNetSurgicalDatasetから特徴量を抽出し、OphNetFeatureDatasetを生成

引数:
- dataset: 入力データセット（OphNetSurgicalDataset）
- batch_size (int): バッチサイズ
- num_workers (int): DataLoaderのワーカー数
- device (str): 使用デバイス

処理ステップ:
1. 画像エンコーダ（timm/maxvit_large_tf_224.in21k）を準備
2. DataLoaderでバッチ処理
3. 各バッチで画像・マスクから特徴量抽出
4. 特徴量・ラベルをリストに蓄積
5. 全特徴量・ラベルを結合しOphNetFeatureDatasetを返却

戻り値:
- OphNetFeatureDataset: 特徴量データセット

## CLIサブコマンド

### extract_feature
- 単一のnpzファイルから特徴量を抽出し、出力npzファイルに保存
- 引数:
  - input_npz: 入力npzファイルパス
  - output_npz: 出力npzファイルパス
  - --batch-size, --num-workers, --device: 各種オプション

### extract_features
- 指定ディレクトリ内のすべてのnpzファイルに対して特徴量抽出を実行し、出力フォルダにnpzを保存
- 引数:
  - input_dir: 入力ディレクトリ（npzファイル群）
  - output_dir: 出力ディレクトリ
  - --batch-size, --num-workers, --device: 各種オプション

## 実装上のポイント

1. 並列処理・バッチ処理による高速化
2. 画像・マスク両方から特徴量抽出
3. 例外処理・エラーログ出力
4. CLIサブコマンドによる柔軟な運用
5. PEP8/コメント充実・保守性重視
