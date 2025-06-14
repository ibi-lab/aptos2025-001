---
applyTo: "view.py"
---

# view.py 詳細説明書

このスクリプトは、OphNet手術フェーズ分類タスクの処理の途中経過・結果を可視化するためのものです。

## 主要な依存パッケージ

- os: ファイルパス操作
- torch: PyTorchモデルとデータの操作
- numpy: 数値計算と配列操作
- matplotlib.pyplot: データ可視化
- pandas: データフレーム操作
- click: CLIインターフェース構築

## 関数の詳細説明

### show_sample(sample, filepath=None)
サンプル（画像群・マスク群・ラベル）を可視化する基本関数。
画像、マスク、ラベルをそれぞれ表示し、画像の基本的な統計情報も表示します。

引数:
- sample (dict): 画像・マスク・ラベルを含む辞書
  - 'images': 画像データ (torch.Tensor)
  - 'masks': マスクデータ (torch.Tensor)
  - 'label': one-hotラベルデータ (torch.Tensor)
- filepath (str, optional): 可視化結果の保存先パス

表示内容:
- 画像: カラーバー付きで表示、平均値と標準偏差を表示
- マスク: カラーバー付きで表示
- ラベル: フェーズインデックスごとの確率を棒グラフで表示


### show_npy関数 (CLIコマンド)
中間ファイル（.npy）の可視化

引数:
- --npy_path (str, required): npyファイルのパス
- -o/--output_img_path (str, default='sample.png'): 出力画像パス

処理ステップ:
1. npyファイルの読み込み
   - allow_pickle=Trueオプションで読み込み
2. データ情報の表示
   - 全体のshapeとtype
   - 各キー（images, masks, label）のshapeとtype
3. show_sample関数で可視化
4. 指定パスに画像を保存

### show_dataset関数 (CLIコマンド)
dataset.pthファイルから特定サンプルを可視化

引数:
- --pth_path (str, required): dataset.pthファイルのパス
- --idx (int, default=0): 表示するサンプルのインデックス

処理ステップ:
1. dataset.pthファイルの読み込み
2. インデックスの範囲チェック
3. 指定サンプルをshow_sample関数で可視化

### show_stats関数 (CLIコマンド)
データセットの統計情報を可視化

引数:
- --pth_path (str, required): dataset.pthファイルのパス

処理ステップ:
1. dataset.pthファイルの読み込み
2. ラベル分布の可視化（左プロット）
   - 各サンプルからラベルのargmaxを取得
   - ヒストグラムを作成
   - 軸ラベルとタイトルを設定
3. 画像統計情報の表示（右プロット）
   - 全サンプルの画像シェイプを表示
   - 画像の平均ピクセル値の分布をヒストグラムで表示
   - 基本的な統計量を表示

### show_prediction関数 (CLIコマンド)
学習済みモデルによる予測結果の可視化

引数:
- --model_ckpt (str, required): モデルのチェックポイントファイルパス
- --pth_path (str, required): dataset.pthファイルのパス
- --idx (int, default=0): 表示するサンプルのインデックス

処理ステップ:
1. 必要なモジュールのインポート
   - torch.nn.functional
   - train.SurgicalPhaseModule
2. データとモデルの準備
   - dataset.pthから指定サンプルを読み込み
   - モデルをチェックポイントから復元
3. 予測の実行
   - モデルを評価モードに設定
   - バッチ次元を追加
   - logitsを計算
   - softmaxで確率に変換
4. 結果の可視化
   - 予測確率と正解ラベルを棒グラフで比較
   - 凡例とタイトルを設定
   - グラフを表示

## 実装上のポイント

1. モジュール構成
   - CLIインターフェースはclick.groupで構築
   - 各機能は独立したコマンドとして実装

2. 可視化の一貫性
   - すべての可視化にmatplotlibを使用
   - 画像・マスク・ラベルの表示形式を統一

3. エラー処理
   - インデックス範囲のチェック
   - ファイル存在確認
   - 適切なエラーメッセージ

4. 保守性
   - PEP8準拠のコードスタイル
   - 豊富なドキュメント文字列
   - モジュール化された関数設計

5. 使いやすさ
   - CLIパラメータに説明的なヘルプメッセージ
   - デフォルト値の適切な設定
   - 直感的なコマンド名
