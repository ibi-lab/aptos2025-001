---
applyTo: "vdata.py"
---

# vdata.py 詳細説明書

このスクリプトは、OphNet手術フェーズ分類タスク用のデータセット定義・抽出スクリプトです。

## 概要

動画フォルダ内の手術動画ファイル（命名規則: `<case_id>-<phase>-<start_timwe>-<end_time>.mp4`）から、以下の処理を行います：

1. ファイル名からメタ情報を抽出してデータフレーム化
2. 各動画を読み込んで、以下の形式のデータを生成
   - images: nframe, height, width, 3の形状を持つRGB画像群
   - masks: nframe, height, width, 3の形状を持つマスク画像群
   - label: one-hotラベル（手術フェーズのインデックス）
4. imagesを生成するときスライディングウィンドウでウィンドウ化して生成
5. 画像・マスク・one-hotラベルを生成して保存

## 主要な依存パッケージ

- os: ファイルパス操作
- random: ランダムサンプリング
- numpy: 数値計算と配列操作
- pandas: データフレーム操作
- torch: PyTorchデータセット定義
- cv2: 画像・動画処理
- click: CLIインターフェース
- tqdm: 進捗表示
- joblib: 並列処理

## クラスとメソッドの詳細

### SlidingWindowExtractor
動画フレーム列からウィンドウを抽出するクラス

#### コンストラクタ引数
- window_size (int, default=16): ウィンドウサイズ
- stride (int, default=4): ストライド幅

#### メソッド

##### extract_windows(video_frames, masks=None, stride=1, window_size=16)
動画フレーム列からウィンドウを抽出するメソッド

引数:
- video_frames (list): 動画フレームのリスト
- masks (list, optional): マスク画像のリスト

処理ステップ:
1. 空のウィンドウリストを初期化
2. フレーム数に基づいてウィンドウを抽出
   - 開始位置をstrideずつ移動
   - window_size分のフレームをスタック
3. マスクが与えられている場合は同様にマスクウィンドウも抽出
4. numpy配列に変換して返却

戻り値:
- マスクなしの場合: windows (np.ndarray)
- マスクありの場合: (windows, mask_windows) (tuple of np.ndarray)

戻り値:
- dict:
  - 'image': 前処理済み画像 (torch.Tensor)
  - 'mask': マスク画像 (torch.Tensor)
  - 'label': one-hotラベル (torch.Tensor)

##### read_video(video_path)
動画ファイルからフレームを読み込む内部メソッド

引数:
- video_path (str): 動画ファイルパス

処理ステップ:
1. OpenCVで動画ファイルを開く
2. framesを読み込み、extract_windowsメソッドでウィンドウ化
   - BGRからRGBに変換   
4. キャプチャを解放

戻り値:
- list: chunkリスト

### OphNetSurgicalDataset (torch.utils.data.Dataset)
手術フェーズ分類用のデータセットクラス

#### クラスメソッド

##### create_from_video(cls, video_path, phase, phase2idx, window_size=16, stride=4, skip_generate_masks=False)
動画ファイルからデータセットを生成するクラスメソッド

引数:
- video_path (str): 動画ファイルパス
- phase (str): 手術フェーズ名
- phase2idx (dict): フェーズ名からインデックスへのマッピング
- window_size (int, default=16): ウィンドウサイズ
- stride (int, default=4): ストライド幅
- skip_generate_masks (bool, default=False): マスク生成をスキップするかどうか
- max_frames (int, optional): 読み込む最大フレーム数。Noneの場合は全フレームを読み込む。
                              指定された場合、動画のランダムな位置から連続したフレームを読み込む

処理ステップ:
1. 動画ファイルを読み込み
   - max_frames指定時は、ランダムな開始位置から指定数まで連続的に読み込み
   - 総フレーム数を確認し、ランダムな開始位置を決定（範囲: 0 〜 総フレーム数-max_frames）
2. フレームをウィンドウ化
3. マスク生成（skip_generate_masks=Falseの場合）
4. データセットインスタンスを生成して返却

戻り値:
- OphNetSurgicalDataset: 生成されたデータセットインスタンス

#### コンストラクタ引数
- chunks (np.ndarray): 動画フレームのチャンク配列
- mask_chunks (np.ndarray, optional): マスク画像のチャンク配列
- labels (np.ndarray): one-hotラベルの配列

#### メソッド

##### __len__()
データセットの長さを返す

##### __getitem__(idx)
指定インデックスのサンプルを取得

##### save_npz(file_path)
データセットをnpz形式で保存します。データの圧縮を行い、ディスク容量を節約します。

引数:
- file_path (str): 保存先のファイルパス

処理ステップ:
1. numpy.savez_compressedを使用してデータを圧縮保存
2. chunks, mask_chunks, labelsを保存
3. 保存完了をログに出力

##### load_npz(file_path)
npzファイルからデータセットを読み込むクラスメソッド

引数:
- file_path (str): 読み込むファイルのパス

処理ステップ:
1. numpy.loadでnpzファイルを読み込み
2. chunks, mask_chunks, labelsを取得
3. 新しいデータセットインスタンスを生成して返却

戻り値:
- OphNetSurgicalDataset: 読み込まれたデータセットインスタンス

例外:
- 読み込みに失敗した場合はエラーをログ出力して再raise

引数:
- idx (int): サンプルのインデックス

処理ステップ:
1. ランダムに1ウィンドウ選択
2. 画像処理
   - 平均化
   - リサイズ (224x224)
   - 正規化 (/255.0)
   - チャンネル順変更
3. one-hotラベル生成
4. 辞書形式で返却

## ユーティリティ関数

### process_video(row, phase2idx, intermediate_pth=None, skip_existing=False)
1動画の処理を行う関数

引数:
- row (pd.Series): 動画メタデータの1行
- phase2idx (dict): フェーズ名からインデックスへのマッピング
- intermediate_pth (str, optional): 中間ファイル保存先ディレクトリ
- skip_existing (bool, default=False): 既存の中間ファイルをスキップするかどうか

処理ステップ:
1. 中間ファイルパスの構築（指定時）
2. 既存ファイルのチェックとロード（skip_existing=True時
3. read_videoで動画フレームを読み込み、chunksを生成
3. chunksをもとにOphNetSurgicalDatasetを生成
4. 中間ファイルとして保存（指定時）

戻り値:
- dict: 生成されたサンプル、もしくはNone（エラー時）

### main(video_dir, output_pth, intermediate_pth=None, skip_existing=False, n_jobs=4, n_samples=None)
メイン処理を行う関数

引数:
- video_dir (str): 動画フォルダパス
- output_pth (str): 出力pthファイルパス
- intermediate_pth (str, optional): 中間ファイル保存先ディレクトリ
- skip_existing (bool, default=False): 既存ファイルをスキップするかどうか
- n_jobs (int, default=4): 並列ジョブ数
- n_samples (int, optional): 抽出するサンプル数。フェーズごとに均等に配分されます

処理ステップ:
1. 動画ファイルの探索
2. ファイル名からメタ情報を抽出
   - case_id, phase, start_time, end_time
3. データフレーム作成
4. phase2idxマッピング構築
5. サンプリング処理（n_samples指定時）
   - 各フェーズの利用可能なサンプル数を確認し、ログ出力
   - フェーズごとの初期目標サンプル数を計算（n_samples / フェーズ数）
   - 余りのサンプル数を、利用可能なフェーズに順次1つずつ配分
   - フェーズごとに実際の利用可能数と目標数を比較
   - 目標数に満たないフェーズがある場合：
     * 不足分を計算
     * 他のフェーズの余剰容量を確認
     * 可能な限り他のフェーズに不足分を再配分
     * 再配分できない場合は警告を出力
   - 各フェーズで目標数に応じてランダムサンプリング実行
   - フェーズごとの最終サンプル数をログ出力
6. 中間保存ディレクトリ作成（指定時）
7. 並列処理でサンプル生成
8. 結果をリスト化してpthファイルに保存

### cli()
CLIエントリポイント

オプション:
- --video_dir (str, required): 動画フォルダパス
- --output_pth (str, default="dataset.pth"): 保存先pthファイルパス
- --intermediate_pth (str, default="mediate"): 中間npy保存ディレクトリ
- --skip_existing (flag): 既存の中間ファイルをスキップ
- --n_jobs (int, default=4): 並列ジョブ数
- --n_samples (int, optional): 抽出するサンプル数。指定した場合、フェーズごとに可能な限り均等に配分されます

## 実装上のポイント

1. データ処理の効率化
   - 並列処理による高速化
   - 中間ファイルの再利用機能

2. メモリ管理
   - 必要なフレームのみ保持
   - マスク生成のスキップオプション

3. エラー処理
   - 各処理段階でのtry-except
   - エラーログの出力
   - エラー時のグレースフル失敗

4. 柔軟性
   - CLIパラメータによる動作カスタマイズ
   - 中間ファイル保存の選択可能性

5. 保守性
   - 豊富なログ出力
   - モジュール化された設計
   - 詳細なドキュメンテーション
