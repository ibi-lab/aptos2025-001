---
applyTo: "vdata.py"
---

# vdata.py 詳細説明書

このスクリプトは、OphNet手術フェーズ分類タスク用のデータセット定義・抽出スクリプトです。

## 概要

動画フォルダ内の手術動画ファイル（命名規則: `<case_id>-<phase>-<start_time>-<end_time>.mp4`）から、以下の処理を行います：

1. ファイル名からメタ情報を抽出してデータフレーム化
2. 各動画を読み込んで、以下の形式のデータを生成
   - images: nframe, height, width, 3の形状を持つRGB画像群
   - masks: nframe, height, width, 3の形状を持つマスク画像群（pupil部と手術器具の銀色部分を抽出）
   - features: 画像特徴量（maxvit_large_tf_224.in21k特徴量）
   - mask_features: マスク特徴量（同上）
   - label: one-hotラベル（手術フェーズのインデックス）
3. imagesを生成するときスライディングウィンドウでウィンドウ化して生成
4. 画像・マスク・特徴量・ラベルを生成して保存

## 主要な依存パッケージ

- os: ファイルパス操作
- random: ランダムサンプリング
- numpy: 数値計算と配列操作
- pandas: データフレーム操作
- torch: PyTorchデータセット定義
- cv2: 画像・動画処理
- timm: 特徴量抽出用モデル
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

##### extract_windows(video_frames, masks=None)
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

### OphNetSurgicalDataset (torch.utils.data.Dataset)
手術フェーズ分類用のデータセットクラス

#### コンストラクタ引数
- chunks (np.ndarray, optional): 動画フレームのチャンク配列
- mask_chunks (np.ndarray, optional): マスク画像のチャンク配列
- labels (np.ndarray): one-hotラベルの配列
- features (np.ndarray, optional): 画像特徴量の配列
- mask_features (np.ndarray, optional): マスク特徴量の配列
- return_features (bool): __getitem__で特徴量を返すかどうか

#### メソッド

##### __len__()
データセットの長さを返す。return_featuresとfeaturesの有無に応じて適切な長さを返す。

##### __getitem__(idx)
指定インデックスのサンプルを取得

戻り値:
特徴量モード (return_features=True) の場合:
- dict:
  - 'features': 画像特徴量 (torch.Tensor)
  - 'mask_features': マスク特徴量 (torch.Tensor)
  - 'label': one-hotラベル (torch.Tensor)

画像モード (return_features=False) の場合:
- dict:
  - 'image': 前処理済み画像 (torch.Tensor)
  - 'mask': マスク画像 (torch.Tensor)
  - 'label': one-hotラベル (torch.Tensor)

##### save_npz(file_path, save_images=True)
データセットをnpz形式で保存します。画像データを保存するかどうかを選択できます。

引数:
- file_path (str): 保存先のファイルパス
- save_images (bool): 画像/マスクデータも保存するかどうか

処理ステップ:
1. numpy.savez_compressedを使用してデータを圧縮保存
2. save_imagesがTrueの場合、chunks, mask_chunksを保存
3. 特徴量が存在する場合、features, mask_featuresを保存
4. labelsを保存
5. 保存完了をログに出力

##### load_npz(file_path)
npzファイルからデータセットを読み込むクラスメソッド

引数:
- file_path (str): 読み込むファイルのパス

処理ステップ:
1. numpy.loadでnpzファイルを読み込み
2. 利用可能なデータ（画像/マスク/特徴量）を取得
3. 新しいデータセットインスタンスを生成して返却

戻り値:
- OphNetSurgicalDataset: 読み込まれたデータセットインスタンス

##### create_from_video(cls, video_path, phase, phase2idx, window_size=16, stride=4, skip_generate_masks=True, max_frames=None, extract_features=False, device='cuda')
動画ファイルからデータセットを生成するクラスメソッド

引数:
- video_path (str): 動画ファイルパス
- phase (str): 手術フェーズ名
- phase2idx (dict): フェーズ名からインデックスへのマッピング
- window_size (int): ウィンドウサイズ
- stride (int): ストライド幅
- skip_generate_masks (bool): マスク生成をスキップするかどうか
- max_frames (int, optional): 読み込む最大フレーム数
- extract_features (bool): 特徴量を抽出するかどうか
- device (str): 特徴量抽出に使用するデバイス

処理ステップ:
1. 動画ファイルを読み込み
   - max_frames指定時は、ランダムな開始位置から指定数まで連続的に読み込み
2. フレームをウィンドウ化
3. マスク生成（skip_generate_masks=Falseの場合）
4. 特徴量抽出（extract_features=Trueの場合）
   - maxvit_large_tf_224.in21kモデルを使用
   - 画像とマスク両方から特徴量を抽出
5. データセットインスタンスを生成して返却

## ユーティリティ関数

### process_video(row, phase2idx, intermediate_pth=None, skip_existing=False, extract_features=False, save_images=True, device='cuda')
1動画の処理を行う関数

引数:
- row (pd.Series): 動画メタデータの1行
- phase2idx (dict): フェーズ名からインデックスへのマッピング
- intermediate_pth (str, optional): 中間ファイル保存先ディレクトリ
- skip_existing (bool): 既存の中間ファイルをスキップするかどうか
- extract_features (bool): 特徴量を抽出するかどうか
- save_images (bool): 画像/マスクデータも保存するかどうか
- device (str): 特徴量抽出に使用するデバイス

処理ステップ:
1. 中間ファイルパスの構築（指定時）
2. 既存ファイルのチェックとロード（skip_existing=True時）
3. 動画ファイルからデータセットを生成
4. 中間ファイルとして保存（指定時）

戻り値:
- OphNetSurgicalDataset: 生成されたデータセット、もしくはNone（エラー時）

### main(video_dir, output_pth, intermediate_pth=None, skip_existing=False, n_jobs=4, n_samples=None, extract_features=False, save_images=True, device='cuda')
メイン処理を行う関数

引数:
- video_dir (str): 動画フォルダパス
- output_pth (str): 保存先pthファイルパス
- intermediate_pth (str, optional): 中間ファイル保存先ディレクトリ
- skip_existing (bool): 既存ファイルをスキップするかどうか
- n_jobs (int): 並列ジョブ数
- n_samples (int, optional): 抽出するサンプル数。フェーズごとに均等に配分されます。
- extract_features (bool): 特徴量を抽出するかどうか
- save_images (bool): 画像/マスクデータも保存するかどうか
- device (str): 特徴量抽出に使用するデバイス

処理ステップ:
1. 動画ファイルの探索
2. ファイル名からメタ情報を抽出
   - case_id, phase, start_time, end_time
3. データフレーム作成
4. phase2idxマッピング構築
5. サンプリング処理（n_samples指定時）
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
- --n_samples (int, optional): 抽出するサンプル数。フェーズごとに均等に配分されます。
- --extract_features (flag): 特徴量を抽出するかどうか
- --no_save_images (flag): 画像/マスクデータを保存しないようにする
- --device (str, default='cuda'): 特徴量抽出に使用するデバイス

## 実装上のポイント

1. データ処理の効率化
   - 並列処理による高速化
   - 中間ファイルの再利用機能
   - 特徴量抽出のバッチ処理

2. メモリ管理
   - 必要なフレームのみ保持
   - マスク生成のスキップオプション
   - 画像データの選択的保存

3. エラー処理
   - 各処理段階でのtry-except
   - エラーログの出力
   - エラー時のグレースフル失敗

4. 柔軟性
   - CLIパラメータによる動作カスタマイズ
   - 中間ファイル保存の選択可能性
   - 特徴量/画像の出力切り替え

5. 保守性
   - 豊富なログ出力
   - モジュール化された設計
   - 詳細なドキュメンテーション

## ユーティリティ関数

### extract_pupil_mask(frame)
瞳孔（暗い円形領域）を抽出してマスクを生成する関数

引数:
- frame (np.ndarray): 入力フレーム（BGR形式）

処理ステップ:
1. グレースケール変換
2. ガウシアンブラーでノイズ除去 (kernel_size=(9,9), sigma=2)
3. 適応的閾値処理で暗い領域を抽出
4. モルフォロジー演算（Opening, Closing）でノイズ除去と領域の整形

戻り値:
- np.ndarray: 瞳孔のマスク（バイナリ）

### extract_instruments_mask(frame)
手術器具の銀色部分を抽出してマスクを生成する関数

引数:
- frame (np.ndarray): 入力フレーム（BGR形式）

処理ステップ:
1. HSV色空間に変換
2. 低彩度・高明度領域を抽出（銀色の特徴）
   - lower = [0, 0, 100]
   - upper = [180, 30, 255]
3. モルフォロジー演算でノイズ除去と領域の整形

戻り値:
- np.ndarray: 器具のマスク（バイナリ）

### generate_combined_mask(frame)
瞳孔と手術器具のマスクを組み合わせて最終的なマスクを生成する関数

引数:
- frame (np.ndarray): 入力フレーム（BGR形式）

処理ステップ:
1. 瞳孔マスクを生成（extract_pupil_mask）
2. 器具マスクを生成（extract_instruments_mask）
3. OR演算でマスクを統合
4. マスクを3チャンネルに拡張（BGR形式）

戻り値:
- np.ndarray: 結合されたマスク（BGR形式）
