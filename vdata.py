#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OphNet手術フェーズ分類タスク用のデータセット定義・抽出スクリプト
動画ファイル（<case_id>-<phase>-<start_time>-<end_time>.mp4）から
画像・マスク・ラベルデータを生成します。
"""

import os
import random
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import click
import logging
from tqdm import tqdm
from joblib import Parallel, delayed

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SlidingWindowExtractor:
    """
    動画フレーム列からウィンドウを抽出するクラス。
    指定されたウィンドウサイズとストライド幅で、フレーム列を時系列ウィンドウに分割します。

    Args:
        window_size (int): ウィンドウに含めるフレーム数
        stride (int): ウィンドウ間のフレーム数の間隔
    """
    def __init__(self, window_size=16, stride=8):
        self.window_size = window_size
        self.stride = stride

    def extract_windows(self, video_frames, masks=None):
        """
        動画フレーム列からウィンドウを抽出します。

        Args:
            video_frames (list): 動画フレームのリスト
            masks (list, optional): マスク画像のリスト

        Returns:
            マスクなしの場合: np.ndarray: ウィンドウ配列
            マスクありの場合: tuple: (ウィンドウ配列, マスクウィンドウ配列)
        """
        if len(video_frames) < self.window_size:
            raise ValueError(f"フレーム数がウィンドウサイズより小さいです: {len(video_frames)} < {self.window_size}")

        windows = []
        mask_windows = [] if masks is not None else None
        num_frames = len(video_frames)

        for start in range(0, num_frames - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(np.stack(video_frames[start:end]))
            if masks is not None:
                if len(masks) != len(video_frames):
                    raise ValueError("マスクとフレームの数が一致しません")
                mask_windows.append(np.stack(masks[start:end]))

        if not windows:
            raise ValueError("ウィンドウを抽出できませんでした")

        if masks is not None:
            return np.array(windows), np.array(mask_windows)
        return np.array(windows)


class OphNetSurgicalDataset(Dataset):
    """
    手術フェーズ分類用のデータセットクラス

    Args:
        chunks (np.ndarray): 動画フレームのチャンク配列
        mask_chunks (np.ndarray): マスク画像のチャンク配列
        labels (np.ndarray): one-hotラベルの配列
    """
    def __init__(self, chunks, mask_chunks, labels):
        self.chunks = chunks
        self.mask_chunks = mask_chunks
        self.labels = labels

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        指定インデックスのサンプルを取得

        Args:
            idx (int): サンプルのインデックス

        Returns:
            dict:
                - 'image': 前処理済み画像 (torch.Tensor)
                - 'mask': マスク画像 (torch.Tensor)
                - 'label': one-hotラベル (torch.Tensor)
        """
        try:
            # ランダムに1ウィンドウ選択
            window = self.chunks[idx]
            mask_window = self.mask_chunks[idx]

            # 画像平均化とマスク生成
            img = window.mean(axis=0).astype(np.uint8)
            mask = img.copy()

            # 画像前処理
            img = cv2.resize(img, (224,224))
            mask = cv2.resize(mask, (224,224))
            img = img.transpose(2,0,1) / 255.0
            mask = mask.transpose(2,0,1)

            # one-hotラベル
            label = self.labels[idx]

            return {
                'image': torch.tensor(img, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)
            }

        except Exception as e:
            logging.error(f"サンプル生成エラー index {idx}: {e}")
            # エラー時はダミーデータを返す
            dummy = torch.zeros((3, 224, 224), dtype=torch.float32)
            label = torch.zeros(self.labels.shape[1], dtype=torch.float32)
            return {'image': dummy, 'mask': dummy, 'label': label}

    @classmethod
    def create_from_video(cls, video_path, phase, phase2idx, window_size=16, stride=4, skip_generate_masks=True, max_frames=None):
        """
        動画ファイルからデータセットを作成するビルダーメソッド

        Args:
            video_path (str): 動画ファイルパス
            phase (str): フェーズラベル
            phase2idx (dict): フェーズ名からインデックスへのマッピング
            window_size (int): ウィンドウサイズ
            stride (int): ストライド幅
            skip_generate_masks (bool): マスク生成をスキップするかどうか。
                                      Trueの場合、画像をそのままマスクとして使用
            max_frames (int, optional): 読み込む最大フレーム数

        Returns:
            OphNetSurgicalDataset: 生成されたデータセットインスタンス

        Raises:
            FileNotFoundError: 動画ファイルが存在しない場合
            RuntimeError: 動画の読み込みに失敗した場合
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"動画ファイルが存在しません: {video_path}")

        # 動画フレーム読み込み
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"動画ファイルを開けません: {video_path}")

        # 総フレーム数を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"フレーム数を取得できません: {video_path}")
        
        # 読み込み開始位置の決定
        start_frame = 0
        if max_frames and total_frames > max_frames:
            # ランダムな開始位置を選択（最後のmax_framesを超えないように）
            max_start = total_frames - max_frames
            start_frame = random.randint(0, max_start)
            logging.debug(f"総フレーム数 {total_frames} から {start_frame} フレーム目からランダムに {max_frames} フレーム読み込みます")
            
        # 開始位置までシーク
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    logging.debug(f"指定フレーム数 {max_frames} の読み込みが完了しました")
                    break
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"フレームを読み込めません: {video_path}")

        logging.debug(f"読み込んだフレーム数: {len(frames)}")

        if skip_generate_masks:
            # マスクとして画像をそのまま使用
            mask_frames = [np.stack([f, f], axis=0) for f in frames]
            logging.debug("マスク生成をスキップし、画像をそのまま使用します")
        else:
            # マスク生成（ここでは単純にフレームを2倍にしてマスクとする）
            mask_frames = frames.copy()

        # スライディングウィンドウ抽出
        sw_extractor = SlidingWindowExtractor(window_size, stride)
        windows, mask_windows = sw_extractor.extract_windows(frames, mask_frames)

        logging.debug(f"生成されたウィンドウ数: {len(windows)}")

        # one-hotラベル生成
        label = np.zeros(len(phase2idx), dtype=np.float32)
        label[phase2idx[phase]] = 1.0
        labels = np.tile(label, (len(windows), 1))  # 全ウィンドウに同じラベルを付与

        return cls(windows, mask_windows, labels)

    def save_npz(self, file_path):
        """
        データセットをnpz形式で保存します。

        Args:
            file_path (str): 保存先のファイルパス
        """
        np.savez_compressed(
            file_path,
            chunks=self.chunks,
            mask_chunks=self.mask_chunks,
            labels=self.labels
        )
        logging.debug(f"データセットを {file_path} に保存しました")

    @classmethod
    def load_npz(cls, file_path):
        """
        npzファイルからデータセットを読み込みます。

        Args:
            file_path (str): 読み込むファイルのパス

        Returns:
            OphNetSurgicalDataset: 読み込まれたデータセットインスタンス
        """
        try:
            data = np.load(file_path)
            return cls(
                chunks=data['chunks'],
                mask_chunks=data['mask_chunks'],
                labels=data['labels']
            )
        except Exception as e:
            logging.error(f"npzファイル読み込みエラー {file_path}: {e}")
            raise


def process_video(row, phase2idx, intermediate_pth=None, skip_existing=False):
    """
    1動画の処理を行う関数

    Args:
        row (pd.Series): 動画メタデータの1行
        phase2idx (dict): フェーズ名からインデックスへのマッピング
        intermediate_pth (str, optional): 中間ファイル保存先ディレクトリ
        skip_existing (bool): 既存ファイルをスキップするかどうか

    Returns:
        OphNetSurgicalDataset: 生成されたデータセット、もしくはNone（エラー時）
    """
    npz_path = None
    if intermediate_pth:
        npz_path = os.path.join(intermediate_pth, os.path.basename(row['video_path']) + '.npz')
        if skip_existing and os.path.exists(npz_path):
            try:
                return OphNetSurgicalDataset.load_npz(npz_path)
            except Exception as e:
                logging.error(f"中間ファイル読み込みエラー {npz_path}: {e}")
                if os.path.exists(npz_path):
                    os.remove(npz_path)

    try:
        # データセットをビルダーメソッドで作成
        dataset = OphNetSurgicalDataset.create_from_video(
            video_path=row['video_path'],
            phase=row['phase'],
            phase2idx=phase2idx,
            window_size=16,
            stride=4,
            skip_generate_masks=False,
            max_frames=128  # 最大フレーム数を制限
        )
        
        # データセット全体をnpz形式で保存
        if dataset is not None and npz_path:
            dataset.save_npz(npz_path)
        return dataset
    except Exception as e:
        logging.error(f"動画処理エラー {row['video_path']}: {e}")
        return None


def main(video_dir, output_pth, intermediate_pth=None, skip_existing=False, n_jobs=4, n_samples=None):
    """
    メイン処理を行う関数

    Args:
        video_dir (str): 動画フォルダパス
        output_pth (str): 出力pthファイルパス
        intermediate_pth (str, optional): 中間ファイル保存先ディレクトリ
        skip_existing (bool): 既存ファイルをスキップするかどうか
        n_jobs (int): 並列ジョブ数
        n_samples (int, optional): 抽出するサンプル数。指定した場合、フェーズごとに均等に配分します。
    """
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"動画フォルダが存在しません: {video_dir}")

    # 動画ファイルの探索
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    logging.debug(f"Found {len(video_files)} video files in {video_dir}")

    if not video_files:
        raise RuntimeError(f"動画ファイルが見つかりません: {video_dir}")

    # メタ情報抽出
    pattern = re.compile(r'([^/\\]+?)-([^/\\]+?)-([\d.]+)-([\d.]+)\.mp4$')
    rows = []
    
    for vf in video_files:
        logging.debug(f"Processing {vf}...")
        try:
            m = pattern.search(os.path.basename(vf))
            if m:
                case_id, phase, start_time, end_time = m.groups()
                rows.append({
                    'video_path': vf,
                    'case_id': case_id,
                    'phase': phase,
                    'start_time': float(start_time),
                    'end_time': float(end_time)
                })
            else:
                logging.warning(f"ファイル名パターン不一致: {vf}")
        except Exception as e:
            logging.error(f"ファイル名解析エラー {vf}: {e}")

    if not rows:
        raise RuntimeError("有効な動画ファイルが見つかりません")

    df = pd.DataFrame(rows)
    logging.debug(f"Created DataFrame with {len(df)} rows")
    phases = sorted(df['phase'].unique())
    phase2idx = {p: i for i, p in enumerate(phases)}

    # n_samplesが指定された場合、フェーズごとに均等にサンプリング
    if n_samples is not None:
        # 各フェーズの利用可能なサンプル数を確認
        phase_available = {phase: len(df[df['phase'] == phase]) for phase in phases}
        logging.debug("各フェーズの利用可能なサンプル数:")
        for phase, count in phase_available.items():
            logging.debug(f"フェーズ {phase}: {count} サンプル")

        # 初期配分数の計算
        samples_per_phase = n_samples // len(phases)
        remaining_samples = n_samples % len(phases)
        
        # フェーズごとの目標サンプル数を計算
        target_samples = {phase: samples_per_phase for phase in phases}
        
        # 余りを配分（利用可能なサンプル数を考慮）
        for phase in phases:
            if remaining_samples > 0 and phase_available[phase] > target_samples[phase]:
                target_samples[phase] += 1
                remaining_samples -= 1

        # 不足分を再配分
        for phase in phases:
            shortage = max(0, target_samples[phase] - phase_available[phase])
            if shortage > 0:
                # 不足分を他のフェーズに再配分
                target_samples[phase] = phase_available[phase]
                to_redistribute = shortage
                
                # 再配分可能なフェーズを探す
                for other_phase in phases:
                    if other_phase == phase:
                        continue
                    # 追加で配分可能な数を計算
                    can_add = min(
                        to_redistribute,
                        phase_available[other_phase] - target_samples[other_phase]
                    )
                    if can_add > 0:
                        target_samples[other_phase] += can_add
                        to_redistribute -= can_add
                        if to_redistribute == 0:
                            break
                
                if to_redistribute > 0:
                    logging.warning(f"フェーズ {phase} の不足分 {to_redistribute} サンプルを再配分できませんでした")

        # サンプリングの実行
        sampled_rows = []
        for phase in phases:
            phase_rows = df[df['phase'] == phase]
            n_phase_samples = target_samples[phase]
            
            if len(phase_rows) <= n_phase_samples:
                sampled_rows.extend(phase_rows.to_dict('records'))
            else:
                sampled_rows.extend(phase_rows.sample(n=n_phase_samples).to_dict('records'))
        
        rows = sampled_rows
        logging.debug(f"フェーズごとに均等サンプリング: 合計 {len(rows)} サンプル")
        
        # フェーズごとの分布を表示
        phase_counts = pd.DataFrame(rows)['phase'].value_counts()
        for phase, count in phase_counts.items():
            logging.debug(f"フェーズ {phase}: {count} サンプル")

    if intermediate_pth:
        os.makedirs(intermediate_pth, exist_ok=True)

    # 並列処理でデータセット生成
    datasets = Parallel(n_jobs=n_jobs)(
        delayed(process_video)(row, phase2idx, intermediate_pth, skip_existing)
        for row in tqdm(rows, desc='Extracting datasets')
    )
    
    # # 有効なデータセットをフィルタリング
    # datasets = [d for d in datasets if d is not None]

    # if not datasets:
    #     raise RuntimeError("有効なデータセットを生成できませんでした")

    # # データセットの結合
    # total_samples = sum(len(d) for d in datasets)
    # logging.debug(f"合計 {total_samples} サンプルを生成しました")

    # # データセット全体を保存（pickleプロトコル5を使用）
    # logging.debug(f"データセットを {output_pth} に保存します")
    # torch.save(datasets, output_pth, pickle_protocol=5)
    # print(f"データセットを {output_pth} に保存しました（合計 {total_samples} サンプル）")
    


@click.command()
@click.option('--video_dir', type=str, required=True, help='動画フォルダパス')
@click.option('--output_pth', type=str, default="dataset.pth", help='保存先pthファイルパス')
@click.option('--intermediate_pth', type=str, default="mediate", help='中間npy保存ディレクトリ')
@click.option('--skip_existing', is_flag=True, help='既存の中間ファイルがあればスキップ')
@click.option('--n_jobs', type=int, default=4, help='並列ジョブ数')
@click.option('--n_samples', type=int, default=None, help='抽出するサンプル数。フェーズごとに均等に配分されます。')
def cli(video_dir, output_pth, intermediate_pth, skip_existing, n_jobs, n_samples):
    """動画フォルダからデータセットを生成するCLIツール"""
    main(video_dir, output_pth, intermediate_pth, skip_existing, n_jobs, n_samples)


if __name__ == '__main__':
    cli()
