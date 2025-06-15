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
    画像/マスクデータと特徴量の両方を扱うことができます

    Args:
        chunks (np.ndarray): 動画フレームのチャンク配列
        mask_chunks (np.ndarray): マスク画像のチャンク配列
        labels (np.ndarray): one-hotラベルの配列
        features (np.ndarray, optional): 画像特徴量の配列
        mask_features (np.ndarray, optional): マスク特徴量の配列
        return_features (bool): __getitem__で特徴量を返すかどうか
    """
    def __init__(self, chunks=None, mask_chunks=None, labels=None, features=None, mask_features=None, return_features=True):
        self.chunks = chunks
        self.mask_chunks = mask_chunks
        self.labels = labels
        self.features = features
        self.mask_features = mask_features
        self.return_features = return_features

    def __len__(self):
        if self.return_features and self.features is not None:
            return len(self.features)
        return len(self.chunks) if self.chunks is not None else 0

    def __getitem__(self, idx):
        """
        指定インデックスのサンプルを取得

        Args:
            idx (int): サンプルのインデックス

        Returns:
            特徴量モード (return_features=True) の場合:
                dict:
                    - 'features': 画像特徴量 (torch.Tensor)
                    - 'mask_features': マスク特徴量 (torch.Tensor)
                    - 'label': one-hotラベル (torch.Tensor)
            画像モード (return_features=False) の場合:
                dict:
                    - 'image': 前処理済み画像 (torch.Tensor)
                    - 'mask': マスク画像 (torch.Tensor)
                    - 'label': one-hotラベル (torch.Tensor)
        """
        try:
            if self.return_features and self.features is not None:
                return {
                    'features': torch.tensor(self.features[idx], dtype=torch.float32),
                    'mask_features': torch.tensor(self.mask_features[idx], dtype=torch.float32) if self.mask_features is not None else None,
                    'label': torch.tensor(self.labels[idx], dtype=torch.float32)
                }
            else:
                chunk = self.chunks[idx].astype(np.float32) / 255.0
                chunk = chunk.transpose(0, 3, 1, 2)
                mask_chunk = None
                if self.mask_chunks is not None:
                    mask_chunk = self.mask_chunks[idx].astype(np.float32) / 255.0
                    mask_chunk = mask_chunk.transpose(0, 3, 1, 2)
                return {
                    'image': torch.tensor(chunk),
                    'mask': torch.tensor(mask_chunk) if mask_chunk is not None else None,
                    'label': torch.tensor(self.labels[idx], dtype=torch.float32)
                }
        except Exception as e:
            logging.error(f"サンプル取得エラー (idx={idx}): {e}")
            raise

    @classmethod
    def create_from_video(cls, video_path, phase, phase2idx, window_size=16, stride=4, 
                     skip_generate_masks=True, max_frames=128, extract_features=True, 
                     return_features=True, device='cuda'):
        """
        動画ファイルからデータセットを生成します

        Args:
            video_path (str): 動画ファイルパス
            phase (str): 手術フェーズ名
            phase2idx (dict): フェーズ名からインデックスへのマッピング
            window_size (int): ウィンドウサイズ
            stride (int): ストライド幅
            skip_generate_masks (bool): マスク生成をスキップするかどうか
            max_frames (int, optional): 読み込む最大フレーム数
            extract_features (bool): 特徴量を抽出するかどうか
            device (str): 特徴量抽出に使用するデバイス

        Returns:
            OphNetSurgicalDataset: 生成されたデータセット
        """
        try:
            # 動画の読み込みと前処理
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if not frames:
                logging.error(f"フレームを読み込めませんでした: {video_path}")
                return None

            # max_framesが指定されている場合、ランダムな位置から連続したフレームを取得
            if max_frames is not None and len(frames) > max_frames:
                start_idx = random.randint(0, len(frames) - max_frames)
                frames = frames[start_idx:start_idx + max_frames]

            # ウィンドウ抽出器の準備
            extractor = SlidingWindowExtractor(window_size=window_size, stride=stride)
            chunks = extractor.extract_windows(frames)

            # マスクの生成（オプション）
            mask_chunks = None
            if not skip_generate_masks:
                masks = [generate_mask(frame) for frame in frames]
                mask_chunks = extractor.extract_windows(masks)

            # ラベルの準備
            phase_idx = phase2idx[phase]
            num_chunks = len(chunks)
            labels = np.zeros((num_chunks, len(phase2idx)))
            labels[:, phase_idx] = 1

            # 特徴量の抽出（オプション）
            features = None
            mask_features = None
            if extract_features:
                import timm
                import torch
                from torch.utils.data import DataLoader

                # 一時的なデータセットを作成
                temp_dataset = cls(chunks=chunks, mask_chunks=mask_chunks, labels=labels)
                
                # DataLoaderの設定
                batch_size = 8
                dataloader = DataLoader(
                    temp_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )

                # 特徴抽出モデルの準備
                model = timm.create_model('maxvit_large_tf_224.in21k', pretrained=True, num_classes=0)
                model = model.to(device)
                model.eval()

                # 特徴量抽出
                features_list = []
                mask_features_list = []

                with torch.no_grad():
                    for batch in tqdm(dataloader, desc="Extracting features"):
                        images = batch['image'].to(device)
                        features_list.append(model(images).cpu().numpy())
                        
                        if batch['mask'] is not None:
                            masks = batch['mask'].to(device)
                            mask_features_list.append(model(masks).cpu().numpy())

                features = np.concatenate(features_list)
                if mask_features_list:
                    mask_features = np.concatenate(mask_features_list)

            # データセットの作成
            return cls(
                chunks=chunks,
                mask_chunks=mask_chunks,
                labels=labels,
                features=features,
                mask_features=mask_features,
                return_features=return_features
            )

        except Exception as e:
            logging.error(f"データセット生成エラー {video_path}: {e}")
            return None

    def save_npz(self, file_path, save_images=False):
        """
        データセットをnpz形式で保存します。

        Args:
            file_path (str): 保存先のファイルパス
            save_images (bool): 画像/マスクデータも保存するかどうか
        """
        save_dict = {
            'labels': self.labels
        }
        
        if save_images and self.chunks is not None:
            save_dict['chunks'] = self.chunks
            if self.mask_chunks is not None:
                save_dict['mask_chunks'] = self.mask_chunks

        if self.features is not None:
            save_dict['features'] = self.features
            if self.mask_features is not None:
                save_dict['mask_features'] = self.mask_features

        np.savez_compressed(file_path, **save_dict)
        logging.debug(f"データセットを {file_path} に保存しました")

    @classmethod
    def load_npz(cls, file_path):
        """
        npzファイルからデータセットを読み込みます

        Args:
            file_path (str): 読み込むファイルのパス

        Returns:
            OphNetSurgicalDataset: 読み込まれたデータセット
        """
        try:
            data = np.load(file_path)
            return_features = 'features' in data or 'mask_features' in data
            if return_features:
                logging.debug(f"特徴量を含むデータセットを読み込み: {file_path}")
            else:
                logging.debug(f"画像/マスクデータのみのデータセットを読み込み: {file_path}")
            return cls(
                chunks=data['chunks'] if 'chunks' in data else None,
                mask_chunks=data['mask_chunks'] if 'mask_chunks' in data else None,
                labels=data['labels'],
                features=data['features'] if 'features' in data else None,
                mask_features=data['mask_features'] if 'mask_features' in data else None
            )
        except Exception as e:
            logging.error(f"npzファイル読み込みエラー {file_path}: {e}")
            raise


def process_video(row, phase2idx, intermediate_pth=None, skip_existing=False, 
               extract_features=True, save_images=False, device='cuda'):
    """
    1動画の処理を行う関数

    Args:
        row (pd.Series): 動画メタデータの1行
        phase2idx (dict): フェーズ名からインデックスへのマッピング
        intermediate_pth (str, optional): 中間ファイル保存先ディレクトリ
        skip_existing (bool): 既存ファイルをスキップするかどうか
        extract_features (bool): 特徴量を抽出するかどうか
        save_images (bool): 画像/マスクデータも保存するかどうか
        device (str): 特徴量抽出に使用するデバイス

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
                logging.warning(f"既存ファイル読み込みエラー {npz_path}: {e}")

    try:
        # データセットをビルダーメソッドで作成
        dataset = OphNetSurgicalDataset.create_from_video(
            video_path=row['video_path'],
            phase=row['phase'],
            phase2idx=phase2idx,
            window_size=16,
            stride=4,
            skip_generate_masks=False,
            max_frames=128,
            extract_features=extract_features,
            device=device
        )
        
        # データセット全体をnpz形式で保存
        if dataset is not None and npz_path:
            dataset.save_npz(npz_path, save_images=save_images)
            
        return dataset
    except Exception as e:
        logging.error(f"動画処理エラー {row['video_path']}: {e}")
        return None


def main(video_dir, output_pth, intermediate_pth=None, skip_existing=False, n_jobs=4, 
         n_samples=None, extract_features=True, save_images=False, device='cuda'):
    """
    メイン処理を行う関数

    Args:
        video_dir (str): 動画フォルダパス
        output_pth (str): 出力pthファイルパス
        intermediate_pth (str, optional): 中間ファイル保存先ディレクトリ
        skip_existing (bool): 既存ファイルをスキップするかどうか
        n_jobs (int): 並列ジョブ数
        n_samples (int, optional): 抽出するサンプル数。フェーズごとに均等に配分されます。
        extract_features (bool): 特徴量を抽出するかどうか
        save_images (bool): 画像/マスクデータも保存するかどうか
        device (str): 特徴量抽出に使用するデバイス
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
        delayed(process_video)(
            row, phase2idx, intermediate_pth, skip_existing,
            extract_features=extract_features, save_images=save_images, device=device
        )
        for row in tqdm(rows, desc='Extracting datasets')
    )
    
    # 有効なデータセットをフィルタリング
    datasets = [d for d in datasets if d is not None]

    if not datasets:
        raise RuntimeError("有効なデータセットを生成できませんでした")
    
    # データセットの結合
    total_samples = sum(len(d) for d in datasets)
    logging.debug(f"合計 {total_samples} サンプルを生成しました")

    # データセット全体を保存（pickleプロトコル5を使用）
    logging.debug(f"データセットを {output_pth} に保存します")
    torch.save(datasets, output_pth, pickle_protocol=5)
    print(f"データセットを {output_pth} に保存しました（合計 {total_samples} サンプル）")

@click.command()
@click.option('--video_dir', type=str, required=True, help='動画フォルダパス')
@click.option('--output_pth', type=str, default="dataset.pth", help='保存先pthファイルパス')
@click.option('--intermediate_pth', type=str, default="mediate", help='中間npy保存ディレクトリ')
@click.option('--skip_existing', is_flag=True, help='既存の中間ファイルがあればスキップ')
@click.option('--n_jobs', type=int, default=4, help='並列ジョブ数')
@click.option('--n_samples', type=int, default=None, help='抽出するサンプル数。フェーズごとに均等に配分されます。')
@click.option('--extract_features', is_flag=True, help='特徴量を抽出するかどうか')
@click.option('--no_save_images', is_flag=True, help='画像/マスクデータを保存しないようにする')
@click.option('--device', type=str, default='cuda', help='特徴量抽出に使用するデバイス')
def cli(video_dir, output_pth, intermediate_pth, skip_existing, n_jobs, n_samples,
        extract_features, no_save_images, device):
    """vdata.pyのCLIエントリポイント"""
    main(video_dir, output_pth, intermediate_pth, skip_existing, n_jobs, n_samples,
         extract_features=extract_features, save_images=not no_save_images, device=device)


if __name__ == '__main__':
    cli()
