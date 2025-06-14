#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vdata.pyで生成された中間ファイル（npz）から特徴量を抽出し、
新しいデータセット形式（特徴量ベース）に変換するスクリプト。
"""

import os
import glob
import torch
import timm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import click
import logging
from tqdm import tqdm
from vdata import OphNetSurgicalDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OphNetFeatureDataset(Dataset):
    """
    特徴量ベースの手術フェーズ分類用データセット
    
    Args:
        features (np.ndarray): 画像特徴量の配列 (N, feature_dim)
        mask_features (np.ndarray): マスク特徴量の配列 (N, feature_dim)
        labels (np.ndarray): one-hotラベルの配列 (N, num_classes)
    """
    def __init__(self, features, mask_features, labels):
        self.features = features
        self.mask_features = mask_features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'mask_features': torch.tensor(self.mask_features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
    
    def save_npz(self, file_path):
        """データセットをnpz形式で保存"""
        np.savez_compressed(
            file_path,
            features=self.features,
            mask_features=self.mask_features,
            labels=self.labels
        )
        logging.info(f"データセットを {file_path} に保存しました")
    
    @classmethod
    def load_npz(cls, file_path):
        """npzファイルからデータセットを読み込み"""
        try:
            data = np.load(file_path)
            return cls(
                features=data['features'],
                mask_features=data['mask_features'],
                labels=data['labels']
            )
        except Exception as e:
            logging.error(f"npzファイル読み込みエラー {file_path}: {e}")
            raise

def extract_features(dataset, batch_size=8, num_workers=4, device='cuda'):
    """
    データセットから特徴量を抽出
    
    Args:
        dataset: 入力データセット（OphNetSurgicalDataset）
        batch_size (int): バッチサイズ
        num_workers (int): DataLoaderのワーカー数
        device (str): 使用デバイス
    
    Returns:
        OphNetFeatureDataset: 特徴量データセット
    """
    # 特徴抽出モデルの準備
    model = timm.create_model('maxvit_large_tf_224.in21k', pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    features_list = []
    mask_features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label']
            
            # 特徴量抽出
            features = model(images)
            mask_features = model(masks)
            
            # CPU上のNumPy配列に変換
            features_list.append(features.cpu().numpy())
            mask_features_list.append(mask_features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    # 特徴量を結合
    features = np.concatenate(features_list)
    mask_features = np.concatenate(mask_features_list)
    labels = np.concatenate(labels_list)
    
    return OphNetFeatureDataset(features, mask_features, labels)

@click.group()
def cli():
    """
    features.pyのCLIエントリポイント。サブコマンドを登録。
    """
    pass

@cli.command('extract_feature')
@click.argument('input_npz', type=click.Path(exists=True))
@click.argument('output_npz', type=str)
@click.option('--batch-size', default=8, help='バッチサイズ')
@click.option('--num-workers', default=4, help='DataLoaderのワーカー数')
@click.option('--device', default='cuda', help='使用デバイス (cuda or cpu)')
def extract_feature(input_npz, output_npz, batch_size, num_workers, device):
    """
    中間ファイル（npz）から特徴量を抽出し、新しいデータセット形式で保存
    """
    try:
        logging.info(f"入力データセットを読み込み: {input_npz}")
        dataset = OphNetSurgicalDataset.load_npz(input_npz)
        logging.info("特徴量抽出を開始")
        feature_dataset = extract_features(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device
        )
        logging.info(f"特徴量データセットを保存: {output_npz}")
        feature_dataset.save_npz(output_npz)
    except Exception as e:
        logging.error(f"処理失敗: {e}")
        raise

@cli.command('extract_features')
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--batch-size', default=8, help='バッチサイズ')
@click.option('--num-workers', default=4, help='DataLoaderのワーカー数')
@click.option('--device', default='cuda', help='使用デバイス (cuda or cpu)')
def extract_features_dir(input_dir, output_dir, batch_size, num_workers, device):
    """
    指定ディレクトリ内のすべてのnpzファイルに対して特徴量抽出を実行し、
    出力フォルダにnpzを出力する。
    """
    os.makedirs(output_dir, exist_ok=True)
    npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])
    for fname in npz_files:
        input_npz = os.path.join(input_dir, fname)
        output_npz = os.path.join(output_dir, fname)
        try:
            logging.info(f"処理開始: {input_npz}")
            dataset = OphNetSurgicalDataset.load_npz(input_npz)
            feature_dataset = extract_features(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device
            )
            feature_dataset.save_npz(output_npz)
            logging.info(f"完了: {output_npz}")
        except Exception as e:
            logging.error(f"{input_npz} の処理失敗: {e}")

if __name__ == '__main__':
    cli()
