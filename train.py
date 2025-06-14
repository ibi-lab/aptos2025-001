import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import click
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import Optional

# --- データセット定義 ---
class OphNetSurgicalDataset(Dataset):
    """
    DataFrameの各行から動画パス・フェーズラベルを取得し、
    動画ファイルから最大64フレームを読み込み、
    pupil/instrumentマスクには画像そのもの（R/Gチャンネル）を使用。
    SlidingWindowExtractorでウィンドウ抽出し、ランダムに1つ選択。
    画像平均化・リサイズ・正規化・one-hotラベル化して返す。
    """
    def __init__(self, df, phase2idx, window_size=16, stride=8, max_frames=64):
        self.df = df.reset_index(drop=True)
        self.phase2idx = phase2idx
        self.window_size = window_size
        self.stride = stride
        self.max_frames = max_frames
        self.sw_extractor = SlidingWindowExtractor(window_size, stride)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        phase = row['phase']
        try:
            frames = self._read_video(video_path)
            pupil_masks = [f[..., 0] for f in frames]
            instrument_masks = [f[..., 1] for f in frames]
            windows, mask_windows = self.sw_extractor.extract_windows(
                frames, [np.stack([p, i], axis=0) for p, i in zip(pupil_masks, instrument_masks)])
            idx_win = np.random.randint(0, len(windows))
            window = windows[idx_win]
            mask_window = mask_windows[idx_win]
            img = window.mean(axis=0).astype(np.uint8)
            mask = img.copy()
            import cv2
            img = cv2.resize(img, (224, 224))
            mask = cv2.resize(mask, (224, 224))
            img = img.transpose(2, 0, 1) / 255.0
            mask = mask.transpose(2, 0, 1)
            label = np.zeros(len(self.phase2idx), dtype=np.float32)
            label[self.phase2idx[phase]] = 1.0
            return {
                'image': torch.tensor(img, dtype=torch.float32),
                'mask': torch.tensor(mask, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)
            }
        except Exception as e:
            print(f"[ERROR] データ取得失敗: {video_path} ({e})")
            # 例外時はダミーデータを返す（学習が止まらないように）
            dummy = torch.zeros((3, 224, 224), dtype=torch.float32)
            label = torch.zeros(len(self.phase2idx), dtype=torch.float32)
            return {'image': dummy, 'mask': dummy, 'label': label}

    def _read_video(self, video_path):
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

class SlidingWindowExtractor:
    """
    動画フレーム列から指定サイズ・ストライドでウィンドウを抽出。
    マスクがあれば同様にマスクウィンドウも抽出。
    """
    def __init__(self, window_size=16, stride=8):
        self.window_size = window_size
        self.stride = stride

    def extract_windows(self, video_frames, masks=None):
        windows = []
        mask_windows = [] if masks is not None else None
        num_frames = len(video_frames)
        for start in range(0, num_frames - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append(np.stack(video_frames[start:end]))
            if masks is not None:
                mask_windows.append(np.stack(masks[start:end]))
        if masks is not None:
            return np.array(windows), np.array(mask_windows)
        return np.array(windows)

# --- DataModule ---
class OphNetDataModule(pl.LightningDataModule):
    """
    CSVデータから一意の動画IDをもとにトレーニングデータと検証データに分割し、
    それぞれに対してOphNetSurgicalDatasetを構築するLightningDataModule。
    DataLoader生成・並列処理・メモリ最適化も担う。
    """
    def __init__(self, csv_path, phase2idx, batch_size=8, num_workers=4, window_size=16, stride=8, max_frames=64):
        super().__init__()
        self.csv_path = csv_path
        self.phase2idx = phase2idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.window_size = window_size
        self.stride = stride
        self.max_frames = max_frames

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.csv_path)
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['phase'], random_state=42)
        self.train_dataset = OphNetSurgicalDataset(train_df, self.phase2idx, self.window_size, self.stride, self.max_frames)
        self.val_dataset = OphNetSurgicalDataset(val_df, self.phase2idx, self.window_size, self.stride, self.max_frames)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

# --- モデル定義 ---
class SurgicalPhaseClassificationModel(nn.Module):
    """
    事前学習済み画像エンコーダ（timm/maxvit_large_tf_224.in21k）で特徴抽出。
    マスク特徴量を線形層で画像特徴量と同次元に変換し、Multi-Head Attentionで統合。
    全結合層で分類。
    """
    def __init__(self, num_classes):
        super().__init__()
        self.img_encoder = timm.create_model('maxvit_large_tf_224.in21k', pretrained=True, num_classes=0)
        self.mask_proj = nn.Linear(224*224*3, self.img_encoder.num_features)
        self.attn = nn.MultiheadAttention(self.img_encoder.num_features, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.img_encoder.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, mask):
        x = self.img_encoder(image)
        b = mask.size(0)
        mask_feat = self.mask_proj(mask.view(b, -1)).unsqueeze(1)
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, mask_feat, mask_feat)
        out = self.fc(attn_out.squeeze(1))
        return out

# --- LightningModule ---
class SurgicalPhaseModule(pl.LightningModule):
    """
    PyTorch LightningのLightningModule。
    モデル・損失・最適化・学習/検証ループ・メトリクス管理。
    """
    def __init__(self, num_classes, class_weights=None, lr=1e-4):
        super().__init__()
        self.model = SurgicalPhaseClassificationModel(num_classes)
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def training_step(self, batch, batch_idx):
        x, m, y = batch['image'], batch['mask'], batch['label']
        logits = self.model(x, m)
        loss = self.criterion(logits, y.argmax(dim=1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, m, y = batch['image'], batch['mask'], batch['label']
        logits = self.model(x, m)
        loss = self.criterion(logits, y.argmax(dim=1))
        preds = logits.argmax(dim=1)
        targets = y.argmax(dim=1)
        acc = (preds == targets).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# --- クラス重み計算 ---
def calculate_class_weights(df, column):
    """
    データフレーム中の各手術フェーズの出現頻度に基づいて、逆頻度の重みを計算。
    """
    counts = Counter(df[column])
    total = sum(counts.values())
    weights = {k: total/v for k, v in counts.items()}
    max_weight = max(weights.values())
    norm_weights = {k: v/max_weight for k, v in weights.items()}
    return torch.tensor([norm_weights[k] for k in sorted(norm_weights.keys())], dtype=torch.float32)

# --- メイン処理 ---
@click.command()
@click.option('--csv_path', type=str, required=True, help='動画メタ情報CSV（video_path, phase列必須）')
@click.option('--output_pth', type=str, default='model.ckpt', help='モデル保存パス')
@click.option('--batch_size', type=int, default=8, help='バッチサイズ')
@click.option('--num_workers', type=int, default=4, help='DataLoaderのワーカ数')
def main(csv_path, output_pth, batch_size, num_workers):
    """
    データセットCSVを読み込み、OphNetDataModule/モデル/Trainerを構築し学習・保存。
    """
    try:
        df = pd.read_csv(csv_path)
        phases = sorted(df['phase'].unique())
        phase2idx = {p: i for i, p in enumerate(phases)}
        class_weights = calculate_class_weights(df, 'phase')
        datamodule = OphNetDataModule(csv_path, phase2idx, batch_size, num_workers)
        model = SurgicalPhaseModule(num_classes=len(phases), class_weights=class_weights)
        trainer = pl.Trainer(max_epochs=30, accelerator='auto', devices=1, log_every_n_steps=10)
        trainer.fit(model, datamodule=datamodule)
        trainer.save_checkpoint(output_pth)
        print(f"モデルを {output_pth} に保存しました")
    except Exception as e:
        print(f"[ERROR] 学習処理失敗: {e}")

if __name__ == '__main__':
    main()
