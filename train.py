import os
import glob
import torch
import pytorch_lightning as pl
import logging
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import Optional
from torch import nn
import timm
import click
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from vdata import OphNetSurgicalDataset, SlidingWindowExtractor
import torchmetrics

# ログレベルを設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DataModule ---
class OphNetDataModule(pl.LightningDataModule):
    """
    CSVデータから一意の動画IDをもとにトレーニングデータと検証データに分割し、
    それぞれに対してOphNetSurgicalDatasetを構築するLightningDataModule。
    DataLoader生成・並列処理・メモリ最適化も担う。
    """
    def __init__(self, npz_dir, batch_size=32, num_workers=4, val_split=0.2):
        super().__init__()
        self.npz_dir = npz_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # .npzファイルの一覧を取得
        npz_files = glob.glob(os.path.join(self.npz_dir, "*.npz"))
        if not npz_files:
            raise ValueError(f"No .npz files found in {self.npz_dir}")

        # train/val分割
        train_files, val_files = train_test_split(
            npz_files, test_size=self.val_split, random_state=42
        )

        logging.debug(f"Found {len(train_files)} training files and {len(val_files)} validation files")

        # トレーニングデータセットの作成
        self.train_dataset = OphNetSurgicalDataset.load_npz(train_files[0])
        for npz_file in tqdm(train_files[1:], desc="Loading training datasets"):
            dataset = OphNetSurgicalDataset.load_npz(npz_file)
            # データセットの結合
            for key in ['features', 'mask_features', 'label']:
                if hasattr(self.train_dataset, key) and getattr(self.train_dataset, key) is not None:
                    if hasattr(dataset, key) and getattr(dataset, key) is not None:
                        setattr(self.train_dataset, key, 
                               np.concatenate([getattr(self.train_dataset, key), getattr(dataset, key)]))

        # 検証データセットの作成
        self.val_dataset = OphNetSurgicalDataset.load_npz(val_files[0])
        for npz_file in tqdm(val_files[1:], desc="Loading validation datasets"):
            dataset = OphNetSurgicalDataset.load_npz(npz_file)
            # データセットの結合
            for key in ['features', 'mask_features', 'label']:
                if hasattr(self.val_dataset, key) and getattr(self.val_dataset, key) is not None:
                    if hasattr(dataset, key) and getattr(dataset, key) is not None:
                        setattr(self.val_dataset, key,
                               np.concatenate([getattr(self.val_dataset, key), getattr(dataset, key)]))

        # 特徴量モードを有効化
        self.train_dataset.return_features = True
        self.val_dataset.return_features = True

        logging.debug(f"Training dataset size: {len(self.train_dataset)}")
        logging.debug(f"Validation dataset size: {len(self.val_dataset)}")

        # データセットが空でないことを確認
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if len(self.val_dataset) == 0:
            raise ValueError("Validation dataset is empty")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

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

# --- Transformerベースのモデル定義 ---
class TransformerFeatureClassifier(nn.Module):
    """
    特徴量をTransformerで統合し、分類を行うモデル。
    過学習対策として以下の機能を追加:
    - 追加のドロップアウト層
    - Layer Normalization
    - Residual connections
    """
    def __init__(self, feature_dim, num_classes, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.2):
        super().__init__()
        
        # 特徴量の次元を調整する線形層とドロップアウト
        # 特徴量の段階的な投影と非線形変換
        self.feature_proj = nn.Sequential(
            # 第1層: 入力次元 → 中間次元
            nn.Linear(feature_dim, dim_feedforward * 2),
            nn.LayerNorm(dim_feedforward * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # 第2層: 中間次元 → 中間次元
            nn.Linear(dim_feedforward * 2, dim_feedforward * 2),
            nn.LayerNorm(dim_feedforward * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # 第3層: 中間次元 → 出力次元
            nn.Linear(dim_feedforward * 2, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.Dropout(dropout)
        )
        
        # マスク特徴量も同様の段階的な投影を適用
        self.mask_feature_proj = nn.Sequential(
            # 第1層: 入力次元 → 中間次元
            nn.Linear(feature_dim, dim_feedforward * 2),
            nn.LayerNorm(dim_feedforward * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # 第2層: 中間次元 → 中間次元
            nn.Linear(dim_feedforward * 2, dim_feedforward * 2),
            nn.LayerNorm(dim_feedforward * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # 第3層: 中間次元 → 出力次元
            nn.Linear(dim_feedforward * 2, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.Dropout(dropout)
        )
        
        # Transformerエンコーダ（ドロップアウトを強化）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward * 2,  # 中間層を広げて表現力を向上
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # ReLUよりも安定した学習が期待できる
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(dim_feedforward)
        )
        
        # 分類ヘッド（多層化して表現力を向上）
        self.classifier = nn.Sequential(
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )
    
    def forward(self, features, mask_features):
        # 特徴量の次元を調整
        x1 = self.feature_proj(features)
        x2 = self.mask_feature_proj(mask_features)
        
        # 特徴量を結合 [batch_size, 2, dim_feedforward]
        x = torch.stack([x1, x2], dim=1)
        
        # Transformerで特徴統合（Residual connection）
        trans_out = self.transformer(x)
        x = x + trans_out  # Residual connection
        
        # Global Average Poolingで系列を統合
        x = x.mean(dim=1)
        
        return self.classifier(x)

# --- LightningModule ---
class SurgicalPhaseModule(pl.LightningModule):
    """
    学習プロセスを管理するLightningModule
    """
    def __init__(self, feature_dim, num_classes, learning_rate=1e-4, weight_decay=1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerFeatureClassifier(feature_dim, num_classes)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss関数にLabel Smoothingを追加
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # メトリクス初期化
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_predictions = []
        self.test_targets = []
        
        # 検証ロスの履歴を保持（Early Stopping用）
        self.best_val_loss = float('inf')
        self.val_loss_cnt = 0
    
    def forward(self, features, mask_features):
        # logging.debug(f"features: {features.shape}, mask_features: {mask_features.shape}")
        return self.model(features, mask_features)
    
    def training_step(self, batch, batch_idx):
        features = batch['features']
        mask_features = batch['mask_features']
        labels = batch['label']
        
        # モデルの予測
        logits = self(features, mask_features)
        loss = self.criterion(logits, labels.argmax(dim=1))
        
        # メトリクスの計算
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels.argmax(dim=1))
        
        # 現在のバッチの精度を計算
        batch_acc = (preds == labels.argmax(dim=1)).float().mean()
        
        # ログ出力
        # バッチごとのメトリクス
        self.log('train/batch_loss', loss, on_step=True, prog_bar=True)
        self.log('train/batch_acc', batch_acc, on_step=True, prog_bar=True)
        
        # エポックごとの累積メトリクス
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # デバッグ出力
        if batch_idx % 100 == 0:  # 100バッチごとに詳細なログを出力
            logging.debug(
                f"Training batch {batch_idx}: "
                f"loss={loss:.4f}, "
                f"acc={batch_acc:.4f}, "
                f"features_shape={features.shape}, "
                f"mask_features_shape={mask_features.shape}"
            )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        features = batch['features']
        mask_features = batch['mask_features']
        labels = batch['label']
        
        # モデルの予測
        logits = self(features, mask_features)
        loss = self.criterion(logits, labels.argmax(dim=1))
        
        # メトリクスの計算
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels.argmax(dim=1))
        
        # WandBへのログ記録
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        テストステップ：予測を行い、結果を保存
        """
        features = batch['features']
        mask_features = batch.get('mask_features', None)
        y = batch['label']
        
        # 予測
        logits = self(features, mask_features)
        loss = F.cross_entropy(logits, y)
        
        # 予測確率とクラスを計算
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        targets = torch.argmax(y, dim=1)
        
        # 予測と正解を保存
        self.test_predictions.extend(probs.cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        
        # メトリクスの計算
        acc = (preds == targets).float().mean()
        
        # ログ出力
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def test_epoch_end(self, outputs):
        """
        テストエポック終了時：平均メトリクスを計算
        """
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        
        # 予測と正解をnumpy配列として保存
        predictions = np.array(self.test_predictions)
        targets = np.array(self.test_targets)
        
        # 結果を保存
        test_results = {
            'test_loss': avg_loss.item(),
            'test_acc': avg_acc.item(),
            'predictions': predictions,
            'targets': targets
        }
        
        # 結果をファイルに保存
        output_dir = self.trainer.checkpoint_callback.dirpath
        results_path = os.path.join(output_dir, 'test_results.npz')
        np.savez(results_path, **test_results)
        
        logging.info(f"Test results saved to {results_path}")
        return test_results
    
    def configure_optimizers(self):
        """
        オプティマイザとスケジューラの設定
        - AdamWオプティマイザ（Weight Decay対応）
        - OneCycleLR scheduler（学習率の動的調整）
        - Early Stoppingのための監視
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # OneCycleLRスケジューラの設定
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.trainer.max_epochs,
            # steps_per_epoch=len(self.trainer.train_dataloader),
            steps_per_epoch=300,  # 仮の値。実際にはトレーニングデータのサイズに依存
            pct_start=0.3,  # 30%までWarm-up
            div_factor=10.0,  # 初期学習率はmax_lrの1/10
            final_div_factor=100.0  # 最終学習率はmax_lrの1/1000
        )
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }
    
    # def on_validation_epoch_end(self, outputs):
    #     """
    #     検証エポック終了時の処理
    #     - Early Stoppingの判定
    #     - 最良モデルの保存
    #     """
    #     # 平均検証損失の計算
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
    #     # Early Stopping判定
    #     if avg_loss < self.best_val_loss:
    #         self.best_val_loss = avg_loss
    #         self.val_loss_cnt = 0
    #     else:
    #         self.val_loss_cnt += 1
            
    #         if self.val_loss_cnt >= 5:  # 5エポック改善がない場合
    #             # 学習率を1/2に削減
    #             for param_group in self.trainer.optimizers[0].param_groups:
    #                 param_group['lr'] *= 0.5
    #             logging.info(f"Reducing learning rate to {param_group['lr']}")
    #             self.val_loss_cnt = 0  # カウンタをリセット
        
    #     # WandBへのログ記録
    #     self.log('val/avg_loss', avg_loss)
    #     self.log('val/best_loss', self.best_val_loss)
    #     self.log('val/no_improve_epochs', self.val_loss_cnt)

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

# --- NPZデータセット定義 ---
class NPZFeatureDataset(Dataset):
    """
    vdata.pyで生成されたnpzファイルから特徴量とラベルを読み込むデータセット
    """
    def __init__(self, npz_files):
        self.npz_files = npz_files
        self.datasets = [OphNetSurgicalDataset.load_npz(f) for f in npz_files]
        self.cumulative_lengths = np.cumsum([len(d) for d in self.datasets])
        logging.debug(f"Initialized NPZFeatureDataset with {len(self.datasets)} files, "
                     f"total samples: {self.cumulative_lengths[-1] if len(self.cumulative_lengths) > 0 else 0}")
    
    def __len__(self):
        if not self.cumulative_lengths.size:
            return 0
        return int(self.cumulative_lengths[-1])  # 最後の累積長が総サンプル数
    
    def __getitem__(self, idx):
        # データセットのインデックスを特定
        dataset_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        
        # 対応するデータセットからサンプルを取得
        dataset = self.datasets[dataset_idx]
        dataset.return_features = True  # 特徴量モードを有効化
        sample = dataset[sample_idx]
        
        return {
            'features': sample['features'],
            'mask_features': sample['mask_features'],
            'label': sample['label']
        }

class NPZDataModule(pl.LightningDataModule):
    """
    npzファイルを管理し、トレーニング/検証用のDataLoaderを提供するDataModule
    """
    def __init__(self, npz_dir, batch_size=32, num_workers=4, val_split=0.2):
        super().__init__()
        self.npz_dir = npz_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        # npzファイルのリストを取得
        npz_files = sorted(glob.glob(os.path.join(self.npz_dir, "*.npz")))
        if not npz_files:
            raise ValueError(f"No npz files found in {self.npz_dir}")
        
        # トレーニング/検証用にファイルを分割
        train_files, val_files = train_test_split(
            npz_files, test_size=self.val_split, random_state=42
        )
        
        # データセットの作成
        self.train_dataset = NPZFeatureDataset(train_files)
        self.val_dataset = NPZFeatureDataset(val_files)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

# --- メイン処理 ---
@click.command()
@click.option('--npz_dir', type=str, required=True, help='npzファイルが保存されているディレクトリ')
@click.option('--output_dir', type=str, default='models', help='モデル保存ディレクトリ')
@click.option('--batch_size', type=int, default=32, help='バッチサイズ')
@click.option('--num_workers', type=int, default=4, help='DataLoaderのワーカー数')
@click.option('--max_epochs', type=int, default=100, help='最大エポック数')
@click.option('--feature_dim', type=int, default=1024, help='特徴量の次元数')
@click.option('--learning_rate', type=float, default=1e-4, help='学習率')
@click.option('--gpus', type=str, default=None, help='使用するGPU番号（カンマ区切り。例: "0,1"）')
@click.option('--device', type=str, default='cuda', help='使用するデバイス（"cuda"または"cpu"）')
@click.option('--use_wandb', is_flag=True, help='Weights & Biasesを使用する')
@click.option('--wandb_project', type=str, default='surgical-phase-classification', help='Weights & Biasesのプロジェクト名')
@click.option('--wandb_run_name', type=str, default=None, help='Weights & Biasesの実験名')
def main(npz_dir, output_dir, batch_size, num_workers, max_epochs, feature_dim, 
         learning_rate, gpus, device, use_wandb, wandb_project, wandb_run_name):
    """
    メイン学習処理
    過学習対策として以下を追加:
    - Early Stoppingコールバック
    - モデルチェックポイントの保存
    - 学習率の自動調整
    """
    # 設定のログ出力
    logging.info(f"""Training configuration:
    - NPZ directory: {npz_dir}
    - Output directory: {output_dir}
    - Batch size: {batch_size}
    - Workers: {num_workers}
    - Epochs: {max_epochs}
    - Device: {device}
    - Use WandB: {use_wandb}
    """)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データモジュールの初期化
    data_module = NPZDataModule(
        npz_dir=npz_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    data_module.setup()
    
    # モデルの作成
    model = SurgicalPhaseModule(
        feature_dim=feature_dim,
        num_classes=35,  # 手術フェーズのクラス数
        learning_rate=learning_rate,
        weight_decay=1e-2  # L2正則化の追加
    )
    
    # ロガーの設定
    loggers = [
        pl_loggers.CSVLogger(output_dir),
        pl_loggers.TensorBoardLogger(output_dir)
    ]
    if use_wandb:
        loggers.append(WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            log_model=True
        ))
    
    # コールバックの設定
    callbacks = [
        # Early Stopping
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            min_delta=1e-4
        ),
        # モデルチェックポイント
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename='{epoch}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,  # 上位3つのモデルを保存
            save_last=True  # 最新のモデルも保存
        ),
        # 学習率のモニタリング
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        # GPUメモリ使用量のモニタリング
        pl.callbacks.DeviceStatsMonitor()
    ]
    
    # トレーナーの設定
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=[int(g) for g in gpus.split(',')] if gpus else 1,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=1.0,  # 勾配クリッピングの追加
        accumulate_grad_batches=2,  # 勾配累積によるバッチサイズの実質的な増加
        precision=16,  # 混合精度学習の有効化
        strategy='ddp' if device == 'cuda' and len(gpus.split(',')) > 1 else "auto"
    )
    
    # 学習の実行
    trainer.fit(model, data_module)
    
    # 最良モデルのパスを出力
    best_model_path = trainer.checkpoint_callback.best_model_path
    logging.info(f"Best model saved at: {best_model_path}")
    
    return best_model_path

if __name__ == "__main__":
    main()
