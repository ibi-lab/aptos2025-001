import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import click
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, precision_score, recall_score, f1_score,
    average_precision_score
)
import torch.nn.functional as F

# --- データセット可視化・中間ファイル可視化・モデル予測可視化 ---
# 各関数・コマンドの役割を明記し、PEP8/コメント充実・保守性向上

def show_sample(images, masks, label, filepath=None, num_samples=8):
    """
    画像群、マスク群、ラベル（one-hot）をmatplotlibで可視化。
    
    Args:
        images (torch.Tensor): 表示する画像群 (N, C, H, W)
        masks (torch.Tensor): 表示するマスク群 (N, C, H, W)
        label (torch.Tensor): one-hotラベル
        filepath (str, optional): 保存先ファイルパス
        num_samples (int, default=4): 表示するサンプル数
    """
    # 表示するサンプル数を制限
    N = min(len(images), num_samples)
    
    # 画像とマスクの統計情報を計算
    img_stats = {
        'mean': images[:N].mean().item(),
        'std': images[:N].std().item(),
        'min': images[:N].min().item(),
        'max': images[:N].max().item()
    }
    
    # プロットのサイズを設定
    fig = plt.figure(figsize=(2 + 3 * N, 8))
    
    # 画像とマスクを2行で表示
    for i in range(N):
        # 画像表示
        ax = plt.subplot(2, N, i + 1)
        img = images[i].numpy().transpose(1, 2, 0)
        im = ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        plt.colorbar(im, ax=ax)
        
        # マスク表示
        ax = plt.subplot(2, N, N + i + 1)
        mask = masks[i].numpy().transpose(1, 2, 0)
        im = ax.imshow(mask)
        ax.set_title(f'Mask {i+1}')
        plt.colorbar(im, ax=ax)
    
    # 画像の下部に統計情報とラベル情報を表示
    phase_idx = label.numpy().argmax()
    fig.text(0.1, 0.02, 
             f'Image Statistics:\nμ={img_stats["mean"]:.3f}, σ={img_stats["std"]:.3f}, '
             f'min={img_stats["min"]:.3f}, max={img_stats["max"]:.3f}\n'
             f'Phase Index: {phase_idx} (confidence: {label[phase_idx]:.3f})',
             verticalalignment='bottom')
    
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
        print(f"Saved visualization to {filepath}")
        plt.close(fig)
    else:
        plt.show()

@click.group()
def cli():
    """
    view.pyのCLIエントリポイント。各可視化コマンドを登録。
    """
    pass

@cli.command()
@click.argument('npz_path', type=click.Path(exists=True))
@click.option('-o', '--output_img_path', type=str, help='可視化画像の保存先パス（省略時は入力ファイルと同名のpng）')
def show_image_from_npz(npz_path, output_img_path):
    """
    vdata.pyで生成されたnpzファイル（中間ファイル）の画像・マスク・ラベルを可視化。
    可視化結果を指定パスに保存。

    Args:
        npz_path: npzファイルパス（必須）
        output_img_path: 出力画像パス（省略時は入力ファイルと同名のpng）
    """
    from vdata import OphNetSurgicalDataset

    # 出力パスが指定されていない場合、入力ファイル名から生成
    if output_img_path is None:
        output_img_path = os.path.splitext(npz_path)[0] + '.png'

    try:
        # データセットをロード
        dataset = OphNetSurgicalDataset.load_npz(npz_path)
        
        if dataset.chunks is None or dataset.mask_chunks is None:
            print("警告: このnpzファイルには画像データが含まれていません")
            return
        
        # チャンクからサンプルを生成
        samples = []
        num_samples = min(4, len(dataset))  # 最大4サンプルまで
        for i in range(num_samples):
            samples.append(dataset[i])
        
        # サンプルデータを整理
        images = torch.stack([s['image'] for s in samples])
        masks = torch.stack([s['mask'] for s in samples])
        label = samples[0]['label']  # ラベルは全て同じなので最初のものを使用
        
        # 可視化
        show_sample(images, masks, label, filepath=output_img_path, num_samples=num_samples)

    except Exception as e:
        print(f"Error loading npz file: {e}")


@cli.command()
@click.argument('npz_path', type=click.Path(exists=True))
def show_features_from_npz(npz_path):
    """
    vdata.pyで生成されたnpzファイル（中間ファイル）の特徴量とラベル情報を標準出力に表示。

    Args:
        npz_path: npzファイルパス（必須）
    """
    from vdata import OphNetSurgicalDataset

    try:
        # データセットをロード
        dataset = OphNetSurgicalDataset.load_npz(npz_path)
        
        # データセット基本情報の表示
        print('=== データセット基本情報 ===')
        print(f'サンプル数: {len(dataset)}')
        
        # 画像/マスクデータの情報表示
        if dataset.chunks is not None:
            print('\n=== 画像データ情報 ===')
            print(f'画像チャンクの形状: {dataset.chunks.shape}')
            print(f'画像の値域: [{dataset.chunks.min():.3f}, {dataset.chunks.max():.3f}]')
            print(f'画像の平均値: {dataset.chunks.mean():.3f}')
            print(f'画像の標準偏差: {dataset.chunks.std():.3f}')
        
        if dataset.mask_chunks is not None:
            print('\n=== マスクデータ情報 ===')
            print(f'マスクチャンクの形状: {dataset.mask_chunks.shape}')
            print(f'マスクの値域: [{dataset.mask_chunks.min():.3f}, {dataset.mask_chunks.max():.3f}]')
            print(f'マスクの平均値: {dataset.mask_chunks.mean():.3f}')
            print(f'マスクの標準偏差: {dataset.mask_chunks.std():.3f}')
        
        # 特徴量データの情報表示
        if dataset.features is not None:
            print('\n=== 画像特徴量情報 ===')
            print(f'特徴量の形状: {dataset.features.shape}')
            print(f'特徴量の値域: [{dataset.features.min():.3f}, {dataset.features.max():.3f}]')
            print(f'特徴量の平均値: {dataset.features.mean():.3f}')
            print(f'特徴量の標準偏差: {dataset.features.std():.3f}')
        
        if dataset.mask_features is not None:
            print('\n=== マスク特徴量情報 ===')
            print(f'マスク特徴量の形状: {dataset.mask_features.shape}')
            print(f'マスク特徴量の値域: [{dataset.mask_features.min():.3f}, {dataset.mask_features.max():.3f}]')
            print(f'マスク特徴量の平均値: {dataset.mask_features.mean():.3f}')
            print(f'マスク特徴量の標準偏差: {dataset.mask_features.std():.3f}')
        
        # ラベル情報の表示
        if dataset.labels is not None:
            print('\n=== ラベル情報 ===')
            print(f'ラベルの形状: {dataset.labels.shape}')
            label_dist = dataset.labels.argmax(axis=1)
            unique, counts = np.unique(label_dist, return_counts=True)
            print('\nラベル分布:')
            for label, count in zip(unique, counts):
                print(f'クラス {label}: {count}サンプル ({count/len(dataset)*100:.1f}%)')

    except Exception as e:
        print(f"Error loading npz file: {e}")

@cli.command()
@click.option('--model_ckpt', type=str, required=True, help='学習済みモデルckpt')
@click.option('--pth_path', type=str, required=True, help='dataset.pthファイルパス')
@click.option('--idx', type=int, default=0, help='表示するサンプルのインデックス')
def show_prediction(model_ckpt, pth_path, idx):
    """
    train.pyで学習したモデルを使い、指定サンプルの予測値と正解ラベルを可視化（棒グラフで比較）
    """
    import torch.nn.functional as F
    from train import SurgicalPhaseModule
    data = torch.load(pth_path)
    sample = data[idx]
    model = SurgicalPhaseModule.load_from_checkpoint(model_ckpt, num_classes=len(sample['label']))
    model.eval()
    with torch.no_grad():
        x = sample['image'].unsqueeze(0)
        m = sample['mask'].unsqueeze(0)
        logits = model.model(x, m)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(probs)), probs, label='Pred')
    plt.bar(np.arange(len(sample['label'])), sample['label'].numpy(), alpha=0.5, label='GT')
    plt.legend()
    plt.title('Prediction vs GT')
    plt.show()

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--output_dir', type=str, help='可視化結果の保存先ディレクトリ')
@click.option('--batch_size', type=int, default=32, help='バッチサイズ')
@click.option('--num_workers', type=int, default=4, help='DataLoaderのワーカー数')
@click.option('--device', type=str, default='cuda', help='使用するデバイス')
def show_training_result(model_path, data_dir, output_dir=None, batch_size=32, num_workers=4, device='cuda'):
    """
    学習済みモデルを使用してテストを実行し、結果を可視化します。
    混同行列、ROC曲線、精度・再現率曲線などを表示します。

    Args:
        model_path (str): モデルのチェックポイントファイルパス
        data_dir (str): テストデータのディレクトリパス
        output_dir (str, optional): 可視化結果の保存先ディレクトリ
        batch_size (int): バッチサイズ
        num_workers (int): DataLoaderのワーカー数
        device (str): 使用するデバイス
    """
    import pytorch_lightning as pl
    from train import SurgicalPhaseModule, NPZDataModule

    # PyTorchの警告を抑制
    import warnings
    warnings.filterwarnings('ignore')

    # データモジュールの準備
    data_module = NPZDataModule(
        npz_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    data_module.setup()

    # モデルのロード
    model = SurgicalPhaseModule.load_from_checkpoint(
        model_path,
        feature_dim=1024,  # モデルの設定に合わせて調整
        num_classes=35
    )
    model = model.to(device)
    model.eval()

    # トレーナーの設定
    trainer = pl.Trainer(
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
        logger=False
    )

    # テストの実行
    test_results = trainer.test(model, data_module)

    # 予測結果の取得
    predictions = np.array(model.test_predictions)
    targets = np.array(model.test_targets)
    test_loss = test_results[0]['test_loss']
    test_acc = test_results[0]['test_acc']

    # 結果概要の表示
    print(f"=== テスト結果の概要 ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")

    # 予測確率とクラスラベルに変換
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)

    # クラスごとの性能評価
    n_classes = predictions.shape[1]
    
    # サブプロットの設定
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)

    # 1. 混同行列
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(true_classes, pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    # 2. ROC曲線
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC curves')
    ax2.legend(loc='lower right')

    # 3. Precision-Recall曲線
    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(targets[:, i], predictions[:, i])
        avg_precision = average_precision_score(targets[:, i], predictions[:, i])
        ax3.plot(recall, precision, label=f'Class {i} (AP = {avg_precision:.2f})')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall curves')
    ax3.legend(loc='lower left')

    # 4. クラスごとの性能メトリクス
    ax4 = fig.add_subplot(gs[1, 1])
    class_metrics = []
    for i in range(n_classes):
        metrics = {
            'Precision': precision_score(true_classes == i, pred_classes == i),
            'Recall': recall_score(true_classes == i, pred_classes == i),
            'F1-score': f1_score(true_classes == i, pred_classes == i)
        }
        class_metrics.append(metrics)
    
    df_metrics = pd.DataFrame(class_metrics)
    df_metrics.index = [f'Class {i}' for i in range(n_classes)]
    
    sns.heatmap(df_metrics, annot=True, fmt='.2f', ax=ax4)
    ax4.set_title('Class-wise Performance Metrics')

    plt.tight_layout()

    # 結果の保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, 'training_results.png'))
        print(f"Results saved to {output_dir}")
    else:
        plt.show()

    plt.close(fig)

if __name__ == '__main__':
    cli()

