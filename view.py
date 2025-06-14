import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import click

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
def show_npz(npz_path, output_img_path):
    """
    vdata.pyで生成されたnpzファイル（中間ファイル）を可視化。
    画像・マスク・ラベル群をmatplotlibで可視化し、指定パスに保存。

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
        print('Dataset info:')
        print(f'サンプル数: {len(dataset)}')
        print(f'Chunks shape: {dataset.chunks.shape}')
        print(f'Mask chunks shape: {dataset.mask_chunks.shape}')
        print(f'Labels shape: {dataset.labels.shape}')
        
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

if __name__ == '__main__':
    cli()

