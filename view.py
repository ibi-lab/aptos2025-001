import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import click

# --- データセット可視化・中間ファイル可視化・モデル予測可視化 ---
# 各関数・コマンドの役割を明記し、PEP8/コメント充実・保守性向上

def show_sample(sample, filepath=None):
    """
    1サンプル（dict: image, mask, label）を画像・マスク・ラベル（one-hot）としてmatplotlibで可視化。
    filepathが指定されていれば画像として保存。
    画像とマスクについての基本的な統計情報も表示。
    """
    img = sample['image'].numpy().transpose(1, 2, 0)
    mask = sample['mask'].numpy().transpose(1, 2, 0)
    label = sample['label'].numpy()
    
    # 画像の統計情報を計算
    img_stats = {
        'mean': img.mean(),
        'std': img.std(),
        'min': img.min(),
        'max': img.max()
    }
    
    fig = plt.figure(figsize=(15, 5))
    
    # 画像表示
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(img)
    ax1.set_title(f'Image\nμ={img_stats["mean"]:.3f}, σ={img_stats["std"]:.3f}')
    plt.colorbar(im1, ax=ax1)
    
    # マスク表示
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(mask)
    ax2.set_title('Mask')
    plt.colorbar(im2, ax=ax2)
    
    # ラベル表示
    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(np.arange(len(label)), label)
    ax3.set_title('Label (one-hot)')
    ax3.set_xlabel('Phase index')
    ax3.set_ylabel('Probability')
    
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
        
        # 最初のサンプルを可視化
        sample = dataset[0]
        show_sample(sample, filepath=output_img_path)
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

