"""
可视化函数
Visualization Functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from evaluation import compute_iou_with_best_perm, evaluate_temporal_consistency

# 颜色定义
# 建议的颜色映射，对齐生成器逻辑：
COLORS = np.array([
    [0.1, 0.1, 0.1],  # 0: 背景 (深灰)
    [1.0, 0.0, 0.0],  # 1: 线粒体1 (红色)
    [0.6, 0.1, 0.8],  # 2: 交叠区 (紫色) -> 对应生成器里的 overlap
    [0.0, 0.4, 1.0],  # 3: 线粒体2 (蓝色)
])

CLASS_NAMES = ['Background', 'Mito 1', 'Overlap', 'Mito 2']


def plot_training_curves(history, output_path, cfg):
    """
    绘制训练曲线

    包含：
    - 总Loss（训练+验证）
    - 各分量Loss
    - IoU
    - Temporal Consistency
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    epochs = range(1, len(history['train_loss']) + 1)
    val_epochs = history.get('val_epochs', list(range(cfg.val_interval, len(epochs) + 1, cfg.val_interval)))

    # 总Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=1.5, alpha=0.8)
    if 'val_loss' in history:
        ax.plot(val_epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 分割Loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_seg'], 'b-', label='Train', linewidth=1.5, alpha=0.8)
    if 'val_seg' in history:
        ax.plot(val_epochs, history['val_seg'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Segmentation Loss (PIT)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 时序Loss
    ax = axes[0, 2]
    ax.plot(epochs, history['train_temporal'], 'b-', label='Train', linewidth=1.5, alpha=0.8)
    if 'val_temporal' in history:
        ax.plot(val_epochs, history['val_temporal'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Temporal Continuity Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 重建Loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_recon'], 'b-', label='Train', linewidth=1.5, alpha=0.8)
    if 'val_recon' in history:
        ax.plot(val_epochs, history['val_recon'], 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # IoU
    ax = axes[1, 1]
    if 'val_iou' in history:
        ax.plot(val_epochs, history['val_iou'], 'g-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(val_epochs,
                        np.array(history['val_iou']) - np.array(
                            history.get('val_iou_std', [0] * len(history['val_iou']))),
                        np.array(history['val_iou']) + np.array(
                            history.get('val_iou_std', [0] * len(history['val_iou']))),
                        alpha=0.2, color='g')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('IoU', fontsize=11)
    ax.set_title('Validation IoU', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Temporal Consistency
    ax = axes[1, 2]
    if 'val_consistency' in history:
        ax.plot(val_epochs, history['val_consistency'], 'm-', linewidth=2, marker='o', markersize=4)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random baseline')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Consistency', fontsize=11)
    ax.set_title('Temporal Consistency', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.vis_dpi, bbox_inches='tight')
    plt.close()
    print(f"训练曲线 -> {output_path}")


def visualize_predictions(model, dataloader, output_path, cfg, device, num_samples=None):
    """
    可视化预测结果

    每个样本显示4行：
    - 输入图像
    - GT标签
    - 预测标签
    - 预测概率差（P(M1) - P(M2)）
    """
    model.eval()

    if num_samples is None:
        num_samples = cfg.vis_num_samples

    # 收集样本
    samples = []
    with torch.no_grad():
        for images, labels, masks1, masks2 in dataloader:
            for i in range(images.shape[0]):
                if len(samples) >= num_samples:
                    break

                img = images[i:i + 1].to(device)
                output = model(img)
                pred = output.argmax(dim=2)[0].cpu().numpy()
                prob = torch.softmax(output, dim=2)[0].cpu().numpy()

                samples.append({
                    'images': images[i].numpy()[:, 0],  # (T, H, W)
                    'labels': labels[i].numpy(),
                    'preds': pred,
                    'probs': prob,
                    'masks1': masks1[i].numpy(),
                    'masks2': masks2[i].numpy(),
                })

            if len(samples) >= num_samples:
                break

    T = samples[0]['images'].shape[0]

    fig, axes = plt.subplots(num_samples * 4, T, figsize=(T * 2.8, num_samples * 9))
    if num_samples == 1:
        axes = axes.reshape(4, T)

    for s, sample in enumerate(samples):
        images = sample['images']
        labels = sample['labels']
        preds = sample['preds']
        probs = sample['probs']
        masks1 = sample['masks1']
        masks2 = sample['masks2']

        # 计算指标
        sample_ious = [compute_iou_with_best_perm(
            torch.LongTensor(preds[t]),
            torch.FloatTensor(masks1[t]),
            torch.FloatTensor(masks2[t])
        ) for t in range(T)]
        avg_iou = np.mean(sample_ious)
        consistency = evaluate_temporal_consistency(torch.LongTensor(preds))

        row_base = s * 4

        for t in range(T):
            # Row 1: 输入图像
            ax = axes[row_base, t]
            img = images[t]
            vmin, vmax = np.percentile(img, [1, 99])
            ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if t == 0:
                ax.set_ylabel(f'Sample {s + 1}\nInput', fontsize=10, fontweight='bold')
            if s == 0:
                ax.set_title(f'Frame {t}', fontsize=11, fontweight='bold')

            # Row 2: GT标签
            ax = axes[row_base + 1, t]
            ax.imshow(COLORS[labels[t]])
            ax.axis('off')
            if t == 0:
                ax.set_ylabel('GT Label', fontsize=10, fontweight='bold')

            # Row 3: 预测标签
            ax = axes[row_base + 2, t]
            ax.imshow(COLORS[preds[t]])
            ax.axis('off')
            if t == 0:
                ax.set_ylabel('Prediction', fontsize=10, fontweight='bold')

            # Row 4: 概率差
            ax = axes[row_base + 3, t]
            prob_m1 = probs[t, 1] + probs[t, 3]
            prob_m2 = probs[t, 2] + probs[t, 3]
            prob_diff = prob_m1 - prob_m2
            im = ax.imshow(prob_diff, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.axis('off')
            if t == 0:
                ax.set_ylabel('P(M1)-P(M2)', fontsize=10, fontweight='bold')

        # 添加指标
        axes[row_base, T - 1].text(
            1.05, 0.5,
            f'IoU: {avg_iou:.3f}\nCons: {consistency:.3f}',
            transform=axes[row_base, T - 1].transAxes,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], label=CLASS_NAMES[i])
        for i in range(4)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.995))

    # 添加colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.2])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(-1, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('P(M1) - P(M2)', fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(output_path, dpi=cfg.vis_dpi, bbox_inches='tight')
    plt.close()
    print(f"预测可视化 -> {output_path}")


def visualize_single_sequence(images, labels, preds, probs, output_path, title=""):
    """
    可视化单个序列
    """
    T = images.shape[0]

    fig, axes = plt.subplots(4, T, figsize=(T * 3, 11))

    for t in range(T):
        # 输入
        ax = axes[0, t]
        img = images[t]
        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.set_title(f'Frame {t}', fontsize=11)
        if t == 0:
            ax.set_ylabel('Input', fontsize=10, fontweight='bold')

        # GT
        ax = axes[1, t]
        ax.imshow(COLORS[labels[t]])
        ax.axis('off')
        if t == 0:
            ax.set_ylabel('GT', fontsize=10, fontweight='bold')

        # Prediction
        ax = axes[2, t]
        ax.imshow(COLORS[preds[t]])
        ax.axis('off')
        if t == 0:
            ax.set_ylabel('Pred', fontsize=10, fontweight='bold')

        # Probability
        ax = axes[3, t]
        prob_m1 = probs[t, 1] + probs[t, 3]
        prob_m2 = probs[t, 2] + probs[t, 3]
        ax.imshow(prob_m1 - prob_m2, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.axis('off')
        if t == 0:
            ax.set_ylabel('P(M1)-P(M2)', fontsize=10, fontweight='bold')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()