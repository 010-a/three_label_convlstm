"""
主训练脚本
Main Training Script

使用方法：
    python train.py

确保数据已处理：
    python scripts/prepare_data.py --input_dir ./data/raw --output_dir ./data/processed
"""

import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from config import Config
from models import ConvLSTMUNet, count_parameters
from datasets import MitoSequenceDataset, create_dataloaders
from losses import CombinedLoss
from evaluation import evaluate_batch
from visualization import plot_training_curves, visualize_predictions


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    loss_components = {'seg': 0, 'temporal': 0, 'recon': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels, masks1, masks2 in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss, losses, use_swap = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 记录
        total_loss += losses['total']
        loss_components['seg'] += losses['seg']
        loss_components['temporal'] += losses['temporal']
        loss_components['recon'] += losses['recon']
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{losses['total']:.4f}",
            'seg': f"{losses['seg']:.4f}",
            'temp': f"{losses['temporal']:.4f}"
        })

    return {
        'loss': total_loss / num_batches,
        'seg': loss_components['seg'] / num_batches,
        'temporal': loss_components['temporal'] / num_batches,
        'recon': loss_components['recon'] / num_batches,
    }


def main():
    # 配置
    cfg = Config
    cfg.print_config()

    # 设置随机种子
    set_seed(cfg.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = cfg.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    cfg.save_config(output_dir / "config.txt")

    # 设备
    device = torch.device(cfg.device)
    print(f"\n设备: {device}")

    # 数据
    print("\n加载数据...")
    train_loader, val_loader = create_dataloaders(cfg)
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")

    # 模型
    print("\n创建模型...")
    model = ConvLSTMUNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_ch=cfg.base_ch,
        lstm_hidden=cfg.lstm_hidden,
        bidirectional=cfg.bidirectional
    ).to(device)

    print(f"模型参数量: {count_parameters(model) / 1e6:.2f}M")

    # Loss
    criterion = CombinedLoss(cfg)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # 学习率调度器
    if cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )

    # 训练历史
    history = {
        'train_loss': [], 'train_seg': [], 'train_temporal': [], 'train_recon': [],
        'val_loss': [], 'val_seg': [], 'val_temporal': [], 'val_recon': [],
        'val_iou': [], 'val_iou_std': [],
        'val_consistency': [], 'val_consistency_std': [],
        'val_epochs': [],
        'lr': [],
    }

    best_iou = 0
    best_consistency = 0
    epochs_without_improvement = 0

    print("\n开始训练...")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 记录训练指标
        history['train_loss'].append(train_metrics['loss'])
        history['train_seg'].append(train_metrics['seg'])
        history['train_temporal'].append(train_metrics['temporal'])
        history['train_recon'].append(train_metrics['recon'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 学习率调度
        scheduler.step()

        # 验证
        if epoch % cfg.val_interval == 0:
            val_metrics = evaluate_batch(model, val_loader, device, criterion)

            history['val_loss'].append(val_metrics['loss'])
            history['val_seg'].append(val_metrics['seg_loss'])
            history['val_temporal'].append(val_metrics['temporal_loss'])
            history['val_recon'].append(val_metrics['recon_loss'])
            history['val_iou'].append(val_metrics['iou'])
            history['val_iou_std'].append(val_metrics['iou_std'])
            history['val_consistency'].append(val_metrics['consistency'])
            history['val_consistency_std'].append(val_metrics['consistency_std'])
            history['val_epochs'].append(epoch)

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch}/{cfg.epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(seg={train_metrics['seg']:.4f}, temp={train_metrics['temporal']:.4f}, recon={train_metrics['recon']:.4f})")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} "
                  f"(seg={val_metrics['seg_loss']:.4f}, temp={val_metrics['temporal_loss']:.4f}, recon={val_metrics['recon_loss']:.4f})")
            print(f"  Val IoU: {val_metrics['iou']:.4f} ± {val_metrics['iou_std']:.4f}")
            print(f"  Val Consistency: {val_metrics['consistency']:.4f} ± {val_metrics['consistency_std']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            improved = False

            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'consistency': val_metrics['consistency'],
                }, output_dir / 'best_iou_model.pth')
                print(f"  -> 保存最佳IoU模型 (IoU={best_iou:.4f})")
                improved = True

            if val_metrics['consistency'] > best_consistency:
                best_consistency = val_metrics['consistency']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_consistency': best_consistency,
                    'iou': val_metrics['iou'],
                }, output_dir / 'best_consistency_model.pth')
                print(f"  -> 保存最佳Consistency模型 (Cons={best_consistency:.4f})")
                improved = True

            if improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += cfg.val_interval

            # 早停
            if epochs_without_improvement >= cfg.early_stop_patience:
                print(f"\n早停: {epochs_without_improvement} epochs无改进")
                break

        # 定期保存checkpoint
        if epoch % cfg.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, output_dir / f'checkpoint_epoch{epoch}.pth')

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"训练完成! 总时间: {total_time / 60:.1f} 分钟")
    print(f"最佳IoU: {best_iou:.4f}")
    print(f"最佳Consistency: {best_consistency:.4f}")

    # 保存最终模型和历史
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, output_dir / 'final_model.pth')

    torch.save(history, output_dir / 'history.pth')

    # 绘制训练曲线
    print("\n生成可视化...")
    plot_training_curves(history, output_dir / 'training_curves.png', cfg)

    # 加载最佳模型进行可视化
    checkpoint = torch.load(output_dir / 'best_iou_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    visualize_predictions(model, val_loader, output_dir / 'predictions.png', cfg, device)

    # 保存结果摘要
    with open(output_dir / 'results.txt', 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"实验: {cfg.exp_name}\n")
        f.write(f"时间: {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"最佳IoU: {best_iou:.4f}\n")
        f.write(f"最佳Consistency: {best_consistency:.4f}\n")
        f.write(f"总训练时间: {total_time / 60:.1f} 分钟\n")
        f.write(f"总Epochs: {epoch}\n\n")
        f.write("配置:\n")
        f.write(f"  batch_size: {cfg.batch_size}\n")
        f.write(f"  lr: {cfg.lr}\n")
        f.write(f"  base_ch: {cfg.base_ch}\n")
        f.write(f"  lstm_hidden: {cfg.lstm_hidden}\n")
        f.write(f"  seg_loss_weight: {cfg.seg_loss_weight}\n")
        f.write(f"  temporal_loss_weight: {cfg.temporal_loss_weight}\n")
        f.write(f"  recon_loss_weight: {cfg.recon_loss_weight}\n")

    print(f"\n所有结果保存至: {output_dir}")


if __name__ == "__main__":
    main()