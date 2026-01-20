"""
评估函数 (修正版)
Evaluation Metrics - Corrected for 3-class labels

标签格式：
- 0: 背景
- 1: 线粒体1独占区 (M1 only)
- 2: 交叠区 (Overlap)
- 3: 线粒体2独占区 (M2 only)

完整mask：
- mask1 = (label == 1) | (label == 2)
- mask2 = (label == 3) | (label == 2)
"""

import torch
import numpy as np


def compute_iou(pred, target, num_classes=4):
    """
    计算每个类别的IoU

    Args:
        pred: (H, W) 预测类别
        target: (H, W) 真实类别
        num_classes: 类别数

    Returns:
        list of IoU for each class
    """
    ious = []
    pred = pred.reshape(-1)
    target = target.reshape(-1)

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(float('nan'))

    return ious


def compute_iou_with_best_perm(pred, masks1, masks2):
    """
    计算IoU，考虑最优排列

    由于使用PIT，M1和M2的身份可能互换。
    这里计算两种匹配方式的IoU，取较大的。

    Args:
        pred: (H, W) 预测的4类标签 (0, 1, 2, 3)
        masks1: (H, W) M1的真实完整mask (amodal)
        masks2: (H, W) M2的真实完整mask (amodal)

    Returns:
        best_iou: 两条线粒体的平均IoU（取最优匹配）

    标签定义：
    - 0: 背景
    - 1: M1独占区
    - 2: 交叠区（M1和M2都存在）
    - 3: M2独占区
    """
    # ========== 关键修正：从预测标签恢复完整mask ==========
    # pred_m1 = M1存在的区域 = M1独占 + 交叠 = (pred == 1) | (pred == 2)
    # pred_m2 = M2存在的区域 = M2独占 + 交叠 = (pred == 3) | (pred == 2)

    if isinstance(pred, torch.Tensor):
        pred_m1 = ((pred == 1) | (pred == 2)).float()
        pred_m2 = ((pred == 3) | (pred == 2)).float()
        gt_m1 = (masks1 > 0.5).float()
        gt_m2 = (masks2 > 0.5).float()
    else:
        pred_m1 = ((pred == 1) | (pred == 2)).astype(np.float32)
        pred_m2 = ((pred == 3) | (pred == 2)).astype(np.float32)
        gt_m1 = (masks1 > 0.5).astype(np.float32)
        gt_m2 = (masks2 > 0.5).astype(np.float32)

    def single_iou(a, b):
        if isinstance(a, torch.Tensor):
            inter = (a * b).sum().item()
            union = (a.sum() + b.sum() - inter).item()
        else:
            inter = (a * b).sum()
            union = a.sum() + b.sum() - inter
        return inter / (union + 1e-6)

    # 两种匹配方式
    # 匹配1: pred_m1 对应 gt_m1, pred_m2 对应 gt_m2
    iou_match1 = (single_iou(pred_m1, gt_m1) + single_iou(pred_m2, gt_m2)) / 2

    # 匹配2: pred_m1 对应 gt_m2, pred_m2 对应 gt_m1 (身份互换)
    iou_match2 = (single_iou(pred_m1, gt_m2) + single_iou(pred_m2, gt_m1)) / 2

    return max(iou_match1, iou_match2)


def evaluate_temporal_consistency(preds):
    """
    评估时序一致性

    检查相邻帧之间，同一线粒体的预测是否在空间上连续。

    Args:
        preds: (T, H, W) 预测的类别序列

    Returns:
        consistency: 0-1之间的值，1表示完美一致

    标签定义：
    - 1: M1独占区
    - 2: 交叠区
    - 3: M2独占区
    """
    T = preds.shape[0]
    consistencies = []

    for t in range(1, T):
        prev = preds[t - 1]
        curr = preds[t]

        # ========== 关键修正：正确提取线粒体区域 ==========
        # M1存在的区域 = M1独占 + 交叠
        # M2存在的区域 = M2独占 + 交叠
        if isinstance(prev, torch.Tensor):
            prev_m1 = (prev == 1) | (prev == 2)
            curr_m1 = (curr == 1) | (curr == 2)
            prev_m2 = (prev == 3) | (prev == 2)
            curr_m2 = (curr == 3) | (curr == 2)
        else:
            prev_m1 = (prev == 1) | (prev == 2)
            curr_m1 = (curr == 1) | (curr == 2)
            prev_m2 = (prev == 3) | (prev == 2)
            curr_m2 = (curr == 3) | (curr == 2)

        # 计算同一身份的重叠 vs 交换身份的重叠
        if isinstance(prev_m1, torch.Tensor):
            same = (prev_m1 & curr_m1).sum() + (prev_m2 & curr_m2).sum()
            swap = (prev_m1 & curr_m2).sum() + (prev_m2 & curr_m1).sum()
        else:
            same = (prev_m1 & curr_m1).sum() + (prev_m2 & curr_m2).sum()
            swap = (prev_m1 & curr_m2).sum() + (prev_m2 & curr_m1).sum()

        if same + swap > 0:
            if isinstance(same, torch.Tensor):
                consistencies.append((same.float() / (same + swap).float()).item())
            else:
                consistencies.append(same / (same + swap))

    return np.mean(consistencies) if consistencies else 0.0


def evaluate_batch(model, dataloader, device, criterion=None):
    """
    评估整个数据集

    Returns:
        metrics: dict containing all metrics
    """
    model.eval()

    all_ious = []
    all_consistencies = []
    total_loss = 0
    loss_components = {'seg': 0, 'temporal': 0, 'recon': 0}
    num_batches = 0

    with torch.no_grad():
        for images, labels, masks1, masks2 in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Loss
            if criterion is not None:
                loss, losses, _ = criterion(outputs, labels)
                total_loss += losses['total']
                loss_components['seg'] += losses['seg']
                loss_components['temporal'] += losses['temporal']
                loss_components['recon'] += losses['recon']

            # 预测
            preds = outputs.argmax(dim=2).cpu()

            B, T = preds.shape[:2]
            for b in range(B):
                # IoU
                sample_ious = []
                for t in range(T):
                    iou = compute_iou_with_best_perm(
                        preds[b, t], masks1[b, t], masks2[b, t]
                    )
                    sample_ious.append(iou)
                all_ious.append(np.mean(sample_ious))

                # Consistency
                cons = evaluate_temporal_consistency(preds[b])
                all_consistencies.append(cons)

            num_batches += 1

    metrics = {
        'iou': np.mean(all_ious),
        'iou_std': np.std(all_ious),
        'consistency': np.mean(all_consistencies),
        'consistency_std': np.std(all_consistencies),
    }

    if criterion is not None:
        metrics['loss'] = total_loss / num_batches
        metrics['seg_loss'] = loss_components['seg'] / num_batches
        metrics['temporal_loss'] = loss_components['temporal'] / num_batches
        metrics['recon_loss'] = loss_components['recon'] / num_batches

    return metrics