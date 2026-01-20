"""
Loss函数定义 (修正版)
Loss Functions - Corrected for 3-class labels

标签格式：
- 0: 背景
- 1: 线粒体1独占区 (M1 only)
- 2: 交叠区 (Overlap) - 两条线粒体共有
- 3: 线粒体2独占区 (M2 only)

完整mask定义：
- mask1 (M1的amodal mask) = (label == 1) | (label == 2)
- mask2 (M2的amodal mask) = (label == 3) | (label == 2)

包含：
1. 序列级PIT Loss（Permutation Invariant Training）
2. 时序连续Loss
3. 重建Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequencePITLoss(nn.Module):
    """
    序列级排列不变Loss

    整个序列只选一个最优排列（M1/M2互换），而不是逐帧选择。
    这保证了帧间的身份一致性。

    交换逻辑：
    - 原始: 1=M1_only, 2=Overlap, 3=M2_only
    - 交换: 1→3, 3→1, 2保持不变（因为交叠区对两个线粒体是共有的）
    """

    def __init__(self, num_classes=4, loss_type='mse'):
        super().__init__()
        self.num_classes = num_classes
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, C, H, W) 预测logits
            target: (B, T, H, W) 标签 (0, 1, 2, 3)

        Returns:
            loss: 标量
            use_swap: (B,) 布尔tensor，表示是否使用了交换排列
        """
        B, T, C, H, W = pred.shape

        pred_softmax = torch.softmax(pred, dim=2)

        # 原始标签的one-hot
        target_onehot1 = torch.zeros_like(pred_softmax)
        target_onehot1.scatter_(2, target.view(B, T, 1, H, W), 1)

        # ========== 关键修正：交换M1和M2 ==========
        # 原始: 0=bg, 1=M1_only, 2=overlap, 3=M2_only
        # 交换: 0=bg, 1=M2_only, 2=overlap, 3=M1_only
        # 即：1↔3，0和2保持不变
        target_swap = target.clone()
        mask1 = (target == 1)  # M1独占区
        mask3 = (target == 3)  # M2独占区

        target_swap[mask1] = 3  # 原来的M1独占 → 变成M2独占
        target_swap[mask3] = 1  # 原来的M2独占 → 变成M1独占
        # label 0 (背景) 和 label 2 (交叠) 保持不变

        target_onehot2 = torch.zeros_like(pred_softmax)
        target_onehot2.scatter_(2, target_swap.view(B, T, 1, H, W), 1)

        # 计算两种排列的loss（整个序列求和）
        if self.loss_type == 'mse':
            loss1 = ((pred_softmax - target_onehot1) ** 2).sum(dim=(1, 2, 3, 4))
            loss2 = ((pred_softmax - target_onehot2) ** 2).sum(dim=(1, 2, 3, 4))
        else:  # dice
            loss1 = self._dice_loss(pred_softmax, target_onehot1)
            loss2 = self._dice_loss(pred_softmax, target_onehot2)

        # 选择更小的排列
        use_swap = (loss2 < loss1)
        final_loss = torch.where(use_swap, loss2, loss1)

        return final_loss.mean(), use_swap

    def _dice_loss(self, pred, target, smooth=1e-5):
        """Dice loss"""
        intersection = (pred * target).sum(dim=(1, 2, 3, 4))
        union = pred.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice


class TemporalContinuityLoss(nn.Module):
    """
    时序连续Loss

    约束相邻帧的同一线粒体位置接近。
    使用dilate+overlap方式，不要求线性轨迹。

    线粒体概率提取（基于三分类标签定义）：
    - M1的概率 = P(class 1) + P(class 2)  # M1独占 + 交叠
    - M2的概率 = P(class 3) + P(class 2)  # M2独占 + 交叠
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, pred):
        """
        Args:
            pred: (B, T, C, H, W) 预测logits

        Returns:
            loss: 标量
        """
        B, T, C, H, W = pred.shape

        pred_softmax = torch.softmax(pred, dim=2)

        # ========== 关键修正：正确提取线粒体概率 ==========
        # 根据标签定义：
        # - Class 0: 背景
        # - Class 1: M1独占区
        # - Class 2: 交叠区（M1和M2都存在）
        # - Class 3: M2独占区
        #
        # 因此：
        # - M1存在的概率 = P(M1独占) + P(交叠) = P(class1) + P(class2)
        # - M2存在的概率 = P(M2独占) + P(交叠) = P(class3) + P(class2)
        prob_mito1 = pred_softmax[:, :, 1] + pred_softmax[:, :, 2]  # (B, T, H, W)
        prob_mito2 = pred_softmax[:, :, 3] + pred_softmax[:, :, 2]  # (B, T, H, W)

        padding = self.kernel_size // 2

        total_loss = 0
        count = 0

        for t in range(T - 1):
            for prob in [prob_mito1, prob_mito2]:
                curr = prob[:, t:t + 1]  # (B, 1, H, W)
                next_ = prob[:, t + 1:t + 2]

                # 膨胀当前帧
                curr_dilated = F.max_pool2d(curr, self.kernel_size, stride=1, padding=padding)

                # 约束：next应该在curr_dilated范围内
                outside = next_ * (1 - curr_dilated)
                loss = (outside ** 2).mean()
                total_loss += loss
                count += 1

                # 反向约束
                next_dilated = F.max_pool2d(next_, self.kernel_size, stride=1, padding=padding)
                outside_rev = curr * (1 - next_dilated)
                loss_rev = (outside_rev ** 2).mean()
                total_loss += loss_rev
                count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred.device)


class ReconstructionLoss(nn.Module):
    """
    重建Loss

    预测的前景区域应该与真实前景区域一致。
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, C, H, W) 预测logits
            target: (B, T, H, W) 标签

        Returns:
            loss: 标量
        """
        pred_softmax = torch.softmax(pred, dim=2)

        # 预测的前景 = 1 - P(background)
        pred_foreground = 1 - pred_softmax[:, :, 0]

        # 真实前景 = label > 0 (任何非背景区域)
        gt_foreground = (target > 0).float()

        # BCE loss
        loss = F.binary_cross_entropy(pred_foreground, gt_foreground)

        return loss


class CombinedLoss(nn.Module):
    """
    组合Loss
    """

    def __init__(self, cfg):
        super().__init__()

        self.seg_loss = SequencePITLoss(
            num_classes=cfg.num_classes,
            loss_type=cfg.seg_loss_type
        )
        self.temporal_loss = TemporalContinuityLoss(
            kernel_size=cfg.temporal_dilate_kernel
        )
        self.recon_loss = ReconstructionLoss()

        self.seg_weight = cfg.seg_loss_weight
        self.temporal_weight = cfg.temporal_loss_weight
        self.recon_weight = cfg.recon_loss_weight

    def forward(self, pred, target):
        """
        Returns:
            total_loss, dict of individual losses, use_swap
        """
        seg_loss, use_swap = self.seg_loss(pred, target)
        temporal_loss = self.temporal_loss(pred)
        recon_loss = self.recon_loss(pred, target)

        total_loss = (
                self.seg_weight * seg_loss +
                self.temporal_weight * temporal_loss +
                self.recon_weight * recon_loss
        )

        losses = {
            'total': total_loss.item(),
            'seg': seg_loss.item(),
            'temporal': temporal_loss.item(),
            'recon': recon_loss.item(),
        }

        return total_loss, losses, use_swap