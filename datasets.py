"""
数据集定义
Dataset Definitions
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class MitoSequenceDataset(Dataset):
    """
    线粒体序列数据集

    数据格式：
    - images: (N, T, H, W) 图像序列
    - labels: (N, T, H, W) 4类标签
    - masks1: (N, T, H, W) mito1的完整mask
    - masks2: (N, T, H, W) mito2的完整mask
    """

    def __init__(self, data_path, augment=False, normalize=True, cfg=None):
        """
        Args:
            data_path: NPZ文件路径
            augment: 是否进行数据增强
            normalize: 是否归一化
            cfg: 配置对象
        """
        data = np.load(data_path)
        self.images = data['images'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        self.masks1 = data['masks1'].astype(np.float32)
        self.masks2 = data['masks2'].astype(np.float32)

        self.augment = augment
        self.normalize = normalize
        self.cfg = cfg

        print(f"加载数据集: {len(self)} 个序列, 形状: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx].copy()  # (T, H, W)
        labels = self.labels[idx].copy()  # (T, H, W)
        masks1 = self.masks1[idx].copy()  # (T, H, W)
        masks2 = self.masks2[idx].copy()  # (T, H, W)

        # 数据增强
        if self.augment:
            images, labels, masks1, masks2 = self._augment(images, labels, masks1, masks2)

        # 归一化（每帧独立）
        if self.normalize:
            images = self._normalize(images)

        # 转换为tensor
        images = torch.FloatTensor(images).unsqueeze(1)  # (T, 1, H, W)
        labels = torch.LongTensor(labels)  # (T, H, W)
        masks1 = torch.FloatTensor(masks1)  # (T, H, W)
        masks2 = torch.FloatTensor(masks2)  # (T, H, W)

        return images, labels, masks1, masks2

    def _normalize(self, images):
        """每帧独立归一化"""
        T = images.shape[0]
        for t in range(T):
            img = images[t]
            mean = img.mean()
            std = img.std()
            if std > 1e-6:
                images[t] = (img - mean) / std
            else:
                images[t] = img - mean
        return images

    def _augment(self, images, labels, masks1, masks2):
        """数据增强"""
        # 水平翻转
        if self.cfg and self.cfg.aug_flip_h and np.random.rand() > 0.5:
            images = images[:, :, ::-1].copy()
            labels = labels[:, :, ::-1].copy()
            masks1 = masks1[:, :, ::-1].copy()
            masks2 = masks2[:, :, ::-1].copy()

        # 垂直翻转
        if self.cfg and self.cfg.aug_flip_v and np.random.rand() > 0.5:
            images = images[:, ::-1, :].copy()
            labels = labels[:, ::-1, :].copy()
            masks1 = masks1[:, ::-1, :].copy()
            masks2 = masks2[:, ::-1, :].copy()

        # 90度旋转
        if self.cfg and self.cfg.aug_rotate and np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])
            images = np.rot90(images, k, axes=(1, 2)).copy()
            labels = np.rot90(labels, k, axes=(1, 2)).copy()
            masks1 = np.rot90(masks1, k, axes=(1, 2)).copy()
            masks2 = np.rot90(masks2, k, axes=(1, 2)).copy()

        # 亮度扰动
        if self.cfg and self.cfg.aug_brightness and np.random.rand() > 0.5:
            factor = np.random.uniform(*self.cfg.brightness_range)
            images = images * factor

        return images, labels, masks1, masks2


def create_dataloaders(cfg):
    """
    创建训练和验证数据加载器
    """
    train_path = cfg.processed_data_dir / "train_data.npz"
    val_path = cfg.processed_data_dir / "val_data.npz"

    train_dataset = MitoSequenceDataset(
        train_path,
        augment=cfg.augmentation,
        normalize=True,
        cfg=cfg
    )

    val_dataset = MitoSequenceDataset(
        val_path,
        augment=False,
        normalize=True,
        cfg=cfg
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    return train_loader, val_loader