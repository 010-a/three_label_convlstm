"""
数据处理脚本
Data Preprocessing Script

功能：
1. 读取TIF序列文件
2. 自动检测前景，分离两条线粒体
3. 生成4类标签（背景、mito1_only、mito2_only、overlap）
4. 划分训练/验证集
5. 保存为NPZ格式

使用方法：
    python scripts/prepare_data.py --input_dir ./data/raw --output_dir ./data/processed

输入数据格式：
    每个TIF文件包含5帧，每帧256x256
    文件命名：任意，程序会自动处理所有.tif文件
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

try:
    from skimage import io as skio
    from skimage import filters, morphology, measure

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: skimage not installed, using basic processing")

try:
    import tifffile

    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def load_tif_sequence(filepath):
    """
    加载TIF序列
    返回: (T, H, W) 数组
    """
    if HAS_TIFFFILE:
        data = tifffile.imread(filepath)
    elif HAS_SKIMAGE:
        data = skio.imread(filepath)
    else:
        raise ImportError("需要安装 tifffile 或 skimage: pip install tifffile scikit-image")

    # 确保是3D数组 (T, H, W)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim == 4:
        # 可能是 (T, H, W, C) 格式，取第一个通道
        data = data[..., 0]

    return data.astype(np.float32)


def normalize_sequence(images):
    """
    归一化序列到 [0, 1]
    """
    min_val = images.min()
    max_val = images.max()
    if max_val - min_val > 1e-6:
        images = (images - min_val) / (max_val - min_val)
    return images


def detect_foreground(image, threshold_ratio=0.1):
    """
    检测前景区域
    """
    threshold = image.max() * threshold_ratio
    foreground = image > threshold

    # 形态学清理
    if HAS_SKIMAGE:
        foreground = morphology.remove_small_objects(foreground, min_size=50)
        foreground = morphology.binary_closing(foreground, morphology.disk(2))

    return foreground


def separate_two_mitos(foreground_mask, image):
    """
    分离两条线粒体

    策略：
    1. 找到所有连通区域
    2. 如果有2个或以上区域，取最大的两个
    3. 如果只有1个区域，尝试用形态学分离
    4. 根据位置或大小分配为mito1和mito2

    返回: mask1, mask2 (两个二值mask)
    """
    if HAS_SKIMAGE:
        labeled = measure.label(foreground_mask)
        regions = measure.regionprops(labeled)
    else:
        # 简化版本：直接返回整个前景作为一个mito
        return foreground_mask, np.zeros_like(foreground_mask)

    if len(regions) == 0:
        return np.zeros_like(foreground_mask), np.zeros_like(foreground_mask)

    if len(regions) == 1:
        # 只有一个连通区域，可能是交叠
        # 尝试用骨架或分水岭分离
        # 简化处理：整体作为交叠区
        mask = (labeled == 1)
        return mask, mask  # 两个mask相同表示完全交叠

    # 按面积排序，取最大的两个
    regions = sorted(regions, key=lambda r: r.area, reverse=True)[:2]

    mask1 = (labeled == regions[0].label)
    mask2 = (labeled == regions[1].label)

    # 根据质心位置排序（确保一致性）
    centroid1 = regions[0].centroid
    centroid2 = regions[1].centroid

    # 用x坐标排序：左边的是mito1
    if centroid1[1] > centroid2[1]:
        mask1, mask2 = mask2, mask1

    return mask1, mask2


def generate_labels_for_sequence(images, threshold_ratio=0.1):
    """
    为整个序列生成标签

    策略：
    1. 对首帧和尾帧（通常分离）进行连通区域分析
    2. 根据位置追踪确定mito1和mito2的身份
    3. 生成4类标签

    返回: labels (T, H, W), masks1 (T, H, W), masks2 (T, H, W)
    """
    T, H, W = images.shape
    labels = np.zeros((T, H, W), dtype=np.int64)
    masks1 = np.zeros((T, H, W), dtype=np.float32)
    masks2 = np.zeros((T, H, W), dtype=np.float32)

    # 首先分析首帧，确定mito1和mito2
    first_fg = detect_foreground(images[0], threshold_ratio)
    m1_first, m2_first = separate_two_mitos(first_fg, images[0])

    # 计算首帧的质心作为参考
    if m1_first.sum() > 0:
        y1, x1 = np.where(m1_first)
        ref_center1 = (y1.mean(), x1.mean())
    else:
        ref_center1 = (H / 2, W / 4)

    if m2_first.sum() > 0:
        y2, x2 = np.where(m2_first)
        ref_center2 = (y2.mean(), x2.mean())
    else:
        ref_center2 = (H / 2, 3 * W / 4)

    # 处理每一帧
    for t in range(T):
        fg = detect_foreground(images[t], threshold_ratio)
        m1, m2 = separate_two_mitos(fg, images[t])

        # 如果两个mask相同（完全交叠）
        if np.array_equal(m1, m2):
            # 整个区域标记为overlap
            masks1[t] = m1.astype(np.float32)
            masks2[t] = m2.astype(np.float32)
        else:
            # 根据与参考质心的距离分配身份
            if m1.sum() > 0:
                y1, x1 = np.where(m1)
                center1 = (y1.mean(), x1.mean())
            else:
                center1 = (0, 0)

            if m2.sum() > 0:
                y2, x2 = np.where(m2)
                center2 = (y2.mean(), x2.mean())
            else:
                center2 = (0, 0)

            # 计算到参考点的距离
            d1_to_ref1 = np.sqrt((center1[0] - ref_center1[0]) ** 2 + (center1[1] - ref_center1[1]) ** 2)
            d1_to_ref2 = np.sqrt((center1[0] - ref_center2[0]) ** 2 + (center1[1] - ref_center2[1]) ** 2)
            d2_to_ref1 = np.sqrt((center2[0] - ref_center1[0]) ** 2 + (center2[1] - ref_center1[1]) ** 2)
            d2_to_ref2 = np.sqrt((center2[0] - ref_center2[0]) ** 2 + (center2[1] - ref_center2[1]) ** 2)

            # 分配：总距离最小的方案
            if d1_to_ref1 + d2_to_ref2 <= d1_to_ref2 + d2_to_ref1:
                masks1[t] = m1.astype(np.float32)
                masks2[t] = m2.astype(np.float32)
            else:
                masks1[t] = m2.astype(np.float32)
                masks2[t] = m1.astype(np.float32)

            # 更新参考质心（用于追踪）
            if masks1[t].sum() > 0:
                y, x = np.where(masks1[t] > 0.5)
                ref_center1 = (y.mean(), x.mean())
            if masks2[t].sum() > 0:
                y, x = np.where(masks2[t] > 0.5)
                ref_center2 = (y.mean(), x.mean())

        # 生成4类标签
        m1_only = (masks1[t] > 0.5) & (masks2[t] <= 0.5)
        m2_only = (masks1[t] <= 0.5) & (masks2[t] > 0.5)
        overlap = (masks1[t] > 0.5) & (masks2[t] > 0.5)

        labels[t][m1_only] = 1
        labels[t][m2_only] = 2
        labels[t][overlap] = 3

    return labels, masks1, masks2


def process_all_tifs(input_dir, output_dir, threshold_ratio=0.1, train_ratio=0.8):
    """
    处理所有TIF文件
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有TIF文件
    tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))

    if len(tif_files) == 0:
        print(f"错误：在 {input_dir} 中没有找到TIF文件")
        return

    print(f"找到 {len(tif_files)} 个TIF文件")

    all_images = []
    all_labels = []
    all_masks1 = []
    all_masks2 = []

    for tif_path in tqdm(tif_files, desc="处理TIF文件"):
        try:
            # 加载
            images = load_tif_sequence(tif_path)

            # 归一化
            images = normalize_sequence(images)

            # 检查尺寸
            T, H, W = images.shape
            if T != 5:
                print(f"  警告: {tif_path.name} 有 {T} 帧（期望5帧），跳过")
                continue

            # 生成标签
            labels, masks1, masks2 = generate_labels_for_sequence(images, threshold_ratio)

            all_images.append(images)
            all_labels.append(labels)
            all_masks1.append(masks1)
            all_masks2.append(masks2)

        except Exception as e:
            print(f"  错误处理 {tif_path.name}: {e}")
            continue

    if len(all_images) == 0:
        print("错误：没有成功处理任何文件")
        return

    # 转换为数组
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    all_masks1 = np.array(all_masks1)
    all_masks2 = np.array(all_masks2)

    print(f"\n成功处理 {len(all_images)} 个序列")
    print(f"  图像形状: {all_images.shape}")
    print(f"  标签形状: {all_labels.shape}")

    # 打乱并划分
    np.random.seed(42)
    indices = np.random.permutation(len(all_images))
    n_train = int(len(indices) * train_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # 保存训练集
    train_path = output_dir / "train_data.npz"
    np.savez_compressed(
        train_path,
        images=all_images[train_idx],
        labels=all_labels[train_idx],
        masks1=all_masks1[train_idx],
        masks2=all_masks2[train_idx],
    )
    print(f"\n训练集: {len(train_idx)} 个序列 -> {train_path}")

    # 保存验证集
    val_path = output_dir / "val_data.npz"
    np.savez_compressed(
        val_path,
        images=all_images[val_idx],
        labels=all_labels[val_idx],
        masks1=all_masks1[val_idx],
        masks2=all_masks2[val_idx],
    )
    print(f"验证集: {len(val_idx)} 个序列 -> {val_path}")

    # 保存数据统计信息
    stats_path = output_dir / "data_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("数据处理统计\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"原始TIF文件数: {len(tif_files)}\n")
        f.write(f"成功处理数: {len(all_images)}\n")
        f.write(f"图像尺寸: {all_images.shape[2]}x{all_images.shape[3]}\n")
        f.write(f"帧数: {all_images.shape[1]}\n\n")
        f.write(f"训练集: {len(train_idx)} 序列\n")
        f.write(f"验证集: {len(val_idx)} 序列\n\n")

        # 类别统计
        f.write("类别分布:\n")
        for c in range(4):
            count = (all_labels == c).sum()
            ratio = count / all_labels.size * 100
            class_name = ["背景", "Mito1_only", "Mito2_only", "Overlap"][c]
            f.write(f"  {class_name}: {count} pixels ({ratio:.2f}%)\n")

    print(f"统计信息 -> {stats_path}")

    return train_path, val_path


def visualize_samples(data_path, output_path, num_samples=4):
    """
    可视化处理后的样本
    """
    import matplotlib.pyplot as plt

    data = np.load(data_path)
    images = data['images']
    labels = data['labels']
    masks1 = data['masks1']
    masks2 = data['masks2']

    num_samples = min(num_samples, len(images))
    T = images.shape[1]

    colors = np.array([
        [0.1, 0.1, 0.1],  # 背景
        [1.0, 0.2, 0.2],  # mito1
        [0.2, 0.4, 1.0],  # mito2
        [0.8, 0.2, 0.8],  # overlap
    ])

    fig, axes = plt.subplots(num_samples * 2, T, figsize=(T * 3, num_samples * 5))

    for s in range(num_samples):
        for t in range(T):
            # 图像
            img = images[s, t]
            vmin, vmax = np.percentile(img, [2, 98])
            axes[s * 2, t].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            axes[s * 2, t].axis('off')
            if t == 0:
                axes[s * 2, t].set_ylabel(f'Sample {s + 1}\nImage', fontsize=10)
            if s == 0:
                axes[s * 2, t].set_title(f'Frame {t}', fontsize=11)

            # 标签
            label_colored = colors[labels[s, t]]
            axes[s * 2 + 1, t].imshow(label_colored)
            axes[s * 2 + 1, t].axis('off')
            if t == 0:
                axes[s * 2 + 1, t].set_ylabel('Label', fontsize=10)

    # 图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], label='Background'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], label='Mito1 only'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[2], label='Mito2 only'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[3], label='Overlap'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化 -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理TIF数据")
    parser.add_argument("--input_dir", type=str, default="./data/raw",
                        help="TIF文件目录")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                        help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="前景检测阈值")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--visualize", action="store_true",
                        help="生成可视化")

    args = parser.parse_args()

    train_path, val_path = process_all_tifs(
        args.input_dir,
        args.output_dir,
        args.threshold,
        args.train_ratio
    )

    if args.visualize and train_path:
        vis_path = Path(args.output_dir) / "data_preview.png"
        visualize_samples(train_path, vis_path)