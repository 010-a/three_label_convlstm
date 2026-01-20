"""
批量预测脚本 (Batch Prediction Script)

功能：
1. 自动遍历指定文件夹下的所有TIF图像。
2. 自动匹配对应的标签文件（如果存在）。
3. 生成科研级可视化对比图。
4. 保存预测结果 Mask。
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# ====================== 【修复1：新增2行核心代码 解决torch.load报错 必须加】 ======================
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

# 尝试导入本地配置，如果没有则定义默认值
try:
    from config import Config
    from models import ConvLSTMUNet
    # 假设 visualization.py 里有这些，如果没有，下文有兜底定义
    from visualization import COLORS, CLASS_NAMES
except ImportError:
    # 如果缺少某些文件，可以在这里定义简单的兜底配置
    print("Warning: Local modules not found, using default configurations.")

    class Config:
        in_channels = 1
        num_classes = 4
        base_ch = 32
        lstm_hidden = 32
        bidirectional = True
        device = 'cuda'

    # 兜底颜色定义 (Bg, M1, Both, M2)
    COLORS = np.array([
        [0, 0, 0],  # Background
        [255, 50, 50],  # M1 (Red)
        [50, 255, 50],  # Both (Green)
        [50, 50, 255]  # M2 (Blue)
    ]) / 255.0
    CLASS_NAMES = ['BG', 'M1', 'Both', 'M2']

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    from skimage import io as skio


def load_model(model_path, device, cfg):
    """加载模型"""
    model = ConvLSTMUNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_ch=cfg.base_ch,
        lstm_hidden=cfg.lstm_hidden,
        bidirectional=cfg.bidirectional
    ).to(device)

    # ====================== 【修复2：torch.load 添加 weights_only=False 解决报错】 ======================
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # 兼容只保存了state_dict的情况

    model.eval()
    print(f"✓ 模型加载成功: {Path(model_path).name}")
    if 'epoch' in checkpoint:
        print(f"  - 训练轮数: {checkpoint['epoch']}")
    if 'best_iou' in checkpoint:
        print(f"  - 最优IoU: {checkpoint['best_iou']:.4f}")
    return model


def load_tif(filepath):
    """加载TIF文件"""
    filepath = str(filepath)
    if HAS_TIFFFILE:
        data = tifffile.imread(filepath)
    else:
        data = skio.imread(filepath)

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    return data.astype(np.float32)


def normalize(images):
    """归一化"""
    images_norm = images.copy()
    T = images_norm.shape[0]
    for t in range(T):
        img = images_norm[t]
        mean, std = img.mean(), img.std()
        if std > 1e-6:
            images_norm[t] = (img - mean) / std
        else:
            images_norm[t] = img - mean
    return images_norm


def predict(model, images, device):
    """预测"""
    images_norm = normalize(images)
    # 转换为tensor (1, T, 1, H, W)
    x = torch.FloatTensor(images_norm).unsqueeze(0).unsqueeze(2).to(device)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=2)[0].cpu().numpy()
        prob = torch.softmax(output, dim=2)[0].cpu().numpy()

    return pred, prob


def visualize_result_publication(images, pred, prob, output_path, gt_labels=None):
    """
    可视化结果 - 科研论文级排版
    Input -> GT (Optional) -> Pred -> Prob
    """
    T = images.shape[0]

    # 定义绘图配置
    plot_configs = [
        {'name': 'Input Image', 'data': images, 'type': 'gray'},
        {'name': 'Ground Truth', 'data': gt_labels, 'type': 'seg'},
        {'name': 'Prediction', 'data': pred, 'type': 'seg'},
        {'name': 'Prob Diff\n(M1 - M2)', 'data': prob, 'type': 'heatmap'}
    ]

    # 过滤掉无数据的行
    valid_rows = [cfg for cfg in plot_configs if cfg['data'] is not None]
    n_rows = len(valid_rows)

    # 字体设置
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # 创建画布
    fig, axes = plt.subplots(n_rows, T, figsize=(T * 2.5 + 1.5, n_rows * 2.6), constrained_layout=True)

    if n_rows == 1: axes = axes[np.newaxis, :]
    if T == 1: axes = axes[:, np.newaxis]

    for row_idx, config in enumerate(valid_rows):
        row_type = config['type']
        row_name = config['name']
        data = config['data']

        for t in range(T):
            ax = axes[row_idx, t]

            if row_type == 'gray':
                img = data[t]
                vmin, vmax = np.percentile(img, [1, 99])
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, aspect='equal')
                if row_idx == 0:
                    ax.set_title(f'Frame {t}', fontweight='bold', pad=10)

            elif row_type == 'seg':
                ax.imshow(COLORS[data[t]], aspect='equal', interpolation='nearest')

            elif row_type == 'heatmap':
                # ====================== 【修复3：修复概率差计算BUG 你的核心需求 P(M1)-P(M2)】 ======================
                p = data[t]
                prob_m1 = p[1] + p[2]  # M1 = 纯M1 + 交叠区
                prob_m2 = p[3] + p[2]  # M2 = 纯M2 + 交叠区
                prob_val = prob_m1 - prob_m2
                im = ax.imshow(prob_val, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

                if t == T - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    cbar = fig.colorbar(im, cax=cax)
                    cbar.set_ticks([-1, 0, 1])
                    cbar.set_label('Confidence', rotation=270, labelpad=12, fontsize=10)
                    cbar.ax.tick_params(labelsize=9)

            ax.axis('off')
            if t == 0:
                ax.text(-0.1, 0.5, row_name, transform=ax.transAxes,
                        fontsize=13, fontweight='bold', va='center', ha='right')

    # 图例
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i], edgecolor='0.2', linewidth=0.5, label=label)
        for i, label in enumerate(CLASS_NAMES)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(CLASS_NAMES),
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"  ✔ 可视化保存: {output_path.name}")


def save_masks(pred, output_dir, filename_base):
    """保存分割结果 + 补充拆分线粒体mask"""
    output_dir = Path(output_dir)

    # 保存完整预测
    if HAS_TIFFFILE:
        tifffile.imwrite(output_dir / f"{filename_base}_pred.tif", pred.astype(np.uint8))
        # 补充保存：线粒体1、线粒体2、交叠区 mask (你的需求)
        mito1_mask = ((pred == 1) | (pred == 2)).astype(np.uint8)*255
        mito2_mask = ((pred == 3) | (pred == 2)).astype(np.uint8)*255
        overlap_mask = (pred == 2).astype(np.uint8)*255
        tifffile.imwrite(output_dir / f"{filename_base}_mito1.tif", mito1_mask)
        tifffile.imwrite(output_dir / f"{filename_base}_mito2.tif", mito2_mask)
        tifffile.imwrite(output_dir / f"{filename_base}_overlap.tif", overlap_mask)
    else:
        np.save(output_dir / f"{filename_base}_pred.npy", pred)

    print(f"  ✔ Mask保存完成")


def main():
    parser = argparse.ArgumentParser(description="批量预测脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径 (.pth)")
    parser.add_argument("--input_dir", type=str, required=True, help="包含原始TIF图像的文件夹路径")
    parser.add_argument("--label_dir", type=str, default=None, help="包含GT标签的文件夹路径 (可选)")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="结果保存目录")
    parser.add_argument("--no_gpu", action="store_true", help="强制使用CPU")

    args = parser.parse_args()

    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    print(f"Using device: {device}")

    # 2. 准备路径
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dir = Path(args.label_dir) if args.label_dir else None

    # 3. 加载模型
    cfg = Config()
    model = load_model(args.model_path, device, cfg)

    # 4. 获取文件列表
    # 支持 .tif and .tiff
    tif_files = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))

    if len(tif_files) == 0:
        print(f"错误: 在 {input_dir} 中未找到 .tif 文件")
        return

    print(f"\n找到 {len(tif_files)} 个文件，开始批量处理...\n")

    # 5. 批量处理循环
    for idx, img_path in enumerate(tif_files):
        print(f"[{idx + 1}/{len(tif_files)}] Processing: {img_path.name}")

        # 加载图像
        try:
            images = load_tif(img_path)
            print(f"  - 图像尺寸: {images.shape} (T, H, W)")
        except Exception as e:
            print(f"  ✗ 读取失败 {img_path.name}: {e}")
            continue

        # 加载对应的标签 (如果存在)
        gt_labels = None
        if label_dir:
            gt_path = label_dir / img_path.name
            if gt_path.exists():
                try:
                    gt_labels = load_tif(gt_path).astype(np.uint8)
                    if gt_labels.ndim == 4: gt_labels = gt_labels[:, 0, :, :]
                except Exception as e:
                    print(f"  ! 标签加载失败 {gt_path.name}: {e}")
            else:
                print(f"  ! 未找到对应标签 {gt_path.name}")

        # 预测
        pred, prob = predict(model, images, device)

        # 保存图片路径
        filename_base = img_path.stem
        vis_save_path = output_dir / f"{filename_base}_vis.png"

        # 可视化
        visualize_result_publication(images, pred, prob, vis_save_path, gt_labels)

        # 保存Mask数据
        save_masks(pred, output_dir, filename_base)

    print(f"\n✅ 全部完成! 结果保存在: {output_dir.absolute()}")


if __name__ == "__main__":
    main()