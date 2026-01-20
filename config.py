"""
真实线粒体数据训练配置
Real Mitochondria Overlap Separation Training Configuration

数据：256x256 5帧 TIF序列（真实线粒体片段合成）
模型：ConvLSTM-UNet
Loss：序列级PIT + 时序连续约束
"""

from pathlib import Path
import torch


class Config:
    # ==================== 实验信息 ====================
    exp_name = "real_mito_separation"
    exp_version = "v1.0"
    description = """
    使用真实线粒体片段合成的运动序列训练分离模型。
    目标：学习在交叠区分离两条线粒体。
    """

    # ==================== 路径配置 ====================
    # 原始数据目录（存放TIF文件）
    raw_data_dir = Path(r"C:\Users\Guo_lab\Desktop\ALL_model\three_label_convlstm\data\synthetic_data\5frame_256x256")

    # 处理后数据目录
    processed_data_dir = Path("./data/processed")

    # 输出目录
    output_dir = Path("./output")

    # ==================== 数据参数 ====================
    img_size = 256  # 图像尺寸
    num_frames = 5  # 每个序列的帧数
    num_classes = 4  # 0=背景, 1=mito1_only, 2=mito2_only, 3=overlap

    # 数据划分
    train_ratio = 0.8  # 训练集比例
    val_ratio = 0.2  # 验证集比例

    # 前景检测阈值（用于自动生成标签）
    foreground_threshold = 0.1  # 相对于最大值的比例

    # ==================== 数据增强 ====================
    augmentation = True
    aug_flip_h = True  # 水平翻转
    aug_flip_v = True  # 垂直翻转
    aug_rotate = True  # 90度旋转
    aug_brightness = True  # 亮度扰动
    brightness_range = (0.7, 1.3)

    # ==================== 模型参数 ====================
    model_type = "ConvLSTM-UNet"
    in_channels = 1
    base_ch = 32  # 基础通道数
    lstm_hidden = 64  # ConvLSTM隐藏层维度
    bidirectional = True  # 双向LSTM

    # ==================== Loss参数 ====================
    # 序列级PIT分割Loss
    seg_loss_weight = 1.0
    seg_loss_type = "mse"  # "mse" or "dice"

    # 时序连续Loss
    temporal_loss_weight = 1.0
    temporal_dilate_kernel = 7

    # 重建Loss
    recon_loss_weight = 1.0

    # ==================== 训练参数 ====================
    batch_size = 2  # 256x256图像，batch稍小
    epochs = 1
    lr = 1e-3
    lr_min = 1e-6
    weight_decay = 1e-4

    # 学习率调度
    scheduler = "cosine"  # "cosine" or "step"

    # 验证和保存
    val_interval = 1  # 每N个epoch验证一次
    save_interval = 50  # 每N个epoch保存一次checkpoint
    early_stop_patience = 50  # 早停耐心值

    # ==================== 其他 ====================
    num_workers = 0
    pin_memory = True
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== 可视化 ====================
    vis_num_samples = 6  # 可视化样本数
    vis_dpi = 150  # 图像DPI

    @classmethod
    def print_config(cls):
        print("=" * 70)
        print(f"  实验名称: {cls.exp_name} ({cls.exp_version})")
        print("=" * 70)
        print(f"  数据: {cls.img_size}x{cls.img_size}, {cls.num_frames}帧")
        print(f"  模型: {cls.model_type}, base_ch={cls.base_ch}, lstm_hidden={cls.lstm_hidden}")
        print(
            f"  Loss权重: seg={cls.seg_loss_weight}, temporal={cls.temporal_loss_weight}, recon={cls.recon_loss_weight}")
        print(f"  训练: batch={cls.batch_size}, epochs={cls.epochs}, lr={cls.lr}")
        print(f"  设备: {cls.device}")
        print("=" * 70)

    @classmethod
    def save_config(cls, path):
        """保存配置到文件"""
        # 修复1：添加 encoding='utf-8' 解决中文编码写入失败问题【核心修复】
        # 修复2：添加 errors='ignore' 兜底，忽略极少数无法编码的特殊字符
        # 优化3：添加 newline='\n' 统一跨平台换行符，Windows/Linux都兼容
        with open(path, 'w', encoding='utf-8', errors='ignore', newline='\n') as f:
            f.write(f"# {cls.exp_name} Configuration\n")
            f.write(f"# Version: {cls.exp_version}\n\n")
            # 修复4：优化遍历逻辑，兼容pathlib.Path对象的序列化，防止写入异常
            for key, value in vars(cls).items():
                if not key.startswith('_') and not callable(value):
                    # 对Path路径对象做特殊处理，转为字符串，保证写入格式整洁
                    if isinstance(value, Path):
                        f.write(f"{key} = Path(r'{value}')\n")
                    else:
                        f.write(f"{key} = {repr(value)}\n")