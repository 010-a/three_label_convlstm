# 真实线粒体交叠分离训练

## 项目概述

使用真实线粒体片段合成的运动序列，训练深度学习模型实现交叠区域的分离。

### 任务定义

- **输入**: 5帧256×256的荧光显微镜图像序列
- **输出**: 每帧的4类分割图（背景、Mito1独占区、Mito2独占区、交叠区）
- **目标**: 在交叠区域正确分配像素归属

### 方法

- **模型**: ConvLSTM-UNet（双向ConvLSTM处理时序依赖）
- **Loss**: 
  - 序列级PIT Loss（允许Mito1/Mito2身份互换，但整个序列一致）
  - 时序连续Loss（相邻帧同一物体位置接近）
  - 重建Loss（前景覆盖验证）

---

## 目录结构

```
training/
├── config.py              # 配置文件
├── models.py              # 模型定义
├── datasets.py            # 数据集定义
├── losses.py              # Loss函数
├── evaluation.py          # 评估函数
├── visualization.py       # 可视化函数
├── train.py               # 训练脚本
├── predict.py             # 预测脚本
├── README.md              # 本文件
│
├── scripts/
│   └── prepare_data.py    # 数据预处理脚本
│
├── data/
│   ├── raw/               # 原始TIF文件（需自行放置）
│   └── processed/         # 处理后的NPZ文件
│
└── output/                # 训练输出
    └── run_YYYYMMDD_HHMMSS/
        ├── config.txt
        ├── best_iou_model.pth
        ├── best_consistency_model.pth
        ├── final_model.pth
        ├── history.pth
        ├── training_curves.png
        ├── predictions.png
        └── results.txt
```

---

## 快速开始

### 1. 环境准备

```bash
pip install torch torchvision numpy matplotlib tqdm tifffile scikit-image
```

### 2. 准备数据

将TIF文件放入 `data/raw/` 目录，然后运行：

```bash
python scripts/prepare_data.py --input_dir ./data/raw --output_dir ./data/processed --visualize
```

这将：
- 读取所有TIF文件
- 自动检测前景并分离两条线粒体
- 生成4类标签
- 划分训练/验证集
- 保存为NPZ格式

### 3. 训练模型

```bash
python train.py
```

训练过程会：
- 显示实时进度
- 每5个epoch验证并报告指标
- 自动保存最佳模型
- 生成训练曲线和预测可视化

### 4. 使用模型预测

```bash
python predict.py \
    --model_path ./output/run_xxx/best_iou_model.pth \
    --input_path ./data/test_data/test.tif \
    --output_dir ./predictions \
    --save_masks
```

---

## 配置说明

主要配置项在 `config.py` 中：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `img_size` | 256 | 图像尺寸 |
| `num_frames` | 5 | 每序列帧数 |
| `base_ch` | 32 | 模型基础通道数 |
| `lstm_hidden` | 64 | ConvLSTM隐藏维度 |
| `batch_size` | 4 | 批大小 |
| `epochs` | 200 | 训练轮数 |
| `lr` | 1e-3 | 学习率 |
| `seg_loss_weight` | 1.0 | 分割Loss权重 |
| `temporal_loss_weight` | 0.5 | 时序Loss权重 |
| `recon_loss_weight` | 0.2 | 重建Loss权重 |

---

## 输出说明

### 训练曲线 (training_curves.png)

- **Total Loss**: 总损失（训练+验证）
- **Segmentation Loss**: PIT分割损失
- **Temporal Loss**: 时序连续损失
- **Reconstruction Loss**: 前景重建损失
- **Validation IoU**: 验证集IoU（考虑最优排列）
- **Temporal Consistency**: 时序一致性（>0.5表示优于随机）

### 预测可视化 (predictions.png)

每个样本4行：
1. **Input**: 输入图像（对比度增强）
2. **GT Label**: 真实标签（红=M1, 蓝=M2, 紫=Overlap）
3. **Prediction**: 模型预测
4. **P(M1)-P(M2)**: 概率差热图（用于分析模型确信度）

---

## 评估指标

### IoU (Intersection over Union)

由于使用PIT，会计算两种Mito1/Mito2匹配方式的IoU，取较大值。

### Temporal Consistency

衡量模型在相邻帧之间是否保持一致的Mito1/Mito2分配：
- **= 0.5**: 随机分配（模型未学会时序一致性）
- **> 0.8**: 良好的时序一致性
- **= 1.0**: 完美一致

---

## 注意事项

1. **数据质量**: 确保TIF文件是5帧序列，每帧包含两条可分离的线粒体
2. **GPU内存**: 256×256图像建议batch_size≤4（8GB GPU）
3. **训练时间**: 约1-2小时（200 epochs，2000个序列）
4. **收敛判断**: 观察IoU稳定且Consistency>0.5

---

## 常见问题

**Q: Temporal Consistency始终约为0.5？**

A: 模型可能没有学会利用时序信息。尝试：
- 增加temporal_loss_weight
- 增加temporal_dilate_kernel
- 检查数据是否有足够的运动

**Q: IoU很高但分离效果不好？**

A: 可能是交叠区太小。检查数据中交叠区的比例。

**Q: 训练loss不下降？**

A: 检查数据预处理是否正确，特别是标签生成。

---

## 版本历史

- v1.0: 初始版本，支持序列级PIT + 时序约束