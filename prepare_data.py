import os
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def prepare_npz_data():
    # 你的数据路径
    base_dir = Path(r"C:\Users\Guo_lab\Desktop\ALL_model\three_label_convlstm\data\synthetic_data\5frame_256x256")
    raw_dir = base_dir / "raw"
    label_dir = base_dir / "label"

    # 输出路径
    output_dir = Path("./data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(list(raw_dir.glob("*.tif")))

    images_list = []
    labels_list = []
    m1_list = []
    m2_list = []

    print("正在读取TIF数据并提取Mask...")
    for rf in tqdm(raw_files):
        lf = label_dir / rf.name
        if not lf.exists(): continue

        img = tifffile.imread(str(rf))  # (T, H, W)
        label = tifffile.imread(str(lf))  # (T, H, W)

        # 根据你的定义提取完整Mask:
        # 1: M1_only, 2: Overlap, 3: M2_only
        mask1 = ((label == 1) | (label == 2)).astype(np.uint8)
        mask2 = ((label == 3) | (label == 2)).astype(np.uint8)

        images_list.append(img)
        labels_list.append(label)
        m1_list.append(mask1)
        m2_list.append(mask2)

    # 转换为numpy数组 (N, T, H, W)
    images_list = np.array(images_list)
    labels_list = np.array(labels_list)
    m1_list = np.array(m1_list)
    m2_list = np.array(m2_list)

    # 划分训练集和验证集
    idx = np.arange(len(images_list))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    print(f"保存数据: 训练集 {len(train_idx)}, 验证集 {len(val_idx)}")

    np.savez(output_dir / "train_data.npz",
             images=images_list[train_idx],
             labels=labels_list[train_idx],
             masks1=m1_list[train_idx],
             masks2=m2_list[train_idx])

    np.savez(output_dir / "val_data.npz",
             images=images_list[val_idx],
             labels=labels_list[val_idx],
             masks1=m1_list[val_idx],
             masks2=m2_list[val_idx])


if __name__ == "__main__":
    prepare_npz_data()