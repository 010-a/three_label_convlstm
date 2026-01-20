"""
线粒体合成数据生成器 - 三分类标签版
Mitochondria Synthetic Data Generator - 3-class Label Version

标签格式：
- 0: 背景
- 奇数 (1, 3, 5, ...): 线粒体独占区 (mito1_only, mito2_only, ...)
- 偶数 (2, 4, 6, ...): 交叠区 (overlap_1_2, overlap_3_4, ...)

对于单对线粒体：
- 0: 背景
- 1: 线粒体1独占区
- 2: 交叠区
- 3: 线粒体2独占区

使用方法：
    python generate_synthetic_3class.py --config config_synthetic.yaml
"""

import os
import cv2
import yaml
import numpy as np
import glob
from tqdm import tqdm
import random
import tifffile
import argparse
from pathlib import Path
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import re


def load_config(config_path: str) -> dict:
    """加载YAML配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class MitoSyntheticGenerator:
    """
    线粒体合成数据生成器

    输出：
    - raw/: 合成图像序列 (TIF)
    - label/: 三分类标签序列 (TIF)
    """

    def __init__(self, config: dict):
        self.cfg = config

        # 基础参数
        self.T = config['sequence']['num_frames']
        self.size = config['sequence']['canvas_size']
        self.h, self.w = self.size, self.size
        self.bit_depth = config['sequence']['bit_depth']
        self.contact_frame = config['sequence']['contact_frame']

        # 设置随机种子
        seed = config['generation'].get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)

        # 加载资源
        self.actors, self.masks, self.names = self._load_resources(
            config['paths']['actors_dir'],
            config['paths']['masks_dir']
        )

        if len(self.actors) < 2:
            raise ValueError(f"需要至少2个线粒体样本，当前只有 {len(self.actors)} 个")

        print(f"=" * 60)
        print(f"线粒体合成数据生成器")
        print(f"=" * 60)
        print(f"✓ 加载 {len(self.actors)} 个线粒体样本")
        print(f"✓ 配置: {self.T}帧, {self.h}×{self.w}, {self.bit_depth}bit")
        print(f"✓ 交叠帧: 第{self.contact_frame}帧 (0-indexed)")
        print(f"=" * 60)

    def _load_resources(self, actors_dir: str, masks_dir: str) -> Tuple[List, List, List]:
        """加载图像和掩码"""
        # 查找所有图像文件
        actor_files = []
        for ext in ['*.tif', '*.tiff', '*.png']:
            actor_files.extend(glob.glob(os.path.join(actors_dir, ext)))
        actor_files = sorted(actor_files)

        actors = []
        masks = []
        names = []

        for f in actor_files:
            basename = os.path.basename(f)
            name = os.path.splitext(basename)[0]

            # 加载actor
            actor = tifffile.imread(f) if f.endswith(('.tif', '.tiff')) else cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if actor is None:
                print(f"  警告: 无法加载 {f}")
                continue

            # 处理多帧图像（取第一帧）
            if actor.ndim == 3:
                actor = actor[0]

            # 查找对应的mask
            # 尝试匹配 raw1 -> label1 的命名模式
            mask_name = name.replace('raw', 'label')
            mask_path = None

            for ext in ['.tif', '.tiff', '.png']:
                candidate = os.path.join(masks_dir, mask_name + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is None:
                # 如果没找到，尝试直接使用相同名称
                for ext in ['.tif', '.tiff', '.png']:
                    candidate = os.path.join(masks_dir, name + ext)
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break

            if mask_path:
                mask = tifffile.imread(mask_path) if mask_path.endswith(('.tif', '.tiff')) else cv2.imread(mask_path,
                                                                                                           cv2.IMREAD_GRAYSCALE)
                if mask.ndim == 3:
                    mask = mask[0]
                # 二值化
                mask = (mask > 0).astype(np.uint8) * 255
            else:
                # 如果没有mask，用阈值生成
                print(f"  警告: 未找到 {name} 的mask，使用阈值生成")
                threshold = actor.max() * 0.1
                mask = (actor > threshold).astype(np.uint8) * 255

            actors.append(actor)
            masks.append(mask)
            names.append(name)

        return actors, masks, names

    def _get_random_sample(self, exclude_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, str, int]:
        """随机获取一个样本"""
        available = list(range(len(self.actors)))
        if exclude_idx is not None and len(available) > 1:
            available.remove(exclude_idx)

        idx = random.choice(available)
        return self.actors[idx].copy(), self.masks[idx].copy(), self.names[idx], idx

    def _rotate_image(self, image: np.ndarray, mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """旋转图像和掩码"""
        if abs(angle) < 0.1:
            return image, mask

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算新尺寸
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # 旋转
        if image.dtype == np.uint16:
            rot_img = cv2.warpAffine(image.astype(np.float32), M, (new_w, new_h)).astype(np.uint16)
        else:
            rot_img = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
        rot_mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST)

        return rot_img, rot_mask

    def _sample_collision_scenario(self) -> dict:
        """采样碰撞场景"""
        margin = 60

        # 碰撞点位置
        col_x = random.randint(margin, self.w - margin)
        col_y = random.randint(margin, self.h - margin)

        # 角度模式
        angle_weights = self.cfg['motion']['angle_mode_weights']
        modes = list(angle_weights.keys())
        probs = [angle_weights[m] for m in modes]
        mode = np.random.choice(modes, p=probs)

        angle_a = random.uniform(0, 2 * np.pi)

        if mode == 'symmetric':
            angle_b = angle_a + np.pi + random.uniform(-0.3, 0.3)
        elif mode == 'perpendicular':
            angle_b = angle_a + random.choice([-1, 1]) * np.pi / 2 + random.uniform(-0.2, 0.2)
        elif mode == 'oblique':
            angle_b = angle_a + random.choice([-1, 1]) * np.pi / 4 + random.uniform(-0.2, 0.2)
        elif mode == 'chase':
            angle_b = angle_a + random.uniform(-0.4, 0.4)
        else:
            angle_b = random.uniform(0, 2 * np.pi)

        # 轨迹类型
        traj_weights = self.cfg['motion']['trajectory_weights']
        traj_types = list(traj_weights.keys())
        traj_probs = [traj_weights[t] for t in traj_types]
        motion_a = np.random.choice(traj_types, p=traj_probs)
        motion_b = np.random.choice(traj_types, p=traj_probs)

        # 速度
        speed_range = self.cfg['motion']['speed_range']
        speed_a = random.uniform(*speed_range)
        speed_b = random.uniform(*speed_range)

        return {
            'collision_pos': (col_x, col_y),
            'angle_a': angle_a,
            'angle_b': angle_b,
            'motion_a': motion_a,
            'motion_b': motion_b,
            'speed_a': speed_a,
            'speed_b': speed_b,
            'angle_mode': mode,
        }

    def _generate_trajectory(self, collision_pos: Tuple[int, int], angle: float,
                             speed: float, motion_type: str) -> List[Tuple[int, int]]:
        """生成轨迹"""
        trajectory = []

        # 从碰撞点倒推起始位置
        dist_before = speed * self.contact_frame
        start_x = collision_pos[0] - dist_before * np.cos(angle)
        start_y = collision_pos[1] - dist_before * np.sin(angle)

        # 曲线参数
        if motion_type == 'curved':
            curve_offset = random.uniform(15, 40) * random.choice([-1, 1])
            perp_angle = angle + np.pi / 2
        elif motion_type == 'wobble':
            wobble_freq = random.uniform(0.6, 1.2)
            wobble_amp = random.uniform(3, 8)

        jitter = self.cfg['optical'].get('position_jitter', 1.0)

        for t in range(self.T):
            # 基础线性位置
            x = start_x + speed * t * np.cos(angle)
            y = start_y + speed * t * np.sin(angle)

            # 非线性扰动
            if motion_type == 'curved':
                norm_t = t / max(self.T - 1, 1)
                parabola = 4 * norm_t * (1 - norm_t)
                x += curve_offset * np.cos(perp_angle) * parabola
                y += curve_offset * np.sin(perp_angle) * parabola

            elif motion_type == 'wobble':
                offset = np.sin(t * wobble_freq * np.pi) * wobble_amp
                perp_angle = angle + np.pi / 2
                x += offset * np.cos(perp_angle)
                y += offset * np.sin(perp_angle)

            # 位置抖动
            x += np.random.normal(0, jitter)
            y += np.random.normal(0, jitter)

            trajectory.append((int(x), int(y)))

        return trajectory

    def _paste_to_canvas(self, canvas: np.ndarray, patch: np.ndarray,
                         center: Tuple[int, int], mode: str = 'add') -> None:
        """将patch粘贴到canvas"""
        ph, pw = patch.shape[:2]
        cx, cy = center

        x1, y1 = cx - pw // 2, cy - ph // 2
        x2, y2 = x1 + pw, y1 + ph

        c_x1, c_y1 = max(0, x1), max(0, y1)
        c_x2, c_y2 = min(self.w, x2), min(self.h, y2)

        if c_x2 <= c_x1 or c_y2 <= c_y1:
            return

        p_x1, p_y1 = c_x1 - x1, c_y1 - y1
        p_x2, p_y2 = p_x1 + (c_x2 - c_x1), p_y1 + (c_y2 - c_y1)

        patch_crop = patch[p_y1:p_y2, p_x1:p_x2]

        if mode == 'add':
            max_val = 65535 if canvas.dtype == np.uint16 else 255
            canvas[c_y1:c_y2, c_x1:c_x2] = np.clip(
                canvas[c_y1:c_y2, c_x1:c_x2].astype(np.float64) + patch_crop.astype(np.float64),
                0, max_val
            ).astype(canvas.dtype)
        elif mode == 'replace':
            mask = patch_crop > 0
            canvas[c_y1:c_y2, c_x1:c_x2][mask] = patch_crop[mask]

    def generate_sequence(self, seq_idx: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        生成一个序列 (8-bit 优化版)
        """
        # 1. 强制使用 uint8，不再判断 16-bit
        dtype = np.uint8
        images = np.zeros((self.T, self.h, self.w), dtype=dtype)
        labels = np.zeros((self.T, self.h, self.w), dtype=np.uint8)

        # 采样场景 (轨迹逻辑不变)
        scenario = self._sample_collision_scenario()
        actor1, mask1, name1, idx1 = self._get_random_sample()
        actor2, mask2, name2, idx2 = self._get_random_sample(exclude_idx=idx1)

        # 随机交换逻辑... (保持不变)
        if random.random() > 0.5:
            actor1, actor2 = actor2, actor1
            mask1, mask2 = mask2, mask1
            name1, name2 = name2, name1
            scenario['angle_a'], scenario['angle_b'] = scenario['angle_b'], scenario['angle_a']
            scenario['speed_a'], scenario['speed_b'] = scenario['speed_b'], scenario['speed_a']
            scenario['motion_a'], scenario['motion_b'] = scenario['motion_b'], scenario['motion_a']

        traj1 = self._generate_trajectory(scenario['collision_pos'], scenario['angle_a'], scenario['speed_a'],
                                          scenario['motion_a'])
        traj2 = self._generate_trajectory(scenario['collision_pos'], scenario['angle_b'], scenario['speed_b'],
                                          scenario['motion_b'])

        # 旋转和亮度衰减参数
        rot_range = self.cfg['motion']['rotation_speed_range']
        init_angle1, init_angle2 = random.uniform(0, 360), random.uniform(0, 360)
        spin1, spin2 = random.uniform(*rot_range), random.uniform(*rot_range)

        optical = self.cfg['optical']
        decay_curve = np.linspace(1.0, random.uniform(*optical['decay_range']), self.T)

        # 渲染每帧
        for t in range(self.T):
            angle1, angle2 = init_angle1 + spin1 * t, init_angle2 + spin2 * t
            rot_actor1, rot_mask1 = self._rotate_image(actor1, mask1, angle1)
            rot_actor2, rot_mask2 = self._rotate_image(actor2, mask2, angle2)

            # 2. 亮度调整：直接针对 8-bit 处理，简化掉之前的 16-bit 判断逻辑
            brightness = decay_curve[t] * random.uniform(*optical['brightness_range'])

            # 使用 float32 计算防止中间过程截断，然后直接 clip 到 255
            rot_actor1 = np.clip(rot_actor1.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
            rot_actor2 = np.clip(rot_actor2.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            # 渲染临时 mask (逻辑不变)
            temp_m1, temp_m2 = np.zeros((self.h, self.w), dtype=np.uint8), np.zeros((self.h, self.w), dtype=np.uint8)
            self._paste_to_canvas(temp_m1, rot_mask1, traj1[t], mode='replace')
            self._paste_to_canvas(temp_m2, rot_mask2, traj2[t], mode='replace')

            # 生成三分类标签 (逻辑不变)
            m1_region, m2_region = temp_m1 > 0, temp_m2 > 0
            labels[t][m1_region & (~m2_region)] = 1
            labels[t][m1_region & m2_region] = 2
            labels[t][m2_region & (~m1_region)] = 3

            # 3. 渲染图像：加性叠加
            self._paste_to_canvas(images[t], rot_actor1, traj1[t], mode='add')
            self._paste_to_canvas(images[t], rot_actor2, traj2[t], mode='add')

            # 4. 噪声处理：统一使用 255 作为基准
            noise_level = random.uniform(*optical['background_noise'])
            # 8-bit 下噪声标准差计算：noise_level * 255
            noise = np.random.normal(0, noise_level * 255, (self.h, self.w))

            # 叠加噪声并转回 uint8
            images[t] = np.clip(images[t].astype(np.float32) + noise, 0, 255).astype(np.uint8)

        metadata = {
            'seq_idx': seq_idx,
            'mito1': name1,
            'mito2': name2,
            'scenario': scenario,
            'trajectories': {'mito1': traj1, 'mito2': traj2},
        }

        return images, labels, metadata

    def run(self):
        """生成数据集"""
        output_dir = Path(self.cfg['paths']['output_dir'])
        raw_dir = output_dir / 'raw'
        label_dir = output_dir / 'label'
        meta_dir = output_dir / 'metadata'

        raw_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        num_sequences = self.cfg['generation']['num_sequences']

        print(f"\n生成 {num_sequences} 个序列...")
        print(f"输出目录: {output_dir}")

        # 统计信息
        label_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for i in tqdm(range(num_sequences)):
            try:
                images, labels, metadata = self.generate_sequence(i)

                # 保存图像
                tifffile.imwrite(str(raw_dir / f'syn_{i:04d}.tif'), images)

                # 保存标签
                tifffile.imwrite(str(label_dir / f'syn_{i:04d}.tif'), labels)

                # 保存元数据
                with open(meta_dir / f'syn_{i:04d}.json', 'w') as f:
                    json.dump(metadata, f, indent=2)

                # 统计
                for c in range(4):
                    label_counts[c] += (labels == c).sum()

            except Exception as e:
                print(f"\n错误生成序列 {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 保存统计信息
        self._save_stats(output_dir, num_sequences, label_counts)

        # 生成预览
        self._generate_preview(output_dir, num_samples=5)

        print(f"\n✓ 完成！共生成 {num_sequences} 个序列")
        print(f"  raw/: {raw_dir}")
        print(f"  label/: {label_dir}")

    def _save_stats(self, output_dir: Path, num_sequences: int, label_counts: dict):
        """保存统计信息"""
        total_pixels = sum(label_counts.values())

        stats_path = output_dir / 'dataset_stats.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("合成数据集统计 (Synthetic Dataset Statistics)\n")
            f.write("=" * 60 + "\n\n")

            f.write("配置:\n")
            f.write(f"  序列数: {num_sequences}\n")
            f.write(f"  帧数: {self.T}\n")
            f.write(f"  尺寸: {self.h}×{self.w}\n")
            f.write(f"  位深: {self.bit_depth}bit\n")
            f.write(f"  交叠帧: 第{self.contact_frame}帧\n\n")

            f.write("标签格式 (三分类):\n")
            f.write("  0: 背景\n")
            f.write("  1: 线粒体1独占区\n")
            f.write("  2: 交叠区\n")
            f.write("  3: 线粒体2独占区\n\n")

            f.write("类别分布:\n")
            class_names = ['背景', '线粒体1独占区', '交叠区', '线粒体2独占区']
            for c in range(4):
                ratio = label_counts[c] / total_pixels * 100 if total_pixels > 0 else 0
                f.write(f"  {class_names[c]}: {label_counts[c]:,} pixels ({ratio:.2f}%)\n")

        print(f"\n统计信息 -> {stats_path}")

    def _generate_preview(self, output_dir: Path, num_samples: int = 5):
        """生成预览图"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("警告: matplotlib未安装，跳过预览生成")
            return

        raw_dir = output_dir / 'raw'
        label_dir = output_dir / 'label'
        preview_dir = output_dir / 'preview'
        preview_dir.mkdir(exist_ok=True)

        # 颜色映射
        colors = np.array([
            [0.1, 0.1, 0.1],  # 0: 背景
            [0.9, 0.2, 0.2],  # 1: 线粒体1独占区 (红)
            [0.8, 0.2, 0.8],  # 2: 交叠区 (紫)
            [0.2, 0.4, 0.9],  # 3: 线粒体2独占区 (蓝)
        ])

        raw_files = sorted(raw_dir.glob('*.tif'))[:num_samples]

        for raw_path in raw_files:
            name = raw_path.stem
            label_path = label_dir / f'{name}.tif'

            if not label_path.exists():
                continue

            images = tifffile.imread(str(raw_path))
            labels = tifffile.imread(str(label_path))

            T = images.shape[0]
            fig, axes = plt.subplots(2, T, figsize=(T * 2.5, 5))

            for t in range(T):
                # 图像
                img = images[t].astype(np.float32)
                vmin, vmax = np.percentile(img, [1, 99])
                axes[0, t].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                axes[0, t].axis('off')
                axes[0, t].set_title(f'Frame {t}', fontsize=10)

                # 标签
                label_colored = colors[labels[t]]
                axes[1, t].imshow(label_colored)
                axes[1, t].axis('off')

            axes[0, 0].set_ylabel('Image', fontsize=10)
            axes[1, 0].set_ylabel('Label', fontsize=10)

            # 图例
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=colors[0], label='Background'),
                plt.Rectangle((0, 0), 1, 1, facecolor=colors[1], label='Mito1 only'),
                plt.Rectangle((0, 0), 1, 1, facecolor=colors[2], label='Overlap'),
                plt.Rectangle((0, 0), 1, 1, facecolor=colors[3], label='Mito2 only'),
            ]
            fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=9)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(preview_dir / f'{name}.png', dpi=120)
            plt.close()

        print(f"预览图 -> {preview_dir}")


def main():
    parser = argparse.ArgumentParser(description="生成线粒体合成数据（三分类标签）")
    parser.add_argument("--config", type=str, required=True,
                        help="配置文件路径 (YAML)")

    args = parser.parse_args()

    config = load_config(args.config)
    generator = MitoSyntheticGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
