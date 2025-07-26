import os
import numpy as np
import random
from typing import Tuple, Literal
from pathlib import Path
from scipy.spatial import ConvexHull


# ==============================================================================
# --- 1. 配置区域 (请根据您的设置修改) ---
# ==============================================================================
# 输入文件夹：存放您所有标准样本的地方
INPUT_DIR = "H:\data\self_data\std"

# 基础输出文件夹：所有生成的缺陷文件夹将创建在这里
BASE_OUTPUT_DIR = "H://data//add_data"

# 每种缺陷的参数范围，您可以在这里微调以获得想要的效果
DENT_BUMP_RADIUS_RANGE = (10.0, 25.0)
DENT_BUMP_HEIGHT_RANGE = (1.5, 4.0)

SCRATCH_LENGTH_RANGE = (50.0, 150.0)
SCRATCH_WIDTH_RANGE = (3.0, 15.0)
SCRATCH_DEPTH_RANGE = (1.0, 3.0)

PITTING_AREA_RADIUS_RANGE = (6.0, 40.0)
PITTING_NUM_PITS_RANGE = (60, 400)
PITTING_PIT_RADIUS_RANGE = (0.5, 2.5)
PITTING_PIT_DEPTH_RANGE = (0.3, 1.0)

WARPING_FACTOR_RANGE = (8e-5, 15e-4)

CHIPPING_RADIUS_RANGE = (8.0, 30.0)  # 缺口大小范围


# ==============================================================================
# --- 2. 核心缺陷实现函数 (我们之前编写的所有 apply_... 函数) ---
# ==============================================================================

# --- 辅助函数：加载和保存 ---
def load_point_cloud(txt_path: Path) -> np.ndarray:
    return np.loadtxt(str(txt_path))


def save_point_cloud(points: np.ndarray, txt_path: Path):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(txt_path), points, fmt="%.8f")


# --- 凹陷/凸起 ---
def apply_elliptical_defect(points, center, radius_a, radius_b, angle, magnitude, is_dent=True):
    relative_coords = points[:, :2] - center[:2]
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    x_rotated = relative_coords[:, 0] * cos_a - relative_coords[:, 1] * sin_a
    y_rotated = relative_coords[:, 0] * sin_a + relative_coords[:, 1] * cos_a

    if radius_a == 0 or radius_b == 0: return points

    ellipse_eq_val = (x_rotated / radius_a) ** 2 + (y_rotated / radius_b) ** 2
    mask = ellipse_eq_val <= 1
    if not np.any(mask): return points

    z_change = magnitude * (1 - ellipse_eq_val[mask])
    if is_dent:
        points[mask, 2] -= z_change
    else:
        points[mask, 2] += z_change
    return points


# --- 划痕 ---
def apply_tapered_scratch_defect(points, p1, p2, max_width, depth):
    line_vec = p2 - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0: return points

    points_vec = points[:, :2] - p1
    t = np.dot(points_vec, line_vec) / line_len_sq
    t_clamped = np.clip(t, 0, 1)

    closest_points = p1 + t_clamped[:, np.newaxis] * line_vec
    distances = np.linalg.norm(points[:, :2] - closest_points, axis=1)

    local_half_width = (max_width / 2.0) * np.sin(t_clamped * np.pi)
    mask = distances < local_half_width
    if not np.any(mask): return points

    distances_in, local_half_width_in = distances[mask], local_half_width[mask]
    safe_mask = local_half_width_in > 1e-6
    if not np.any(safe_mask): return points

    depth_reduction = depth * (1 - (distances_in[safe_mask] / local_half_width_in[safe_mask]) ** 2)
    original_indices = np.where(mask)[0][safe_mask]
    points[original_indices, 2] -= depth_reduction
    return points


# --- 腐蚀/麻点 ---
def apply_single_dent_defect(points, center, radius, depth):
    dist_xy = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    mask = dist_xy < radius
    if not np.any(mask): return points
    distances_in_radius = dist_xy[mask]
    depth_reduction = depth * (1 - (distances_in_radius / radius) ** 2)
    points[mask, 2] -= depth_reduction
    return points


# --- 翘曲 ---
def apply_warping_defect(points, warp_factor, warp_axis: Literal['x', 'y']):
    axis_index = 0 if warp_axis == 'x' else 1
    center_coord = (points[:, axis_index].min() + points[:, axis_index].max()) / 2.0
    relative_coords = points[:, axis_index] - center_coord
    deformation = warp_factor * (relative_coords ** 2)
    points[:, 2] += deformation
    return points


# --- 崩边/缺口 ---
def find_edge_point_indices(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3: return np.array([], dtype=int)
    try:
        return ConvexHull(points[:, :2]).vertices
    except Exception:
        return np.array([], dtype=int)


def apply_chipping_defect(points, chip_center, chip_radius):
    distances = np.linalg.norm(points - chip_center, axis=1)
    mask_to_keep = distances > chip_radius
    return points[mask_to_keep]


# ==============================================================================
# --- 3. 缺陷生成逻辑封装 (每个函数负责生成一种缺陷) ---
# ==============================================================================
def create_dent_or_bump(points: np.ndarray, is_dent: bool) -> np.ndarray:
    if points.shape[0] == 0: return points
    center_idx = random.randint(0, points.shape[0] - 1)
    center = points[center_idx]

    r1 = random.uniform(*DENT_BUMP_RADIUS_RANGE)
    r2 = random.uniform(*DENT_BUMP_RADIUS_RANGE)
    magnitude = random.uniform(*DENT_BUMP_HEIGHT_RANGE)
    angle = random.uniform(0, np.pi)

    return apply_elliptical_defect(points, center, max(r1, r2), min(r1, r2), angle, magnitude, is_dent)


def create_scratch(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0: return points
    start_idx = random.randint(0, points.shape[0] - 1)
    p1 = points[start_idx][:2]

    length = random.uniform(*SCRATCH_LENGTH_RANGE)
    angle = random.uniform(0, 2 * np.pi)
    p2 = p1 + np.array([np.cos(angle), np.sin(angle)]) * length

    max_width = random.uniform(*SCRATCH_WIDTH_RANGE)
    depth = random.uniform(*SCRATCH_DEPTH_RANGE)

    return apply_tapered_scratch_defect(points, p1, p2, max_width, depth)


def create_pitting(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0: return points
    center_idx = random.randint(0, points.shape[0] - 1)
    area_center = points[center_idx]
    area_radius = random.uniform(*PITTING_AREA_RADIUS_RANGE)

    dist_to_area_center = np.linalg.norm(points[:, :2] - area_center[:2], axis=1)
    indices_in_area = np.where(dist_to_area_center < area_radius)[0]
    if indices_in_area.size == 0: return points

    num_pits = random.randint(*PITTING_NUM_PITS_RANGE)
    if indices_in_area.size < num_pits:
        pit_center_indices = indices_in_area
    else:
        pit_center_indices = np.random.choice(indices_in_area, size=num_pits, replace=False)

    for idx in pit_center_indices:
        pit_center = points[idx]
        pit_radius = random.uniform(*PITTING_PIT_RADIUS_RANGE)
        pit_depth = random.uniform(*PITTING_PIT_DEPTH_RANGE)
        points = apply_single_dent_defect(points, pit_center, pit_radius, pit_depth)
    return points


def create_warping(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0: return points
    warp_factor = random.uniform(*WARPING_FACTOR_RANGE)
    if random.random() < 0.5: warp_factor *= -1
    warp_axis = random.choice(['x', 'y'])
    return apply_warping_defect(points, warp_factor, warp_axis)


def create_chipping(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3: return points
    edge_indices = find_edge_point_indices(points)
    if edge_indices.size == 0:
        print("    [警告] 无法找到边缘点，跳过崩边缺陷。")
        return points

    center_idx = random.choice(edge_indices)
    chip_center = points[center_idx]
    chip_radius = random.uniform(*CHIPPING_RADIUS_RANGE)
    return apply_chipping_defect(points, chip_center, chip_radius)


# ==============================================================================
# --- 4. 主执行流程 ---
# ==============================================================================
def main():
    """主函数，执行批量处理任务。"""
    input_path = Path(INPUT_DIR)
    base_output_path = Path(BASE_OUTPUT_DIR)

    if not input_path.is_dir():
        print(f"错误：输入文件夹 '{INPUT_DIR}' 不存在。请创建它并放入样本文件。")
        return

    # 定义缺陷类型和对应的生成函数及输出文件夹
    defect_map = {
        "1_dent": create_dent_or_bump,
        "2_bump": create_dent_or_bump,
        "3_scratch": create_scratch,
        "4_pitting": create_pitting,
        "5_warping": create_warping,
        "6_chipping": create_chipping
    }

    # 获取所有源文件
    source_files = list(input_path.glob("*.txt"))
    if not source_files:
        print(f"警告：在输入文件夹 '{INPUT_DIR}' 中没有找到任何 .txt 文件。")
        return

    print(f"找到 {len(source_files)} 个标准样本，准备开始处理...")

    # 遍历每个源文件
    for i, file_path in enumerate(source_files):
        print(f"\n--- [{i + 1}/{len(source_files)}] 正在处理: {file_path.name} ---")

        try:
            points_original = load_point_cloud(file_path)
            if points_original.shape[0] == 0:
                print(f"  [警告] 文件为空，已跳过。")
                continue

            # 为每个缺陷类型生成并保存一个样本
            for name, func in defect_map.items():
                output_dir = base_output_path / name
                output_file_path = output_dir / file_path.name

                print(f"  -> 正在生成 {name} ...")

                # 复制原始点云以防被修改
                points_copy = points_original.copy()

                # 特殊处理凹陷和凸起
                if name == "1_dent":
                    defective_points = func(points_copy, is_dent=True)
                elif name == "2_bump":
                    defective_points = func(points_copy, is_dent=False)
                else:
                    defective_points = func(points_copy)

                save_point_cloud(defective_points, output_file_path)
                print(f"     成功保存至: {output_file_path}")

        except Exception as e:
            print(f"  [错误] 处理文件 {file_path.name} 时发生严重错误: {e}")
            import traceback
            traceback.print_exc()  # 打印详细的错误追溯信息
            continue  # 继续处理下一个文件

    print("\n所有文件处理完毕！")


if __name__ == '__main__':
    main()