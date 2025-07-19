# 划痕

import numpy as np
from typing import Tuple


# --- 基础辅助函数 ---
def load_point_cloud(txt_path: str) -> np.ndarray:
    """从 .txt 文件加载点云数据。"""
    return np.loadtxt(txt_path)


def save_point_cloud(points: np.ndarray, txt_path: str):
    """将点云数据保存到 .txt 文件。"""
    np.savetxt(txt_path, points, fmt="%.8f")


# --- 新的、宽度可变的划痕函数 ---
def apply_tapered_scratch_defect(
        points: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        max_width: float,  # 改为最大宽度
        depth: float
) -> np.ndarray:
    """
    对点云应用一个两端渐窄的、U型剖面的划痕缺陷。
    """
    line_vec = p2 - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        return points

    points_vec = points[:, :2] - p1

    # t 是每个点在线段方向上的投影比例，范围可以是(-inf, +inf)
    t = np.dot(points_vec, line_vec) / line_len_sq

    # t_clamped 将 t 约束在 [0, 1] 范围，用于计算点到线段的最近点
    t_clamped = np.clip(t, 0, 1)
    closest_points_on_segment = p1 + t_clamped[:, np.newaxis] * line_vec
    distances_to_segment = np.linalg.norm(points[:, :2] - closest_points_on_segment, axis=1)

    # --- 核心改动：计算随位置变化的局部宽度 ---
    # 我们只关心在[0,1]区间内的t值，区间外的宽度应为0
    # np.clip(t, 0, 1) 确保了这一点
    # local_half_width = (max_width / 2) * sin(t_clamped * pi)
    local_half_width = (max_width / 2.0) * np.sin(t_clamped * np.pi)

    # 判断点是否在可变宽度的划痕内部
    mask = distances_to_segment < local_half_width

    if not np.any(mask):
        return points

    # --- 针对在划痕内的点，计算深度衰减 ---
    distances_in_scratch = distances_to_segment[mask]
    local_half_width_in_scratch = local_half_width[mask]

    # 安全检查，防止除以零（在t=0或t=1处，宽度为0）
    safe_mask = local_half_width_in_scratch > 1e-6
    if not np.any(safe_mask):
        return points

    # 计算U型剖面，深度正比于 (1 - (相对距离)^2)
    depth_reduction = depth * (1 - (distances_in_scratch[safe_mask] / local_half_width_in_scratch[safe_mask]) ** 2)

    # 获取原始掩码中需要应用安全掩码的索引
    original_indices = np.where(mask)[0]
    safe_indices = original_indices[safe_mask]

    # 应用深度变化
    points[safe_indices, 2] -= depth_reduction

    return points


# --- 生成器函数也同步更新 ---
def generate_scratch_sample(
        input_path: str,
        output_path: str,
        length_range: Tuple[float, float] = (30.0, 80.0),
        max_width_range: Tuple[float, float] = (2.0, 5.0),  # 改为最大宽度范围
        depth_range: Tuple[float, float] = (0.5, 2.0)
):
    """加载点云，并生成一个带随机、宽度可变划痕的样本。"""
    points = load_point_cloud(input_path)
    if points.shape[0] == 0:
        print(f"错误：输入文件 {input_path} 为空。")
        return

    scratch_length = np.random.uniform(length_range[0], length_range[1])
    scratch_max_width = np.random.uniform(max_width_range[0], max_width_range[1])
    scratch_depth = np.random.uniform(depth_range[0], depth_range[1])

    start_point_idx = np.random.randint(0, points.shape[0])
    p1_3d = points[start_point_idx]
    p1_2d = p1_3d[:2]

    angle_rad = np.random.uniform(0, 2 * np.pi)
    p2_2d = p1_2d + np.array([np.cos(angle_rad), np.sin(angle_rad)]) * scratch_length

    print(f"生成锥形划痕参数: 长度={scratch_length:.2f}, 最大宽度={scratch_max_width:.2f}, 深度={scratch_depth:.2f}")
    print(f"划痕起点(xy): {p1_2d}")
    print(f"划痕终点(xy): {p2_2d}")

    # 调用新的、宽度可变的划痕函数
    defective_points = apply_tapered_scratch_defect(
        points.copy(), p1_2d, p2_2d, scratch_max_width, scratch_depth
    )

    save_point_cloud(defective_points, output_path)
    print(f"带锥形划痕的缺陷样本已成功保存至 {output_path}")
# --- 主程序入口 ---
if __name__ == '__main__':
    input_file = "E:\data\self_data\std\std_001.txt"
    output_file = "E:\data\self_data\sunken\sunken_002.txt"

    generate_scratch_sample(
        input_file,
        output_file,
        length_range=(50.0, 150.0),
        max_width_range=(3.0, 15.0), # 中间最宽处的宽度
        depth_range=(1.0, 3.0)
    )