import numpy as np
from typing import Tuple


# --- 基础辅助函数 (无变化) ---
def load_point_cloud(txt_path: str) -> np.ndarray:
    """从 .txt 文件加载点云数据。"""
    return np.loadtxt(txt_path)


def save_point_cloud(points: np.ndarray, txt_path: str):
    """将点云数据保存到 .txt 文件。"""
    np.savetxt(txt_path, points, fmt="%.8f")


# --- 辅助函数：创建一个微小的圆形凹陷 ---
def apply_single_dent_defect(
        points: np.ndarray,
        center: np.ndarray,
        radius: float,
        depth: float
) -> np.ndarray:
    """
    在点云上应用一个微小的、平滑的圆形凹陷。这是腐蚀效果的基本单元。
    """
    dist_xy = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    mask = dist_xy < radius

    if not np.any(mask):
        return points

    distances_in_radius = dist_xy[mask]
    depth_reduction = depth * (1 - (distances_in_radius / radius) ** 2)
    points[mask, 2] -= depth_reduction
    return points


# --- 核心函数：创建表面腐蚀/麻点效果 ---
def apply_pitting_defect(
        points: np.ndarray,
        area_center: np.ndarray,
        area_radius: float,
        num_pits: int,
        pit_radius_range: Tuple[float, float],
        pit_depth_range: Tuple[float, float]
) -> np.ndarray:
    """
    在一个指定区域内应用表面腐蚀（大量随机小凹坑）效果。

    Args:
        points (np.ndarray): 输入的点云。
        area_center (np.ndarray): 腐蚀区域的中心点。
        area_radius (float): 腐蚀区域的半径。
        num_pits (int): 凹坑的数量，控制腐蚀密度。
        pit_radius_range (Tuple[float, float]): 每个小凹坑的半径随机范围。
        pit_depth_range (Tuple[float, float]): 每个小凹坑的深度随机范围。

    Returns:
        np.ndarray: 应用了腐蚀效果的点云。
    """
    # 1. 找到在主要腐蚀区域内的所有点的索引
    dist_to_area_center = np.linalg.norm(points[:, :2] - area_center[:2], axis=1)
    indices_in_area = np.where(dist_to_area_center < area_radius)[0]

    if indices_in_area.size == 0:
        print("警告：在指定的腐蚀区域内没有点。")
        return points

    # 2. 从区域内的点中，随机选择 N 个作为小凹坑的中心
    # replace=False 确保我们不重复选择同一个点作为中心（虽然影响不大）
    num_available_points = indices_in_area.size
    # 如果可用点比要生成的坑还少，就用所有可用点
    if num_available_points < num_pits:
        pit_center_indices = indices_in_area
    else:
        pit_center_indices = np.random.choice(indices_in_area, size=num_pits, replace=False)

    print(f"在腐蚀区域内找到 {num_available_points} 个点, 将生成 {pit_center_indices.size} 个凹坑。")

    # 3. 循环生成大量微小凹陷并叠加效果
    for i, idx in enumerate(pit_center_indices):
        if (i + 1) % 100 == 0:  # 每100个打印一次进度
            print(f"  正在生成第 {i + 1}/{pit_center_indices.size} 个凹坑...")

        pit_center = points[idx]
        pit_radius = np.random.uniform(pit_radius_range[0], pit_radius_range[1])
        pit_depth = np.random.uniform(pit_depth_range[0], pit_depth_range[1])

        # 直接在原始点云上反复修改，实现效果叠加
        points = apply_single_dent_defect(points, pit_center, pit_radius, pit_depth)

    return points


# --- 缺陷样本生成器 ---
def generate_pitting_sample(
        input_path: str,
        output_path: str
):
    """加载点云，并生成一个带随机腐蚀效果的样本。"""
    points = load_point_cloud(input_path)
    if points.shape[0] == 0:
        print(f"错误：输入文件 {input_path} 为空。")
        return

    # 1. 随机定义整个腐蚀区域的参数
    # 在点云内部随机选择一个腐蚀区域的中心
    area_center_idx = np.random.randint(0, points.shape[0])
    area_center = points[area_center_idx]
    area_radius = np.random.uniform(6.0, 40.0)  # 控制整个腐蚀区域的大小。

    # 2. 定义小凹坑的参数
    num_pits = np.random.randint(60, 400)  # 凹坑数量/密度
    pit_radius_range = (0.5, 2.5)  # 每个小坑的半径范围
    pit_depth_range = (0.3, 1.0)  # 每个小坑的深度范围

    print(f"生成腐蚀区域参数: 中心={area_center[:2]}, 半径={area_radius:.2f}, 凹坑数量={num_pits}")

    # 3. 应用腐蚀缺陷
    defective_points = apply_pitting_defect(
        points.copy(),
        area_center,
        area_radius,
        num_pits,
        pit_radius_range,
        pit_depth_range
    )

    # 4. 保存文件
    save_point_cloud(defective_points, output_path)
    print(f"带腐蚀效果的缺陷样本已成功保存至 {output_path}")


# --- 主程序入口 ---
if __name__ == '__main__':
    input_file = "E:\data\self_data\std\std_001.txt"
    output_file = "E:\data\self_data\sunken\sunken_005.txt"

    print("--- 步骤2: 生成腐蚀缺陷样本 ---")
    generate_pitting_sample(input_file, output_file)