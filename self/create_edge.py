import numpy as np
from typing import Tuple
from scipy.spatial import ConvexHull


# --- 基础辅助函数 (无变化) ---
def load_point_cloud(txt_path: str) -> np.ndarray:
    return np.loadtxt(txt_path)


def save_point_cloud(points: np.ndarray, txt_path: str):
    np.savetxt(txt_path, points, fmt="%.8f")


# --- 步骤1：找到边缘点 ---
def find_edge_point_indices(points: np.ndarray) -> np.ndarray:
    """
    使用2D凸包算法找到点云的边缘点索引。
    """
    if points.shape[0] < 3:
        return np.array([], dtype=int)

    # 将点云投影到XY平面
    points_2d = points[:, :2]

    try:
        # 计算凸包， hull.vertices 包含了作为凸包顶点的点的索引
        hull = ConvexHull(points_2d)
        edge_indices = hull.vertices
        return edge_indices
    except Exception as e:
        print(f"计算凸包时出错: {e}")
        return np.array([], dtype=int)


# --- 步骤2：应用缺口缺陷 ---
def apply_chipping_defect(
        points: np.ndarray,
        chip_center: np.ndarray,
        chip_radius: float
) -> np.ndarray:
    """
    以 chip_center 为中心，移除一个球形区域内的所有点，模拟缺口。

    Args:
        points (np.ndarray): 输入的点云。
        chip_center (np.ndarray): 缺口的3D中心点。
        chip_radius (float): 缺口的半径（虚拟球体半径）。

    Returns:
        np.ndarray: 移除了部分点后的新点云。
    """
    # 计算所有点到缺口中心的3D距离
    distances = np.linalg.norm(points - chip_center, axis=1)

    # 创建一个布尔掩码，标记所有不在球体内的点（即需要保留的点）
    mask_to_keep = distances > chip_radius

    # 返回被筛选后的点云
    return points[mask_to_keep]


# --- 缺陷样本生成器 ---
def generate_chipping_sample(
        input_path: str,
        output_path: str,
        chip_radius_range: Tuple[float, float] = (10.0, 25.0)
):
    """加载点云，并在其边缘创建一个随机的崩边/缺口缺陷。"""
    points = load_point_cloud(input_path)
    if points.shape[0] == 0:
        print(f"错误：输入文件 {input_path} 为空。")
        return

    print(f"原始点数: {points.shape[0]}")

    # 1. 找到所有可能的边缘点
    edge_indices = find_edge_point_indices(points)

    if edge_indices.size == 0:
        print("错误：未能找到任何边缘点，无法创建缺口。")
        return

    print(f"已识别出 {edge_indices.size} 个边缘点。")

    # 2. 从边缘点中随机选择一个作为缺口中心
    chip_center_idx = np.random.choice(edge_indices)
    chip_center = points[chip_center_idx]

    # 3. 随机化缺口大小
    chip_radius = np.random.uniform(chip_radius_range[0], chip_radius_range[1])

    print(f"生成缺口参数: 中心点索引={chip_center_idx}, 半径={chip_radius:.2f}")
    print(f"缺口中心坐标(xyz): {chip_center}")

    # 4. 应用缺口缺陷
    defective_points = apply_chipping_defect(
        points.copy(),
        chip_center,
        chip_radius
    )

    print(f"缺陷生成后点数: {defective_points.shape[0]}")
    print(f"移除了 {points.shape[0] - defective_points.shape[0]} 个点。")

    # 5. 保存文件
    save_point_cloud(defective_points, output_path)
    print(f"带缺口效果的缺陷样本已成功保存至 {output_path}")


# --- 主程序入口 ---
if __name__ == '__main__':
    input_file = "E:\data\self_data\std\std_001.txt"
    output_file = "E:\data\self_data\sunken\sunken_008.txt"
    generate_chipping_sample(
        input_file,
        output_file,
        chip_radius_range=(8.0, 30.0)  # 缺口大小范围
    )