import numpy as np
from typing import Tuple


def load_point_cloud(txt_path: str) -> np.ndarray:
    """从 .txt 文件加载点云数据。"""
    return np.loadtxt(txt_path)


def save_point_cloud(points: np.ndarray, txt_path: str):
    """将点云数据保存到 .txt 文件。"""
    np.savetxt(txt_path, points, fmt="%.8f")


def apply_dent_defect(
        points: np.ndarray,
        center: np.ndarray,
        radius: float,
        max_depth: float,
        smooth: bool = True
) -> np.ndarray:
    """
    对点云应用一个凹陷缺陷。

    Args:
        points (np.ndarray): 输入的点云数组 (N, 3)。
        center (np.ndarray): 缺陷中心的坐标 (x, y, z)。
        radius (float): 缺陷影响的半径。
        max_depth (float): 缺陷中心的最大深度。
        smooth (bool): 是否使用平滑过渡的凹陷。True为抛物线形，False为圆柱形。

    Returns:
        np.ndarray: 应用了缺陷后的点云数组。
    """
    # 计算所有点到中心点在 xy 平面上的距离
    dist_xy = np.linalg.norm(points[:, :2] - center[:2], axis=1)

    # 找到在半径范围内的点的索引
    mask = dist_xy < radius

    if not np.any(mask):
        print("警告：选定的缺陷区域内没有点。")
        return points

    if smooth:
        # --- 平滑凹陷（抛物线形状）---
        # 计算每个受影响点的深度，距离中心越近，深度越大
        # 公式: depth = max_depth * (1 - (dist/radius)^2)
        # 这种方式可以避免在循环中操作，利用numpy的向量化特性，效率更高
        distances_in_radius = dist_xy[mask]
        depth_reduction = max_depth * (1 - (distances_in_radius / radius) ** 2)
        points[mask, 2] -= depth_reduction
    else:
        # --- 原始的平底凹陷（圆柱形）---
        points[mask, 2] -= max_depth

    return points


# 椭圆凹陷
def apply_elliptical_dent_defect(
        points: np.ndarray,
        center: np.ndarray,
        radius_a: float,  # 长轴半径
        radius_b: float,  # 短轴半径
        angle: float,  # 旋转角度 (弧度)
        max_depth: float
) -> np.ndarray:
    """
    对点云应用一个平滑的椭圆形凹陷缺陷。

    Args:
        points (np.ndarray): 输入的点云数组 (N, 3)。
        center (np.ndarray): 缺陷中心的坐标 (x, y, z)。
        radius_a (float): 椭圆的长轴半径。
        radius_b (float): 椭圆的短轴半径。
        angle (float): 椭圆从x轴正方向逆时针旋转的角度（弧度）。
        max_depth (float): 缺陷中心的最大深度。

    Returns:
        np.ndarray: 应用了缺陷后的点云数组。
    """
    # 1. 坐标变换：将所有点移动到以缺陷中心为原点的坐标系
    relative_coords = points[:, :2] - center[:2]

    # 2. 旋转变换：将所有点绕原点进行反向旋转，相当于把椭圆摆正
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    # 使用旋转矩阵进行高效的向量化计算
    # x' = x*cos(θ) - y*sin(θ)
    # y' = x*sin(θ) + y*cos(θ)
    x_rotated = relative_coords[:, 0] * cos_a - relative_coords[:, 1] * sin_a
    y_rotated = relative_coords[:, 0] * sin_a + relative_coords[:, 1] * cos_a

    # 3. 判断哪些点在椭圆内部
    # 使用标准椭圆方程: (x'/a)^2 + (y'/b)^2 <= 1
    # 为了避免除以零
    if radius_a == 0 or radius_b == 0:
        return points

    ellipse_eq_val = (x_rotated / radius_a) ** 2 + (y_rotated / radius_b) ** 2
    mask = ellipse_eq_val <= 1

    if not np.any(mask):
        print("警告：选定的缺陷区域内没有点。")
        return points

    # 4. 计算并应用平滑的深度衰减
    # ellipse_eq_val 的值在[0, 1]之间，中心为0，边缘为1，正好可以用来计算衰减
    # 公式: depth = max_depth * (1 - ellipse_eq_val)
    depth_reduction = max_depth * (1 - ellipse_eq_val[mask])
    points[mask, 2] -= depth_reduction

    return points


def generate_defective_sample(
        input_path: str,
        output_path: str,
        radius_range: Tuple[float, float] = (5.0, 10.0),
        depth_range: Tuple[float, float] = (1.0, 3.0)
):
    """
    加载一个标准点云样本，生成一个带有人工凹陷缺陷的样本。

    Args:
        input_path (str): 标准样本的文件路径。
        output_path (str): 生成的缺陷样本的保存路径。
        radius_range (Tuple[float, float]): 缺陷半径的随机范围 (min, max)。
        depth_range (Tuple[float, float]): 缺陷最大深度的随机范围 (min, max)。
    """
    points = load_point_cloud(input_path)

    if points.shape[0] == 0:
        print(f"错误：输入文件 {input_path} 为空。")
        return

    # 1. 随机化椭圆的参数
    # 从范围中随机选择两个半径，一个作为长轴，一个作为短轴
    r1 = np.random.uniform(radius_range[0], radius_range[1])
    r2 = np.random.uniform(radius_range[0], radius_range[1])

    radius_a = max(r1, r2)  # 长轴
    radius_b = min(r1, r2)  # 短轴

    # 随机生成一个旋转角度（0到180度）
    angle_deg = np.random.uniform(0, 180)
    angle_rad = np.deg2rad(angle_deg)  # 转换为弧度

    dent_depth = np.random.uniform(depth_range[0], depth_range[1])

    # 2. 从现有数据点中随机选择一个作为缺陷中心
    center_index = np.random.randint(0, points.shape[0])
    defect_center = points[center_index]

    print(f"生成椭圆缺陷参数: a={radius_a:.2f}, b={radius_b:.2f}, "
          f"角度={angle_deg:.2f}°, 最大深度={dent_depth:.2f}")
    print(f"选择的缺陷中心点 (xyz): {defect_center}")

    # 3. 应用椭圆凹陷缺陷
    defective_points = apply_elliptical_dent_defect(
        points.copy(),
        defect_center,
        radius_a,
        radius_b,
        angle_rad,
        dent_depth
    )

    # 4. 保存新的点云
    save_point_cloud(defective_points, output_path)
    print(f"缺陷样本已成功保存至 {output_path}")


# --- 使用示例 ---
if __name__ == '__main__':

    input_file = "E:\data\self_data\std\std_001.txt"
    output_file = "E:\data\self_data\sunken\sunken_001.txt"

    # 生成缺陷样本
    generate_defective_sample(
        input_file,
        output_file,
        radius_range=(8.0, 15.0),  # 随机半径范围
        depth_range=(2.0, 5.0)  # 随机深度范围
    )