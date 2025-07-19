import numpy as np
from typing import Tuple


# --- 基础辅助函数 (无变化) ---
def load_point_cloud(txt_path: str) -> np.ndarray:
    """从 .txt 文件加载点云数据。"""
    return np.loadtxt(txt_path)


def save_point_cloud(points: np.ndarray, txt_path: str):
    """将点云数据保存到 .txt 文件。"""
    np.savetxt(txt_path, points, fmt="%.8f")


# --- 凸起/鼓包缺陷核心函数 ---
def apply_elliptical_bump_defect(
        points: np.ndarray,
        center: np.ndarray,
        radius_a: float,  # 长轴半径
        radius_b: float,  # 短轴半径
        angle: float,  # 旋转角度 (弧度)
        max_height: float  # 改为最大高度
) -> np.ndarray:
    """
    对点云应用一个平滑的椭圆形凸起/鼓包缺陷。

    Args:
        points (np.ndarray): 输入的点云数组 (N, 3)。
        center (np.ndarray): 缺陷中心的坐标 (x, y, z)。
        radius_a (float): 椭圆的长轴半径。
        radius_b (float): 椭圆的短轴半径。
        angle (float): 椭圆从x轴正方向逆时针旋转的角度（弧度）。
        max_height (float): 缺陷中心的最大凸起高度。

    Returns:
        np.ndarray: 应用了缺陷后的点云数组。
    """
    # 1. 坐标变换到以缺陷中心为原点
    relative_coords = points[:, :2] - center[:2]

    # 2. 旋转变换
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    x_rotated = relative_coords[:, 0] * cos_a - relative_coords[:, 1] * sin_a
    y_rotated = relative_coords[:, 0] * sin_a + relative_coords[:, 1] * cos_a

    # 3. 判断点是否在椭圆内部
    if radius_a == 0 or radius_b == 0:
        return points
    ellipse_eq_val = (x_rotated / radius_a) ** 2 + (y_rotated / radius_b) ** 2
    mask = ellipse_eq_val <= 1

    if not np.any(mask):
        print("警告：选定的缺陷区域内没有点。")
        return points

    # 4. 计算并应用平滑的高度增加
    # 公式: height = max_height * (1 - ellipse_eq_val)
    # ellipse_eq_val 在中心为0，边缘为1，正好可以用于平滑过渡
    height_increase = max_height * (1 - ellipse_eq_val[mask])

    # --- 核心区别：Z轴坐标是增加而不是减少 ---
    points[mask, 2] += height_increase

    return points


# --- 缺陷样本生成器 ---
def generate_bump_sample(
        input_path: str,
        output_path: str,
        radius_range: Tuple[float, float] = (5.0, 15.0),
        height_range: Tuple[float, float] = (1.0, 3.0)
):
    """加载一个标准点云样本，生成一个带有人工椭圆鼓包缺陷的样本。"""
    points = load_point_cloud(input_path)
    if points.shape[0] == 0:
        print(f"错误：输入文件 {input_path} 为空。")
        return

    # 1. 随机化椭圆的参数
    r1 = np.random.uniform(radius_range[0], radius_range[1])
    r2 = np.random.uniform(radius_range[0], radius_range[1])
    radius_a = max(r1, r2)
    radius_b = min(r1, r2)

    angle_deg = np.random.uniform(0, 180)
    angle_rad = np.deg2rad(angle_deg)

    bump_height = np.random.uniform(height_range[0], height_range[1])

    # 2. 从现有数据点中随机选择一个作为缺陷中心
    center_index = np.random.randint(0, points.shape[0])
    defect_center = points[center_index]

    print(f"生成鼓包缺陷参数: a={radius_a:.2f}, b={radius_b:.2f}, "
          f"角度={angle_deg:.2f}°, 最大高度={bump_height:.2f}")
    print(f"选择的缺陷中心点 (xyz): {defect_center}")

    # 3. 应用椭圆鼓包缺陷
    defective_points = apply_elliptical_bump_defect(
        points.copy(),
        defect_center,
        radius_a,
        radius_b,
        angle_rad,
        bump_height
    )

    # 4. 保存新的点云
    save_point_cloud(defective_points, output_path)
    print(f"带鼓包的缺陷样本已成功保存至 {output_path}")


# --- 主程序入口 ---
if __name__ == '__main__':
    input_file = "E:\data\self_data\std\std_001.txt"
    output_file = "E:\data\self_data\sunken\sunken_003.txt"
    generate_bump_sample(
        input_file,
        output_file,
        radius_range=(5.0, 15.0),
        height_range=(1.0, 6.0)
    )