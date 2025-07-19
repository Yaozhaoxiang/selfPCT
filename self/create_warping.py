import numpy as np
from typing import Tuple, Literal


# --- 基础辅助函数 (无变化) ---
def load_point_cloud(txt_path: str) -> np.ndarray:
    """从 .txt 文件加载点云数据。"""
    return np.loadtxt(txt_path)


def save_point_cloud(points: np.ndarray, txt_path: str):
    """将点云数据保存到 .txt 文件。"""
    np.savetxt(txt_path, points, fmt="%.8f")


# --- 核心函数：应用翘曲/弯曲变形 ---
def apply_warping_defect(
        points: np.ndarray,
        warp_factor: float,
        warp_axis: Literal['x', 'y'] = 'x'
) -> np.ndarray:
    """
    对整个点云应用一个整体的、抛物线形的翘曲/弯曲变形。

    Args:
        points (np.ndarray): 输入的点云。
        warp_factor (float): 翘曲系数。一个很小的正数或负数，控制弯曲程度和方向。
        warp_axis (Literal['x', 'y']): 'x' 或 'y'，决定沿着哪个轴进行弯曲。

    Returns:
        np.ndarray: 应用了翘曲变形后的点云。
    """
    if points.shape[0] == 0:
        return points

    # 1. 为了让弯曲对称，计算点云在指定轴上的几何中心
    if warp_axis == 'x':
        axis_index = 0
    elif warp_axis == 'y':
        axis_index = 1
    else:
        raise ValueError("warp_axis 必须是 'x' 或 'y'")

    center_coord = (points[:, axis_index].min() + points[:, axis_index].max()) / 2.0

    # 2. 计算每个点相对于中心的坐标
    relative_coords = points[:, axis_index] - center_coord

    # 3. 应用抛物线变形函数 f(r) = k * r^2
    deformation = warp_factor * (relative_coords ** 2)

    # 4. 将变形量加到Z轴坐标上
    points[:, 2] += deformation

    return points


# --- 缺陷样本生成器 ---
def generate_warping_sample(
        input_path: str,
        output_path: str,
        warp_factor_range: Tuple[float, float] = (1e-5, 5e-5)  # 注意：翘曲系数通常非常小
):
    """加载点云，并生成一个带随机翘曲缺陷的样本。"""
    points = load_point_cloud(input_path)
    if points.shape[0] == 0:
        print(f"错误：输入文件 {input_path} 为空。")
        return

    # 1. 随机化翘曲参数
    # 随机选择一个翘曲系数
    warp_factor = np.random.uniform(warp_factor_range[0], warp_factor_range[1])
    # 随机决定是向上弯曲还是向下弯曲
    if np.random.rand() < 0.5:
        warp_factor *= -1

    # 随机选择弯曲的轴向
    warp_axis = np.random.choice(['x', 'y'])

    direction = "向上(U形)" if warp_factor > 0 else "向下(倒U形)"
    print(f"生成翘曲缺陷参数: 系数={warp_factor:.2e}, 沿 {warp_axis.upper()} 轴弯曲, 方向: {direction}")

    # 2. 应用翘曲缺陷
    defective_points = apply_warping_defect(
        points.copy(),
        warp_factor,
        warp_axis=warp_axis
    )

    # 3. 保存文件
    save_point_cloud(defective_points, output_path)
    print(f"带翘曲效果的缺陷样本已成功保存至 {output_path}")


# --- 主程序入口 ---
if __name__ == '__main__':
    input_file = "E:\data\self_data\std\std_001.txt"
    output_file = "E:\data\self_data\sunken\sunken_006.txt"

    generate_warping_sample(
        input_file,
        output_file,
        # 可以调整这个范围来控制弯曲的剧烈程度
        warp_factor_range=(8e-5, 15e-4)
    )