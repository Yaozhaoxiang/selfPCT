
import numpy as np
import os

def load_point_cloud(file_path):
    """
    加载点云文件（.txt），每行格式为 x y z
    """
    return np.loadtxt(file_path)

def downsample_point_cloud(points, num_samples=10000):
    """
    随机下采样点云
    - points: (N, 3) 的 numpy 数组
    - num_samples: 目标点数
    """
    N = points.shape[0]
    if N <= num_samples:
        return points
    else:
        indices = np.random.choice(N, num_samples, replace=False)
        return points[indices]

def save_point_cloud(points, save_path):
    """
    保存点云为 .txt 文件
    """
    np.savetxt(save_path, points, fmt='%.6f')

if __name__ == '__main__':
    input_file = r'E:\data\data\flaw_samples_1\group_1.txt'
    output_file = r'E:\data\data\flaw_samples_1\test.txt'

    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        exit(1)

    # 加载点云
    point_cloud = load_point_cloud(input_file)

    # 下采样
    sampled_point_cloud = downsample_point_cloud(point_cloud, num_samples=10000)

    # 保存下采样结果
    save_point_cloud(sampled_point_cloud, output_file)

    print(f"下采样完成，保存为: {output_file}")



