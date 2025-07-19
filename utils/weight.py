import numpy as np
import torch

def compute_class_weights(train_dataset, device=None):
    """
    根据训练数据集中各类别样本的数量，计算每类的权重（用于加权交叉熵等场景）。

    参数:
        train_dataset: 包含 `datapath` 和 `classes` 属性的数据集对象
        device: 可选，权重最终所在的设备（如 'cuda' 或 'cpu'）

    返回:
        class_weights: Tensor[n_classes]，归一化后的类别权重张量
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取类别总数
    num_classes = len(train_dataset.classes)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    # 统计每类的样本数量
    for item in train_dataset.datapath:
        class_name = item[0]
        class_index = train_dataset.classes[class_name]
        class_counts[class_index] += 1

    print(f"[INFO] 各类别样本数量: {class_counts}")

    # 计算逆频率权重
    epsilon = 1e-6
    weights = 1.0 / (class_counts + epsilon)
    weights = weights / np.sum(weights) * num_classes  # 归一化

    class_weights = torch.FloatTensor(weights).to(device)
    print(f"[INFO] 计算得到的类别权重: {class_weights.cpu().numpy()}")

    return class_weights
