import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import io
import PIL.Image
from torchvision.transforms import ToTensor


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    生成混淆矩阵的图像，并返回一个可供 TensorBoard 使用的图像张量。

    Args:
        y_true (list or np.array): 真实标签列表。
        y_pred (list or np.array): 预测标签列表。
        class_names (list): 类别名称列表，用于坐标轴标签。

    Returns:
        PIL.Image: 混淆矩阵的图像。
    """
    # 1. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 归一化，显示百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 2. 使用 matplotlib 和 seaborn 绘图
    fig, ax = plt.subplots(figsize=(12, 10))  # 您可以根据类别数量调整图像大小
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)

    # 添加标签和标题
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix (Normalized)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()  # 自动调整布局

    # 3. 将 matplotlib 图像转换为 PIL Image，以便 TensorBoard 处理
    # 创建一个内存中的二进制流缓冲区
    buf = io.BytesIO()
    # 将图像保存到缓冲区
    fig.savefig(buf, format='png')
    # 关闭图像以释放内存
    plt.close(fig)
    # 将缓冲区指针移到开头
    buf.seek(0)
    # 从缓冲区创建 PIL Image
    image = PIL.Image.open(buf)

    return image