import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. T-Net (Transformation Network) 模块
# ---------------------------------------------------------------------------
# T-Net是一个迷你的PointNet，其作用是学习一个仿射变换矩阵，
# 以对输入的点云坐标或特征进行对齐，从而增强模型的旋转不变性。

class TNet(nn.Module):
    def __init__(self, k=3):
        """
        初始化T-Net。
        k: 变换矩阵的维度。对于输入变换，k=3；对于特征变换，k=64。
        """
        super(TNet, self).__init__()
        self.k = k
        # 使用1D卷积实现共享MLP
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # 全连接层用于从全局特征中回归出变换矩阵
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 初始化变换矩阵为单位矩阵，有助于训练初期的稳定性
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, x):
        # x: 输入张量，形状 [B, k, N]
        batch_size = x.size(0)

        # 通过共享MLP提取特征
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # 最大池化得到全局特征
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.view(-1, 1024)

        # 通过全连接层回归变换矩阵
        x = self.relu(self.bn4(self.fc1(global_feature)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 将输出的向量重塑为 kx_k 的矩阵
        transform_matrix = x.view(batch_size, self.k, self.k)
        return transform_matrix


# ---------------------------------------------------------------------------
# 2. PointNet 分类模型主体
# ---------------------------------------------------------------------------

class PointNetCls(nn.Module):
    def __init__(self, num_class=40, use_feature_transform=True):
        """
        初始化PointNet分类模型。
        num_class: 分类的类别数。
        use_feature_transform: 是否使用特征空间的T-Net。
        """
        super(PointNetCls, self).__init__()
        self.use_feature_transform = use_feature_transform

        # 输入变换T-Net (k=3)
        self.input_tnet = TNet(k=3)

        # 第一组共享MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        # 如果使用特征变换，则实例化特征T-Net (k=64)
        if self.use_feature_transform:
            self.feature_tnet = TNet(k=64)

        # 第二组共享MLP
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # 分类头 (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_class)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: 输入点云，形状 [B, C, N]，例如 [B, 3, 1024]
        num_points = x.size(2)

        # 1. 输入变换
        input_transform_matrix = self.input_tnet(x)
        # 将点云与变换矩阵相乘
        x = x.transpose(2, 1)  # [B, N, C]
        x = torch.bmm(x, input_transform_matrix)
        x = x.transpose(2, 1)  # [B, C, N]

        # 2. 第一组MLP
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, N]

        # 3. 特征变换
        feature_transform_matrix = None
        if self.use_feature_transform:
            feature_transform_matrix = self.feature_tnet(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, feature_transform_matrix)
            x = x.transpose(2, 1)

        # 4. 第二组MLP
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = self.bn3(self.conv3(x))  # [B, 1024, N]

        # 5. 全局最大池化
        global_feature = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        global_feature = global_feature.view(-1, 1024)  # [B, 1024]

        # 6. 分类
        logits = self.classifier(global_feature)

        # 返回最终的分类得分和用于计算正则化损失的特征变换矩阵
        return logits, feature_transform_matrix


# ---------------------------------------------------------------------------
# 3. 特征变换正则化损失函数
# ---------------------------------------------------------------------------
# 这个损失函数鼓励特征变换矩阵接近于一个正交矩阵，
# 因为正交变换（如旋转）不会丢失信息。

def feature_transform_regularizer(trans_matrix):
    """
    计算特征变换矩阵的正则化损失。
    trans_matrix: T-Net输出的特征变换矩阵，形状 [B, K, K]
    """
    batch_size, K, _ = trans_matrix.shape
    # I - A*A^T
    identity = torch.eye(K).to(trans_matrix.device).unsqueeze(0).repeat(batch_size, 1, 1)
    loss = torch.mean(torch.norm(torch.bmm(trans_matrix, trans_matrix.transpose(2, 1)) - identity, dim=(1, 2)))
    return loss


# ---------------------------------------------------------------------------
# 4. 使用示例
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # 创建一个模拟的点云数据
    # B=2, N=1024, C=3
    dummy_input = torch.rand(2, 3, 1024)

    # 实例化模型
    # 假设进行40分类 (ModelNet40)
    pointnet_model = PointNetCls(num_class=40, use_feature_transform=True)

    # 前向传播
    logits, trans_feat_matrix = pointnet_model(dummy_input)

    # 打印输出形状
    print("PointNet模型已成功组装并通过测试！")
    print("输入形状:", dummy_input.shape)
    print("输出Logits形状:", logits.shape)  # 预期: torch.Size([2, 40])
    print("特征变换矩阵形状:", trans_feat_matrix.shape)  # 预期: torch.Size([2, 64, 64])

    # 计算损失的示例
    # 假设我们有一个目标标签
    dummy_target = torch.randint(0, 40, (2,))

    # 分类损失 (例如交叉熵)
    classification_loss = F.cross_entropy(logits, dummy_target)

    # 特征变换正则化损失
    # 乘以一个权重因子 (如0.001) 来平衡两个损失
    if trans_feat_matrix is not None:
        regularization_loss = feature_transform_regularizer(trans_feat_matrix)
        total_loss = classification_loss + 0.001 * regularization_loss
    else:
        total_loss = classification_loss

    print("\n损失计算示例:")
    print("分类损失:", classification_loss.item())
    if trans_feat_matrix is not None:
        print("正则化损失:", regularization_loss.item())
    print("总损失:", total_loss.item())
