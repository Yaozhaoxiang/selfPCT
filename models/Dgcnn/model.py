import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. 核心工具函数
# ---------------------------------------------------------------------------

def knn(x, k):
    """
    在特征空间中查找k个最近邻。
    x: 输入特征，形状 [B, C, N]
    k: 邻居数量
    返回: 邻居的索引，形状 [B, N, k]
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    计算EdgeConv的输入特征。
    x: 输入特征，形状 [B, C, N]
    k: 邻居数量
    idx: 可选的预计算邻居索引
    返回: 边特征，形状 [B, 2*C, N, k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    # feature.shape: [B*N, C] -> [B, N, C]
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # 中心点特征
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # 拼接中心点特征和邻居点与中心点的差值特征
    feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # [B, 2*C, N, k]


# ---------------------------------------------------------------------------
# 2. DGCNN 分类模型主体
# ---------------------------------------------------------------------------

class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        num_class = 7
        k = 10
        dropout = 0.5
        """
        初始化DGCNN分类模型。
        num_class: 分类的类别数。
        k: k-NN中的邻居数。
        dropout: Dropout层的概率。
        """
        super(PointTransformerCls, self).__init__()
        self.k = k
        self.dropout = dropout

        # EdgeConv 模块
        # 使用2D卷积实现共享MLP
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))

        # 最终的特征聚合层
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        # 分类头 (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(self.dropout),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        # x: 输入点云，形状 [B, C, N]，例如 [B, 3, 1024]
        batch_size = x.size(0)

        # --- EdgeConv 1 ---
        # 第一次在坐标空间中找邻居
        x1_graph = get_graph_feature(x, k=self.k)  # [B, 6, N, k]
        x1 = self.conv1(x1_graph)  # [B, 64, N, k]
        x1 = x1.max(dim=-1, keepdim=False)[0]  # [B, 64, N]

        # --- EdgeConv 2 ---
        # 第二次在64维特征空间中动态找邻居
        x2_graph = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2_graph)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        # --- EdgeConv 3 ---
        x3_graph = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3_graph)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        # --- EdgeConv 4 ---
        x4_graph = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4_graph)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        # --- 特征聚合 ---
        # 将所有EdgeConv层的输出拼接起来
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # [B, 64+64+128+256, N] = [B, 512, N]

        # 通过一个MLP进一步融合特征
        x = self.conv5(x_cat)  # [B, 1024, N]

        # --- 全局池化 ---
        # 同时使用最大池化和平均池化
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)

        # 拼接两种全局特征
        global_feature = torch.cat((x_max, x_avg), 1)  # [B, 2048]

        # --- 分类 ---
        logits = self.classifier(global_feature)

        return logits



