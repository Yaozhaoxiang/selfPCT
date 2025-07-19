import torch
import torch.nn as nn
from pointnet_util import farthest_point_sample, index_points, square_distance

# xyz: [B, N, 3]       # 每个点的坐标（输入点云）
# points: [B, N, C]    # 每个点的特征（如坐标、颜色、法线等）
# npoint: int          # 要采样的点数 S
# nsample: int         # 每个采样点的邻域中采样的点数 K

def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint 

    # 使用 Farthest Point Sampling 从原始点集中选出 npoint 个分布均匀的“代表性点”。
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    new_xyz = index_points(xyz, fps_idx) # [B, npoint, 3] 坐标

    new_points = index_points(points, fps_idx) # [B, npoint, C]

    # 每个关键点计算它与所有点的欧式距离，并找到最近的 K 个邻居。
    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    # 提取邻域内的点特征
    grouped_points = index_points(points, idx) # [B, npoint, K, C]
    # 相对特征：邻居点特征 - 中心点特征
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)  # [B, npoint, K, C]

    # 拼接相对特征 + 中心点特征 [B, S, K, 2C]
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    # new_xyz	采样后的 S 个关键点坐标	[B, S, 3]
    # new_points	每个关键点邻域的 K 个点特征（相对特征 + 中心点复制）	[B, S, K, 2C]
    return new_xyz, new_points


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.num_class
        d_points = cfg.input_dim
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    # x.shape = [B, N, C]，比如 B=32, N=1024, C=3
    # 假设 cfg.input_dim = 3，cfg.num_class = 40（比如 ModelNet40 分类）。
    def forward(self, x):
        xyz = x[..., :3] # 提取前三个特征维度 shape: [B, N, 3]
        x = x.permute(0, 2, 1) # x: [B, C, N] => [B, 3, 1024]
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N   [B, 3, 1024] → [B, 64, 1024]
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = x.permute(0, 2, 1)   # [B, 1024, 64]


        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)  # new_points.shape = [B, 512, 32, 64 + 64] = [B, 512, 32, 128]
        feature_0 = self.gather_local_0(new_feature) # [B, 128, 512]
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature) # new_feature = [B, 256, 32, 128 + 128] = [B, 256, 32, 256]
        feature_1 = self.gather_local_1(new_feature)  # [B, 256, 256]
        
        x = self.pt_last(feature_1) # [B, 4*256, 256] = [B, 1024, 256]

        x = torch.cat([x, feature_1], dim=1)  # [B, 1024+256, 256] = [B, 1280, 256]
        x = self.conv_fuse(x) # [B, 1280, 256] → [B, 1024, 256]
        x = torch.max(x, 2)[0]  # [B, 1024]
        x = x.view(batch_size, -1)

        x = self.relu(self.bn6(self.linear1(x))) # [B, 512]
        x = self.dp1(x)
        x = self.relu(self.bn7(self.linear2(x))) # [B, 256]
        x = self.dp2(x)
        x = self.linear3(x) #[B, num_class]

        return x