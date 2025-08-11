import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 步骤 1: 定义所有我们需要的工具函数和基础模块
# ---------------------------------------------------------------------------

# --- 工具函数 (来自 pointnet_util.py) ---
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = points.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    dists = square_distance(new_xyz, xyz)
    idx = dists.argsort()[:, :, :nsample]
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    grouped_features = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, grouped_features


# --- 我们设计的先进模块 (此处省略了SA_Layer, Point_AttnConv等依赖模块的定义) ---
# (在实际运行中，需要将它们全部定义好)
class SA_Layer_Efficient_Offset(nn.Module):
    def __init__(self, channels, pool_ratio=4):
        super().__init__()
        self.pool_ratio = pool_ratio
        if self.pool_ratio > 1: self.pool = nn.AvgPool1d(kernel_size=pool_ratio, stride=pool_ratio)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k, x_v = self.k_conv(x), self.v_conv(x)
        if self.pool_ratio > 1: x_k, x_v = self.pool(x_k), self.pool(x_v)
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(attention, x_v.permute(0, 2, 1)).permute(0, 2, 1)
        x_offset = self.act(self.after_norm(self.trans_conv(x - x_r)))
        return x_offset


class Point_AttnConv_Complete(nn.Module):
    def __init__(self, channels, key_channels, kernel_size=3):
        super().__init__()
        self.key_channels = key_channels
        self.q_conv = nn.Conv1d(channels, key_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, key_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_dwconv = nn.Conv1d(key_channels, key_channels, kernel_size, padding=(kernel_size - 1) // 2,
                                  groups=key_channels)
        self.v_dwconv = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=channels)
        self.attention_map_generator = nn.Sequential(nn.Conv1d(key_channels, key_channels, 1), nn.SiLU(),
                                                     nn.Conv1d(key_channels, key_channels, 1), nn.Tanh())
        self.temperature = nn.Parameter(torch.log(torch.ones(1, key_channels, 1) * 10))
        self.softmax = nn.Softmax(dim=-1)
        self.attn_proj = nn.Conv1d(key_channels, channels, 1)

    def forward(self, x):
        q, k, v = self.q_conv(x), self.k_conv(x), self.v_conv(x)
        q_local, v_local = self.q_dwconv(q), self.v_dwconv(v)
        attn_map_raw = q_local * k
        attn_map_gated = self.attention_map_generator(attn_map_raw)
        attention_logits = attn_map_gated * self.temperature.exp()
        attention_weights = self.softmax(attention_logits)
        attention_projected = self.attn_proj(attention_weights)
        x_out = attention_projected * v_local
        return x_out


class Local_Offset_Generator(nn.Module):
    def __init__(self, channels, key_channels, kernel_size=3):
        super().__init__()
        self.local_feature_extractor = Point_AttnConv_Complete(channels=channels, key_channels=key_channels,
                                                               kernel_size=kernel_size)
        self.offset_processor = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels),
                                              nn.ReLU())

    def forward(self, x):
        feature_local = self.local_feature_extractor(x)
        raw_offset = x - feature_local
        processed_offset = self.offset_processor(raw_offset)
        return processed_offset


class Point_ConvFFN_Faithful(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv1d(in_channels, hidden_channels, 1, bias=False)
        self.act = act_layer()
        self.dwconv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2,
                                groups=hidden_channels, bias=False)
        self.fc2 = nn.Conv1d(hidden_channels, out_channels, 1, bias=False)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# --- 【核心修改】为我们的主模块添加“开关” ---
class Clo_Point_Block_Ablation(nn.Module):
    def __init__(self, channels, key_channels, pool_ratio=4, ffn_expand_ratio=4,
                 use_global_branch=True, use_local_branch=True):
        super().__init__()
        self.use_global_branch = use_global_branch
        self.use_local_branch = use_local_branch

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        if self.use_global_branch:
            self.global_offset_generator = SA_Layer_Efficient_Offset(channels=channels, pool_ratio=pool_ratio)

        if self.use_local_branch:
            self.local_offset_generator = Local_Offset_Generator(channels=channels, key_channels=key_channels)

        if self.use_global_branch and self.use_local_branch:
            self.fusion_conv = nn.Conv1d(channels, channels, 1, bias=False)
            self.fusion_bn = nn.BatchNorm1d(channels)

        self.ffn = Point_ConvFFN_Faithful(
            in_channels=channels, hidden_channels=channels * ffn_expand_ratio, out_channels=channels)

    def forward(self, x):
        x_norm1 = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        offset_global = self.global_offset_generator(x_norm1) if self.use_global_branch else 0
        offset_local = self.local_offset_generator(x_norm1) if self.use_local_branch else 0

        if self.use_global_branch and self.use_local_branch:
            total_offset = self.fusion_bn(self.fusion_conv(offset_global + offset_local))
        else:
            total_offset = offset_global + offset_local  # 如果只有一个分支，直接使用其输出

        x_after_branches = x + total_offset

        x_norm2 = self.norm2(x_after_branches.permute(0, 2, 1)).permute(0, 2, 1)
        offset_ffn = self.ffn(x_norm2)
        x_final = x_after_branches + offset_ffn

        return x_final


# --- 【新增模块】用于对比实验的标准注意力模块 ---
class StandardAttentionBlock(nn.Module):
    def __init__(self, channels, pool_ratio=4, ffn_expand_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        # 使用一个标准的自注意力层 (非偏移量版本)
        self.attn = SA_Layer_Efficient(channels=channels, pool_ratio=pool_ratio)
        self.ffn = Point_ConvFFN_Faithful(
            in_channels=channels, hidden_channels=channels * ffn_expand_ratio, out_channels=channels)

    def forward(self, x):
        x_norm1 = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_after_attn = self.attn(x_norm1)  # 注意这里是标准残差连接

        x_norm2 = self.norm2(x_after_attn.permute(0, 2, 1)).permute(0, 2, 1)
        x_after_ffn = self.ffn(x_norm2)
        x_final = x_after_attn + x_after_ffn
        return x_final


# 辅助类，用于标准注意力
class SA_Layer_Efficient(nn.Module):
    def __init__(self, channels, pool_ratio=4):
        super().__init__()
        self.pool_ratio = pool_ratio
        if self.pool_ratio > 1: self.pool = nn.AvgPool1d(kernel_size=pool_ratio, stride=pool_ratio)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k, x_v = self.k_conv(x), self.v_conv(x)
        if self.pool_ratio > 1: x_k, x_v = self.pool(x_k), self.pool(x_v)
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(attention, x_v.permute(0, 2, 1)).permute(0, 2, 1)
        # 标准注意力返回的是 x + F(x)，而不是偏移量
        return x + self.act(self.after_norm(self.trans_conv(x_r)))


# --- 层次化模块现在也接受ablation_cfg ---
class SG_Trans_Block_Ablation(nn.Module):
    def __init__(self, npoint, nsample, in_channels, out_channels, key_channels, pool_ratio, ffn_expand_ratio,
                 ablation_cfg):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample

        self.local_op = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        # 根据ablation_cfg决定使用哪个Transformer块
        if ablation_cfg.get('use_standard_attention', False):
            self.transformer_block = StandardAttentionBlock(
                channels=out_channels, pool_ratio=pool_ratio, ffn_expand_ratio=ffn_expand_ratio)
        else:
            self.transformer_block = Clo_Point_Block_Ablation(
                channels=out_channels, key_channels=key_channels, pool_ratio=pool_ratio,
                ffn_expand_ratio=ffn_expand_ratio,
                use_global_branch=ablation_cfg.get('use_global', True),
                use_local_branch=ablation_cfg.get('use_local', True))

    def forward(self, xyz, features):
        features_permuted = features.permute(0, 2, 1)
        new_xyz, grouped_features = sample_and_group(self.npoint, self.nsample, xyz, features_permuted)
        grouped_features = grouped_features.permute(0, 3, 2, 1)
        new_features = self.local_op(grouped_features)
        new_features = torch.max(new_features, 2)[0]
        processed_features = self.transformer_block(new_features)
        return new_xyz, processed_features


# --- 最终的主模型，现在接受ablation_cfg ---
class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 从配置对象中解析参数
        num_class = cfg.num_class
        input_dim = cfg.input_dim
        ablation_cfg = cfg.ablation_cfg
        if ablation_cfg is None:
            ablation_cfg = {}  # 默认为完整模型

        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.stage1 = SG_Trans_Block_Ablation(npoint=512, nsample=32, in_channels=64, out_channels=128, key_channels=32,
                                              pool_ratio=4, ffn_expand_ratio=4, ablation_cfg=ablation_cfg)
        self.stage2 = SG_Trans_Block_Ablation(npoint=256, nsample=32, in_channels=128, out_channels=256,
                                              key_channels=64, pool_ratio=4, ffn_expand_ratio=4,
                                              ablation_cfg=ablation_cfg)
        self.stage3 = SG_Trans_Block_Ablation(npoint=128, nsample=32, in_channels=256, out_channels=512,
                                              key_channels=128, pool_ratio=4, ffn_expand_ratio=4,
                                              ablation_cfg=ablation_cfg)

        fused_feature_dim = 128 + 256 + 512

        self.classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_class))

    def forward(self, x):
        # 1. 保留原始形状的点云用于几何操作 (如 sample_and_group)
        xyz = x  # xyz shape is [B, N, C]

        # 2. 将 x 的维度重排以适应卷积层
        # from [B, N, C] -> [B, C, N]
        features = x.permute(0, 2, 1)  # features shape is now [B, C, N]

        # 3. 将正确形状的张量送入 stem
        features = self.stem(features)  # Input [B, C, N] -> Output [B, 64, N]

        # 4. 依次通过层次化主干网络
        xyz1, features1 = self.stage1(xyz, features)
        xyz2, features2 = self.stage2(xyz1, features1)
        xyz3, features3 = self.stage3(xyz2, features2)

        # 5. 多尺度特征融合
        global_feat1 = torch.max(features1, 2)[0]
        global_feat2 = torch.max(features2, 2)[0]
        global_feat3 = torch.max(features3, 2)[0]

        fused_global_feature = torch.cat([global_feat1, global_feat2, global_feat3], dim=1)

        # 6. 分类
        logits = self.classifier(fused_global_feature)
        return logits


# --- 使用示例 ---
if __name__ == '__main__':
    dummy_input = torch.rand(2, 3, 1024)

    # 实验 (a): 完整模型 (默认)
    print("--- 实验 (a): 完整模型 ---")
    model_full =  PointTransformerCls(num_class=40)
    output_full = model_full(dummy_input)
    print("输出形状:", output_full.shape)

    # 实验 (b): 移除局部分支
    print("\n--- 实验 (b): 仅全局分支 ---")
    ablation_cfg_b = {'use_local': False}
    model_global_only =  PointTransformerCls(num_class=40, ablation_cfg=ablation_cfg_b)
    output_global_only = model_global_only(dummy_input)
    print("输出形状:", output_global_only.shape)

    # 实验 (c): 移除全局分支
    print("\n--- 实验 (c): 仅局部分支 ---")
    ablation_cfg_c = {'use_global': False}
    model_local_only =  PointTransformerCls(num_class=40, ablation_cfg=ablation_cfg_c)
    output_local_only = model_local_only(dummy_input)
    print("输出形状:", output_local_only.shape)

    # 实验 (d): 使用标准自注意力
    print("\n--- 实验 (d): 标准自注意力 ---")
    ablation_cfg_d = {'use_standard_attention': True}
    model_standard =  PointTransformerCls(num_class=40, ablation_cfg=ablation_cfg_d)
    output_standard = model_standard(dummy_input)
    print("输出形状:", output_standard.shape)
