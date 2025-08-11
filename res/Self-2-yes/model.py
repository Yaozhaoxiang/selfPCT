import torch
import torch.nn as nn
from pointnet_util import farthest_point_sample, index_points, square_distance


# 提取全局特征 [B, C, N] -》 [B, C, N]
class SA_Layer_Efficient_Offset(nn.Module):

    def __init__(self, channels, pool_ratio=4):
        super().__init__()
        self.pool_ratio = pool_ratio

        # 定义一个简单的平均池化层
        # kernel_size 和 stride 设为 pool_ratio，实现不重叠的池化
        if self.pool_ratio > 1:
            self.pool = nn.AvgPool1d(kernel_size=pool_ratio, stride=pool_ratio)

        self.qk_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv = self.qk_conv
        self.k_conv = self.qk_conv

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, C, N]

        # Q 的形状保持不变
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c'

        # K 和 V 先进行变换，然后进行池化
        x_k = self.k_conv(x)  # b, c', n
        x_v = self.v_conv(x)  # b, c, n

        if self.pool_ratio > 1:
            x_k = self.pool(x_k)  # b, c', n / pool_ratio
            x_v = self.pool(x_v)  # b, c,  n / pool_ratio

        # 计算注意力，此时 N_kv = n / pool_ratio
        energy = torch.bmm(x_q, x_k)  # b, n, n_kv
        attention = self.softmax(energy)
        # 我们可以选择保留或去掉 PCT 的 L1-norm
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        # 加权聚合，V也需要permute来匹配矩阵乘法
        # attention: [b, n, n_kv], x_v.permute: [b, n_kv, c]
        x_r = torch.bmm(attention, x_v.permute(0, 2, 1)).permute(0, 2, 1)  # b, c, n

        # 核心改动：只返回计算出的偏移量，不加回x
        x_offset = self.act(self.after_norm(self.trans_conv(x - x_r)))
        return x_offset # B,C,N

# 提取局部特征 [B, C, N] -》 [B, C, N]
class Point_AttnConv_Complete(nn.Module):

    def __init__(self, channels, key_channels, kernel_size=3):
        super().__init__()
        self.key_channels = key_channels

        # 1. 生成 q, k, v 的线性变换层
        self.q_conv = nn.Conv1d(channels, key_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, key_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)

        # 2. 对 Q 和 V 进行局部特征聚合的 DWConv
        #    注意：原论文图中是对Q和V操作，我们也遵循此设计
        self.q_dwconv = nn.Conv1d(key_channels, key_channels, kernel_size, padding=(kernel_size - 1) // 2,
                                  groups=key_channels)
        self.v_dwconv = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=channels)

        # 3. 上下文感知的注意力图生成器 (FC -> Swish -> FC -> Tanh)
        #    我们用 1x1 卷积来实现 FC (全连接层)
        self.attention_map_generator = nn.Sequential(
            nn.Conv1d(key_channels, key_channels, 1),
            nn.SiLU(),  # Swish激活函数在PyTorch中通常用 nn.SiLU()
            nn.Conv1d(key_channels, key_channels, 1),
            nn.Tanh()
        )

        # 可学习的温度系数，用于缩放注意力分数
        self.temperature = nn.Parameter(torch.log(torch.ones(1, key_channels, 1) * 10))

        self.softmax = nn.Softmax(dim=-1)  # 注意力是在点(N)的维度上计算

        # 新增一个1x1卷积，用于将 attention 的通道从 key_channels 映射回 channels
        self.attn_proj = nn.Conv1d(key_channels, channels, 1)

    def forward(self, x):
        # x: [B, C, N]

        # 步骤1: 生成 Q, K, V
        q = self.q_conv(x)  # [B, key_channels, N]
        k = self.k_conv(x)  # [B, key_channels, N]
        v = self.v_conv(x)  # [B, C, N]

        # 步骤2: Q 和 V 经过 DWConv 进行局部特征聚合
        q_local = self.q_dwconv(q)  # [B, key_channels, N]
        v_local = self.v_dwconv(v)  # [B, C, N]

        # 步骤3: 生成注意力图的核心
        # 3.1: 局部化的q与全局的k进行交互（哈达玛积）
        attn_map_raw = q_local * k  # [B, key_channels, N]

        # 3.2: 通过非线性变换和Tanh门控
        attn_map_gated = self.attention_map_generator(attn_map_raw)  # [B, key_channels, N]

        # 步骤4: 计算最终的注意力权重
        # 4.1: 应用可学习的温度系数
        # temperature.exp()确保温度为正
        attention_logits = attn_map_gated * self.temperature.exp() # [B, key_channels, N]
        attention_weights = self.softmax(attention_logits)  # [B, key_channels, N]

        # 将注意力权重的通道数投影回 C
        attention_projected = self.attn_proj(attention_weights)  # [B, key_channels, N] -> [B, channels, N]
        # 步骤5: 将注意力权重应用到局部化的 Value 上
        x_out = attention_projected * v_local  # [B, C, N] * [B, C, N]

        return x_out

# 局部偏移量  [B, C, N] -》 [B, C, N]
class Local_Offset_Generator(nn.Module):
    """
    一个专门的局部分支，其唯一职责就是计算"局部偏移量"。
    它内部封装了 Point_AttnConv_Complete 和处理偏移量的LBR网络。
    """

    def __init__(self, channels, key_channels, kernel_size=3):
        super().__init__()

        # 1. 核心的局部特征提取器
        self.local_feature_extractor = Point_AttnConv_Complete(
            channels=channels,
            key_channels=key_channels,
            kernel_size=kernel_size
        )

        # 2. 用于处理偏移量的 LBR (Linear-BatchNorm-ReLU) 网络
        #    我们用一个简单的1x1卷积来实现
        self.offset_processor = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x 是输入 Fin, 形状: [B, C, N]

        # 步骤1: 使用 AttnConv 提取局部特征 [B, C, N]
        feature_local = self.local_feature_extractor(x)

        # 步骤2: 计算原始偏移量 (Fin - F_local) [B, C, N]
        raw_offset = x - feature_local

        # 步骤3: 将原始偏移量送入LBR网络进行加工和非线性变换 [B, C, N]
        processed_offset = self.offset_processor(raw_offset)

        # 步骤4: 返回最终计算出的"局部修正建议" -> ΔF_local[B, C, N]
        return processed_offset

class Point_ConvFFN_Faithful(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        # 1. 升维层 (Pointwise Conv)
        self.fc1 = nn.Conv1d(in_channels, hidden_channels, 1, bias=False)
        self.act = act_layer()

        # 2. 深度可分离卷积层，用于在FFN内部混合局部信息
        self.dwconv = nn.Conv1d(hidden_channels, hidden_channels,
                                kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2,
                                groups=hidden_channels,  # Depth-wise
                                bias=False)

        # 3. 降维层 (Pointwise Conv)
        self.fc2 = nn.Conv1d(hidden_channels, out_channels, 1, bias=False)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        # x: [B, C, N]

        # 升维
        x = self.fc1(x)  # 升维:       [B, C_in, N] → [B, hidden, N]
        x = self.act(x)  # 激活:       [B, hidden, N]

        # 局部信息混合
        x = self.dwconv(x)  # 局部混合:    [B, hidden, N]

        # 降维
        x = self.fc2(x)  # 降维:       [B, hidden, N] → [B, C_out, N]
        x = self.drop(x)  # Dropout:    [B, C_out, N]

        return x

class Clo_Point_Block_Final_V2_With_Norm(nn.Module):
    """
    最终版模块 V2 的再升级版，加入了标准的 Pre-LN 结构以增强训练稳定性。
    """

    def __init__(self, channels, key_channels, pool_ratio=4, ffn_expand_ratio=4):
        super().__init__()

        # 两个归一化层，分别用于注意力和FFN之前
        # 对于点云 [B, C, N] 的数据，LayerNorm作用在通道(C)维度上
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # 全局和局部分支保持不变
        self.global_offset_generator = SA_Layer_Efficient_Offset(channels=channels, pool_ratio=pool_ratio)
        self.local_offset_generator = Local_Offset_Generator(channels=channels, key_channels=key_channels)

        # 融合层保持不变
        self.fusion_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.fusion_bn = nn.BatchNorm1d(channels)

        # FFN部分保持不变
        self.ffn = Point_ConvFFN_Faithful(
            in_channels=channels,
            hidden_channels=channels * ffn_expand_ratio,
            out_channels=channels
        )

    def forward(self, x):    # x : [B, C, N]


        # --- Pre-LN for Attention Branches ---
        # 归一化需要作用在 last dimension，所以需要 permute
        x_norm1 = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 两个分支都使用归一化后的 x_norm1 作为输入
        offset_global = self.global_offset_generator(x_norm1)  # [B, C, N]
        offset_local = self.local_offset_generator(x_norm1)  # [B, C, N]

        total_offset = self.fusion_bn(self.fusion_conv(offset_global + offset_local))

        # 第一个残差连接
        x_after_branches = x + total_offset

        # --- Pre-LN for FFN Branch ---
        x_norm2 = self.norm2(x_after_branches.permute(0, 2, 1)).permute(0, 2, 1)

        offset_ffn = self.ffn(x_norm2) # [B,C,N]

        # 第二个残差连接
        x_final = x_after_branches + offset_ffn

        return x_final

# 采样 [B, S, K, 2C]
def sample_and_group(npoint, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint

    # 使用 Farthest Point Sampling 从原始点集中选出 npoint 个分布均匀的“代表性点”。
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3] 坐标

    new_points = index_points(points, fps_idx)  # [B, npoint, C]

    # 每个关键点计算它与所有点的欧式距离，并找到最近的 K 个邻居。
    dists = square_distance(new_xyz, xyz)  # B x npoint x N
    idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    # 提取邻域内的点特征
    grouped_points = index_points(points, idx)  # [B, npoint, K, C]
    # 相对特征：邻居点特征 - 中心点特征
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)  # [B, npoint, K, C]

    # 拼接相对特征 + 中心点特征 [B, S, K, 2C]
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    # new_xyz	采样后的 S 个关键点坐标	[B, S, 3]
    # new_points	每个关键点邻域的 K 个点特征（相对特征 + 中心点复制）	[B, S, K, 2C]
    return new_xyz, new_points



class SG_Trans_Block_V2(nn.Module):

    def __init__(self, npoint, nsample, in_channels, out_channels, key_channels, pool_ratio, ffn_expand_ratio):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample

        # 1. 新的局部聚合器 (Local Aggregator)
        # 它的输入通道数是 2 * in_channels，因为 sample_and_group 进行了拼接
        # 输出通道数是我们为下一阶段指定的 out_channels
        self.local_op = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 2. Transformer 模块保持不变
        self.transformer_block = Clo_Point_Block_Final_V2_With_Norm(
            channels=out_channels,
            key_channels=key_channels,
            pool_ratio=pool_ratio,
            ffn_expand_ratio=ffn_expand_ratio
        )

    def forward(self, xyz, features):
        # xyz: [B, N, 3]
        # features: [B, C_in, N]

        features_permuted = features.permute(0, 2, 1)  # [B, N, C_in]

        # 1. 调用您提供的 sample_and_group 函数
        new_xyz, grouped_features = sample_and_group(
            self.npoint, self.nsample, xyz, features_permuted
        )
        # grouped_features 的形状: [B, npoint, nsample, 2 * C_in]

        # 2. 局部特征聚合
        # 调整维度以适应 Conv2d
        grouped_features = grouped_features.permute(0, 3, 2, 1)  # [B, 2 * C_in, nsample, npoint]

        # 通过我们新的 local_op 进行处理
        new_features = self.local_op(grouped_features)  # [B, C_out, nsample, npoint]

        # 最大池化，聚合邻域信息
        new_features = torch.max(new_features, 2)[0]  # [B, C_out, npoint]

        # 3. Transformer 特征提炼
        processed_features = self.transformer_block(new_features)

        return new_xyz, processed_features


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 从配置对象中解析参数
        num_class = cfg.num_class
        input_dim = cfg.input_dim

        # 初始特征嵌入层: 将输入的3维坐标提升到64维特征
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # 堆叠多个层次化的 SG-Trans 模块，构成网络主干
        # Stage 1: 2048 -> 2048 points, 64 -> 256 channels
        self.stage1 = SG_Trans_Block_V2(npoint=2048, nsample=64, in_channels=64, out_channels=256, key_channels=64,
                                     pool_ratio=4, ffn_expand_ratio=4)

        # Stage 2: 2048 -> 1024 points, 256 -> 256 channels
        self.stage2 = SG_Trans_Block_V2(npoint=1024, nsample=64, in_channels=256, out_channels=256, key_channels=64,
                                     pool_ratio=4, ffn_expand_ratio=4)

        # Stage 3 (您新增的层): 1024 -> 512 points, 256 -> 256 channels
        # 作用：在1024点的尺度上，进行更深层次的特征提炼
        self.stage3 = SG_Trans_Block_V2(npoint=512, nsample=64, in_channels=256, out_channels=256, key_channels=64,
                                     pool_ratio=4, ffn_expand_ratio=4)

        # Stage 4 (原来的stage3): 512 -> 256 points, 256 -> 256 channels
        self.stage4 = SG_Trans_Block_V2(npoint=256, nsample=64, in_channels=256, out_channels=256, key_channels=64,
                                     pool_ratio=4, ffn_expand_ratio=4)

        # 分类头的输入维度现在是 256 * 4 = 1024
        fused_feature_dim = 256 * 4

        self.classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        # x 从 DataLoader 传来，形状为 [B, N, C], e.g., [16, 2048, 3]

        # 1. 保留原始形状的点云用于几何操作 (如 sample_and_group)
        xyz = x  # xyz 的形状是 [B, N, 3]

        # 2. 【关键修正】将 x 的维度重排以适应卷积层
        # 从 [B, N, C] -> [B, C, N]
        features = x.permute(0, 2, 1)  # 现在 features 的形状是 [16, 3, 2048]
        features = self.stem(features)  # 输入 [B, 3, 2048] -> 输出 [B, 64, 2048]

        # --- 通过层次化主干网络，并保存每个阶段的输出 ---
        xyz1, features1 = self.stage1(xyz, features)  # -> features1: [B, 256, 512]
        xyz2, features2 = self.stage2(xyz1, features1)  # -> features2: [B, 256, 256]
        xyz3, features3 = self.stage3(xyz2, features2)  # -> features3: [B, 256, 128]
        xyz4, features4 = self.stage4(xyz3, features3)  # -> features4: [B, 256, 64]

        # --- 对每个阶段的输出进行全局池化 ---
        global_feat1 = torch.max(features1, 2)[0]  # [B, 256]
        global_feat2 = torch.max(features2, 2)[0]  # [B, 256]
        global_feat3 = torch.max(features3, 2)[0]  # [B, 256]
        global_feat4 = torch.max(features4, 2)[0]  # [B, 256]

        # --- 将所有全局特征拼接起来 ---
        fused_global_feature = torch.cat([global_feat1, global_feat2, global_feat3, global_feat4], dim=1)  # [B, 1024]

        # --- 分类 ---
        logits = self.classifier(fused_global_feature)

        return logits


# --- 使用示例 ---
if __name__ == '__main__':
    # 假设输入点云有1024个点
    dummy_input = torch.rand(2, 1024, 3)

    # 实例化模型
    model = PointTransformerCls(num_class=40)

    # 前向传播
    output_logits = model(dummy_input)

    print("宽体融合版模型已成功组装并通过测试！")
    print("输入形状:", dummy_input.shape)
    print("输出Logits形状:", output_logits.shape)
    # 预期输出: torch.Size([2, 40])