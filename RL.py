import torch
import torch.nn as nn


class EdgeSelectionRL(nn.Module):
    """基于强化学习的边选择模块，动态生成图结构的邻接矩阵
    新增马尔可夫链机制：通过状态转移动态调整边选择概率"""

    def __init__(self, bn_features, hidden_size, topk):
        super().__init__()
        self.bn_features = bn_features  # 瓶颈层特征维度，例如：64
        self.hidden_size = hidden_size  # 隐藏层维度，例如：64
        self.topk = topk  # 每个节点最大连接数，例如：10

        # 策略网络结构（保持不变）
        self.actor = nn.Sequential(
            nn.Linear(bn_features * 2, hidden_size),  # 处理节点对特征
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 输出边存在可能性评分
        )
        self.edge_probs = None  # 当前批次的边概率矩阵

    def forward(self, xa):
        """前向传播逻辑保持不变，生成原始边概率矩阵,(16, 62, 64)"""
        batch_size, channels, bn_features = xa.shape  # 解析输入张量的形状
        # xa: (batch_size, channels, bn_features)，例如 (16, 62, 64)
        # - batch_size: 批次大小，例如 16
        # - channels: 每个图的节点数，例如 62
        # - bn_features: 每个节点的特征维度，例如 64

        # 生成节点对的组合特征
        xa_expanded1 = xa.unsqueeze(2).expand(-1, -1, channels, -1)
        # xa_expanded1: (batch_size, channels, channels, bn_features)，例如 (16, 62, 62, 64)
        # 在第2维插入新维度，并复制特征，形成每个节点与其他节点的组合
        xa_expanded2 = xa.unsqueeze(1).expand(-1, channels, -1, -1)
        # xa_expanded2: (batch_size, channels, channels, bn_features)，例如 (16, 62, 62, 64)
        # 在第1维插入新维度，复制特征，形成与其他节点的组合
        edge_feats = torch.cat([xa_expanded1, xa_expanded2], dim=-1)
        # edge_feats: (batch_size, channels, channels, bn_features*2)，例如 (16, 62, 62, 128)
        # 将两个节点的特征拼接，表示每对节点 (i, j) 的特征

        # 计算边存在逻辑值并通过Sigmoid转换
        edge_logits = self.actor(edge_feats).squeeze(-1)
        # edge_logits: (batch_size, channels, channels)，例如 (16, 62, 62)
        # 策略网络输出每对节点的边存在性评分，squeeze 去除最后一维 (1)
        self.edge_probs = torch.sigmoid(edge_logits)
        # self.edge_probs: (batch_size, channels, channels)，例如 (16, 62, 62)
        # 通过sigmoid将评分转换为概率值（0到1之间）
        return self.edge_probs

    def select_topk_edges(self, edge_probs):
        """基于马尔可夫链的边选择算法
        每次选择一条边后，增强相邻边的选择概率"""
        batch_size, channels, _ = edge_probs.shape  # edge_probs: (batch_size, channels, channels)
        device = edge_probs.device  # 获取张量所在的设备（CPU或GPU）

        # 初始化邻接矩阵掩码和临时概率矩阵
        adj_mask = torch.zeros_like(edge_probs)
        # adj_mask: (batch_size, channels, channels)，初始化为全0，用于记录选择的边
        current_probs = edge_probs.clone()
        # current_probs: (batch_size, channels, channels)，复制边概率矩阵，用于动态调整

        # 马尔可夫链参数：已选边对相邻边的影响系数
        alpha = 1.2  # 通过实验调整该参数，大于1表示增强相邻边的概率

        for _ in range(self.topk):  # 循环选择 topk 条边
            # 获取当前每个样本的最大概率边
            flattened = current_probs.view(batch_size, -1)
            # flattened: (batch_size, channels*channels)，例如 (16, 62*62)，展平为二维张量
            topk_indices = torch.argmax(flattened, dim=1)  # [batch_size]，例如 (16,)
            # 找到每个样本中当前概率最大的边的索引

            # 将一维索引转换为二维坐标
            rows = topk_indices // channels  # 行坐标，例如节点 i
            cols = topk_indices % channels  # 列坐标，例如节点 j

            # 更新邻接矩阵掩码
            batch_indices = torch.arange(batch_size, device=device)  # [0, 1, ..., batch_size-1]
            adj_mask[batch_indices, rows, cols] = 1.0  # 将选中的边位置置为1

            # 将已选边概率置为极小值防止重复选择
            current_probs[batch_indices, rows, cols] = -float('inf')
            # 将已选边的概率设为负无穷，避免重复选择

            # 马尔可夫状态转移：增强选中节点相关边的概率
            # 创建行掩码并增强概率
            row_mask = torch.zeros_like(current_probs)
            row_mask[batch_indices, rows, :] = 1  # 标记与选中行（节点 i）相关的边
            current_probs[row_mask.bool()] *= alpha  # 增强这些边的概率

            # 创建列掩码并增强概率
            col_mask = torch.zeros_like(current_probs)
            col_mask[batch_indices, :, cols] = 1  # 标记与选中列（节点 j）相关的边
            current_probs[col_mask.bool()] *= alpha  # 增强这些边的概率

        return adj_mask  # 返回最终的邻接矩阵掩码