import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, GraphConv, DenseSAGEConv, dense_diff_pool, DenseGCNConv
from torch.nn import Linear, Dropout, Conv2d, MaxPool2d
from torch_geometric.utils import to_dense_batch
from RL import EdgeSelectionRL  # 假设存在的强化学习边选择模块

device = torch.device('cuda', 0)  # 使用GPU加速


# 梯度反转层（实现领域对抗训练的核心组件）
class GradientReversalLayer(torch.autograd.Function):
    """
    功能：在前向传播中保持输入不变，在反向传播中对梯度进行反转
    参数：
        x: 输入张量
        alpha: 梯度反转系数（控制对抗强度）
    """

    @staticmethod
    def forward(ctx, x, alpha):
        # 前向传播：直接传递输入
        ctx.alpha = alpha  # 保存系数用于反向传播
        return x.clone()  # 克隆张量避免原地操作

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播：对梯度进行反转
        reversed_grad = grad_output.neg() * ctx.alpha  # 负梯度乘以系数
        return reversed_grad, None  # 返回反转后的梯度，None表示不需要alpha的梯度


# 渐进式梯度反转模块
class Progressive_GRL(torch.nn.Module):
    """
    功能：管理梯度反转系数alpha的渐进式增长
    参数：
        alpha: 最大反转系数（默认1.0）
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.max_alpha = alpha
        self.current_alpha = 0.0  # 初始系数为0，随训练过程逐步增加

    def update_alpha(self, epoch, total_epochs):
        """根据当前epoch更新alpha值（线性增长策略）"""
        self.current_alpha = self.max_alpha * (epoch / total_epochs)

    def forward(self, x):
        """应用带当前alpha值的梯度反转层"""
        return GradientReversalLayer.apply(x, self.current_alpha)


# 自组织图卷积模块（RSCG）
class RSCGC(torch.nn.Module):
    """
    功能：通过强化学习进行边选择的自组织图卷积模块
    参数：
        in_features: 输入特征维度（例如65*32）
        bn_features: 瓶颈层特征维度（压缩维度）
        out_features: 输出特征维度
        topk: 保留的topk边数量
    """
    def __init__(self, in_features, bn_features, out_features, topk):
        super().__init__()
        self.channels = 62  # EEG通道数（固定）
        self.in_features = in_features #65*32
        self.bn_features = bn_features #64
        self.out_features = out_features #32
        self.topk = topk #10

        # 强化学习边选择模块
        self.rl_agent = EdgeSelectionRL(
            bn_features=bn_features,
            hidden_size=64,
            topk=topk
        )
        # 特征变换层
        self.bnlin = Linear(in_features, bn_features)  # 瓶颈层压缩特征
        self.gconv = DenseGCNConv(in_features, out_features)  # 图卷积层

    def forward(self, x):
        """
        输入形状: (batch*channels, conv_out_ch, height, width)
        例如: ([992, 32, 1, 65])，其中992 = batch_size * 62通道
        """
        # 形状变换: [batch*62, 32, 1, 65] -> [batch, 62, 32*65]
        batch_size = x.shape[0] // self.channels
        x = x.reshape(batch_size, self.channels, -1)  # [16, 62, 32*65]

        # 通过瓶颈层和非线性激活
        xa = torch.tanh(self.bnlin(x))  # [16, 62, 64]

        # 使用RL代理选择边
        edge_probs = self.rl_agent(xa)  # 生成边概率矩阵 [16, 62, 62]
        adj_mask = self.rl_agent.select_topk_edges(edge_probs)  # 选择topk边的掩码,adj_mask torch.Size([16, 62, 62])

        # 构建邻接矩阵
        adj = torch.matmul(xa, xa.transpose(2, 1))  # 相似度矩阵 [16, 62, 62]
        adj = torch.softmax(adj, dim=2)  # 归一化
        adj = adj * adj_mask.to(adj.device)  # 应用选择的边
        # x = torch.Size([16, 62, 2080])
        # adj = torch.Size([16, 62, 62])
        # x = torch.Size([16, 62, 960])
        # adj = torch.Size([16, 62, 62])
        # x = torch.Size([16, 62, 256])
        # adj = torch.Size([16, 62, 62])
        # 图卷积操作
        x = F.relu(self.gconv(x, adj))  # [16, 62, 32]
        return x,adj


# 领域分类器
class DomainClassifier(torch.nn.Module):
    """
    功能：区分特征来自源域还是目标域（领域对抗训练）
    参数：
        input_dim: 输入特征维度
    """
    def __init__(self, input_dim):
        super().__init__()
        self.fc = Linear(input_dim, 5)  # 假设有5个不同的领域需要区分

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)  # 返回领域概率分布


# 主网络架构
class EEGAlignNet(torch.nn.Module):
    """
    功能：跨领域EEG信号对齐与分类的主网络
    结构：
        - 三级CNN特征提取
        - 三级自组织图卷积
        - 领域对抗模块
        - 分类输出层
    """

    def __init__(self):
        super().__init__()
        # ----------------- 特征提取部分 -----------------
        # 第一级卷积块
        self.conv1 = Conv2d(1, 32, (5, 5))  # 输入: [bs,1,5,265], 输出: [bs,32,1,261]
        self.drop1 = Dropout(0.1)
        self.pool1 = MaxPool2d((1, 4))  # 输出: [bs,32,1,65]
        self.rscgc1 = RSCGC(65 * 32, 64, 32, 10)  # 输入: [bs*62,32,1,65]

        # 第二级卷积块
        self.conv2 = Conv2d(32, 64, (1, 5))  # 输出: [bs,64,1,61]
        self.drop2 = Dropout(0.1)
        self.pool2 = MaxPool2d((1, 4))  # 输出: [bs,64,1,15]
        self.rscgc2 = RSCGC(15 * 64, 64, 32, 10)

        # 第三级卷积块
        self.conv3 = Conv2d(64, 128, (1, 5))  # 输出: [bs,128,1,11]
        self.drop3 = Dropout(0.1)
        self.pool3 = MaxPool2d((1, 4))  # 输出: [bs,128,1,2]
        self.rscgc3 = RSCGC(2 * 128, 64, 32, 10)

        # ----------------- 对齐与分类部分 -----------------
        self.drop4 = Dropout(0.1)
        self.grl = Progressive_GRL(alpha=1.0)  # 渐进梯度反转层
        self.domain_classifier = DomainClassifier(62 * 32 * 3)  # 输入: 62*32*3=5952
        self.linend = Linear(62 * 32 * 3, 3)  # 最终分类层（3类）

    def forward(self, x, edge_index, batch):
        """
        输入:
            x: 原始EEG信号 [num_nodes, features]
            batch: 批次索引
        流程:
            1. 转换为密集批次
            2. 三级CNN特征提取
            3. 三级图卷积处理
            4. 领域对抗训练
            5. 分类输出
        """
        # 转换为密集批次格式
        x, mask = to_dense_batch(x, batch)  # x: [batch_size, max_nodes, features]
        x = x.reshape(-1, 1, 5, 265)  # [batch_size, 1, 5, 265]

        # ----------------- 特征提取流程 -----------------
        # 第一级处理
        x = F.relu(self.conv1(x))  # [bs,32,1,261]
        x = self.drop1(x)
        x = self.pool1(x)  # [bs,32,1,65] -> sogc输入形状[bs*62,32,1,65]
        x1,adj1 = self.rscgc1(x)  # [bs,62,32]

        # 第二级处理
        x = F.relu(self.conv2(x))  # [bs,64,1,61]
        x = self.drop2(x)
        x = self.pool2(x)  # [bs,64,1,15]
        x2,adj2 = self.rscgc2(x)  # [bs,62,32]

        # 第三级处理
        x = F.relu(self.conv3(x))  # [bs,128,1,11]
        x = self.drop3(x)
        x = self.pool3(x)  # [bs,128,1,2]
        x3,adj3 = self.rscgc3(x)  # [bs,62,32]

        # ----------------- 特征融合与对齐 -----------------
        features = torch.cat([x1, x2, x3], dim=1)  # [bs, 62*3, 32]
        features = self.drop4(features)
        flat_features = features.reshape(features.size(0), -1)  # [bs, 62*32*3]

        # 领域对抗训练
        reversed_features = self.grl(flat_features)  # 梯度反转后的特征
        domain_output = self.domain_classifier(reversed_features)  # 领域预测 [bs,5]

        # 分类输出
        class_output = self.linend(flat_features)  # 原始分类输出 [bs,3]
        pred = F.softmax(class_output, dim=1)  # 分类概率 [bs,3]

        return (
            class_output,  # 分类原始输出（用于计算交叉熵损失）
            pred,  # 分类概率分布
            domain_output.squeeze(),  # 领域预测结果（用于领域对抗损失）
            flat_features,  # 过渡特征
            [adj1, adj2, adj3]
        )