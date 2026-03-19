import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from RL_mindspore import EdgeSelectionRL 

class GRL_Function(nn.Cell):
    """
    梯度反转层 (Gradient Reversal Layer) 的实现
    """
    def __init__(self):
        super(GRL_Function, self).__init__()
    def construct(self, x, alpha):
        return x
    def bprop(self, x, alpha, out, dout):
        # 反向传播时对梯度取反并乘以系数 alpha
        return (dout * -alpha, ops.zeros_like(alpha))

class Progressive_GRL(nn.Cell):
    """
    管理梯度反转系数 alpha 的渐进式增长
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.max_alpha = alpha
        self.current_alpha = ms.Parameter(ms.Tensor(0.0, ms.float32), requires_grad=False)
        self.grl = GRL_Function()
    def update_alpha(self, epoch, total_epochs):
        new_alpha = self.max_alpha * (epoch / total_epochs)
        ops.assign(self.current_alpha, ms.Tensor(new_alpha, ms.float32))
    def construct(self, x):
        return self.grl(x, self.current_alpha)

class DenseGCNConv_MS(nn.Cell):
    """
    [核心对齐] 密集图卷积层
    复刻了 PyTorch Geometric 的数值特性，特别是针对孤立节点的放大效果
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 显式指定初始化方式，与 PyTorch 像素级对齐
        self.weight = ms.Parameter(init.initializer(init.XavierUniform(), [in_channels, out_channels]))
        self.bias = ms.Parameter(init.initializer(init.Zero(), [out_channels]))

    def construct(self, x, adj):
        bs, num_nodes, _ = adj.shape
        I = ops.eye(num_nodes, num_nodes, ms.float32).expand_dims(0).broadcast_to(adj.shape)
        
        # 1. 强制自环处理 (Self-loop)
        adj = adj * (1.0 - I) + I
        
        # 2. 计算度矩阵
        degree = adj.sum(axis=-1)
        
        # 【神级复刻点】：PyTorch 默认 clamp(min=1e-5)，
        # 当节点无连接时，1/sqrt(1e-5) 会产生约 316.22 倍的特征放大效果
        degree_clamped = ops.maximum(degree, ms.Tensor(1e-5, ms.float32))
        degree_inv_sqrt = ops.pow(degree_clamped, -0.5)
        
        # 3. 对称归一化邻接矩阵
        deg_inv_sqrt_col = ops.ExpandDims()(degree_inv_sqrt, -1)
        deg_inv_sqrt_row = ops.ExpandDims()(degree_inv_sqrt, -2)
        adj_norm = deg_inv_sqrt_col * adj * deg_inv_sqrt_row
        
        # 4. 线性变换与聚合
        xw = ops.matmul(x, self.weight)
        out = ops.matmul(adj_norm, xw) + self.bias
        return out

class RSCGC(nn.Cell):
    """
    自组织图卷积模块 (Reinforced Self-Constructed Graph Convolution)
    """
    def __init__(self, in_features, bn_features, out_features, topk):
        super().__init__()
        self.channels = 62
        self.in_features = in_features
        self.bn_features = bn_features
        self.out_features = out_features
        self.topk = topk

        self.rl_agent = EdgeSelectionRL(bn_features=bn_features, hidden_size=64, topk=topk)
        
        # 【对齐初始化】：指定 HeUniform (Kaiming Uniform)
        bound = 1.0 / math.sqrt(in_features)
        self.bnlin = nn.Dense(in_features, bn_features, 
                             weight_init=init.HeUniform(math.sqrt(5)), 
                             bias_init=init.Uniform(bound))
        self.gconv = DenseGCNConv_MS(in_features, out_features)

    def construct(self, x):
        batch_size = x.shape[0] // self.channels
        x = x.reshape(batch_size, self.channels, -1)
        
        # 瓶颈层特征提取
        xa = ops.tanh(self.bnlin(x))
        
        # 强化学习动态选边
        edge_probs = self.rl_agent(xa)
        adj_mask = self.rl_agent.select_topk_edges(edge_probs)
        
        # 构建邻接矩阵
        adj = ops.matmul(xa, xa.transpose(0, 2, 1))
        adj = ops.softmax(adj, axis=2)
        adj = adj * adj_mask
        
        x = ops.relu(self.gconv(x, adj))
        return x, adj, edge_probs

class DomainClassifier(nn.Cell):
    """
    领域分类器 (用于 DANN 训练)
    """
    def __init__(self, input_dim):
        super().__init__()
        bound = 1.0 / math.sqrt(input_dim)
        self.fc = nn.Dense(input_dim, 5, 
                          weight_init=init.HeUniform(math.sqrt(5)), 
                          bias_init=init.Uniform(bound))
    def construct(self, x):
        return ops.softmax(self.fc(x), axis=1)

class EEGAlignNet(nn.Cell):
    """
    主网络架构：包含三级特征提取与 RSCG 模块
    """
    def __init__(self):
        super().__init__()
        # 定义各层的初始化范围 (bounds)
        bound1 = 1.0 / math.sqrt(1 * 5 * 5)      
        bound2 = 1.0 / math.sqrt(32 * 1 * 5)     
        bound3 = 1.0 / math.sqrt(64 * 1 * 5)     
        dense_bound = 1.0 / math.sqrt(62 * 32 * 3) 
        
        # 第一级
        self.conv1 = nn.Conv2d(1, 32, (5, 5), pad_mode='valid', has_bias=True, 
                              weight_init=init.HeUniform(math.sqrt(5)), bias_init=init.Uniform(bound1))
        self.drop1 = nn.Dropout(p=0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.rscgc1 = RSCGC(65 * 32, 64, 32, 10)

        # 第二级
        self.conv2 = nn.Conv2d(32, 64, (1, 5), pad_mode='valid', has_bias=True, 
                              weight_init=init.HeUniform(math.sqrt(5)), bias_init=init.Uniform(bound2))
        self.drop2 = nn.Dropout(p=0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.rscgc2 = RSCGC(15 * 64, 64, 32, 10)

        # 第三级
        self.conv3 = nn.Conv2d(64, 128, (1, 5), pad_mode='valid', has_bias=True, 
                              weight_init=init.HeUniform(math.sqrt(5)), bias_init=init.Uniform(bound3))
        self.drop3 = nn.Dropout(p=0.1)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.rscgc3 = RSCGC(2 * 128, 64, 32, 10)

        self.drop4 = nn.Dropout(p=0.1)
        self.grl = Progressive_GRL(alpha=1.0)
        self.domain_classifier = DomainClassifier(62 * 32 * 3)
        
        # 最终分类层
        self.linend = nn.Dense(62 * 32 * 3, 3, 
                              weight_init=init.HeUniform(math.sqrt(5)), 
                              bias_init=init.Uniform(dense_bound))

    def construct(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size * 62, 1, 5, 265)

        # 特征提取流水线
        x = ops.relu(self.conv1(x))
        x = self.drop1(x)
        x = self.pool1(x)
        x1, adj1, prob1 = self.rscgc1(x)

        x = ops.relu(self.conv2(x))
        x = self.drop2(x)
        x = self.pool2(x)
        x2, adj2, prob2 = self.rscgc2(x)

        x = ops.relu(self.conv3(x))
        x = self.drop3(x)
        x = self.pool3(x)
        x3, adj3, prob3 = self.rscgc3(x)

        # 特征融合
        features = ops.Concat(axis=1)([x1, x2, x3])
        features = self.drop4(features)
        flat_features = features.reshape(features.shape[0], -1)

        # 领域判别分支
        reversed_features = self.grl(flat_features)
        domain_output = self.domain_classifier(reversed_features)

        # 情感分类分支
        class_output = self.linend(flat_features)
        pred = ops.softmax(class_output, axis=1)

        return (class_output, pred, domain_output.squeeze(), flat_features, 
                [adj1, adj2, adj3], [prob1, prob2, prob3])