import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init

class EdgeSelectionRL(nn.Cell):
    def __init__(self, bn_features, hidden_size, topk):
        super(EdgeSelectionRL, self).__init__()
        self.bn_features = bn_features
        self.hidden_size = hidden_size
        self.topk = topk

        # 【核心对齐】：复刻 PyTorch 的 HeUniform 初始化
        bound1 = 1.0 / math.sqrt(bn_features * 2)
        bound2 = 1.0 / math.sqrt(hidden_size)
        self.actor = nn.SequentialCell(
            nn.Dense(bn_features * 2, hidden_size, 
                     weight_init=init.HeUniform(math.sqrt(5)), bias_init=init.Uniform(bound1)),
            nn.ReLU(),
            nn.Dense(hidden_size, 1, 
                     weight_init=init.HeUniform(math.sqrt(5)), bias_init=init.Uniform(bound2))
        )
        self.edge_probs = None
        self.tensor_scatter_update = ops.TensorScatterUpdate()

    def construct(self, xa):
        batch_size, channels, bn_features = xa.shape
        xa_expanded1 = xa.expand_dims(2).broadcast_to((batch_size, channels, channels, bn_features))
        xa_expanded2 = xa.expand_dims(1).broadcast_to((batch_size, channels, channels, bn_features))
        edge_feats = ops.concat((xa_expanded1, xa_expanded2), axis=-1)
        edge_logits = self.actor(edge_feats).squeeze(-1)
        self.edge_probs = ops.sigmoid(edge_logits)
        return self.edge_probs

    def select_topk_edges(self, edge_probs):
        batch_size, channels, _ = edge_probs.shape
        adj_mask = ops.zeros_like(edge_probs)
        current_probs = edge_probs 

        alpha = 1.2
        batch_indices = ops.arange(batch_size, dtype=ms.int32)

        for _ in range(self.topk):
            flattened = current_probs.reshape(batch_size, -1)
            topk_indices = ops.Argmax(axis=1)(flattened)

            rows = (topk_indices // channels).astype(ms.int32)
            cols = (topk_indices % channels).astype(ms.int32)

            indices = ops.stack((batch_indices, rows, cols), axis=1)

            updates_ones = ops.ones(batch_size, adj_mask.dtype)
            adj_mask = self.tensor_scatter_update(adj_mask, indices, updates_ones)

            updates_min = ops.fill(current_probs.dtype, (batch_size,), -1e9)
            current_probs = self.tensor_scatter_update(current_probs, indices, updates_min)

            row_onehot = ops.one_hot(rows, channels, ops.scalar_to_tensor(1.0, current_probs.dtype), ops.scalar_to_tensor(0.0, current_probs.dtype))
            col_onehot = ops.one_hot(cols, channels, ops.scalar_to_tensor(1.0, current_probs.dtype), ops.scalar_to_tensor(0.0, current_probs.dtype))

            row_mask = row_onehot.expand_dims(2).broadcast_to(current_probs.shape)
            col_mask = col_onehot.expand_dims(1).broadcast_to(current_probs.shape)

            multiplier = (1.0 + (alpha - 1.0) * row_mask) * (1.0 + (alpha - 1.0) * col_mask)
            current_probs = current_probs * multiplier

        return adj_mask