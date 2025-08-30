import torch
# MK_MMD
def mmd_rbf(source, target, source_labels=None, target_labels=None, kernel_mul=2.0, kernel_num=5, max_samples=1000,
            fix_sigma=None):
    """
    改进版自适应多核 MMD，支持类条件对齐和动态带宽调整

    参数:
    - source/target: 源域/目标域特征 (batch_size, feature_dim)
    - source_labels/target_labels: 类标签 (batch_size,)
    - kernel_mul: 核带宽倍数
    - kernel_num: 核数量
    - max_samples: 中位数估计最大采样数
    - fix_sigma: 固定带宽 (None时自动计算)

    返回:
    - mmd_loss: 加权后的 MMD 损失
    """
    # 类条件对齐模式
    if source_labels is not None and target_labels is not None:
        return _class_conditional_mmd(
            source, target, source_labels, target_labels,
            kernel_mul, kernel_num, max_samples, fix_sigma
        )

    # 计算基础核矩阵
    xx_kernel = _compute_kernel(source, source,
                                kernel_mul, kernel_num,
                                max_samples, fix_sigma)
    yy_kernel = _compute_kernel(target, target,
                                kernel_mul, kernel_num,
                                max_samples, fix_sigma)
    xy_kernel = _compute_kernel(source, target,
                                kernel_mul, kernel_num,
                                max_samples, fix_sigma)

    # 计算 MMD
    mmd = (xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean())
    return abs(mmd)


def _class_conditional_mmd(source, target, source_labels, target_labels,
                           kernel_mul, kernel_num, max_samples, fix_sigma):
    """类条件 MMD 计算"""
    classes = torch.unique(torch.cat([source_labels, target_labels]))
    total_mmd = 0.0
    valid_classes = 0

    for cls in classes:
        src_mask = (source_labels == cls)
        tgt_mask = (target_labels == cls)

        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            continue

        # 计算单类 MMD
        cls_mmd = mmd_rbf(source[src_mask], target[tgt_mask],
                          None, None, kernel_mul,
                          kernel_num, max_samples, fix_sigma)
        total_mmd += cls_mmd
        valid_classes += 1

    return total_mmd / valid_classes if valid_classes > 0 else 0.0


def _compute_kernel(x, y, kernel_mul, kernel_num, max_samples, fix_sigma):
    """自适应带宽核矩阵计算"""
    n_x, n_y = x.size(0), y.size(0)

    # 动态带宽估计
    if fix_sigma is None:
        # 随机采样估计中位数
        if max_samples < n_x * n_y:
            indices = torch.randperm(n_x * n_y)[:max_samples]
            x_flat = x.view(-1, x.shape[-1])
            y_flat = y.view(-1, y.shape[-1])
            sampled_pairs = x_flat[indices // n_y] - y_flat[indices % n_y]
            sigma = torch.median(torch.norm(sampled_pairs, dim=1) ** 2)
        else:
            sigma = torch.median(torch.cdist(x, y) ** 2)

        base_sigma = 2.0 * sigma  # 经验缩放因子
        sigma_list = [base_sigma * (kernel_mul ** i) for i in range(kernel_num)]
    else:
        sigma_list = [fix_sigma * (kernel_mul ** i) for i in range(kernel_num)]

    # 数值稳定性处理
    sigma_list = [max(s, 1e-6) for s in sigma_list]

    # 向量化核计算
    x = x.view(n_x, 1, -1)
    y = y.view(1, n_y, -1)
    l2_distance = torch.sum((x - y) ** 2, dim=2)  # (n_x, n_y)

    kernel_val = 0.0
    for sigma in sigma_list:
        kernel_val += torch.exp(-l2_distance / (2 * sigma))

    return kernel_val