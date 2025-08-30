import os  # 导入os模块，用于文件路径处理
from MMD import mmd_rbf  # 导入MMD计算函数，用于特征对齐
import numpy as np  # 导入numpy，进行数组计算
import pandas as pd  # 导入pandas，用于数据存储
import torch  # 导入PyTorch核心库
from torch_geometric.data import DataLoader  # 从PyG导入DataLoader，用于批量加载图数据
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, auc  # 导入评估指标
from datapipe import build_dataset, get_dataset  # 导入自定义数据集构建函数
from Net import EEGAlignNet  # 导入自定义模型EEGAlignNet
import random  # 导入random模块，用于随机数生成
import torch.nn.functional as F  # 导入PyTorch的功能模块
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
from itertools import cycle


# 设置随机种子函数，确保实验可复现
def set_random_seed(seed=42):
    np.random.seed(seed)  # 设置numpy随机种子
    random.seed(seed)  # 设置Python随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    torch.backends.cudnn.deterministic = True  # 确保GPU计算确定性
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN优化，避免非确定性


# 定义全局参数
subjects = 15  # 被试数量，用于LOSO交叉验证
epochs = 200  # 训练轮数
classes = 3  # 分类任务的类别数
Network = EEGAlignNet  # 使用EEGAlignNet模型
device = torch.device('cuda', 0)  # 使用第一个GPU设备
version = 1  # 日志文件版本号

set_random_seed(42)  # 设置随机种子为42，确保结果可复现

# 创建结果目录和日志文件，避免覆盖已有文件
result_dir = './result/'
os.makedirs(result_dir, exist_ok=True)  # 确保结果目录存在

while True:
    dfile = os.path.join(result_dir, f'{Network.__name__}_LOG_{version:.0f}.csv')  # 日志文件路径
    if not os.path.exists(dfile):  # 如果文件不存在
        break  # 跳出循环
    version += 1  # 增加版本号

df = pd.DataFrame()  # 创建空DataFrame用于存储结果
df.to_csv(dfile, index=False)  # 初始化CSV文件


# 定义调度器类，用于动态调整训练参数
class ProgressiveDomainScheduler:
    def __init__(self, total_epochs, max_mix_domains=3):
        self.total_epochs = total_epochs  # 总训练轮数
        self.max_mix_domains = max_mix_domains  # 最大混合域数量（这里为3）

    def get_params(self, epoch):
        progress = epoch / self.total_epochs
        mix_ratio = min(1.0, progress)
        mmd_weight = 0.1
        domain_weight = 0.1
        return {'mix_ratio': mix_ratio, 'mmd_weight': mmd_weight, 'domain_weight': domain_weight}


# 定义训练函数
def train(model, train_loader, target_loader, mix_loaders, crit, domain_crit, optimizer, scheduler, epoch):
    model.train()  # 设置模型为训练模式
    loss_all = 0  # 累计分类损失
    domain_loss_all = 0  # 累计域判别损失
    mmd_loss_all = 0  # 累计MMD损失

    # 获取当前训练参数
    params = scheduler.get_params(epoch)  # 从调度器获取参数
    active_mix = int(len(mix_loaders) * params['mix_ratio'])  # 计算当前使用的混合域数量
    active_mix_loaders = mix_loaders[:max(1, active_mix)]  # 动态选择混合域，至少使用1个

    # 创建动态数据迭代器，根据当前选择的混合域数量组合数据
    dynamic_loader = zip(train_loader, target_loader, *active_mix_loaders)

    for batch_idx, (source_data, target_data, *mix_data) in enumerate(dynamic_loader):
        # 将数据移到GPU
        source_data = source_data.to(device)
        target_data = target_data.to(device)
        mix_data = [m.to(device) for m in mix_data]  # 动态处理多个混合域数据

        optimizer.zero_grad()  # 清零梯度

        # === 源域处理,cls损失计算 ===
        label = torch.argmax(source_data.y.view(-1, classes), dim=1)  # 获取源域标签
        class_out, _, domain_out, source_feature, _ = model(source_data.x, source_data.edge_index, source_data.batch)
        loss_cls = crit(class_out, label)  # 计算分类损失

        # === 目标域处理 ===
        _, _, target_domain_out, target_feature, _ = model(target_data.x, target_data.edge_index, target_data.batch)

        # === 混合域处理 ===
        mix_features = []  # 存储混合域特征
        mix_domain_outs = []  # 存储混合域判别输出
        for m_data in mix_data:
            _, _, m_domain_out, m_feat, _ = model(m_data.x, m_data.edge_index, m_data.batch)
            mix_features.append(m_feat)  # 收集混合域特征用于MMD计算
            mix_domain_outs.append(m_domain_out)  # 收集混合域判别输出

        # === 动态MMD损失计算 ===
        mmd_loss = 0
        for mix_feat in mix_features:
            mmd_loss += mmd_rbf(source_feature, mix_feat) * params['mmd_weight']  # 源域与每个混合域的MMD
        mmd_loss += mmd_rbf(source_feature, target_feature) * params['mmd_weight']  # 源域与目标域的MMD

        # === 动态域判别损失 ===
        domain_preds = [domain_out, target_domain_out] + mix_domain_outs  # 合并所有域的判别输出
        domain_labels = [
            torch.zeros(source_data.num_graphs, device=device),  # 源域标签为0
            torch.ones(target_data.num_graphs, device=device) * (len(active_mix_loaders) + 1)  # 目标域标签为最大值
        ]
        for i in range(len(active_mix_loaders)):
            domain_labels.append(torch.ones(mix_data[i].num_graphs, device=device) * (i + 1))  # 混合域标签为1,2,3

        domain_preds = torch.cat(domain_preds)  # 合并预测输出
        domain_labels = torch.cat(domain_labels).long()  # 合并标签并转为长整型
        loss_domain = domain_crit(domain_preds, domain_labels) * params['domain_weight']  # 计算域判别损失

        # === RL奖励计算 ===
        with torch.no_grad():
            _, pred, _, _, adj_list = model(source_data.x, source_data.edge_index, source_data.batch)
            source_y = source_data.y[::3]  # 假设每3个样本取一个标签（根据数据格式调整）
            if source_y.dim() == 1 or source_y.size(1) == 1:
                reward = accuracy_score(
                    (source_y - 1).cpu().numpy(),  # 标签从1开始，减1调整为从0开始
                    torch.argmax(pred, dim=1).cpu().numpy()
                )
            else:
                reward = accuracy_score(
                    torch.argmax(source_y, dim=1).cpu().numpy(),
                    torch.argmax(pred, dim=1).cpu().numpy()
                )
            connectivity_rewards = []
            lambda_coeff = 0.01  # 平衡系数
            eps = 1e-7
            for adj in adj_list:
                gram_matrix = torch.bmm(adj, adj.transpose(1, 2)) + eps * torch.eye(adj.size(1), device=adj.device)
                sign, logdet = torch.slogdet(gram_matrix)
                connectivity_rewards.append(logdet)
            graph_reward = lambda_coeff * torch.mean(torch.stack(connectivity_rewards))
            reward += graph_reward

        # === 策略梯度损失 ===
        edge_probs = model.rscgc1.rl_agent.edge_probs + model.rscgc2.rl_agent.edge_probs + model.rscgc3.rl_agent.edge_probs  # 获取RL代理的边概率
        if not isinstance(edge_probs, torch.Tensor):
            edge_probs = torch.tensor(edge_probs, device=device)  # 确保为张量
        rl_loss = -torch.log(edge_probs + 1e-7).mean() * reward  # 计算RL损失

        # === 总损失 ===
        total_loss = loss_cls + loss_domain + mmd_loss + rl_loss  # 合并所有损失
        total_loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # === 累计损失 ===
        loss_all += loss_cls.item() * source_data.num_graphs
        domain_loss_all += loss_domain.item() * domain_labels.size(0)
        mmd_loss_all += mmd_loss.item() * domain_labels.size(0)

    # 返回平均损失
    return (loss_all / len(train_loader.dataset),
            domain_loss_all / len(train_loader.dataset),
            mmd_loss_all / len(train_loader.dataset))


# 评估函数，返回指标、预测和标签
def evaluate(model, loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, classes)
            data = data.to(device)
            _, pred, _, _, _ = model(data.x, data.edge_index, data.batch)
            pred = pred.detach().cpu().numpy()
            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label.numpy())  # 修正：直接保存numpy数组

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    f1 = f1_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=1), average='macro')
    acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=-1))
    return AUC, acc, f1, predictions, labels


# 主函数
def main():
    build_dataset(subjects)
    print('Cross Validation')
    result_data = []
    best_acc_results = []

    # 为绘图准备的数据存储
    all_subjects_train_auc_history = []
    all_subjects_train_acc_history = []
    all_subjects_test_auc_history = []
    all_subjects_test_acc_history = []

    # 为ROC曲线准备的数据存储
    all_best_train_preds, all_best_train_labels = [], []
    all_best_test_preds, all_best_test_labels = [], []

    domain_crit = torch.nn.CrossEntropyLoss()
    scheduler = ProgressiveDomainScheduler(total_epochs=epochs)

    for cv_n in range(subjects):
        best_val_acc = 0.0
        best_epoch = 0

        best_train_preds_fold, best_train_labels_fold = None, None
        best_test_preds_fold, best_test_labels_fold = None, None

        current_subject_train_auc, current_subject_train_acc = [], []
        current_subject_test_auc, current_subject_test_acc = [], []

        train_dataset, test_dataset, mix_dataset1, mix_dataset2, mix_dataset3 = get_dataset(subjects, cv_n)
        _, target_dataset, _, _, _ = get_dataset(subjects, cv_n)

        train_loader = DataLoader(train_dataset, 16, shuffle=True)
        target_loader = DataLoader(target_dataset, 16)
        test_loader = DataLoader(test_dataset, 16)
        mix_loader1 = DataLoader(mix_dataset1, 16)
        mix_loader2 = DataLoader(mix_dataset2, 16)
        mix_loader3 = DataLoader(mix_dataset3, 16)
        mix_loaders = [mix_loader1, mix_loader2, mix_loader3]

        model = EEGAlignNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        crit = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            loss, domain_loss, mmd_loss = train(model, train_loader, target_loader, mix_loaders, crit, domain_crit,
                                                optimizer, scheduler, epoch)

            train_AUC, train_acc, _, train_preds, train_labels = evaluate(model, train_loader)
            current_subject_train_auc.append(train_AUC)
            current_subject_train_acc.append(train_acc)

            test_AUC, test_acc, _, test_preds, test_labels = evaluate(model, test_loader)
            current_subject_test_auc.append(test_AUC)
            current_subject_test_acc.append(test_acc)

            if test_acc > best_val_acc:
                best_val_acc = test_acc
                best_epoch = epoch + 1
                best_train_preds_fold, best_train_labels_fold = train_preds, train_labels
                best_test_preds_fold, best_test_labels_fold = test_preds, test_labels

            print(
                f'CV{cv_n:02d}, EP{epoch + 1:03d}, Loss:{loss:.4f} | Train AUC:{train_AUC:.4f}, Acc:{train_acc:.4f} | Test AUC:{test_AUC:.4f}, Acc:{test_acc:.4f} | Best Acc: {best_val_acc:.4f}')

            if best_val_acc == 1:
                print("This subject perfect!")
                break

        all_subjects_train_auc_history.append(current_subject_train_auc)
        all_subjects_train_acc_history.append(current_subject_train_acc)
        all_subjects_test_auc_history.append(current_subject_test_auc)
        all_subjects_test_acc_history.append(current_subject_test_acc)

        all_best_train_preds.append(best_train_preds_fold)
        all_best_train_labels.append(best_train_labels_fold)
        all_best_test_preds.append(best_test_preds_fold)
        all_best_test_labels.append(best_test_labels_fold)

        best_acc_results.append(best_val_acc)
        result_data.append([cv_n, best_epoch, best_val_acc])
        df = pd.DataFrame(result_data, columns=['Subject', 'Best_Epoch', 'Best_Vacc'])
        df.to_csv(dfile, index=False)

    print("\n=== Final Results ===")
    print(f"Mean Vacc: {np.mean(best_acc_results):.4f} ± {np.std(best_acc_results):.4f}")
    print("Individual Results:")
    for subj, acc in enumerate(best_acc_results):
        print(f"Subject {subj:02d}: {acc:.4f}")

    # === 绘图与数据保存部分 ===
    print("\nGenerating plots and saving data...")

    # --- 1. 处理并保存 AUC vs. Epochs 的数据 ---
    # 训练集
    train_auc_df = pd.DataFrame(all_subjects_train_auc_history).transpose()
    train_auc_df.columns = [f'Subject_{i + 1}' for i in range(subjects)]
    train_auc_filepath = os.path.join(result_dir, 'Train_AUC_vs_Epochs_Data.txt')
    train_auc_df.to_csv(train_auc_filepath, sep='\t', index_label='Epoch')
    print(f"Data for Train AUC plot saved to: {train_auc_filepath}")

    # 测试集
    test_auc_df = pd.DataFrame(all_subjects_test_auc_history).transpose()
    test_auc_df.columns = [f'Subject_{i + 1}' for i in range(subjects)]
    test_auc_filepath = os.path.join(result_dir, 'Test_AUC_vs_Epochs_Data.txt')
    test_auc_df.to_csv(test_auc_filepath, sep='\t', index_label='Epoch')
    print(f"Data for Test AUC plot saved to: {test_auc_filepath}")

    # --- 2. 准备绘图函数 ---
    def plot_metric_history(data_df, title, ylabel, save_path):
        plt.figure(figsize=(12, 8))
        colors = plt.get_cmap('tab20', subjects)
        for i, col in enumerate(data_df.columns):
            plt.plot(data_df.index + 1, data_df[col], label=col, color=colors(i))
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")

    # --- 3. 绘制 AUC vs. Epochs 图 ---
    plot_metric_history(train_auc_df, 'Train AUC vs. Epochs', 'AUC',
                        os.path.join(result_dir, 'Train_AUC_vs_Epochs.png'))
    plot_metric_history(test_auc_df, 'Test AUC vs. Epochs', 'AUC',
                        os.path.join(result_dir, 'Test_AUC_vs_Epochs.png'))

    # (可选) 绘制准确率曲线图
    # train_acc_df = pd.DataFrame(all_subjects_train_acc_history).transpose()
    # test_acc_df = pd.DataFrame(all_subjects_test_acc_history).transpose()
    # train_acc_df.columns = [f'Subject_{i + 1}' for i in range(subjects)]
    # test_acc_df.columns = [f'Subject_{i + 1}' for i in range(subjects)]
    # plot_metric_history(train_acc_df, 'Train Accuracy vs. Epochs', 'Accuracy', os.path.join(result_dir, 'Train_Accuracy_vs_Epochs.png'))
    # plot_metric_history(test_acc_df, 'Test Accuracy vs. Epochs', 'Accuracy', os.path.join(result_dir, 'Test_Accuracy_vs_Epochs.png'))

    # --- 4. 处理并保存 ROC 曲线的数据 ---
    # 训练集
    train_roc_labels = np.vstack(all_best_train_labels)
    train_roc_preds = np.vstack(all_best_train_preds)
    train_roc_labels_filepath = os.path.join(result_dir, 'Train_ROC_Data_Labels.txt')
    train_roc_preds_filepath = os.path.join(result_dir, 'Train_ROC_Data_Preds.txt')
    np.savetxt(train_roc_labels_filepath, train_roc_labels, fmt='%d')
    np.savetxt(train_roc_preds_filepath, train_roc_preds, fmt='%.8f')
    print(f"Data for Train ROC plot saved to: {train_roc_labels_filepath} and {train_roc_preds_filepath}")

    # 测试集
    test_roc_labels = np.vstack(all_best_test_labels)
    test_roc_preds = np.vstack(all_best_test_preds)
    test_roc_labels_filepath = os.path.join(result_dir, 'Test_ROC_Data_Labels.txt')
    test_roc_preds_filepath = os.path.join(result_dir, 'Test_ROC_Data_Preds.txt')
    np.savetxt(test_roc_labels_filepath, test_roc_labels, fmt='%d')
    np.savetxt(test_roc_preds_filepath, test_roc_preds, fmt='%.8f')
    print(f"Data for Test ROC plot saved to: {test_roc_labels_filepath} and {test_roc_preds_filepath}")

    # --- 5. 准备 ROC 绘图函数 ---
    def plot_roc_curves(y_true, y_score, plot_title, save_path):
        n_classes = y_true.shape[1]
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.figure(figsize=(10, 8))
        lw = 2
        plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.3f})',
                 color='navy', linestyle=':', linewidth=4)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw, label=f'ROC curve of class {i} (area = {roc_auc[i]:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Aggregated ROC - {plot_title}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")

    # --- 6. 绘制 ROC 曲线图 ---
    plot_roc_curves(train_roc_labels, train_roc_preds, "Train Data (from Best Epochs)",
                    os.path.join(result_dir, 'Train_ROC_Curve.png'))
    plot_roc_curves(test_roc_labels, test_roc_preds, "Test Data (from Best Epochs)",
                    os.path.join(result_dir, 'Test_ROC_Curve.png'))


if __name__ == '__main__':
    main()