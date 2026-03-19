import os
import sys
import numpy as np
import pandas as pd
import random
import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from itertools import cycle 

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from MMD_mindspore import mmd_rbf
from datapipe_mindspore import build_dataset, get_dataset
from Net_mindspore import EEGAlignNet

# 设置运行环境
ms.set_context(mode=ms.PYNATIVE_MODE) 
ms.set_device("Ascend", 0)             

# 全局超参数
subjects = 15
epochs = 200
classes = 3
version = 1
result_dir = './result/'
os.makedirs(result_dir, exist_ok=True)

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)

set_random_seed(42)

# 自动生成日志文件名
while True:
    dfile = os.path.join(result_dir, f'EEGAlignNet_MS_LOG_{version:.0f}.csv')
    if not os.path.exists(dfile):
        break
    version += 1

df = pd.DataFrame()
df.to_csv(dfile, index=False)

class ProgressiveDomainScheduler:
    def __init__(self, total_epochs, max_mix_domains=3):
        self.total_epochs = total_epochs
        self.max_mix_domains = max_mix_domains

    def get_params(self, epoch):
        progress = epoch / self.total_epochs
        mix_ratio = min(1.0, progress)
        return {'mix_ratio': mix_ratio, 'mmd_weight': 0.1, 'domain_weight': 0.1}

def forward_fn(source_x, source_y, target_x, mix_x_list, params, model, crit, domain_crit):
    label = ops.Argmax(axis=1)(source_y.view(-1, classes))
    class_out, pred, domain_out, source_feature, adj_list, ep_list = model(source_x)
    loss_cls = crit(class_out, label)
    _, _, target_domain_out, target_feature, _, _ = model(target_x)
    mix_features = []
    mix_domain_outs = []
    for m_x in mix_x_list:
        _, _, m_domain_out, m_feat, _, _ = model(m_x)
        mix_features.append(m_feat)
        mix_domain_outs.append(m_domain_out)
    mmd_loss = ms.Tensor(0.0, ms.float32)
    for mix_feat in mix_features:
        mmd_loss += mmd_rbf(source_feature, mix_feat) * params['mmd_weight']
    mmd_loss += mmd_rbf(source_feature, target_feature) * params['mmd_weight']
    domain_preds = [domain_out, target_domain_out] + mix_domain_outs
    domain_labels = [ops.zeros(source_x.shape[0], ms.int32), ops.ones(target_x.shape[0], ms.int32) * (len(mix_x_list) + 1)]
    for i in range(len(mix_x_list)):
        domain_labels.append(ops.ones(mix_x_list[i].shape[0], ms.int32) * (i + 1))
    domain_preds = ops.Concat(axis=0)(domain_preds)
    domain_labels = ops.Concat(axis=0)(domain_labels)
    loss_domain = domain_crit(domain_preds, domain_labels) * params['domain_weight']
    total_loss = loss_cls + loss_domain + mmd_loss
    return total_loss, loss_cls, loss_domain, mmd_loss

def train(model, train_loader, target_loader, mix_loaders, crit, domain_crit, optimizer, grad_fn, scheduler, epoch):
    model.set_train(True)
    loss_all = 0
    num_samples = 0
    params = scheduler.get_params(epoch)
    active_mix_count = int(len(mix_loaders) * params['mix_ratio'])
    active_mix_loaders = mix_loaders[:max(1, active_mix_count)]
    train_iter = train_loader.create_tuple_iterator()
    target_iter = target_loader.create_tuple_iterator()
    mix_iters = [loader.create_tuple_iterator() for loader in active_mix_loaders]
    dynamic_loader = zip(train_iter, cycle(target_iter), *[cycle(m) for m in mix_iters])
    for batch_data in dynamic_loader:
        source_x, source_y = batch_data[0]
        target_x, _ = batch_data[1]
        mix_data = batch_data[2:]
        mix_x_list = [m[0] for m in mix_data]
        (total_loss, loss_cls, loss_domain, mmd_loss), grads = grad_fn(
            source_x, source_y, target_x, mix_x_list, params, model, crit, domain_crit)
        optimizer(grads)
        batch_size = source_x.shape[0]
        loss_all += loss_cls.asnumpy() * batch_size
        num_samples += batch_size
    return loss_all / num_samples, 0, 0

def evaluate(model, loader):
    model.set_train(False)
    predictions = []
    labels = []
    for data, label in loader.create_tuple_iterator():
        label_np = label.view(-1, classes).asnumpy()
        _, pred, _, _, _, _ = model(data)
        predictions.append(np.squeeze(pred.asnumpy()))
        labels.append(label_np)
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)
    AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(predictions, axis=-1))
    return AUC, acc

def main():
    build_dataset(subjects)
    print('Cross Validation Started (MindSpore Ascend NPU)')
    result_data = []
    best_acc_results = []
    domain_crit = nn.CrossEntropyLoss()
    scheduler = ProgressiveDomainScheduler(total_epochs=epochs)

    for cv_n in range(subjects):
        best_val_acc = 0.0
        train_ds, target_ds, mix_ds1, mix_ds2, mix_ds3 = get_dataset(subjects, cv_n)
        batch_size = 16
        train_loader = train_ds.batch(batch_size, drop_remainder=False)
        target_loader = target_ds.batch(batch_size, drop_remainder=False)
        mix_loader1 = mix_ds1.batch(batch_size, drop_remainder=False)
        mix_loader2 = mix_ds2.batch(batch_size, drop_remainder=False)
        mix_loader3 = mix_ds3.batch(batch_size, drop_remainder=False)
        mix_loaders = [mix_loader1, mix_loader2, mix_loader3]

        model = EEGAlignNet()
        trainable_params = [p for p in model.trainable_params() if 'rl_agent' not in p.name]
        optimizer = nn.Adam(trainable_params, learning_rate=1e-4)
        crit = nn.CrossEntropyLoss()
        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # --- 修改后的打印逻辑启动 ---
        print(f"\n🚀 [Subject {cv_n:02d}] 训练进度:")
        
        for epoch in range(1, epochs + 1):
            loss, _, _ = train(model, train_loader, target_loader, mix_loaders, crit, domain_crit,
                               optimizer, grad_fn, scheduler, epoch)
            _, train_acc = evaluate(model, train_loader)
            _, test_acc = evaluate(model, target_loader)

            if test_acc > best_val_acc:
                best_val_acc = test_acc
                save_path = os.path.join(result_dir, f'best_model_cv{cv_n:02d}.ckpt')
                ms.save_checkpoint(model, save_path)
            
            # 1. 达到满分 1.0 的逻辑：拉满进度条并跳出
            if best_val_acc >= 1.0:
                best_val_acc = 1.0 # 确保数值显示为1.0
                bar = '█' * 40
                print(f"Sub {cv_n:02d}: |{bar}| 100% [{epoch:03d}/{epochs}] (Early Stop: 1.0)")
                break
                
            # 2. 定期刷新进度条（每 20 个 Epoch 打印一行）
            if epoch % 20 == 0 or epoch == epochs:
                progress = int((epoch / epochs) * 40)
                bar = '█' * progress + '-' * (40 - progress)
                percent = (epoch / epochs) * 100
                print(f"Sub {cv_n:02d}: |{bar}| {percent:3.0f}% [{epoch:03d}/{epochs}]")

        # 3. 循环结束后打印最终 ACC
        print(f"✅ Subject {cv_n:02d} 训练完成 | 最终 ACC: {best_val_acc:.4f}")
        print("-" * 55)
        # --- 修改结束 ---

        best_acc_results.append(best_val_acc)
        result_data.append([cv_n, best_val_acc])
        df = pd.DataFrame(result_data, columns=['Subject', 'Best_Vacc'])
        df.to_csv(dfile, index=False)

    print("\n" + "="*40)
    print("      Individual Subject Results       ")
    print("="*40)
    for res in result_data:
        print(f"Subject {res[0]:02d}: Acc = {res[1]:.4f}")
    print("-" * 40)

    print("\n=== Final Experiment Results ===")
    print(f"Mean Vacc: {np.mean(best_acc_results):.4f} ± {np.std(best_acc_results):.4f}")

if __name__ == '__main__':
    main()