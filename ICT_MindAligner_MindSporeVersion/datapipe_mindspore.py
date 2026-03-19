import os
import glob
import numpy as np
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import mindspore.dataset as ds

subjects = 15
classes = 3
version = 1

# ================= 1. 数据预处理与增强函数 (纯 NumPy 实现) =================

def normalize(data):
    mee = np.mean(data, 0)
    data = data - mee
    stdd = np.std(data, 0)
    data = data / (stdd + 1e-7)
    return data

def add_gaussian_noise(data, mean=0.0, std=1.0):
    noise = np.random.normal(mean, std, size=data.shape)
    return data + noise

def calculate_cosine_similarity(X, testX):
    X_reshaped = X.reshape(X.shape[0], -1)
    testX_reshaped = testX.reshape(testX.shape[0], -1)
    return cosine_similarity(X_reshaped, testX_reshaped)

def EEG_CutMix(X, testX, similarities, num_individuals=4):
    mixed_data = []
    for i in range(testX.shape[0]):
        sim_scores = similarities[:, i]
        top_indices = np.argsort(sim_scores)[-num_individuals:]
        top_samples = X[top_indices]
        for sample in top_samples:
            cut_point = np.random.randint(0, sample.shape[1])
            mixed_sample = np.concatenate((testX[i, :, :cut_point], sample[:, cut_point:]), axis=1)
            mixed_data.append(mixed_sample)
    return np.array(mixed_data)

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    return np.reshape(categorical, output_shape)

def get_data():
    """读取原始 SEED .mat 文件"""
    # ⚠️ 已经为你修正了带 _Dataset 的正确路径
    path = '/home/ma-user/work/SEED_Dataset/ExtractedFeatures/' 
    
    
    if not os.path.exists(os.path.join(path, 'label.mat')):
         raise FileNotFoundError(f"找不到 SEED 数据集路径: {path}。")

    label = sio.loadmat(os.path.join(path, 'label.mat'))['label']
    sub_mov = []
    sub_label = []
    files = sorted(glob.glob(os.path.join(path, '*')))
    files = [f for f in files if 'label.mat' not in f and 'readme.md' not in f]

    print("开始读取 SEED 原始 .mat 文件...")
    for sub_i in tqdm(range(subjects)):
        sub_files = files[sub_i * 3: sub_i * 3 + 3]
        mov_data = []
        for f in sub_files:
            data = sio.loadmat(f, verify_compressed_data_integrity=False)
            keys = data.keys()
            de_mov = [k for k in keys if 'de_movingAve' in k]
            mov_datai = []
            for t in range(15):
                temp_data = data[de_mov[t]].transpose(0, 2, 1)
                data_length = temp_data.shape[-1]
                mov_i = np.zeros((62, 5, 265))
                mov_i[:, :, :data_length] = temp_data
                mov_i = mov_i.reshape(62, -1)
                mov_datai.append(mov_i)
            mov_data.append(np.array(mov_datai))
        mov_data = np.vstack(mov_data)
        mov_data = normalize(mov_data)
        sub_mov.append(mov_data)
        sub_label.append(np.hstack([label, label, label]).squeeze())
    return np.array(sub_mov), np.array(sub_label)

# ================= 2. 核心：生成并保存 .npz 数据集 (方案 A 保底版) =================

def build_dataset(subjects):
    save_dir = './processed_ms'
    os.makedirs(save_dir, exist_ok=True)
    
    check_path = os.path.join(save_dir, f'V_{version}_Train_CV{subjects}_0.npz')
    if os.path.exists(check_path):
        print(f"检测到预处理数据已存在于 {save_dir}，跳过 build_dataset 阶段。")
        return

    print("未检测到 .npz 数据，开始构建方案 A (原版复刻) 数据集...")
    mov_coefs, labels = get_data()
    
    for sub_i in range(subjects):
        print(f"正在处理被试 {sub_i} 的交叉验证数据...")
        index_list = list(range(subjects))
        del index_list[sub_i]
        
        X = mov_coefs[index_list, :].reshape(-1, 62, 1325)
        Y = to_categorical(np.unique(labels[index_list, :].reshape(-1), return_inverse=True)[1], classes)
        
        testX = mov_coefs[sub_i, :].reshape(-1, 62, 1325)
        testY = to_categorical(np.unique(labels[sub_i, :].reshape(-1), return_inverse=True)[1], classes)

        similarities = calculate_cosine_similarity(X, testX)
        mix1 = add_gaussian_noise(EEG_CutMix(X, testX, similarities), 0, 0.6)
        mix2 = add_gaussian_noise(EEG_CutMix(X, testX, similarities), 0, 0.8)
        mix3 = add_gaussian_noise(EEG_CutMix(X, testX, similarities), 0, 1.0)
        
        # 【方案 A 核心动作】：强制截断，只取前 testY.shape[0] 个样本，100% 对齐 PyTorch 的 Bug 现象
        num_test = testY.shape[0]

        np.savez(os.path.join(save_dir, f'V_{version}_Train_CV{subjects}_{sub_i}.npz'), X=X, Y=Y)
        np.savez(os.path.join(save_dir, f'V_{version}_Test_CV{subjects}_{sub_i}.npz'), X=testX, Y=testY)
        
        # 注意这里：强制切片 [:num_test]
        np.savez(os.path.join(save_dir, f'V_{version}_Mix1_CV{subjects}_{sub_i}.npz'), X=mix1[:num_test], Y=testY)
        np.savez(os.path.join(save_dir, f'V_{version}_Mix2_CV{subjects}_{sub_i}.npz'), X=mix2[:num_test], Y=testY)
        np.savez(os.path.join(save_dir, f'V_{version}_Mix3_CV{subjects}_{sub_i}.npz'), X=mix3[:num_test], Y=testY)

# ================= 3. DataLoader =================

def get_dataset(subjects, cv_n, version=1):
    base_path = f'./processed_ms/V_{version}'
    paths = {
        'train': f'{base_path}_Train_CV{subjects}_{cv_n}.npz',
        'target': f'{base_path}_Test_CV{subjects}_{cv_n}.npz',
        'mix1': f'{base_path}_Mix1_CV{subjects}_{cv_n}.npz',
        'mix2': f'{base_path}_Mix2_CV{subjects}_{cv_n}.npz',
        'mix3': f'{base_path}_Mix3_CV{subjects}_{cv_n}.npz'
    }

    def load_npz_to_ds(file_path, shuffle=True):
        data = np.load(file_path)
        x = data['X'].astype(np.float32)
        y = data['Y'].astype(np.float32)
        # 去掉智能补丁，直接高速加载
        dataset = ds.NumpySlicesDataset(data=(x, y), column_names=["data", "label"], shuffle=shuffle)
        return dataset


    return (load_npz_to_ds(paths['train'], True), 
            load_npz_to_ds(paths['target'], False), 
            load_npz_to_ds(paths['mix1'], True), 
            load_npz_to_ds(paths['mix2'], True), 
            load_npz_to_ds(paths['mix3'], True))