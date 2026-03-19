import os  # 导入os模块，用于与操作系统交互
import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库
from torch_geometric.data import Data, InMemoryDataset  # 导入PyG库中的Data类和InMemoryDataset基类
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import scipy.io as sio  # 导入scipy库中的io模块，用于MAT文件的读取
import glob  # 导入glob模块，用于文件路径操作
from sklearn.metrics.pairwise import cosine_similarity

subjects = 15  # 使用的被试数量，用于LOSO（Leave-One-Subject-Out）交叉验证
classes = 3  # 分类的类别数
version = 1  # 设置版本号


def add_gaussian_noise(data, mean=0.0, std=1.0):
    """
    为输入数据添加随机高斯噪声。

    参数:
    - data: 输入数据，形状为 (数据数量, 通道数, 特征点)
    - mean: 高斯噪声的均值，默认为 0.0
    - std: 高斯噪声的标准差，默认为 1.0

    返回:
    - noisy_data: 添加噪声后的数据，形状与输入数据相同
    """
    # 生成与输入数据形状相同的高斯噪声
    noise = np.random.normal(mean, std, size=data.shape)
    noisy_data = data + noise

    return noisy_data

def calculate_cosine_similarity(X, testX):
    """
    计算 X 和 testX 之间的余弦相似度。

    参数:
    - X: 训练集数据，形状为 (630, 62, 1325)
    - testX: 测试集数据，形状为 (45, 62, 1325)

    返回:
    - similarities: 余弦相似度矩阵，形状为 (630, 45)
    """
    # 将数据重塑为 (样本数, 通道数 * 特征点)
    X_reshaped = X.reshape(X.shape[0], -1)
    testX_reshaped = testX.reshape(testX.shape[0], -1)
    # 计算余弦相似度
    similarities = cosine_similarity(X_reshaped, testX_reshaped)
    return similarities

def EEG_CutMix(X, testX, similarities, num_individuals=4):
    """
    执行 EEG-CutMix 操作。

    参数:
    - X: 训练集数据，形状为 (630, 62, 1325)
    - testX: 测试集数据，形状为 (45, 62, 1325)
    - similarities: 余弦相似度矩阵，形状为 (630, 45)
    - num_individuals: 选择的相似度最高的个体数量

    返回:
    - mixed_data: 混合后的数据，形状为 (180, 62, 1325)
    """
    mixed_data = []
    # 遍历 testX 中的每个样本
    for i in range(testX.shape[0]):
        # 获取当前 testX 样本的余弦相似度
        sim_scores = similarities[:, i]
        # 找到相似度最高的 num_individuals 个样本的索引
        top_indices = np.argsort(sim_scores)[-num_individuals:]
        # 从 X 中取出这些样本
        top_samples = X[top_indices]
        # 对每个样本进行 EEG-CutMix 操作
        for sample in top_samples:
            # 随机选择一个切割点
            cut_point = np.random.randint(0, sample.shape[1])
            # 执行 CutMix 操作
            mixed_sample = np.concatenate((testX[i, :, :cut_point], sample[:, cut_point:]), axis=1)
            mixed_data.append(mixed_sample)
    # 将混合数据转换为 numpy 数组
    mixed_data = np.array(mixed_data) # mixed_data shape:  (180, 62, 1325)
    return mixed_data

def to_categorical(y, num_classes=None, dtype='float32'):
    """
    将标签转换为one-hot编码。
    参数:
    - y: 标签数组
    - num_classes: 类别总数（默认为None）
    - dtype: 输出数据类型（默认为float32）
    返回:
    - one-hot编码的标签
    """
    y = np.array(y, dtype='int16')  # 将标签数组转为整数类型
    input_shape = y.shape  # 获取输入标签的形状
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])  # 如果标签数组的最后一维是1，则去除
    y = y.ravel()  # 将标签展平为1D数组
    if not num_classes:
        num_classes = np.max(y) + 1  # 默认类别数为标签的最大值+1
    n = y.shape[0]  # 获取样本数
    categorical = np.zeros((n, num_classes), dtype=dtype)  # 创建一个全零数组用于存储one-hot编码
    categorical[np.arange(n), y] = 1  # 设置one-hot编码
    output_shape = input_shape + (num_classes,)  # 计算输出形状
    categorical = np.reshape(categorical, output_shape)  # 重新调整数组形状
    return categorical  # 返回one-hot编码的标签

def to_categorical(y, num_classes=None, dtype='float32'):
    """
    将标签转换为one-hot编码。
    参数:
    - y: 标签数组
    - num_classes: 类别总数（默认为None）
    - dtype: 输出数据类型（默认为float32）
    返回:
    - one-hot编码的标签
    """
    y = np.array(y, dtype='int16')  # 将标签数组转为整数类型
    input_shape = y.shape  # 获取输入标签的形状
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])  # 如果标签数组的最后一维是1，则去除
    y = y.ravel()  # 将标签展平为1D数组
    if not num_classes:
        num_classes = np.max(y) + 1  # 默认类别数为标签的最大值+1
    n = y.shape[0]  # 获取样本数
    categorical = np.zeros((n, num_classes), dtype=dtype)  # 创建一个全零数组用于存储one-hot编码
    categorical[np.arange(n), y] = 1  # 设置one-hot编码
    output_shape = input_shape + (num_classes,)  # 计算输出形状
    categorical = np.reshape(categorical, output_shape)  # 重新调整数组形状
    return categorical  # 返回one-hot编码的标签


class EmotionDataset(InMemoryDataset):
    """
    自定义的情感数据集类，用于存储训练和测试数据。
    继承自InMemoryDataset，用于存储和处理数据，适配PyTorch Geometric框架。
    """

    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, edge_index=None, transform=None, pre_transform=None):
        """
        初始化函数
        参数:
        - stage: 数据集阶段（'Train' 或 'Test'）
        - root: 数据根目录
        - subjects: 被试数量
        - sub_i: 当前被试索引
        - X: 特征数据
        - Y: 标签数据
        - edge_index: 边的索引（默认为None）
        """
        self.stage = stage  # 存储数据集阶段（训练或测试）
        self.subjects = subjects  # 存储被试数量
        self.sub_i = sub_i  # 存储当前被试的索引
        self.X = X  # 存储特征数据
        self.Y = Y  # 存储标签数据
        self.edge_index = edge_index  # 存储边的索引（如果有的话）
        super().__init__(root, transform, pre_transform)  # 调用基类的构造函数
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)  # 加载已处理的数据

    @property
    def processed_file_names(self):
        """
        返回处理后的文件名，格式为 `V_版本号_阶段_LOSO_被试编号.dataset`。
        """
        return ['./V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(version, self.stage, self.subjects, self.sub_i)]  # 格式化并返回文件名

    def process(self):
        """
        处理数据并保存为PyG的格式。
        """
        data_list = []  # 用于存储每个样本的数据
        num_samples = np.shape(self.Y)[0]  # 获取样本数量
        for sample_id in tqdm(range(num_samples)):  # 遍历所有样本
            x = self.X[sample_id, :, :]  # 获取样本的特征数据
            x = torch.FloatTensor(x)  # 将特征数据转换为PyTorch的浮点张量
            y = torch.FloatTensor(self.Y[sample_id, :])  # 获取样本的标签数据，并转换为浮点张量
            data = Data(x=x, y=y)  # 创建PyG的数据对象，包含特征和标签
            data_list.append(data)  # 将数据添加到列表中
        data, slices = self.collate(data_list)  # 将数据列表合并为PyG支持的格式
        torch.save((data, slices), self.processed_paths[0])  # 将合并后的数据保存到磁盘


def normalize(data):
    """
    对数据进行归一化。
    参数:
    - data: 原始数据
    返回:
    - data: 归一化后的数据
    """
    mee = np.mean(data, 0)  # 计算均值
    data = data - mee  # 去均值
    stdd = np.std(data, 0)  # 计算标准差
    data = data / (stdd + 1e-7)  # 标准化，避免除零错误
    return data  # 返回归一化后的数据


def get_data():
    """
    加载和处理SEED数据集。
    返回:
    - sub_mov: 归一化的特征数据
    - sub_label: 对应的标签数据
    """
    path = 'E:/EEG_data/SEED/ExtractedFeatures/'  # 数据路径
    label = sio.loadmat(path + 'label.mat')['label']  # 加载标签文件
    sub_mov = []  # 用于存储所有被试的特征数据
    sub_label = []  # 用于存储所有被试的标签数据
    files = sorted(glob.glob(os.path.join(path, '*')))  # 获取文件路径并排序
    for sub_i in range(subjects):  # 遍历每个被试
        sub_files = files[sub_i * 3: sub_i * 3 + 3]  # 获取当前被试的文件
        mov_data = []  # 用于存储当前被试的特征数据
        for f in sub_files:  # 遍历每个文件
            data = sio.loadmat(f, verify_compressed_data_integrity=False)  # 读取.mat文件
            keys = data.keys()  # 获取文件中的所有键
            de_mov = [k for k in keys if 'de_movingAve' in k]  # 筛选包含特征数据的键
            mov_datai = []  # 用于存储当前文件的特征数据
            for t in range(15):  # 遍历一个.mat文件的15个时间段
                temp_data = data[de_mov[t]].transpose(0, 2, 1)  # 调整数据维度
                data_length = temp_data.shape[-1]  # 获取数据长度
                mov_i = np.zeros((62, 5, 265))  # 初始化矩阵
                mov_i[:, :, :data_length] = temp_data  # 填充矩阵
                mov_i = mov_i.reshape(62, -1)  # 将数据重塑为一维
                mov_datai.append(mov_i)  # 将一个电影数据添加到列表中
            mov_data.append(np.array(mov_datai))  # 将一个.mat文件所有电影的数据合并
        mov_data = np.vstack(mov_data)  # 合并一个人3个文件的数据
        mov_data = normalize(mov_data)  # 归一化数据
        sub_mov.append(mov_data)  # 添加到总数据列表中
        sub_label.append(np.hstack([label, label, label]).squeeze())  # 将标签复制并展平
    sub_mov = np.array(sub_mov)  # 将数据转为numpy数组
    sub_label = np.array(sub_label)  # 将标签转为numpy数组
    return sub_mov, sub_label  # 返回处理好的数据和标签


def build_dataset(subjects):
    """
    构建数据集并保存到磁盘。
    参数:
    - subjects: 被试数量
    """
    load_flag = True  # 初始化加载标志，防止数据重复加载
    for sub_i in range(subjects):  # 遍历每个被试
        # 构造当前被试的文件路径
        path = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(version, 'Train', subjects, sub_i)
        if not os.path.exists(path):  # 如果文件未存在
            if load_flag:  # 如果还未加载数据
                mov_coefs, labels = get_data()  # 加载数据
                used_coefs = mov_coefs  # 临时保存数据
                load_flag = False  # 更新标志
            # 构建训练集和测试集
            index_list = list(range(subjects))  # 获取所有被试的索引
            del index_list[sub_i]  # 删除当前被试
            test_index = sub_i  # 当前被试作为测试集
            train_index = index_list  # 剩余被试作为训练集
            # train_index = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
            # test_index = 7
            # used_coefs = (15, 45, 62, 1325)
            # 构建训练集数据
            X = used_coefs[train_index, :].reshape(-1, 62, 265 * 5)  # 重塑数据
            Y = labels[train_index, :].reshape(-1)  # 展平标签
            # 构建测试集数据
            testX = used_coefs[test_index, :].reshape(-1, 62, 265 * 5)  # 重塑数据
            testY = labels[test_index, :].reshape(-1)  # 展平标签
            # 转换为独热编码
            _, Y = np.unique(Y, return_inverse=True)  # 标签标准化
            Y = to_categorical(Y, classes)  # 独热编码
            _, testY = np.unique(testY, return_inverse=True)  # 标签标准化
            testY = to_categorical(testY, classes)  # 独热编码
            # X =  (630, 62, 1325)
            # testX =  (45, 62, 1325)
            similarities = calculate_cosine_similarity(X,testX)
            mixed_data1 = EEG_CutMix(X,testX,similarities)
            mixed_data2 = EEG_CutMix(X,testX,similarities)
            mixed_data3 = EEG_CutMix(X, testX, similarities)
            mixed_data1 = add_gaussian_noise(mixed_data1,0,0.6)
            mixed_data2 = add_gaussian_noise(mixed_data2, 0, 0.8)
            mixed_data3 = add_gaussian_noise(mixed_data3, 0, 1)
            NoneY = testY #这个是没有用的，只是为了适配
            # 创建EmotionDataset实例
            train_dataset = EmotionDataset('Train', './', subjects, sub_i, X, Y)  # 创建训练集数据集
            test_dataset = EmotionDataset('Test', './', subjects, sub_i, testX, testY)  # 创建测试集数据集
            mix_dataset1 = EmotionDataset('Mix1', './', subjects, sub_i, mixed_data1, testY)
            mix_dataset2 = EmotionDataset('Mix2', './', subjects, sub_i, mixed_data2, testY)
            mix_dataset3 = EmotionDataset('Mix3', './', subjects, sub_i, mixed_data3, testY)


def get_dataset(subjects, sub_i):
    train_path = f'./processed/V_{version}_Train_CV{subjects}_{sub_i}.dataset'
    target_path = f'./processed/V_{version}_Test_CV{subjects}_{sub_i}.dataset'
    max_path = f'./processed/V_{version}_Mix_CV{subjects}_{sub_i}.dataset'
    print("train_path ", train_path)  # 输出训练集路径
    print("target_path ", target_path)  # 输出目标领域路径
    print("max_path ", max_path)
    train_dataset = EmotionDataset('Train', './', subjects, sub_i)  # 获取训练集
    target_dataset = EmotionDataset('Test', './', subjects, sub_i)  # 获取目标领域数据
    mix_dataset1 = EmotionDataset('Mix1', './', subjects, sub_i)
    mix_dataset2 = EmotionDataset('Mix2', './', subjects, sub_i)
    mix_dataset3 = EmotionDataset('Mix3', './', subjects, sub_i)

    return train_dataset, target_dataset, mix_dataset1, mix_dataset2, mix_dataset3  # 返回训练集和目标领域数据
