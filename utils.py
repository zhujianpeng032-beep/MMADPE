import numpy as np
import random
import torch
from torch_geometric.data import Data, Batch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.utils import resample
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from sklearn.decomposition import PCA
from torch_geometric.utils import degree
from torch_geometric.utils import get_laplacian, to_dense_adj

def move_to_device(obj, device):  # 将字典里的所有数据转移到指定的设备device上
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(list(obj), device))
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def k_fold(data, args):
    k = args.k_fold

    # k折交叉验证
    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']

    # 确定总的正样本数量
    total_positives = np.sum(Y == 1)

    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # 从训练集的负样本中下采样，使其数量为正样本数量的10倍
        X_train_pos = X_train[Y_train == 1]
        X_train_neg = X_train[Y_train == 0]
        Y_train_pos = Y_train[Y_train == 1]
        Y_train_neg = Y_train[Y_train == 0]
        X_train_neg = resample(X_train_neg, replace=False, n_samples=len(X_train_pos) * 1, random_state=42)
        Y_train_neg = resample(Y_train_neg, replace=False, n_samples=len(X_train_pos) * 1, random_state=42)

        # 合并正负样本
        X_train = np.concatenate([X_train_pos, X_train_neg])
        Y_train = np.concatenate([Y_train_pos, Y_train_neg])

        # 在测试集中，使负样本数量与正样本数量相等
        X_test_pos = X_test[Y_test == 1]
        X_test_neg = X_test[Y_test == 0]
        Y_test_pos = Y_test[Y_test == 1]
        Y_test_neg = Y_test[Y_test == 0]
        X_test_neg = resample(X_test_neg, replace=False, n_samples=len(X_test_pos), random_state=42)
        Y_test_neg = resample(Y_test_neg, replace=False, n_samples=len(X_test_pos), random_state=42)

        # 合并正负样本
        X_test = np.concatenate([X_test_pos, X_test_neg])
        Y_test = np.concatenate([Y_test_pos, Y_test_neg])

        # 扩展维度并转换类型
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')

        # 添加到列表中
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    # 把十折创建出来的训练集和测试集保存到原始数据集的目录之下
    for i in range(k):
        # 创建目录
        fold_dir = os.path.join(args.data_dir, 'fold', str(i))
        os.makedirs(fold_dir, exist_ok=True)  # exist_ok=True 表示如果目录已存在，则不会引发异常

        X_train1 = pd.DataFrame(data=np.concatenate((X_train_all[i], Y_train_all[i]), axis=1),
                                columns=['drug', 'disease', 'label'])
        X_train1.to_csv(os.path.join(fold_dir, 'data_train.csv'))

        X_test1 = pd.DataFrame(data=np.concatenate((X_test_all[i], Y_test_all[i]), axis=1),
                               columns=['drug', 'disease', 'label'])
        X_test1.to_csv(os.path.join(fold_dir, 'data_test.csv'))

    data['X_train'] = X_train_all  # 训练集索引
    data['X_test'] = X_test_all    # 测试集索引
    data['Y_train'] = Y_train_all  # 训练集标签
    data['Y_test'] = Y_test_all    # 测试集标签
    return data


def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()  # 转置
    values = torch.ones(len(edges))  # 创建一个大小为len(edges)大小的行向量，值全部为1
    # 创建一个参数为size大小的矩阵adj，按照edges_tensor里面描述的关系将矩阵adj对应点位的数值转换为另一个传入的参数values，其他的皆为0
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()  # 若为C数据集，则adj格式为tensor(663,409)
    return adj  # 代表药物和疾病关系矩阵


def data_processing(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))  # 药物和疾病关联矩阵
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])    # 获得药物和疾病关联矩阵中值大于等于1的索引,其实就是索引为1的，只是为了方便罢了
            else:
                zero_index.append([i, j])   # 获得药物和疾病关联矩阵中值为0的索引，其实就是所有药物和疾病没有关联的索引
    random.seed(args.random_seed)
    args = process(args)
    random.shuffle(one_index)  # 将one_index里面元素的顺序打乱
    random.shuffle(zero_index)  # 将zero_index里面元素的顺序打乱

    index = np.array(one_index + zero_index, dtype=int)  # 获得药物和疾病关联矩阵中所有样本的索引,即所有正负样本的索引
    # 生成一个标签列表,前len(one_index)长度的标签设置为1,后len(zero_index)长度的标签设置为0
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)

    # 获得药物和疾病关联矩阵中所有样本的列表,第三列的意思是标签,用1来表示正样本,0来表示负样本
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]    # 药物和疾病关联矩阵中正样本的列表
    drdi_n = samples[samples[:, 2] == 0, :]    # 药物和疾病关联矩阵中负样本的列表

    drs_mean = (data['drf'] + data['drg']) / 2
    dis_mean = (data['dip'] + data['dig']) / 2

    drs = np.where(data['drf'] == 0, data['drg'], drs_mean)  # 大小为663*663
    dis = np.where(data['dip'] == 0, data['dig'], dis_mean)  # 大小为409*409

# ----------------------------------------药物边索引和边权重的构造-----------------------------------------------------------

    drs_non_zero_positions = np.where(drs > args.similarity)  # 获取drs中大于0.3的索引
    # 转化为tensor格式
    drs_edge_index = torch.tensor(drs_non_zero_positions, dtype=torch.long)  # 药物相似性网络的边索引 需要
    # 将边索引从2×E转换为E×2，以便用于NumPy的高级索引
    drs_edge_index_np = drs_edge_index.numpy().transpose()
    # 获取边权重
    drs_edge_weights = drs[drs_edge_index_np[:, 0], drs_edge_index_np[:, 1]]
    # 转化为tensor格式
    drs_edge_weights_tensor = torch.from_numpy(drs_edge_weights)  # 药物相似性网络的边权重的一维张量
    # 升维为 [num_edges, 1]
    drs_edge_attr = drs_edge_weights_tensor.unsqueeze(-1).float()  # 或 edge_weights.view(-1, 1) 需要

# ----------------------------------------------结束---------------------------------------------------------------------

# -----------------------------------------疾病边索引和边权重的构造----------------------------------------------------------

    # 获取dis中大于0.3的索引
    dis_non_zero_positions = np.where(dis > args.similarity)  # 使用np.where替代np.nonzero并添加条件
    # 转化为tensor格式
    dis_edge_index = torch.tensor(dis_non_zero_positions, dtype=torch.long)  # 疾病相似性网络的边索引 需要
    # 将边索引从2×E转换为E×2，以便用于NumPy的高级索引
    dis_edge_index_np = dis_edge_index.numpy().transpose()
    # 获取边权重
    dis_edge_weights = dis[dis_edge_index_np[:, 0], dis_edge_index_np[:, 1]]
    # 转化为tensor格式
    dis_edge_weights_tensor = torch.from_numpy(dis_edge_weights)  # 疾病相似性网络的边权重的一维张量
    dis_edge_attr = dis_edge_weights_tensor.unsqueeze(-1).float()  # 或 edge_weights.view(-1, 1) 需要

# ----------------------------------------------结束---------------------------------------------------------------------

# ------------------------------   药物和疾病的关联特征矩阵初始构造，使用低维嵌入------------------------------ ------------------

    with torch.no_grad():  # 防止梯度传播
        drug_embedder = BinaryToDense(input_dim=drdi_matrix.shape[1], output_dim=128)
        disease_embedder = BinaryToDense(input_dim=drdi_matrix.T.shape[1], output_dim=128)
        drug_reshape_association_matrix = drug_embedder(drdi_matrix.float())  # [663,128]
        disease_reshape_association_matrix = disease_embedder(drdi_matrix.T.float())  # [409,128]

# ----------------------------------------------结束---------------------------------------------------------------------

# --------------------------------------多跳邻域聚合策略来构造最终的药物和疾病的关联特征矩阵------------------------ --------------

        drs_balance = torch.from_numpy(np.where(drs > args.similarity, 1, 0))  # 在这个相似性矩阵中,相似性大于0.4的为1,小于0.4的为0,且为tensor
        dis_balance = torch.from_numpy(np.where(dis > args.similarity, 1, 0))  # 在这个相似性矩阵中,相似性大于0.4的为1,小于0.4的为0,且为tensor
        drs_balance_edge_index = torch.nonzero(drs_balance).T  # 获取不为0的索引
        dis_balance_edge_index = torch.nonzero(dis_balance).T  # 获取不为0的索引
        drug_multi_hop_aggregation = multi_hop_agg(drs_balance, drug_reshape_association_matrix, args.hops)   #  drs_balance代表药物的可达图
        disease_multi_hop_aggregation = multi_hop_agg(dis_balance, disease_reshape_association_matrix, args.hops)   #  dis_balance代表疾病的可达图

# ----------------------------------------------结束---------------------------------------------------------------------

# ----------------------------使用拉普拉斯位置编码和随机游走位置编码生成拉普拉斯特征向量和随机游走特征向量-----------------------------

    Drug_feature = data['Drug_feature']  # 药物Mol2vec语义特征 [663,64]
    Dis_feature = data['Dis_feature']  # 疾病Mesh语义特征 [409,64]

    drs_pe_EquivStableLapPE = compute_laplacian_pe(Drug_feature.shape[0], drs_edge_index, 16)  # [663,16] 构造的药物拉普拉斯特征向量
    dis_pe_EquivStableLapPE = compute_laplacian_pe(Dis_feature.shape[0], dis_edge_index, 16)  # [409,16] 构造的疾病拉普拉斯特征向量
    drs_pe_RWSE = compute_rwse(Drug_feature.shape[0], drs_edge_index, 8)  # [663,8] 构造的药物随机游走特征向量
    dis_pe_RWSE = compute_rwse(Dis_feature.shape[0], dis_edge_index, 8)  # [409,8] 构造的疾病随机游走特征向量
    drs_batch = torch.zeros(Drug_feature.shape[0], dtype=torch.long)  # 单图全0，多图需连续且无跨图连接
    dis_batch = torch.zeros(Dis_feature.shape[0], dtype=torch.long)  # 单图全0，多图需连续且无跨图连接

# ----------------------------------------------结束---------------------------------------------------------------------

# ------------------------------------构造药物和疾病的Data数据类型，好让GraphGPS进行处理---------------------------------------

    dis_data = Data(
        x=Dis_feature,
        edge_index=dis_edge_index,
        edge_attr=dis_edge_attr,
        pe=dis_pe_EquivStableLapPE,  # 生成得到的位置编码
        rwse=dis_pe_RWSE,  # 生成得到的位置编码
        batch=dis_batch
    )

    drs_data = Data(
        x=Drug_feature,
        edge_index=drs_edge_index,
        edge_attr=drs_edge_attr,
        pe=drs_pe_EquivStableLapPE,  # 生成得到的位置编码
        rwse=drs_pe_RWSE,  # 生成得到的位置编码
        batch=drs_batch
    )


# ----------------------------------------------结束---------------------------------------------------------------------

    data['drs_balance_edge_index'] = drs_balance_edge_index
    data['dis_balance_edge_index'] = dis_balance_edge_index
    data['drs_balance'] = drs_balance  # 663*128
    data['dis_balance'] = dis_balance  # 409*128
    data['drug_multi_hop_aggregation'] = drug_multi_hop_aggregation  # 大小为8*663*128
    data['disease_multi_hop_aggregation'] = disease_multi_hop_aggregation  # 大小为8*409*128
    data['drug_disease_matrix'] = drdi_matrix  # 大小为663*409
    data['drs'] = drs   # 最终得到的药物相似性矩阵  大小为663*663(若为C数据集)
    data['dis'] = dis   # 最终得到的疾病相似性矩阵  大小为409*409(若为C数据集)
    data['all_samples'] = samples  # 药物和疾病关联矩阵中所有样本的列表,第三列的意思是标签,用1来表示正样本,0来表示负样本
    data['all_drdi'] = samples[:, :2]  # 药物和疾病关联矩阵中所有样本的列表
    data['all_drdi_p'] = drdi_p   # 药物和疾病关联矩阵中所有正样本的列表
    data['all_drdi_n'] = drdi_n   # 药物和疾病关联矩阵中所有负样本的列表
    data['all_label'] = label     # 所有标签
    data['all_label_p'] = label_p  # 所有正标签

    return data, drs_data, dis_data


def get_data(args):
    data = dict()

    drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()  # 获取指纹相似性
    drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()  # 获取药物的高斯相似性

    dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()  # 获取疾病表型相似性
    dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()  # 获取疾病的高斯相似性

    data['drug_number'] = int(drf.shape[0])  # 药物数量
    data['disease_number'] = int(dip.shape[0])  # 疾病数量

    data['drf'] = drf
    data['drg'] = drg
    data['dip'] = dip
    data['dig'] = dig

    # 里面包含了所有药物和疾病关联的边
    data['drdi'] = pd.read_csv(args.data_dir + 'DrugDiseaseAssociation.csv', usecols=[0, 2], dtype=int, skiprows=0).to_numpy()
    # 药物mol2vec特征
    drug_feature = pd.read_csv(args.data_dir + 'Drug_mol2vec.csv', header=None).iloc[:, 1:]
    disease_feature = pd.read_csv(args.data_dir + 'DiseaseFeature.csv', header=None).iloc[:, 1:]

# -------------------------------对药物mol2vec和疾病mesh语义的特征向量进行处理，处理成相同的维数----------------------------------

    # PCA Dimensionality Reduction
    pca = PCA(n_components=args.feature_output, random_state=42)  # PCA降维 输出显示是64维

    PCA_drug_feature = pca.fit_transform(drug_feature.values)  # 对药物特征向量进行PCA降维
    PCA_dis_feature = pca.fit_transform(disease_feature.values)  # 对疾病特征向量进行PCA降维
    Dis_feature = torch.FloatTensor(PCA_dis_feature)  # [409,64]
    Drug_feature = torch.FloatTensor(PCA_drug_feature)  # [663,64]

# -----------------------------------------------结束--------------------------------------------------------------------

    data['Dis_feature'] = Dis_feature
    # 疾病Mesh语义特征
    data['Drug_feature'] = Drug_feature

    return data


def print_metrics(name, values):
    mean_value = np.mean(values)
    std_value = np.std(values)
    print(f'{name}: {values}')
    print(f'Mean {name}: {mean_value} ({std_value})')


class BinaryToDense(nn.Module):  # 将二值矩阵转换为高维向量
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        # 核心结构：线性层 + BatchNorm + SiLU激活
        self.embed = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.SiLU(),  # 比ReLU更平滑
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 初始化权重（修正点：使用ReLU的增益替代SiLU）
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.embed:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')  # 关键修改
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # 输入x: [num_nodes, input_dim] (二值矩阵)
        return self.embed(x.float())


# 多跳邻域聚合策略
def multi_hop_agg(adj: torch.Tensor, features: torch.Tensor, k: int) -> torch.Tensor:  # 参数为邻接矩阵，特征矩阵，k为跳数
    """
    多跳邻域聚合 (输出k+1跳特征，第一维为跳数)
    Args:
        adj: 邻接矩阵 [num_nodes, num_nodes]
        features: 节点特征 [num_nodes, feat_dim]
        k: 聚合跳数
    Returns:
        hops: 聚合结果 [k+1, num_nodes, feat_dim]
    """
    # 对称归一化邻接矩阵
    deg = adj.sum(1).clamp(min=1)
    norm_adj = adj * (deg.pow(-0.5).view(-1, 1) * deg.pow(-0.5).view(1, -1))

    # 初始化输出 (k+1跳)
    hops = torch.empty(k + 1, *features.shape, device=features.device)
    hops[0] = features  # 第0跳=自身

    # 迭代计算1-k跳
    for i in range(1, k + 1):
        features = torch.mm(norm_adj, features)
        hops[i] = features

    return hops


class PolynomialDecayLR(LRScheduler):  # 余弦退火学习率
    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                    self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


def compute_laplacian_pe(num_nodes, edge_index, k=16):
    # 计算归一化拉普拉斯矩阵
    edge_index, edge_weight = get_laplacian(
        edge_index, normalization='sym', num_nodes=num_nodes
    )
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]

    # 特征分解（忽略零特征值）
    eigvals, eigvecs = torch.linalg.eigh(L)
    eigvecs = eigvecs[:, eigvals > 1e-8]  # 过滤零特征值
    pe = eigvecs[:, :k]  # 取前k维

    # 归一化（可选）
    pe = (pe - pe.mean(dim=0)) / (pe.std(dim=0) + 1e-6)
    return pe.float()


def compute_rwse(num_nodes, edge_index, k=8):
    # 计算度数
    deg = degree(edge_index[0], num_nodes=num_nodes).float()

    # 示例：使用度数的对数和高阶统计量（实际任务需调整）
    pe = torch.stack([
        deg.log(),  # 对数度数
        (deg + 1e-6).pow(-0.5),  # 逆平方根度数
        deg.pow(0.5),  # 平方根度数
        torch.clamp(deg, max=10).float(),  # 截断度数
    ], dim=-1)

    # 如果维度不足k，通过重复或线性变换扩展
    if pe.size(1) < k:
        repeat_times = (k + pe.size(1) - 1) // pe.size(1)
        pe = pe.repeat(1, repeat_times)[:, :k]
    else:
        pe = pe[:, :k]

    # 归一化（可选）
    pe = (pe - pe.mean(dim=0)) / (pe.std(dim=0) + 1e-6)
    return pe.float()

def process(args):

    args.hops = {1: 5, 5: 1}.get(args.hops, args.hops)

    return args

