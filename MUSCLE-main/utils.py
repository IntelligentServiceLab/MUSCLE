import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def metrics(uids, predictions, topk, test_labels):
    """
    计算召回率（Recall）、归一化折损累积增益（NDCG）、精确率（Precision）和 F1 分数。
    :param uids: 用户ID列表
    :param predictions: 预测的物品列表
    :param topk: 取前k个物品进行评估
    :param test_labels: 测试集的真实标签
    :return: 召回率、NDCG、精确率和 F1 分数
    """
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    all_precision = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_precision = all_precision + hit/topk
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    recall = all_recall/user_num
    precision = all_precision/user_num
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return recall, all_ndcg/user_num, precision, f1_score

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将SciPy稀疏矩阵转换为PyTorch稀疏张量。
    :param sparse_mx: SciPy稀疏矩阵
    :return: PyTorch稀疏张量
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class TrnData(data.Dataset):
    """
    训练数据集类。
    """
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        """
        负采样操作。
        """
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        """
        返回数据集的长度。
        :return: 数据集长度
        """
        return len(self.rows)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。
        :param idx: 索引
        :return: 用户ID、正样本物品ID、负样本物品ID
        """
        return self.rows[idx], self.cols[idx], self.negs[idx]

    # utils.py 底部添加
def build_interaction_adj(train_matrix):
    import scipy.sparse as sp
    n_m, n_a = train_matrix.shape
    # 构建大邻接矩阵 (M+A, M+A)
    adj_mat = sp.dok_matrix((n_m + n_a, n_m + n_a), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = train_matrix.tolil()
    adj_mat[:n_m, n_m:] = R
    adj_mat[n_m:, :n_m] = R.T

    # 归一化 D^-0.5 * A * D^-0.5
    # 归一化 D^-0.5 * A * D^-0.5
    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv = np.zeros_like(rowsum)  # 先全部初始化为 0

    # 寻找度数大于 0 的有效节点
    mask = rowsum > 0
    # 只对这些有效节点进行 -0.5 次方计算，完美避开 0
    d_inv[mask] = np.power(rowsum[mask], -0.5)

    d_mat = sp.diags(d_inv)
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    # 调用你 utils.py 原有的转换函数
    return scipy_sparse_mat_to_torch_sparse_tensor(norm_adj).to(device)