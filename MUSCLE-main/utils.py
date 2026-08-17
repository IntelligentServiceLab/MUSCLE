import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def metrics(uids, predictions, topk, test_labels):
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
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

def build_interaction_adj(train_matrix):
    import scipy.sparse as sp
    n_m, n_a = train_matrix.shape
    adj_mat = sp.dok_matrix((n_m + n_a, n_m + n_a), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = train_matrix.tolil()
    adj_mat[:n_m, n_m:] = R
    adj_mat[n_m:, :n_m] = R.T

    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv = np.zeros_like(rowsum)
    mask = rowsum > 0
    d_inv[mask] = np.power(rowsum[mask], -0.5)

    d_mat = sp.diags(d_inv)
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    return scipy_sparse_mat_to_torch_sparse_tensor(norm_adj).to(device)
