import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dcg_at_k(scores, k):
    """计算 DCG@k"""
    scores = np.asarray(scores, dtype=float)[:k]
    if scores.size == 0:
        return 0.0
    return np.sum((2 ** scores - 1) / np.log2(np.arange(2, scores.size + 2)))


def ndcg_at_k(predicted_scores, true_scores, k):
    """计算 NDCG@k"""
    sorted_true_scores = [true for _, true in sorted(zip(predicted_scores, true_scores), reverse=True)]
    dcg = dcg_at_k(sorted_true_scores, k)
    idcg = dcg_at_k(sorted(true_scores, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


# ------------------------- 模型定义 -------------------------
class LightGCN(nn.Module):
    """处理Mashup-Mashup和API-API同构图的LightGCN"""

    def __init__(self, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.to(device)

    def forward(self, adj_mashup, adj_api, mashup_emb, api_emb):
        # 确保邻接矩阵与嵌入在同一设备
        adj_mashup = adj_mashup.to(mashup_emb.device)
        adj_api = adj_api.to(api_emb.device)

        # Mashup图传播
        mashup_emb_layers = [mashup_emb]
        for _ in range(self.n_layers):
            mashup_emb = torch.sparse.mm(adj_mashup, mashup_emb_layers[-1])
            mashup_emb_layers.append(mashup_emb)

        # API图传播
        api_emb_layers = [api_emb]
        for _ in range(self.n_layers):
            api_emb = torch.sparse.mm(adj_api, api_emb_layers[-1])
            api_emb_layers.append(api_emb)

        # 多阶平均
        final_mashup = torch.mean(torch.stack(mashup_emb_layers), dim=0)
        final_api = torch.mean(torch.stack(api_emb_layers), dim=0)
        final_mashup = F.normalize(final_mashup, p=2, dim=1)
        final_api = F.normalize(final_api, p=2, dim=1)

        return final_mashup, final_api


class MUSCLE(nn.Module):
    """
    我们将 TSSGCF 痕迹全面替换为 MVSF (Multi-View Semantic Fusion)
    """

    def __init__(self, num_mashups, num_apis, emb_dim, n_layers):
        super().__init__()
        self.num_mashups = num_mashups
        self.num_apis = num_apis
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        # 【修复1】：找回被我弄丢的图节点 Embedding 初始化
        self.mashup_emb = nn.Embedding(num_mashups, emb_dim).to(device)
        self.api_emb = nn.Embedding(num_apis, emb_dim).to(device)

        self.lightgcn = LightGCN(emb_dim, n_layers)

        # 【修复2】：恢复你原本正确的 384 维度 MLP
        self.text_mlp = nn.Sequential(
            nn.Linear(384, emb_dim),
        ).to(device)
        self.to(device)

    def interaction_gcn_forward(self, norm_adj, combined_embeddings):
        all_emb = [combined_embeddings]
        for _ in range(self.lightgcn.n_layers):
            combined_embeddings = torch.sparse.mm(norm_adj, combined_embeddings)
            all_emb.append(combined_embeddings)
        return torch.mean(torch.stack(all_emb), dim=0)

    def forward(self, adj_mashup, adj_api, interaction_adj, mashup_text_emb, api_text_emb):
        # 确保输入文本嵌入在正确设备
        mashup_text_emb = mashup_text_emb.to(device)
        api_text_emb = api_text_emb.to(device)

        # 提取当前特征
        mashup_emb = self.mashup_emb.weight
        api_emb = self.api_emb.weight

        # 【修复3】：恢复正确的 4 个参数传入 LightGCN
        g_mashup, g_api = self.lightgcn(adj_mashup, adj_api, mashup_emb, api_emb)

        # 【修复4】：恢复正确的 ego_embeddings 进行交互图传播
        ego_embeddings = torch.cat([mashup_emb, api_emb], dim=0)
        inter_res = self.interaction_gcn_forward(interaction_adj, ego_embeddings)
        inter_m, inter_a = torch.split(inter_res, [mashup_emb.shape[0], api_emb.shape[0]])

        # 第一次融合
        g_fused_mashup = (g_mashup + inter_m) / 2.0
        g_fused_api = (g_api + inter_a) / 2.0

        # 文本嵌入转换
        mashup_text_emb = self.text_mlp(mashup_text_emb)
        api_text_emb = self.text_mlp(api_text_emb)

        t_mashup = F.normalize(mashup_text_emb, p=2, dim=-1)
        t_api = F.normalize(api_text_emb, p=2, dim=-1)

        # 多模态融合
        final_mashup = (g_fused_mashup + t_mashup) / 2.0
        final_api = (g_fused_api + t_api) / 2.0

        return final_mashup, final_api, g_fused_mashup, g_fused_api, t_mashup, t_api, g_mashup, inter_m, g_api, inter_a

    def pred(self, adj_mashup, adj_api, interaction_adj, mashup_text_emb, api_text_emb, test_uids, test_mapping,
             train_mapping, top_k=5):
        test_uids = test_uids.to(device)
        forward_res = self.forward(adj_mashup, adj_api, interaction_adj, mashup_text_emb, api_text_emb)
        final_mashup = forward_res[0]
        final_api = forward_res[1]
        mashup_emb = final_mashup[test_uids]
        scores = torch.matmul(mashup_emb, final_api.T)

        all_recall = 0.0
        all_precision = 0.0
        all_ndcg = 0.0

        for i in range(test_uids.shape[0]):
            hit = 0
            user_id = test_uids[i].item()
            user_scores = scores[i]
            real_api = train_mapping[user_id]

            if isinstance(real_api, torch.Tensor):
                real_api = real_api.to(user_scores.device)
            else:
                real_api = torch.tensor(real_api, device=user_scores.device)

            user_scores[real_api] = -1e18
            _, top_k_indices = torch.topk(user_scores, top_k, dim=0, largest=True, sorted=True)
            needy_api = test_mapping[user_id]
            pred_indices = top_k_indices.cpu().numpy()

            for api in pred_indices:
                if api in needy_api:
                    hit += 1

            all_recall += hit / len(needy_api) if len(needy_api) > 0 else 0.0
            all_precision += hit / top_k
            ground_truth = [1 if api in needy_api else 0 for api in pred_indices]
            all_ndcg += ndcg_at_k(pred_indices, ground_truth, top_k)

        return (all_recall / test_uids.shape[0],
                all_precision / test_uids.shape[0],
                all_ndcg / test_uids.shape[0])



# ------------------------- 损失函数 -------------------------
def bpr_loss(pos_scores, neg_scores):
    """基于相似性得分的BPR损失"""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()




class TextualSimilarityLoss(nn.Module):
    def __init__(self, mashup_full_sim, api_full_sim, alpha1, alpha2, beta, lambda_tss):
        super().__init__()
        # 修正张量创建方式，避免警告
        if isinstance(mashup_full_sim, torch.Tensor):
            self.mashup_sim = mashup_full_sim.clone().detach().to(device, dtype=torch.float32)
        else:
            self.mashup_sim = torch.tensor(mashup_full_sim, dtype=torch.float32, device=device)

        if isinstance(api_full_sim, torch.Tensor):
            self.api_sim = api_full_sim.clone().detach().to(device, dtype=torch.float32)
        else:
            self.api_sim = torch.tensor(api_full_sim, dtype=torch.float32, device=device)

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.lambda_tss = lambda_tss
        self.to(device)

    def forward(self, e_m, e_a):
        # 确保输入嵌入与损失计算在同一设备
        e_m = e_m.to(self.mashup_sim.device)
        e_a = e_a.to(self.api_sim.device)

        # Mashup部分损失计算
        e_m_norm = F.normalize(e_m, p=2, dim=1)
        cos_m = torch.mm(e_m_norm, e_m_norm.T)
        # 动态生成掩码，确保与相似矩阵同设备
        mask_m = (self.mashup_sim < self.alpha1) & (
            ~torch.eye(e_m.size(0), dtype=torch.bool, device=self.mashup_sim.device))
        sigma_alpha_m = F.relu(self.alpha1 - self.mashup_sim)
        sigma_cos_m = F.relu(cos_m)
        loss_mss = (sigma_alpha_m[mask_m] ** self.beta * sigma_cos_m[mask_m] ** (self.beta + 1)).sum()

        # API部分损失计算
        e_a_norm = F.normalize(e_a, p=2, dim=1)
        cos_a = torch.mm(e_a_norm, e_a_norm.T)
        mask_a = (self.api_sim < self.alpha2) & (~torch.eye(e_a.size(0), dtype=torch.bool, device=self.api_sim.device))
        sigma_alpha_a = F.relu(self.alpha2 - self.api_sim)
        sigma_cos_a = F.relu(cos_a)
        loss_atss = (sigma_alpha_a[mask_a] ** self.beta * sigma_cos_a[mask_a] ** (self.beta + 1)).sum()

        return self.lambda_tss * (loss_mss + loss_atss)


# ------------------------- 工具函数 -------------------------
def sparse_matrix_to_tensor(sparse_mat):
    """将SciPy稀疏矩阵转换为PyTorch稀疏张量并放到全局设备"""
    sparse_mat = sparse_mat.tocoo()
    indices = torch.LongTensor(np.vstack((sparse_mat.row, sparse_mat.col))).to(device)
    values = torch.FloatTensor(sparse_mat.data).to(device)
    return torch.sparse_coo_tensor(indices, values, sparse_mat.shape, device=device)



def cross_modal_contrastive_loss(graph_emb, text_emb, temperature=0.2):
    """
    跨模态对比学习 (InfoNCE Loss)
    用于对齐 图融合Embedding 和 文本Embedding
    """
    # 确保L2归一化
    graph_emb = F.normalize(graph_emb, p=2, dim=1)
    text_emb = F.normalize(text_emb, p=2, dim=1)

    # 计算相似度矩阵 (N, N)
    logits = torch.matmul(graph_emb, text_emb.T) / temperature

    # 正样本对角线 (自己图特征与自己的文本特征互为正样本)
    labels = torch.arange(logits.size(0)).to(logits.device)

    # 交叉熵损失 (同时优化 G->T 和 T->G 两个方向)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)

    return (loss_g2t + loss_t2g) / 2.0
