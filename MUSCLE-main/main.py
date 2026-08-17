import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm

from MUSCLE import MUSCLE, bpr_loss, cross_modal_contrastive_loss

from processing import get_sim_matrix, get_bert_emb, get_train_mapping, get_test_mapping, compute_full_sim_matrix
from utils import build_interaction_adj, TrnData

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mashup_mapping = {}
api_mapping = {}
for i in range(2289):
    mashup_mapping[i] = 0
for i in range(956):
    api_mapping[i] = 0

if __name__ == "__main__":
    num_mashups = 2289
    num_apis = 956
    emb_dim = 256  
    n_layers = 5 
    epochs = 500  
    batch_size = 256 
    lr = 1e-4 
    weight_decay = 1e-4
    l2_reg_weight = 1e-5
    cl_weight = 0 
    cl_temperature = 0.2
    sel_cl_weight = 0 
    top_ratio = 0 
    top_k = 5 

    mashup_des_emb, api_des_emb = get_bert_emb()
    mashup_sim_matrix, api_sim_matrix = get_sim_matrix(mashup_des_emb, api_des_emb)

    mashup_full_sim_matrix = compute_full_sim_matrix(mashup_des_emb)
    api_full_sim_matrix = compute_full_sim_matrix(api_des_emb)

    test_mapping = get_test_mapping()
    train_mapping = get_train_mapping()
    f = open('./Data/trnMat.pkl', 'rb')
    train = pickle.load(f)

    interaction_adj = build_interaction_adj(train)
    train_csr = (train != 0).astype(np.float32)

    mashup_degrees = np.array(train.sum(axis=1)).squeeze()
    api_degrees = np.array(train.sum(axis=0)).squeeze()

    top_m_cutoff = int(len(mashup_degrees) * top_ratio)
    top_a_cutoff = int(len(api_degrees) * top_ratio)

    top_mashups = torch.LongTensor(np.argsort(mashup_degrees)[::-1][:top_m_cutoff].copy()).to(device)
    top_apis = torch.LongTensor(np.argsort(api_degrees)[::-1][:top_a_cutoff].copy()).to(device)

    rowD = np.array(train.sum(1)).squeeze()
    colD = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):
        train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)

    train = train.tocoo()
    train_data = TrnData(train)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    model = MUSCLE(num_mashups, num_apis, emb_dim=emb_dim, n_layers=n_layers)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader.dataset.neg_sampling()

    best_recall = 0.0
    for epoch in range(epochs):
        for batch in train_loader:
            u, pos, neg = batch
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)

            final_mashup, final_api, g_fused_m, g_fused_a, t_m, t_a, g_m, inter_m, g_a, inter_a = model(
                mashup_sim_matrix, api_sim_matrix, interaction_adj, mashup_des_emb, api_des_emb
            )

            pos_scores = torch.mul(final_mashup[u], final_api[pos]).sum(dim=1)
            neg_scores = torch.mul(final_mashup[u], final_api[neg]).sum(dim=1)
            optimizer.zero_grad()

            loss_bpr = bpr_loss(pos_scores, neg_scores)

            cl_loss_m = cross_modal_contrastive_loss(g_fused_m[u], t_m[u], temperature=cl_temperature)
            api_batch_idx = torch.cat([pos, neg], dim=0).unique()
            cl_loss_a = cross_modal_contrastive_loss(g_fused_a[api_batch_idx], t_a[api_batch_idx],
                                                     temperature=cl_temperature)
            loss_cl = cl_loss_m + cl_loss_a

            valid_m = u[torch.isin(u, top_mashups)]
            if len(valid_m) > 1:
                sel_cl_loss_m = cross_modal_contrastive_loss(g_m[valid_m], inter_m[valid_m], temperature=cl_temperature)
            else:
                sel_cl_loss_m = torch.tensor(0.0, device=device)

            valid_a = api_batch_idx[torch.isin(api_batch_idx, top_apis)]
            if len(valid_a) > 1:
                sel_cl_loss_a = cross_modal_contrastive_loss(g_a[valid_a], inter_a[valid_a], temperature=cl_temperature)
            else:
                sel_cl_loss_a = torch.tensor(0.0, device=device)
            loss_sel_cl = sel_cl_loss_m + sel_cl_loss_a

            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param) ** 2

            loss = loss_bpr + cl_weight * loss_cl + sel_cl_weight * loss_sel_cl + l2_reg_weight * l2_reg
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            test_uids = np.array([i for i in range(2289)])
            batch_no = int(np.ceil(len(test_uids) / batch_size))
            all_recall = 0
            all_ndcg = 0
            all_precision = 0

            for batch in tqdm(range(batch_no)):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, len(test_uids))

                test_uids_input = torch.LongTensor(test_uids[start:end])

                recall, precision, ndcg = model.pred(
                    mashup_sim_matrix, api_sim_matrix, interaction_adj,
                    mashup_des_emb, api_des_emb, test_uids_input,
                    test_mapping, train_mapping, top_k=top_k
                )

                all_recall += recall
                all_ndcg += ndcg
                all_precision += precision

            Recall = all_recall / batch_no
            Precision = all_precision / batch_no
            NDCG = all_ndcg / batch_no
            F1 = 2 * (Recall * Precision) / (Recall + Precision) if (Recall + Precision) > 0 else 0

            if Recall > best_recall:
                best_recall = Recall
                print(f"--> [Model Saved] New best Recall score: {best_recall:.4f} at epoch {epoch}")
                torch.save(model.state_dict(), 'best_muscle_model.pth')
            print("TEST   EPOCH:", epoch, " Recall: ", Recall, " NDCG: ", NDCG, " Precision: ", Precision, " F1: ", F1)
