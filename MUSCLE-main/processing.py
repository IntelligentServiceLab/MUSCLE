# 导入所需的库
import pickle
from distutils.command.config import config
import sklearn.preprocessing
import torch.utils.data as data
from sentence_transformers import SentenceTransformer
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import glob
import random
# from MLP_MAIN import MLP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(8080)
# SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

mashup_mapping = {}
api_mapping = {}
for i in range(2289):
    mashup_mapping[i] = 0
for i in range(956):
    api_mapping[i] = 0


def check_for_pt_files(file_name,folder_path):
    # 使用 glob 模块匹配文件路径模式
    pt_files = glob.glob(os.path.join(folder_path, f'{file_name}.pt'))
    # 检查是否有匹配的文件
    if pt_files:
      return True
    else:
      return False

def get_bert_emb():
    flag1 = check_for_pt_files('mashup_descr_emb', "../Data")
    flag2 = check_for_pt_files('api_descr_emb', "../Data")
    # print(flag1,flag2)
    mashup_data = pd.read_csv("Data/Mashup_desc.csv", encoding='UTF-8', header=0)  # 使用Mashups.csv文件
    api_data = pd.read_csv("Data/API_desc.csv", encoding='UTF-8', header=0)  # 使用APIs.csv文件

    mashup_descr = mashup_data['description']
    api_descr = api_data['description']

    # 打印描述数据的形状（行数，列数）
    print("shape of mashup_desc ", mashup_descr.shape)
    print("shape of api_desc ", api_descr.shape)

    mashup_descr_emb = bert_convert_emb(mashup_descr) if flag1==False else torch.load('Data/mashup_descr_emb.pt', map_location=device)

    api_descr_emb = bert_convert_emb(api_descr) if flag2==False else torch.load('Data/api_descr_emb.pt', map_location=device)
    if not flag1:
        torch.save(mashup_descr_emb, 'Data/mashup_descr_emb.pt')
    if not flag2:
        torch.save(api_descr_emb, 'Data/api_descr_emb.pt')
    return mashup_descr_emb, api_descr_emb

def bert_convert_emb(descriptions):
    all_sentence_vectors = model.encode(descriptions.astype(str).tolist())
    all_sentence_vectors = torch.tensor(all_sentence_vectors)
    print(all_sentence_vectors.shape)
    return all_sentence_vectors

def get_test_mapping():
    test_mapping = {}
    with open("Data/test.txt", 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            items = list(map(int,items))
            test_mapping[items[0]] = items[1:]
    return test_mapping

def get_train_mapping():
    train_mapping = {}
    with open("Data/train.txt", 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            items = list(map(int,items))
            for item in items[1:]:
                api_mapping[item] += 1
            mashup_mapping[items[0]] += len(items[1:])
            train_mapping[items[0]] = items[1:]
    return train_mapping

def build_similarity_matrix(embeddings, threshold, batch_size=128):
    """构建相似矩阵（分批次计算），返回torch.sparse.FloatTensor
    Args:
        embeddings: PyTorch Tensor of shape (num_entities, embedding_dim)
        threshold: 相似度阈值
        batch_size: 批处理大小
    Returns:
        torch.sparse.FloatTensor: 稀疏相似度矩阵 (N, N)
    """
    # 确保输入是PyTorch Tensor
    if not torch.is_tensor(embeddings):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    # L2归一化（直接在PyTorch中完成）
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # (N, d)
    N = embeddings_norm.shape[0]  # 实体数量
    indices = []
    values = []

    # 分批次计算余弦相似度
    for i in range(0, N, batch_size):
        batch_emb = embeddings_norm[i:i + batch_size]  # (batch_size, d)
        # 计算余弦相似度（PyTorch矩阵乘法）
        batch_sim = torch.mm(batch_emb, embeddings_norm.T)  # (batch_size, N)
        weighted_sim = batch_sim
        # 应用阈值并收集非零元素
        mask = weighted_sim >= threshold
        mask.diagonal().fill_(1)  # 强制对角线为1（自连接）
        rows, cols = torch.where(mask)
        rows += i
        indices.append(torch.stack([rows, cols]))
        values.append(weighted_sim[mask])
        print(f"Processed {min(i + batch_size, N)}/{N}")

    # 合并所有批次的非零元素
    indices = torch.cat(indices, dim=1)  # (2, nnz)
    values = torch.cat(values)  # (nnz,)
    # 创建稀疏张量
    sparse_sim = torch.sparse.FloatTensor(
        indices,
        values,
        torch.Size([N, N])
    ).coalesce()

    return sparse_sim


import torch



def get_sim_matrix(mashup_des_emb,api_des_emb):
    # 获取预处理数据
    # 加载SBERT嵌入向量
    # 构建相似矩阵（示例阈值α1=0.6, α2=0.5）
    mashup_sim_matrix = build_similarity_matrix(mashup_des_emb, threshold=0.34)
    api_sim_matrix = build_similarity_matrix(api_des_emb, threshold=0.34)
    return mashup_sim_matrix,api_sim_matrix

def compute_full_sim_matrix(embeddings):
    """计算未应用阈值的完整相似矩阵 """
    # L2归一化
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    sim_matrix = torch.mm(embeddings_norm, embeddings_norm.T)  # [N, N]
    weighted_sim = sim_matrix
    return weighted_sim


if __name__ == '__main__':
    mashup_des_emb,api_des_emb = get_bert_emb()
    mashup_sim_matrix,api_sim_matrix=get_sim_matrix(mashup_des_emb,api_des_emb)
    print(mashup_sim_matrix,api_sim_matrix)