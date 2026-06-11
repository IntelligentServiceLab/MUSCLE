import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

# 导入你原来的模块
from MUSCLE import MUSCLE
from processing import get_sim_matrix, get_bert_emb, get_test_mapping, get_train_mapping
from utils import build_interaction_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print("=" * 50)
    print("1. Loading Data and Environments...")
    print("=" * 50)

    # 保持和 main.py 一致的超参数
    num_mashups = 2289
    num_apis = 956
    emb_dim = 256
    n_layers = 5

    # 加载数据 (和 main.py 开头的操作一模一样)
    mashup_des_emb, api_des_emb = get_bert_emb()
    mashup_sim_matrix, api_sim_matrix = get_sim_matrix(mashup_des_emb, api_des_emb)
    test_mapping = get_test_mapping()
    train_mapping = get_train_mapping()

    with open('./Data/trnMat.pkl', 'rb') as f:
        train = pickle.load(f)
    interaction_adj = build_interaction_adj(train)

    print("\n" + "=" * 50)
    print("2. Loading the Best Model Weights...")
    print("=" * 50)

    # 模型文件列表
    model_files = [
        ("lightGCN", "TSSGCF.pth"),
        ("IWAR", "bad1.pth"),
        ("TSSGCF", "bad2.pth"),
        ("MUSCLE", "MUSCLE.pth")
    ]

    print("\n" + "=" * 50)
    print("3. Executing Circular Density Visualization (Visual Analysis)...")
    print("=" * 50)

    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    import seaborn as sns
    import matplotlib.colors as mcolors

    # ================= 全局样式设置 =================
    sns.set_style("white")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = True

    # 重新调配颜色映射 (起点纯白，终点截断避免死黑)
    base_cmap = plt.cm.Blues
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc_Blues', base_cmap(np.linspace(0.0, 0.8, 100))
    )

    # ================= 排版与渲染 =================
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 5.5),
        gridspec_kw={'height_ratios': [2.5, 1]}
    )
    fig.patch.set_facecolor('white')

    for col, (model_name, model_path) in enumerate(model_files):

        print(f"Loading {model_name}...")

        # 初始化模型结构
        model = MUSCLE(num_mashups, num_apis, emb_dim=emb_dim, n_layers=n_layers)
        model.to(device)

        # 加载模型权重
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Successfully loaded '{model_path}'!")
        except FileNotFoundError:
            print(f"Error: '{model_path}' not found!")
            continue

        model.eval()  # 设置为评估模式

        # 提取图融合后的最终 Embeddings
        with torch.no_grad():
            forward_res = model(
                mashup_sim_matrix,
                api_sim_matrix,
                interaction_adj,
                mashup_des_emb,
                api_des_emb
            )
            final_api_embeddings = forward_res[1].cpu().numpy()

        # 1. 降维到 2 维以获取方向性
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(final_api_embeddings)

        # 2. 归一化到单位圆上
        norms = np.linalg.norm(embs_2d, axis=1, keepdims=True)
        normalized_embs = embs_2d / (norms + 1e-8)
        x = normalized_embs[:, 0]
        y = normalized_embs[:, 1]

        # 3. 计算角度 (Angles)
        angles = np.arctan2(y, x)

        ax_top = axes[0, col]
        ax_bottom = axes[1, col]

        # --- 上方：二维圆环核密度图 (2D KDE Plot) ---
        ax_top.set_facecolor('white')
        sns.kdeplot(
            x=x,
            y=y,
            ax=ax_top,
            cmap=custom_cmap,
            fill=True,
            thresh=0.02,  # 刚好连起低密度区域
            levels=100,
            bw_adjust=0.25  # 降低带宽，实现“去油瘦身”
        )

        ax_top.set_title(model_name, fontsize=15, fontweight='bold', pad=10)
        ax_top.set_aspect('equal')
        ax_top.set_xlim([-1.25, 1.25])
        ax_top.set_ylim([-1.25, 1.25])
        ax_top.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_top.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_top.set_xlabel('Features', fontsize=12)
        ax_top.set_ylabel('Features', fontsize=12) if col == 0 else ax_top.set_ylabel('')
        ax_top.tick_params(axis='both', labelsize=10)

        # --- 下方：一维角度密度曲线 (1D KDE Plot) ---
        ax_bottom.set_facecolor('white')

        x_eval = np.linspace(-np.pi, np.pi, 500)
        angles_ext = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])

        kde = KernelDensity(kernel='gaussian', bandwidth=0.12)
        kde.fit(angles_ext[:, np.newaxis])
        log_dens = kde.score_samples(x_eval[:, np.newaxis])
        y_eval = np.exp(log_dens)

        y_eval_norm = y_eval / np.max(y_eval) * 0.85

        ax_bottom.plot(x_eval, y_eval_norm, color='#1f77b4', linewidth=1.5)
        ax_bottom.fill_between(x_eval, y_eval_norm, alpha=0.2, color='#1f77b4')

        ax_bottom.set_xlim([-np.pi, np.pi])
        ax_bottom.set_ylim([0, 1.00])
        ax_bottom.set_yticks([0.0, 0.5, 1.0])
        ax_bottom.set_xticks([-2, 0, 2])
        ax_bottom.set_xticklabels(['-2', '0', '2'])
        ax_bottom.set_xlabel('Angles', fontsize=12)
        ax_bottom.set_ylabel('Density', fontsize=12) if col == 0 else ax_bottom.set_ylabel('')
        ax_bottom.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.savefig("visual_analysis.png", dpi=600, bbox_inches='tight')
    print("-> Plot saved as 'visual_analysis.png'!")


if __name__ == "__main__":
    main()