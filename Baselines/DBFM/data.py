
import numpy as np
import pandas as pd
import pickle
from scipy import sparse
import time
from tqdm import tqdm

t0 = time.time()

# ===================== 1. 读取数据 =====================

train_df = pd.read_csv("data/train_baskets.csv", encoding="ISO-8859-1")
test_df  = pd.read_csv("data/test_baskets.csv",  encoding="ISO-8859-1")

USER_COL  = "user_id"
ITEM_COL  = "item_id"
ORDER_COL = "transaction_number"

# 合并构建全局词表（训练/测试共享同一 ID 空间）
all_users = pd.concat([train_df[USER_COL], test_df[USER_COL]]).unique()
all_items = pd.concat([train_df[ITEM_COL], test_df[ITEM_COL]]).unique()

user2idx = {u: i for i, u in enumerate(all_users)}
item2idx = {v: i for i, v in enumerate(all_items)}

sum_id   = len(all_users)
sum_item = len(all_items)
feat_dim = sum_id + sum_item + sum_item   # 用户段 | 购物篮段 | 目标商品段

print(f"用户数: {sum_id}, 商品数: {sum_item}, 特征维度: {feat_dim}")
print(f"读取耗时: {time.time()-t0:.1f}s")


# ===================== 2. 构建四元组 =====================

def build_tuples(df, is_train=True):
    """
    按 (用户, 订单) 分组，每个商品轮流作目标项目，其余为购物篮。
    训练集额外生成 2 条负样本（目标商品替换为随机非购买商品）。
    """
    groups = df.groupby([USER_COL, ORDER_COL])[ITEM_COL].apply(list)
    all_item_indices = np.arange(sum_item, dtype=np.int32)

    pos_tuples = []
    neg_tuples = []

    for (user, _order), items in tqdm(
            groups.items(),
            total=len(groups),
            desc="build tuples"
    ):
        uid = user2idx.get(user)
        if uid is None:
            continue

        item_idxs = [item2idx[v] for v in items if v in item2idx]
        if not item_idxs:
            continue

        item_arr = np.array(item_idxs, dtype=np.int32)

        # ── 负采样候选集 ──────────────────────────────
        if is_train:
            candidates = np.setdiff1d(
                all_item_indices,
                np.unique(item_arr),   
                assume_unique=True
            )

        for i in range(len(item_arr)):
            target_idx = int(item_arr[i])


            mask = np.ones(len(item_arr), dtype=bool)
            mask[i] = False
            basket = item_arr[mask].tolist()

            pos_tuples.append((uid, basket, target_idx, 1))

            if is_train:
                if len(candidates) >= 2:
                    neg_picks = np.random.choice(
                        candidates, size=2, replace=False
                    )
                elif len(candidates) == 1:
                    neg_picks = candidates
                else:
                    continue   

                for neg in neg_picks:
                    neg_tuples.append((uid, basket, int(neg), -1))

    return pos_tuples, neg_tuples


print("构建训练四元组...")
pos_train, neg_train = build_tuples(train_df, is_train=True)
print(f"  正样本: {len(pos_train)}, 负样本: {len(neg_train)}")

print("构建测试四元组...")
pos_test, _ = build_tuples(test_df, is_train=False)
print(f"  测试样本: {len(pos_test)}")


# ===================== 3. 向量化编码为稀疏矩阵 =====================

def tuples_to_sparse(tuples, feat_dim, sum_id, sum_item, with_label=False):
    """
    将四元组列表向量化编码为 CSR 稀疏矩阵。

    特征布局：
      [0 : sum_id)                     → 用户 one-hot
      [sum_id : sum_id+sum_item)        → 购物篮 one-hot（多热）
      [sum_id+sum_item : feat_dim)      → 目标商品 one-hot
      最后一列（可选）                   → 标签 (1 / -1)


    """
    total_cols = feat_dim + (1 if with_label else 0)
    n = len(tuples)

    # 每条样本非零数：1(用户) + len(basket) + 1(目标) [+ 1(标签)]
    nnz = sum(
        1 + len(t[1]) + 1 + (1 if with_label else 0)
        for t in tuples
    )

    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.ones(nnz, dtype=np.float32)

    ptr = 0
    for row_idx, (uid, basket, target, label) in tqdm(
            enumerate(tuples),
            total=n,
            desc="sparse encoding"
    ):
        # 用户
        rows[ptr] = row_idx
        cols[ptr] = uid
        ptr += 1

        # 购物篮（多热）
        for b in basket:
            rows[ptr] = row_idx
            cols[ptr] = sum_id + b
            ptr += 1

        # 目标商品
        rows[ptr] = row_idx
        cols[ptr] = sum_id + sum_item + target
        ptr += 1

        # 标签（1 或 -1，均为非零，不污染稀疏矩阵）
        if with_label:
            rows[ptr] = row_idx
            cols[ptr] = feat_dim
            vals[ptr] = float(label)
            ptr += 1

    return sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n, total_cols)
    )


print("编码训练集稀疏矩阵...")
all_train = pos_train + neg_train
np.random.shuffle(all_train)
train_mat = tuples_to_sparse(
    all_train, feat_dim, sum_id, sum_item, with_label=True
)
print(f"  训练集 shape: {train_mat.shape}, "
      f"密度: {train_mat.nnz / (train_mat.shape[0]*train_mat.shape[1]):.6f}, "
      f"空间节省约 {train_mat.shape[0]*train_mat.shape[1] / train_mat.nnz:.0f}x")

print("编码测试集稀疏矩阵...")
test_mat = tuples_to_sparse(
    pos_test, feat_dim, sum_id, sum_item, with_label=False
)
print(f"  测试集 shape: {test_mat.shape}")


# ===================== 4. 保存 =====================

print("保存文件...")
sparse.save_npz("feature_sparse.npz", train_mat)
sparse.save_npz("feature_test_sparse.npz", test_mat)

metadata = {
    "user2idx":  user2idx,
    "item2idx":  item2idx,
    "all_users": all_users.tolist(),
    "all_items": all_items.tolist(),
    "sum_id":    sum_id,
    "sum_item":  sum_item,
    "feat_dim":  feat_dim,
}
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"\n总耗时: {time.time()-t0:.1f}s")
print("输出文件：")
print("  feature_sparse.npz       ← 训练集（含标签列，label=1/-1）")
print("  feature_test_sparse.npz  ← 测试集")
print("  metadata.pkl             ← 词表与元数据")


