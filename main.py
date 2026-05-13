import numpy as np
import pandas as pd
import pickle
from itertools import chain
import copy
import cupy as cp
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import vstack

# Read data
data_train = pd.read_csv('../../datasets/Dunnhumby/train_baskets.csv')
data_test = pd.read_csv('../../datasets/Dunnhumby/test_baskets.csv')

# Change float to int
data_train['item_id'] = data_train['item_id'].astype(int)
data_train['user_id'] = data_train['user_id'].astype(int)
data_train['transaction_number'] = data_train['transaction_number'].astype(int)

data_test['item_id'] = data_test['item_id'].astype(int)
data_test['user_id'] = data_test['user_id'].astype(int)
data_test['transaction_number'] = data_test['transaction_number'].astype(int)

num_nearest_neighbors = 250
group_decay_rate = -0.9
history_rate = 0.1
alpha = 0.1  # Personal component
item_size = max(data_train['item_id'].max(),data_test['item_id'].max())
user_size = data_train['user_id'].nunique()
test_number = data_test.shape[0]
train_users = np.unique(data_train['user_id'])
train_items = np.unique(data_train['item_id'])
item_list = data_test['item_id'].unique().tolist()


df_train_item = data_train.groupby(['item_id'])
df_test_user = data_test.groupby(['user_id'])
# Create a dictionary for [Item: User]
data_dict = {}
for index, content in df_train_item:
    item_id = content['item_id'].values[0]
    user_id = content['user_id'].unique().tolist()
    if item_id in item_list:
        data_dict[item_id] = user_id

# Create a dictionary for [User: itme]
user_to_item_test_dict= {}
for index, content in df_test_user:
    item_id = content['item_id'].values
    user_id = content['user_id'].values[0]
    user_to_item_test_dict[user_id] = item_id

# ----------------History component----------------------
# user -> row index 映射,转成 0、1、2、3.. 这样的连续索引
user_index_map = {user: i for i, user in enumerate(train_users)}

user_item_matrix = np.zeros((user_size, item_size),dtype=np.float32)

# 先按 user 和 item 分组
grouped = data_train.groupby(['user_id', 'item_id'])

for (user, item), g in grouped:
    #每次循环拿到一个用户ID，用映射表把它换成对应的矩阵行索引。
    i = user_index_map[user]  # user → row index
    j = item - 1

    n = len(g)

    decay_vec = np.arange(n, 0, -1, dtype=np.float32) ** group_decay_rate

    user_item_matrix[i, j] += decay_vec.sum()


print('----历史得分组件加载完成----')

# 共现字典
with open('../../datasets/Dunnhumby/data_chunk.pkl', 'rb') as file:
    data_chunk=pickle.load(file)
print('----data_chunk---载入完成')

# Temporal dynamics
def temporal_decay_sum_history(data_chunk, item_size, within_decay_rate):
    sum_history = [{} for _ in range(len(data_chunk))]

    for user_index, user in enumerate(data_chunk):

        for key, co_items in user.items():
            num_baskets = len(co_items)
            decay_vec_ = np.arange(num_baskets, 0, -1) ** within_decay_rate

            # flatten items
            flat_items = np.concatenate(co_items) - 1

            # 每个 basket 的长度
            lens = np.array([len(b) for b in co_items])

            # 为每个 item 生成对应的权重
            weights = np.repeat(decay_vec_, lens)

            # 累加
            his_matrix = np.zeros(item_size)
            np.add.at(his_matrix, flat_items, weights)

            sum_history[user_index][key] = csr_matrix(his_matrix)

    return sum_history

# zero-shot
# zero-shot：把稀疏向量转dense再送GPU，其余不变


def max_k(search_set, item, item_for_user, num_nearest_neighbors, data_chunk, alpha):
    count = [len(list(chain(*data_chunk[user - 1][item]))) for user in item_for_user]
    count_index = np.argsort(count)[-num_nearest_neighbors:][::-1]
    user_popular = np.array(item_for_user)[count_index]

    # 批量stack成 (250, 2596)，一次toarray，一次传GPU
    batch = vstack([search_set[user - 1][item] for user in user_popular])  # sparse (250, 2596)
    batch_dense = batch.toarray()  # 一次转dense
    vecs = cp.asarray(batch_dense)  # 一次传GPU

    history = vecs.sum(axis=0)
    merge_history = cp.asnumpy((history / len(user_popular) * (1 - alpha)))
    return merge_history

def cosine_similarity_vector_matrix(vector, matrix):
    vector_magnitude = cp.linalg.norm(vector)
    matrix_magnitudes = cp.linalg.norm(matrix, axis=1)
    dot_products = cp.dot(matrix, vector)

    cosine_similarities = dot_products / (vector_magnitude * matrix_magnitudes)

    return cosine_similarities


def KNN(target_set, vector,item, k, item_for_user, alpha):
    # 原来: cp.asarray(vector) 直接转，vector已是dense numpy
    # 优化: vector是稀疏，先转dense
    vector = cp.asarray(vector.toarray()[0])
    selected_set = cp.empty((len(item_for_user), item_size), dtype=cp.float16)

    for i, user in enumerate(item_for_user):
        # 原来: cp.asarray(target_set[(user-1)][item])
        # 优化: 先把稀疏转dense
        selected_set[i] = cp.asarray(target_set[(user - 1)][item].toarray()[0])

    selected_set_transformed = (selected_set > 0).astype(int)
    vector_transformed = (vector > 0).astype(int)
    similarities = cosine_similarity_vector_matrix(vector_transformed, selected_set_transformed)
    indices = cp.argsort(similarities, kind='stable')[::-1][:k + 1]

    history = cp.sum(selected_set[indices[1:]], axis=0)
    count = len(indices[1:]) + 0.001

    merge_history = cp.asnumpy((vector * alpha + (history / count) * (1 - alpha)).astype(np.float16))
    return merge_history

# get score
temporal_decay_sum_history_training = temporal_decay_sum_history(data_chunk, item_size,group_decay_rate)
temporal_decay_new = copy.deepcopy(temporal_decay_sum_history_training)

# # renew score
for user, item in tqdm(user_to_item_test_dict.items()):
    for key in item:
        if key not in data_dict.keys():
            temporal_decay_new[user - 1][key] = np.zeros([item_size]).astype(np.float16)

        elif key not in temporal_decay_sum_history_training[user - 1].keys():
            item_for_user = data_dict[key]  # all users who have the item
            temporal_decay_new[user - 1][key] = \
                max_k(temporal_decay_sum_history_training, key, item_for_user, num_nearest_neighbors,
                      data_chunk, alpha)
        elif key in temporal_decay_sum_history_training[user - 1].keys():
            vector = temporal_decay_sum_history_training[user - 1][key]
            item_for_user = data_dict[key]  # all users who have the item
            temporal_decay_new[user - 1][key] = \
                KNN(temporal_decay_sum_history_training, vector, key, num_nearest_neighbors, item_for_user, alpha)
print('finishing vector generation---------')

# temporal_decay_new = [{} for _ in range(user_size)]
#
# for user, item in tqdm(user_to_item_test_dict.items()):
#     for key in item:
#
#         if key not in data_dict:
#             temporal_decay_new[user-1][key] = np.zeros(item_size, dtype=np.float16)
#
#         elif key not in temporal_decay_sum_history_training[user-1]:
#             item_for_user = data_dict[key]
#             temporal_decay_new[user-1][key] = max_k(temporal_decay_sum_history_training,key,item_for_user,num_nearest_neighbors,data_chunk,alpha
#             )
#
#         else:
#             vector = temporal_decay_sum_history_training[user-1][key]
#             item_for_user = data_dict[key]
#             temporal_decay_new[user-1][key] = KNN(temporal_decay_sum_history_training,vector,key,num_nearest_neighbors,item_for_user,alpha
#             )
# print('finishing vector generation---------')


# -------------------start test------------------

for k in range(1, 11):
    print(f"\n========== 第 {k} 轮测试 (文件后缀 _{k}) ==========")

    hit_num, mrr_num = 0, 0

    # -------------------购物意图得分----------------------
    with open(f'../../datasets/Dunnhumby/item_intent_embeddings_384_{k}.pkl', 'rb') as file:
        item_intent = pickle.load(file)

    # 初始化完整矩阵（所有 item 都有位置）
    all_embeddings = np.zeros((item_size, 384), dtype=np.float32)

    # 取出 index，需要-1
    indices = np.array([item['StockCode'] - 1 for item in item_intent], dtype=np.int32)

    # 取出 embedding
    embeddings = np.vstack([item['embedding'] for item in item_intent]).astype(np.float32)

    # 一次性填入
    all_embeddings[indices] = embeddings


    # -------导入购物篮的购物意图-------
    with open(f'../../datasets/Dunnhumby/basket_embeddings_384_{k}.pkl', 'rb') as file:
        basket_intents = pickle.load(file)  # basket 编号从 0 开始


    eps = 1e-10
    all_norm = np.linalg.norm(all_embeddings, axis=1).astype(np.float32)
    all_norm = np.where(all_norm == 0, eps, all_norm)

    alpha_list = np.array([0.8])
    beta_list  = np.array([0.1])

    alpha_grid = alpha_list.astype(np.float16)[:, None, None]
    beta_grid  = beta_list.astype(np.float16)[None, :, None]

    print("alpha grid:", alpha_grid.shape)
    print("beta grid:", beta_grid.shape)

    # 初始化
    HR10 = np.zeros((len(alpha_list), len(beta_list)))
    HR20 = np.zeros((len(alpha_list), len(beta_list)))
    HR50 = np.zeros((len(alpha_list), len(beta_list)))

    MRR10 = np.zeros((len(alpha_list), len(beta_list)))
    MRR20 = np.zeros((len(alpha_list), len(beta_list)))
    MRR50 = np.zeros((len(alpha_list), len(beta_list)))

    count = 0
    for user, items in tqdm(user_to_item_test_dict.items()):

        for target_item in items:

            given_item = np.setdiff1d(items, target_item)
            basket_intent = basket_intents[count]['shopping_intent_embedding']

            basket_intent_norm = np.linalg.norm(basket_intent).astype(np.float32)
            basket_intent_score = (all_embeddings @ basket_intent) / (all_norm * basket_intent_norm)

            output_vectors = np.stack(
                [temporal_decay_new[user-1][kk] for kk in given_item]
            )

            output_vectors[:, given_item-1] = 0

            history_score = user_item_matrix[user-1].copy()
            history_score[given_item-1] = 0

            output_add = (
                (1-history_rate)*output_vectors +
                history_rate*history_score
            ).astype(np.float16)

            selected_embeddings = all_embeddings[given_item - 1]
            selected_similarity = (
                selected_embeddings @ all_embeddings.T
            ) / (np.linalg.norm(selected_embeddings, axis=1, keepdims=True) * all_norm)

            basket_intent_score = np.clip(basket_intent_score, 0, None)
            selected_similarity = np.clip(selected_similarity, 0, None)

            A = output_add
            B = (output_add * selected_similarity).astype(np.float16)

            part1 = A[None,:,:] + alpha_grid * B[None,:,:]
            intent_result = part1[:,None,:,:] * (1 + beta_grid[:,:,None] * basket_intent_score)
            scores = intent_result.max(axis=2)

            top_idx = np.argpartition(-scores, 50, axis=2)[:, :, :50]

            for i in range(len(alpha_list)):
                for j in range(len(beta_list)):

                    top_items = top_idx[i, j]
                    top_items = top_items[np.argsort(-scores[i, j, top_items])]

                    top50 = top_items[:50]
                    top20 = top_items[:20]
                    top10 = top_items[:10]

                    if target_item-1 in top50:
                        HR50[i, j] += 1
                        rank = np.where(top50 == target_item-1)[0][0] + 1
                        MRR50[i, j] += 1 / rank

                    if target_item-1 in top20:
                        HR20[i, j] += 1
                        rank = np.where(top20 == target_item-1)[0][0] + 1
                        MRR20[i, j] += 1 / rank

                    if target_item-1 in top10:
                        HR10[i, j] += 1
                        rank = np.where(top10 == target_item-1)[0][0] + 1
                        MRR10[i, j] += 1 / rank

            count += 1

    # -------------------输出本轮结果-------------------
    results = []

    HR10  /= test_number
    HR20  /= test_number
    HR50  /= test_number
    MRR10 /= test_number
    MRR20 /= test_number
    MRR50 /= test_number

    for i, alpha in enumerate(alpha_list):
        for j, beta in enumerate(beta_list):
            results.append({
                "file_k": k,
                "alpha": alpha,
                "beta": beta,
                "HR10":  HR10[i, j],
                "HR20":  HR20[i, j],
                "HR50":  HR50[i, j],
                "MRR10": MRR10[i, j],
                "MRR20": MRR20[i, j],
                "MRR50": MRR50[i, j]
            })
    print(results)
