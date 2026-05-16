import os
import csv
import pickle
import numpy as np
import tensorflow as tf
from scipy import sparse
from tqdm import tqdm
import time


# =====================================================
# GPU 设置
# =====================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# =====================================================
# 超参数
# =====================================================

VEC_DIM    = 64
EPOCHS     = 20
BATCH_SIZE = 1024
LR         = 0.0005
REG        = 0.008
START_EVAL = 6
EVAL_EVERY = 2
INFER_BATCH = 2048
EVAL_SAMPLE = 3000


# 训练过程中每次评估只随机抽取 EVAL_SAMPLE 条样本



MODEL_DIR  = "saved_model_sample"
MODEL_NAME = "deepfm.weights.h5"


# =====================================================
# 加载训练数据
# =====================================================

print("加载训练数据...")
train_sparse   = sparse.load_npz("feature_sparse.npz")
x_train_sparse = train_sparse[:, :-1].astype(np.float32)
y_train        = train_sparse[:, -1].toarray().reshape(-1).astype(np.float32)

print(f"x_train_sparse: {x_train_sparse.shape}")
print(f"y_train: {y_train.shape}, unique labels: {np.unique(y_train)}")


# =====================================================
# 加载测试数据
# =====================================================

print("加载测试集...")
test_sparse = sparse.load_npz("feature_test_sparse.npz")
n_test      = test_sparse.shape[0]
print(f"测试集: {test_sparse.shape}")


# =====================================================
# metadata
# =====================================================

with open("metadata.pkl", "rb") as f:
    meta = pickle.load(f)

sum_id   = meta["sum_id"]
sum_item = meta["sum_item"]
feat_dim = meta["feat_dim"]

USER_END   = sum_id
BASKET_END = sum_id + sum_item

print(f"用户数: {sum_id}, 商品数: {sum_item}, 特征维度: {feat_dim}")


# =====================================================
# Dataset
# =====================================================

indices = np.arange(x_train_sparse.shape[0])

train_ds = (
    tf.data.Dataset
    .from_tensor_slices(indices)
    .shuffle(200_000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# =====================================================
# DeepFM
# =====================================================

class DeepFM(tf.keras.Model):

    def __init__(self,
                 feat_dim,
                 vec_dim,
                 user_end,
                 basket_end,
                 deep_layers=(64, 32, 16, 8),
                 reg=1e-4):

        super().__init__()

        self.feat_dim   = feat_dim
        self.vec_dim    = vec_dim
        self.user_end   = user_end
        self.basket_end = basket_end

        regularizer = tf.keras.regularizers.l2(reg)

        self.bias = self.add_weight(
            name='bias', shape=(1,), initializer='zeros'
        )
        self.W = self.add_weight(
            name='linear_weight',
            shape=(feat_dim, 1),
            initializer='random_normal',
            regularizer=regularizer
        )
        self.V = self.add_weight(
            name='feature_embedding',
            shape=(feat_dim, vec_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01),
            regularizer=regularizer
        )

        self.dnn_layers = [
            tf.keras.layers.Dense(
                units, activation='relu',
                kernel_regularizer=regularizer
            )
            for units in deep_layers
        ]

        self.dropout   = tf.keras.layers.Dropout(0.2)
        self.out_layer = tf.keras.layers.Dense(1)

    def fm_cross(self, x, v):
        xv = tf.matmul(x, v)
        return 0.5 * tf.reduce_sum(
            tf.square(xv) - tf.matmul(tf.square(x), tf.square(v)),
            axis=1, keepdims=True
        )

    def call(self, x, training=False):

        ue = self.user_end
        be = self.basket_end

        y1    = self.bias + tf.matmul(x, self.W)

        x_u   = tf.concat([x[:, :ue],   x[:, be:]], axis=1)
        v_u   = tf.concat([self.V[:ue],  self.V[be:]], axis=0)
        y2_u  = self.fm_cross(x_u, v_u)

        x_b   = tf.concat([x[:, ue:be],   x[:, be:]], axis=1)
        v_b   = tf.concat([self.V[ue:be], self.V[be:]], axis=0)
        y2_b  = self.fm_cross(x_b, v_b)

        y2_bb = self.fm_cross(x[:, ue:be], self.V[ue:be])
        y2    = y2_u + y2_b + y2_bb

        user_dense = tf.matmul(x[:, :ue],  self.V[:ue])
        item_dense = tf.matmul(x[:, be:],  self.V[be:])

        basket_x    = x[:, ue:be]
        nonzero_cnt = tf.reduce_sum(
            tf.cast(basket_x > 0, tf.float32), axis=1, keepdims=True
        ) + 1e-8
        basket_dense = tf.matmul(basket_x, self.V[ue:be]) / nonzero_cnt

        deep_input = tf.concat([user_dense, item_dense, basket_dense], axis=1)

        h = deep_input
        for layer in self.dnn_layers:
            h = layer(h)
            h = self.dropout(h, training=training)

        out = self.out_layer(tf.concat([y1, y2, h], axis=1))
        return out


# =====================================================
# BPR loss
# =====================================================

def bpr_loss(y_pred, y_sign):
    return tf.reduce_sum(
        -tf.math.log(tf.sigmoid(y_pred[:, 0] * y_sign) + 1e-8)
    )


# =====================================================
# 模型 & 优化器
# =====================================================

model     = DeepFM(feat_dim=feat_dim, vec_dim=VEC_DIM,
                   user_end=USER_END, basket_end=BASKET_END, reg=REG)
optimizer = tf.keras.optimizers.Adam(LR)


# =====================================================
# train step
# =====================================================

@tf.function
def train_step(batch_x, batch_y):
    with tf.GradientTape() as tape:
        pred     = model(batch_x, training=True)
        loss     = bpr_loss(pred, batch_y)
        reg_loss = tf.add_n(model.losses)
        total    = loss + reg_loss
    grads = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total


# =====================================================
# 推理函数
# =====================================================

@tf.function(input_signature=[
    tf.TensorSpec(shape=[None, feat_dim], dtype=tf.float32)
])
def infer_step(x):
    return model(x, training=False)


# =====================================================
# evaluate（方案一：抽样）
# =====================================================

def evaluate_model(model, final=False):
    """
    final=False：使用 EVAL_SAMPLE 条随机样本快速评估（训练过程中使用）
    final=True ：使用全量测试集精确评估（训练结束后调用）
    """

    Ks = [5, 10, 20]
    recall_dict = {k: 0   for k in Ks}
    map_dict    = {k: 0.0 for k in Ks}
    # ndcg_dict   = {k: 0.0 for k in Ks}

    # ── 决定评估样本索引 ──────────────────────────────────────────────
    if (not final) and EVAL_SAMPLE and n_test > EVAL_SAMPLE:
        eval_indices = np.random.choice(n_test, EVAL_SAMPLE, replace=False)
        n_eval       = EVAL_SAMPLE
        mode_str     = f"抽样评估 ({EVAL_SAMPLE}/{n_test})"
    else:
        eval_indices = np.arange(n_test)
        n_eval       = n_test
        mode_str     = f"全量评估 ({n_test})"

    item_eye = np.eye(sum_item, dtype=np.float32)

    print(f"\nStart Evaluation [{mode_str}]...")

    for idx in tqdm(eval_indices, desc="Evaluating", ncols=100):

        sample   = test_sparse[idx].toarray().reshape(-1).astype(np.float32)
        ub       = sample[:BASKET_END]
        true_idx = int(np.argmax(sample[BASKET_END:]))

        scores = []
        for start in range(0, sum_item, INFER_BATCH):
            end   = min(start + INFER_BATCH, sum_item)
            bs    = end - start
            batch = np.empty((bs, feat_dim), dtype=np.float32)
            batch[:, :BASKET_END] = ub
            batch[:, BASKET_END:] = item_eye[start:end]
            scores.append(infer_step(batch).numpy().reshape(-1))

        scores = np.concatenate(scores)

        top_k_max = max(Ks)
        if sum_item >= top_k_max:
            top_raw = np.argpartition(scores, -top_k_max)[-top_k_max:]
        else:
            top_raw = np.arange(sum_item)

        top_idx = top_raw[np.argsort(scores[top_raw])[::-1]]

        for k in Ks:
            topk = top_idx[:k]
            if true_idx in topk:
                recall_dict[k] += 1
                rank = int(np.where(topk == true_idx)[0][0]) + 1
                map_dict[k]  += 1.0 / rank
                # ndcg_dict[k] += 1.0 / np.log2(rank + 1)

    result = {}
    print("\n" + "=" * 60)
    print(f"Evaluation Result [{mode_str}]")
    print("=" * 60)

    for k in Ks:
        recall = recall_dict[k] / n_eval
        map_k  = map_dict[k]   / n_eval
        # ndcg_k = ndcg_dict[k]  / n_eval

        result[f"Recall@{k}"] = recall
        result[f"MAP@{k}"]    = map_k
        # result[f"NDCG@{k}"]   = ndcg_k

        print(
            f"Recall@{k:<3d} = {recall:.4f}    "
            f"MAP@{k:<3d} = {map_k:.4f}    "
            # f"NDCG@{k:<3d} = {ndcg_k:.4f}"
        )

    print("=" * 60)
    return result


# =====================================================
# 训练主循环
# =====================================================

best_recall      = 0.0
all_eval_results = []
os.makedirs(MODEL_DIR, exist_ok=True)

num_batches = int(np.ceil(x_train_sparse.shape[0] / BATCH_SIZE))

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    epoch_loss = []

    train_bar = tqdm(train_ds, total=num_batches,
                     desc=f"Training Epoch {epoch+1}", ncols=100)

    for batch_idx in train_bar:
        batch_idx_np = batch_idx.numpy()
        bx   = x_train_sparse[batch_idx_np].toarray().astype(np.float32)
        by   = y_train[batch_idx_np]
        loss = train_step(bx, by)
        epoch_loss.append(float(loss.numpy()))
        train_bar.set_postfix(loss=f"{epoch_loss[-1]:.4f}")

    mean_loss = np.mean(epoch_loss)
    print(f"Epoch {epoch+1} Mean Loss = {mean_loss:.4f}")

    should_eval = (
        (epoch + 1 >= START_EVAL)
        and ((epoch + 1 - START_EVAL) % EVAL_EVERY == 0)
    )

    if should_eval:

        t_eval      = time.time()
        eval_result = evaluate_model(model, final=False)   # 抽样评估
        print(f"评估耗时: {time.time()-t_eval:.1f}s")

        record = {"epoch": epoch + 1, **eval_result}
        all_eval_results.append(record)

        ckpt_name = (
            f"epoch{epoch+1:03d}"
            f"_R5={eval_result['Recall@5']:.4f}"
            f"_R20={eval_result['Recall@20']:.4f}"
            f".weights.h5"
        )
        model.save_weights(os.path.join(MODEL_DIR, ckpt_name))
        print(f"✓ Model saved → {ckpt_name}")

        if eval_result["Recall@5"] > best_recall:
            best_recall = eval_result["Recall@5"]
            model.save_weights(os.path.join(MODEL_DIR, "best_" + MODEL_NAME))
            print(f"New best Recall@5 = {best_recall:.4f} → best_{MODEL_NAME}")


# =====================================================
# 训练结束：全量最终评估
# =====================================================

print("\n\n" + "=" * 70)
print("训练结束，开始全量最终评估...")
print("=" * 70)

# 加载最优模型进行全量评估
best_path = os.path.join(MODEL_DIR, "best_" + MODEL_NAME)
if os.path.exists(best_path + ".index"):
    model.load_weights(best_path)
    print(f"已加载最优模型: best_{MODEL_NAME}")

t_final      = time.time()
final_result = evaluate_model(model, final=True)   # 全量评估
print(f"全量评估耗时: {time.time()-t_final:.1f}s")


# =====================================================
# 汇总所有训练过程中的评估结果
# =====================================================

print(f"\n{'='*80}")
print("Training Evaluation Summary (抽样结果，仅供参考趋势)")
print(f"{'='*80}")
print(f"{'Epoch':<8} {'R@5':<8} {'R@10':<8} {'R@20':<8} "
      f"{'MAP@5':<8} {'MAP@10':<8} {'MAP@20':<8} ")
      # f"{'NDCG@5':<8} {'NDCG@10':<8} {'NDCG@20':<8}")
print("-" * 80)

for r in all_eval_results:
    print(
        f"{r['epoch']:<8} "
        f"{r['Recall@5']:<8.4f} {r['Recall@10']:<8.4f} {r['Recall@20']:<8.4f} "
        f"{r['MAP@5']:<8.4f} {r['MAP@10']:<8.4f} {r['MAP@20']:<8.4f} "
        # f"{r['NDCG@5']:<8.4f} {r['NDCG@10']:<8.4f} {r['NDCG@20']:<8.4f}"
    )

print(f"{'='*80}")
print(f"最终全量 Recall@5  = {final_result['Recall@5']:.4f}")
print(f"最终全量 Recall@20 = {final_result['Recall@20']:.4f}")

