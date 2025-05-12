from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import numpy as np

def find_optimal_k(data, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=760)
        model.fit(data)
        score = silhouette_score(data, model.labels_)
        scores.append((k, score))
    return scores

# 得到聚类的模型 - 可以知道新数据属于哪个聚类， labels将数据分组 在组内进行划分数据
def apply_clustering(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=760)
    labels = model.fit_predict(data)
    return model, labels

# 按 cluster 分别划分 train/val/test 0.7/0.1/0.2 ---- 相同的类 进行数据划分
def split_by_cluster(X, y, cluster_labels, test_size=0.5, val_ratio=0.6):
    split_data = {}
    for c in set(cluster_labels):
        idx = (cluster_labels == c)
        X_cluster = X[idx]
        y_cluster = y[idx]

        X_train, X_temp, y_train, y_temp = train_test_split(X_cluster, y_cluster, test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_ratio, random_state=42)

        split_data[c] = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }

    return split_data

# 统计每个聚类的样本数量
def print_cluster_distribution(cluster_labels):
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print("\n📊 每个 Cluster 的样本数量:")
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"Cluster {cluster_id}: {count} samples")