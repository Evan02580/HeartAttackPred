import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from loadData import read_data

# 1. 选择最优K
def find_optimal_k(data, max_k=10):
    inertia = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        inertia.append(model.inertia_)
        score = silhouette_score(data, model.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.show()

# 2. 聚类
def apply_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels

# 3. 主流程
if __name__ == "__main__":
    file_path = "./datasets/heart-attack-risk-prediction-dataset.csv"
    data, labels = read_data(file_path)

    find_optimal_k(data, max_k=10)

    cluster_labels = apply_clustering(data, n_clusters=4)

    # 可视化
    reduced_data = PCA(n_components=2).fit_transform(data)
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=cluster_labels, palette="Set2")
    plt.title('Cluster Visualization with PCA')
    plt.show()
