import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

from loadData import read_data_all
from cluster import apply_clustering
from cluster import print_cluster_distribution

# 1. 选择最优K; labels are not included in data
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


# 主流程
if __name__ == "__main__":
    file_path = "./datasets/heart-attack-risk-prediction-dataset.csv"
    data, labels = read_data_all(file_path)

    find_optimal_k(data, max_k=10) #画图，输出结果是4

    model, cluster_labels = apply_clustering(data, n_clusters=4)  # 给出聚类数量 进行KMean++聚类
    print_cluster_distribution(cluster_labels)

    # 可视化
    reduced_data = PCA(n_components=2).fit_transform(data)
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=cluster_labels, palette="Set2")
    plt.title('Cluster Visualization with PCA')
    plt.show()

"""
聚类数量：4个 - silhouette_score图像得出来的
"""
