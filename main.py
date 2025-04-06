from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from loadData import read_data
from cluster import find_optimal_k, apply_clustering
from randomForest import train_rf_by_cluster, evaluate_rf_by_cluster
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Loading Data
file_path = "./datasets/heart-attack-risk-prediction-dataset.csv"
X_train, y_train, X_val, y_val, X_test, y_test = read_data(file_path)

# 2. Optimal K
scores = find_optimal_k(X_train, max_k=8)
k_values, silhouette_scores = zip(*scores)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.show(block=False)
plt.pause(2)


# 3. Clustering
best_k = 3
cluster_model = KMeans(n_clusters=best_k, random_state=42)
train_clusters = cluster_model.fit_predict(X_train)

# 4. Training Random Forest by Cluster
models, train_scores = train_rf_by_cluster(X_train, y_train, train_clusters)
print("\nTraining Results by Cluster:")
for c, metrics in tqdm(train_scores.items()):
    print(f"Cluster {c} - F1: {metrics['F1']:.4f}, Acc: {metrics['Accuracy']:.4f}, AUC: {metrics['AUC']:.4f}")

# 5. Predicting Clusters
val_clusters = cluster_model.predict(X_val)
test_clusters = cluster_model.predict(X_test)


val_metrics = evaluate_rf_by_cluster(models, X_val, y_val, val_clusters)
print("\nValidation Set Metrics (avg):", val_metrics)

test_metrics = evaluate_rf_by_cluster(models, X_test, y_test, test_clusters)
print("\nTest Set Metrics (avg):", test_metrics)
