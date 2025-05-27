# main.py
"""
聚类 ➜ 按 cluster 训练随机森林 ➜ 评估 ➜ 解释 (SHAP + 聚类画像)
"""

from sklearn.cluster import KMeans

from loadData_chz import read_data
from randomForest import train_rf_by_cluster, evaluate_rf_by_cluster
from explain_rf_shap_chz import explain_rf_by_cluster
from Huazhe_SHAP.cluster_profile import plot_cluster_profiles

# 1. 读取数据
file_path = "../datasets/heart.csv"
X_train, y_train, X_test, y_test, scaler, feature_names = read_data(
    file_path, label_col="HeartDisease")

# 2. KMeans 聚类 (k=4)
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=760)
train_clusters = kmeans.fit_predict(X_train)
test_clusters  = kmeans.predict(X_test)

# 3. 按 cluster 训练随机森林
n_estimators = 30
models, train_scores = train_rf_by_cluster(
    X_train, y_train, train_clusters, n_estimators=n_estimators)

print("\n=== Train Metrics by Cluster ===")
for c, m in train_scores.items():
    print(f"Cluster {c}: F1={m['F1']:.4f} | Acc={m['Accuracy']:.4f} | "
          f"AUC={m['AUC']:.4f} | BalAcc={m['Balanced Accuracy']:.4f}")

# 4. 测试集评估（加权平均）
test_metrics = evaluate_rf_by_cluster(models, X_test, y_test, test_clusters)
print("\n=== Test Metrics (weighted) ===")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# ------------------------------------------------------------------
# 5. 解释性分析
# ------------------------------------------------------------------

# 5-1 SHAP（全局 & 局部）
X_train_orig = scaler.inverse_transform(X_train)  # 还原到原尺度
explain_rf_by_cluster(models, X_train_orig, train_clusters, feature_names)

# 5-2 聚类画像（雷达图 + 均值 CSV）
plot_cluster_profiles(X_train_orig, train_clusters, feature_names)

print("\n解释性文件已生成：shap_plots/  &  cluster_profiles/")
