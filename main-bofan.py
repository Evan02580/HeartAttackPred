import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from loadData import read_data_all
from cluster import apply_clustering
from cluster import split_by_cluster
from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    file_path = "./datasets/heart-attack-risk-prediction-dataset.csv"

    # Step 1: 读取所有数据
    X_all, y_all = read_data_all(file_path)

    # Step 2: 全数据聚类
    best_k = 4
    model, cluster_labels = apply_clustering(X_all, best_k) # 数据划分

    # Step 3: 按 cluster 分别划分数据
    split_data = split_by_cluster(X_all, y_all, cluster_labels)

    # Step 4: 遍历每个 cluster 并训练模型RF
    n_estimators_list = [110,20,50,100,200]  # 100 展现出比较好的 但0.6 - 0.68之间
    for n_estimators in n_estimators_list:
        print(f"\n===== Random Forest (n_estimators={n_estimators}) =====")

        for c, data in split_data.items():
            print(f"\n--- Cluster {c} ---")
            X_train = data["X_train"]
            y_train = data["y_train"]
            X_val = data["X_val"]
            y_val = data["y_val"]
            X_test = data["X_test"]
            y_test = data["y_test"]

            clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            clf.fit(X_train, y_train)

            for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
                y_pred = clf.predict(X)
                y_prob = clf.predict_proba(X)[:, 1]
                print(
                    f"{name} - F1: {f1_score(y, y_pred):.4f}, "
                    f"Acc: {accuracy_score(y, y_pred):.4f}, "
                    f"BalAcc: {balanced_accuracy_score(y, y_pred):.4f}, "
                    f"AUC: {roc_auc_score(y, y_prob):.4f}")

            #  加入 SMOTE 过采样 这个python版本不行 得3.7 我不能动我的项目环境 我得跑cv
"""
# 确保是 numpy array，避免 pandas index 报错 这里是手动采样部分 反而变差了
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)

            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            if len(class_counts) == 2:
                majority_class = unique_classes[np.argmax(class_counts)]
                minority_class = unique_classes[np.argmin(class_counts)]
                num_to_add = np.max(class_counts) - np.min(class_counts)

                idx_minority = np.where(y_train == minority_class)[0]
                repeat_indices = np.random.choice(idx_minority, size=num_to_add, replace=True)

                X_train = np.vstack((X_train, X_train[repeat_indices]))
                y_train = np.concatenate((y_train, y_train[repeat_indices]))
"""






"""
 # Step 4: 遍历每个 cluster 并训练 XGBoost 模型
    n_estimators_list = [5, 10, 15, 20,25,30,40,50]
    for n_estimators in n_estimators_list:
        print(f"\n===== XGBoost (n_estimators={n_estimators}) =====")

        for c, data in split_data.items():
            print(f"\n--- Cluster {c} ---")
            X_train = np.asarray(data["X_train"])
            y_train = np.asarray(data["y_train"])
            X_val = np.asarray(data["X_val"])
            y_val = np.asarray(data["y_val"])
            X_test = np.asarray(data["X_test"])
            y_test = np.asarray(data["y_test"])

            # 手动过采样（复制少数类样本）
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            if len(class_counts) == 2:
                majority_class = unique_classes[np.argmax(class_counts)]
                minority_class = unique_classes[np.argmin(class_counts)]
                num_to_add = np.max(class_counts) - np.min(class_counts)

                idx_minority = np.where(y_train == minority_class)[0]
                repeat_indices = np.random.choice(idx_minority, size=num_to_add, replace=True)

                X_train = np.vstack((X_train, X_train[repeat_indices]))
                y_train = np.concatenate((y_train, y_train[repeat_indices]))

            # 自动设置 scale_pos_weight
            pos = np.sum(y_train == 1)
            neg = np.sum(y_train == 0)
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            clf = XGBClassifier(
                n_estimators=n_estimators,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                random_state=42
            )
            clf.fit(X_train, y_train)

            for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
                y_pred = clf.predict(X)
                y_prob = clf.predict_proba(X)[:, 1]
                print(
                    f"{name} - F1: {f1_score(y, y_pred):.4f}, Acc: {accuracy_score(y, y_pred):.4f}, BalAcc: {balanced_accuracy_score(y, y_pred):.4f}, AUC: {roc_auc_score(y, y_prob):.4f}")
                """
