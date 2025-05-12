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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE



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
    n_estimators_list = [20,30,50,80,100]  # 100 展现出比较好的 但0.6 - 0.68之间
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

            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)

            # 手动过采样（复制少数类样本） 尝试了欠采样- 效果非常差
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            if len(class_counts) == 2:
                majority_class = unique_classes[np.argmax(class_counts)]
                minority_class = unique_classes[np.argmin(class_counts)]
                num_to_add = np.max(class_counts) - np.min(class_counts)

                idx_minority = np.where(y_train == minority_class)[0]
                repeat_indices = np.random.choice(idx_minority, size=num_to_add, replace=True)

                X_train = np.vstack((X_train, X_train[repeat_indices]))
                y_train = np.concatenate((y_train, y_train[repeat_indices]))

                # 🔍 过采样后，统计新 y_train 中 0 和 1 的数量
                count_0 = np.sum(y_train == 0)
                count_1 = np.sum(y_train == 1)
                ratio_1 = count_1 / (count_0 + count_1 + 1e-6)
                print(f"📊 过采样后 Cluster {c} 样本分布：0类={count_0}, 1类={count_1}, 1类占比={ratio_1:.2%}")

            # 自动设置 scale_pos_weight
            pos = np.sum(y_train == 1)
            neg = np.sum(y_train == 0)
            scale_pos_weight = neg / pos if pos > 0 else 1.0

            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            xgb = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)
            # 组合 Voting 模型（软投票）
            ensemble = VotingClassifier(estimators=[
                ('rf', rf),
                ('xgb', xgb)
            ], voting='soft')
            # 拟合模型
            ensemble.fit(X_train, y_train)

            #clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            #clf.fit(X_train, y_train)

            for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
                y_pred = ensemble.predict(X)
                y_prob = ensemble.predict_proba(X)[:, 1]
                print(
                    f"{name} - F1: {f1_score(y, y_pred):.4f}, "
                    f"Acc: {accuracy_score(y, y_pred):.4f}, "
                    f"BalAcc: {balanced_accuracy_score(y, y_pred):.4f}, "
                    f"AUC: {roc_auc_score(y, y_prob):.4f}")

            #  加入 SMOTE 过采样 这个python版本不行 得3.7 我不能动我的项目环境 我得跑cv



"""
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
    n_estimators_list = [20,30,50,80,100]  # 100 展现出比较好的 但0.6 - 0.68之间
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

            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            count_0 = np.sum(y_train == 0)
            count_1 = np.sum(y_train == 1)
            ratio_1 = count_1 / (count_0 + count_1 + 1e-6)
            print(f"📊 SMOTE后 Cluster {c} 样本分布：0类={count_0}, 1类={count_1}, 1类占比={ratio_1:.2%}")
except ValueError as e:
            print(f"⚠️ SMOTE失败，Cluster {c} 数据过少: {e}")


            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            xgb = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)
            # 组合 Voting 模型（软投票）
            ensemble = VotingClassifier(estimators=[
                ('rf', rf),
                ('xgb', xgb)
            ], voting='soft')
            # 拟合模型
            ensemble.fit(X_train, y_train)

            #clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            #clf.fit(X_train, y_train)

            for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
                y_pred = ensemble.predict(X)
                y_prob = ensemble.predict_proba(X)[:, 1]
                print(
                    f"{name} - F1: {f1_score(y, y_pred):.4f}, "
                    f"Acc: {accuracy_score(y, y_pred):.4f}, "
                    f"BalAcc: {balanced_accuracy_score(y, y_pred):.4f}, "
                    f"AUC: {roc_auc_score(y, y_prob):.4f}")

            #  加入 SMOTE 过采样 这个python版本不行 得3.7 我不能动我的项目环境 我得跑cv
"""
