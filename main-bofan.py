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
    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all)

    # ✅ Step 2: 对整个数据集进行 SMOTE 处理（先平衡类别）
    try:
        smote = SMOTE(random_state=42)
        X_all, y_all = smote.fit_resample(X_all, y_all)
        count_0 = np.sum(y_all == 0)
        count_1 = np.sum(y_all == 1)
        ratio_1 = count_1 / (count_0 + count_1 + 1e-6)
        print(f"📊 全数据 SMOTE后样本分布：0类={count_0}, 1类={count_1}, 1类占比={ratio_1:.2%}")
    except ValueError as e:
        print(f"⚠️ 全数据 SMOTE失败: {e}")

    # ✅ Step 3: 聚类处理
    best_k = 4
    model, cluster_labels = apply_clustering(X_all, best_k)

    # ✅ Step 4: 按 cluster 分别划分数据（内部不再做 SMOTE）
    split_data = split_by_cluster(X_all, y_all, cluster_labels)

    # Step 5: 遍历每个 cluster 并训练 Voting 模型
    n_estimators_list = [10]
    for n_estimators in n_estimators_list:
        print(f"\n===== Random Forest (n_estimators={n_estimators}) =====")
        total_test_metrics = []
        for c, data in split_data.items():
            print(f"\n--- Cluster {c} ---")

            X_train = np.asarray(data["X_train"])
            y_train = np.asarray(data["y_train"])
            X_val = data["X_val"]
            y_val = data["y_val"]
            X_test = data["X_test"]
            y_test = data["y_test"]
            print(f"Cluster {c} - "
                  f"Train samples: {len(X_train)}, "
                  f"Val samples: {len(X_val)}, "
                  f"Test samples: {len(X_test)}")

            # 不再对每个 cluster 内部做 SMOTE

            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            xgb = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)
            ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
            ensemble.fit(X_train, y_train)

            for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
                y_pred = ensemble.predict(X)
                y_prob = ensemble.predict_proba(X)[:, 1]
                metrics = {"F1": f1_score(y, y_pred),
                           "Accuracy": accuracy_score(y, y_pred),
                           "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
                           "AUC": roc_auc_score(y, y_prob),
                           "Samples": len(X)}
                print(
                    f"{name} - F1: {metrics["F1"]:.4f}, "
                    f"Acc: {metrics["Accuracy"]:.4f}, "
                    f"BalAcc: {metrics["Balanced Accuracy"]:.4f}, "
                    f"AUC: {metrics["AUC"]:.4f}")
                if name == "Test":
                    total_test_metrics.append(metrics)
        # 加权平均



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
    n_estimators_list = [5]  # 100 展现出比较好的 但0.6 - 0.68之间
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
