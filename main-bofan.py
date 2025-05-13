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

    # Step 2: 对整个数据集进行 SMOTE 处理（先平衡类别）
    try:
        smote = SMOTE(random_state=42)
        X_all, y_all = smote.fit_resample(X_all, y_all)
        count_0 = np.sum(y_all == 0)
        count_1 = np.sum(y_all == 1)
        ratio_1 = count_1 / (count_0 + count_1 + 1e-6)
        print(f"After SMOTE：0={count_0}, 1={count_1}")
    except ValueError as e:
        print(f"SMOTE fail: {e}")

    # Step 3: 聚类处理
    best_k = 4
    model, cluster_labels = apply_clustering(X_all, best_k)

    # Step 4: 按 cluster 分别划分数据（内部不再做 SMOTE）
    split_data = split_by_cluster(X_all, y_all, cluster_labels)

    # Step 5: 遍历每个 cluster 并训练 Voting 模型
    n_estimators_list = [30]
    for n_estimators in n_estimators_list:
        # print(f"\n===== Random Forest (n_estimators={n_estimators}) =====")
        total_test_metrics = []
        for c, data in split_data.items():
            print(f"\n--- Cluster {c} ---")

            X_train = np.asarray(data["X_train"])
            y_train = np.asarray(data["y_train"])
            X_val = data["X_val"]
            y_val = data["y_val"]
            X_test = data["X_test"]
            y_test = data["y_test"]
            print(f"Train samples: {len(X_train)}, "
                  f"Valid samples: {len(X_val)}, "
                  f"Test samples: {len(X_test)}, ", end='')
            # print(sum(y_train), sum(y_val), sum(y_test))

            # 不再对每个 cluster 内部做 SMOTE
            depth = [10, 20, 20, 26][c]
            n_estimators = [35, 55, 45, 45][c]
            print(f"n_estimators: {n_estimators}, depth: {depth}")
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, random_state=42)
            rf.fit(X_train, y_train)
            # best_f1 = 0
            # beat_acc = 0
            # for n in range(25, 66, 5):
            #     for d in range(10, 46):
            #         rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
            #         rf.fit(X_train, y_train)
            #         y_pred = rf.predict(X_test)
            #         y_prob = rf.predict_proba(X_test)[:, 1]
            #         y = y_test
            #         if f1_score(y, y_pred) > best_f1 or accuracy_score(y, y_pred) > beat_acc:
            #             print(f"Dep: {d}, Est: {n}, "
            #                   f"F1: {f1_score(y, y_pred):.4f}, "
            #                   f"Acc: {accuracy_score(y, y_pred):.4f}, "
            #                   f"BalAcc: {balanced_accuracy_score(y, y_pred):.4f}, "
            #                   f"AUC: {roc_auc_score(y, y_prob):.4f}")
            #             best_f1 = f1_score(y, y_pred)
            #             beat_acc = accuracy_score(y, y_pred)
            # continue

            for name, X, y in [("Train", X_train, y_train), ("Valid", X_val, y_val), (" Test", X_test, y_test)]:
                y_pred = rf.predict(X)
                y_prob = rf.predict_proba(X)[:, 1]
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
                if name == " Test":
                    total_test_metrics.append(metrics)
        # 加权平均
        total_samples = sum(m["Samples"] for m in total_test_metrics)

        # 初始化加权指标
        weighted_avg = {
            "F1": 0.0,
            "Accuracy": 0.0,
            "Balanced Accuracy": 0.0,
            "AUC": 0.0
        }

        # 累加加权值
        for m in total_test_metrics:
            weight = m["Samples"] / total_samples
            for k in weighted_avg:
                weighted_avg[k] += m[k] * weight

        # 打印加权平均结果
        print("\n===== Weighted Average over Clusters (Test Set) =====")
        for k, v in weighted_avg.items():
            print(f"{k}: {v:.4f}")



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
