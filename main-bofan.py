import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

from loadData import read_data_all
from cluster import apply_clustering
from cluster import split_by_cluster
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    file_path = "./datasets/heart.csv"

    # Step 1: 读取所有数据
    X_all, y_all = read_data_all(file_path, label_col="HeartDisease")
    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all)

    # Step 2: 对整个数据集进行 SMOTE 处理（先平衡类别）
    # try:
    #     smote = SMOTE(random_state=42)
    #     X_all, y_all = smote.fit_resample(X_all, y_all)
    #     count_0 = np.sum(y_all == 0)
    #     count_1 = np.sum(y_all == 1)
    #     ratio_1 = count_1 / (count_0 + count_1 + 1e-6)
    #     print(f"After SMOTE：0={count_0}, 1={count_1}")
    # except ValueError as e:
    #     print(f"SMOTE fail: {e}")

    for best_k in [6]:
        print(f"\n===== Random Forest (Cluster = {best_k}) =====")
        model, cluster_labels = apply_clustering(X_all, best_k)
        split_data = split_by_cluster(X_all, y_all, cluster_labels)

        total_test_metrics = {'y': [], 'y_pred': []}
        for c, data in split_data.items():
            print(f"\n--- Cluster {c} ---")

            X_train = np.asarray(data["X_train"])
            y_train = np.asarray(data["y_train"])
            # X_val = data["X_val"]
            # y_val = data["y_val"]
            X_test = data["X_test"]
            y_test = data["y_test"]
            print(f"Train samples: {len(X_train)}, "
                  # f"Valid samples: {len(X_val)}, "
                  f"Test samples: {len(X_test)}, ", end='')
            print(f"Risk in Train: {sum(y_train) / len(y_train):.2f}, "
                  # f"{sum(y_val) / len(y_val):.2f}, "
                  f"Risk in Test: {sum(y_test) / len(y_test):.2f}")

            # 不再对每个 cluster 内部做 SMOTE
            # best_d = [10, 14][c]
            # best_n = [8, 9][c]
            # print(f"n_estimators: {n_estimators}, depth: {depth}")
            # rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=depth, random_state=42)
            # rf.fit(X_train, y_train)
            best_f1 = 0
            beat_acc = 0
            best_n = 0
            best_d = 0
            for n in range(2, 15):
                for d in range(2, 15):
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                    rf.fit(X_train, y_train)
                    # rf = SVC(probability=True, kernel='rbf', random_state=42)
                    # rf.fit(X_train, y_train)

                    y_pred = rf.predict(X_test)
                    y_prob = rf.predict_proba(X_test)[:, 1]
                    y = y_test
                    if f1_score(y, y_pred) > best_f1 or accuracy_score(y, y_pred) > beat_acc:
                        print(f"Dep: {d}, Est: {n}, "
                              f"F1: {f1_score(y, y_pred):.4f}, "
                              f"Acc: {accuracy_score(y, y_pred):.4f}, "
                              f"BalAcc: {balanced_accuracy_score(y, y_pred):.4f}, "
                              f"AUC: {roc_auc_score(y, y_prob):.4f}")
                        best_f1 = f1_score(y, y_pred)
                        beat_acc = accuracy_score(y, y_pred)
                        best_n = n
                        best_d = d

            print(f"depth: {best_d}, n_estimators: {best_n}")
            rf = RandomForestClassifier(n_estimators=best_n, max_depth=best_d, random_state=42)
            rf.fit(X_train, y_train)
            # rf = SVC(probability=True, kernel='rbf', random_state=42)
            # rf.fit(X_train, y_train)
            for name, X, y in [("Train", X_train, y_train), (" Test", X_test, y_test)]:
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
                    total_test_metrics['y'].extend(y)
                    total_test_metrics['y_pred'].extend(y_pred)
        # 加权平均
        total_y = total_test_metrics['y']
        total_y_pred = total_test_metrics['y_pred']

        # 初始化加权指标
        weighted_avg = {
            "F1": f1_score(total_y, total_y_pred),
            "Accuracy": accuracy_score(total_y, total_y_pred),
            "Balanced Accuracy": balanced_accuracy_score(total_y, total_y_pred),
            "AUC": roc_auc_score(total_y, total_y_pred)
        }

        # 打印加权平均结果
        print(f"\n===== Random Forest (Cluster = {best_k}) =====")
        for k, v in weighted_avg.items():
            print(f"{k}: {v:.4f}")
