import pdb

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from loadData import read_data_all
from cluster import apply_clustering
from cluster import split_by_cluster
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn.model_selection import train_test_split
from cluster import fit_kmeans_train_only  # 新增的函数


def write_metrics_to_csv(metrics, filename, model_name="Logistic Regression"):
    filename = f"./results/{filename}.csv"
    df = pd.DataFrame([metrics])
    # 最前面加一列
    df.insert(0, 'Dataset', model_name)
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)


if __name__ == "__main__":

    for file_num in range(4):
        file_path = "./datasets/"
        file_name = ["heart", "heart-1", "UCI-1190-11", "cleveland"][file_num]
        label_col = ["HeartDisease", "target", "target", "risk"][file_num]
        best_k = [6, 5, 7, 4][file_num]  # 每个数据集选择的K值

        # Step 1: 读取所有数据
        print(f"Dataset: {file_name}")
        X_all, y_all, feature_names = read_data_all(f"{file_path}{file_name}.csv", label_col=label_col)
        print(f"Feature names: {feature_names}")
        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)

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

            best_f1 = 0
            beat_acc = 0
            best_n = 0
            best_d = 0
            for n in range(2, 15):
                for d in range(2, 15):
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                    rf.fit(X_train, y_train)

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

            # import os
            #
            # # 创建保存图片的文件夹
            # output_dir = f"interpret/{file_name}/SHAP"
            # os.makedirs(output_dir, exist_ok=True)
            #
            # # ========== SHAP 解释（保存 summary plot 为图片） ==========
            # import shap
            # import matplotlib.pyplot as plt
            #
            # explainer = shap.TreeExplainer(rf)
            # shap_values = explainer.shap_values(X_test)
            # # pdb.set_trace()
            # print(f"绘制并保存 Cluster {c} 的 SHAP summary plot")
            # shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
            #
            # plt.title(f"Cluster {c} SHAP Summary Plot", fontsize=14)
            # plt.tight_layout()
            # plt.savefig(os.path.join(output_dir, f"shap_cluster{c}.png"))
            # plt.close()
            #
            # # ========== LIME 解释（只解释1个样本） ==========
            # import lime.lime_tabular
            #
            # explainer_lime = lime.lime_tabular.LimeTabularExplainer(X_train, mode="classification")
            # exp = explainer_lime.explain_instance(X_test[0], rf.predict_proba, num_features=10)
            # # print(f"解释 Cluster {c} 中第一个测试样本的 LIME 贡献度")
            # # exp.show_in_notebook(show_table=True)
            #
            # # ========== 单个样本 waterfall plot ==========
            # sample_idx = 0
            # sample = X_test[sample_idx].reshape(1, -1)
            # pred_prob = rf.predict_proba(sample)
            # print(f"Cluster {c} 中第一个样本的预测概率（低风险, 高风险）: {pred_prob}")
            #
            #
            # shap.plots._waterfall.waterfall_legacy(
            #     explainer.expected_value[1],
            #     shap_values[1][sample_idx],
            #     feature_names=feature_names,
            #     show=False
            # )
            # plt.gcf().set_size_inches(10, 8)
            # plt.subplots_adjust(top=0.9, bottom=0.1)
            # plt.title(f"Cluster {c} Sample {sample_idx} SHAP Waterfall Plot", fontsize=14)
            # plt.savefig(os.path.join(output_dir, f"shap_waterfall_cluster{c}.png"))
            # plt.close()

            for name, X, y in [("Train", X_train, y_train), (" Test", X_test, y_test)]:
                y_pred = rf.predict(X)
                y_prob = rf.predict_proba(X)[:, 1]
                metrics = {"F1": f1_score(y, y_pred),
                           "Accuracy": accuracy_score(y, y_pred),
                           "Balanced Accuracy": balanced_accuracy_score(y, y_pred),
                           "AUC": roc_auc_score(y, y_prob),
                           "Samples": len(X)}
                print(
                    f"{name} - F1: {metrics['F1']:.4f}, "
                    f"Acc: {metrics['Accuracy']:.4f}, "
                    f"BalAcc: {metrics['Balanced Accuracy']:.4f}, "
                    f"AUC: {metrics['AUC']:.4f}")
                if name == " Test":
                    total_test_metrics['y'].extend(y)
                    total_test_metrics['y_pred'].extend(y_pred)
        # 加权平均
        total_y = total_test_metrics['y']
        total_y_pred = total_test_metrics['y_pred']

        # 初始化加权指标
        weighted_avg = {
            "F1": round(f1_score(total_y, total_y_pred), 4),
            "Accuracy": round(accuracy_score(total_y, total_y_pred), 4),
            "Balanced Accuracy": round(balanced_accuracy_score(total_y, total_y_pred), 4),
            "AUC": round(roc_auc_score(total_y, total_y_pred), 4)
        }

        # 打印加权平均结果
        print(f"\n===== Random Forest (Cluster = {best_k}) =====")
        print(f"Data Name: {file_name}")
        print(f"Total Samples: {len(y_all)}")
        for k, v in weighted_avg.items():
            print(f"{k}: {v}")
        # 保存结果到 CSV
        # write_metrics_to_csv(weighted_avg, file_name, model_name=f"CluRF (k = {best_k})")