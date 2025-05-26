# NN.py
"""
多层感知机 (MLP) 早停版 + LIME 可解释性
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, accuracy_score,
                             roc_auc_score, balanced_accuracy_score)
import pandas as pd
from loadData_chz import read_data  # 新版返回 scaler & feature_names


def write_metrics_to_csv(metrics, filename, model_name="Logistic Regression"):
    filename = f"../results/{filename}.csv"
    df = pd.DataFrame([metrics])
    # 最前面加一列
    df.insert(0, 'Dataset', model_name)
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)


# 1. 读取并预处理数据
file_num = 0

file_path = "../datasets/"
file_name = ["heart", "heart-1", "UCI-1190-11", "statlog_heart"][file_num]
label_col = ["HeartDisease", "target", "target", "target"][file_num]

X_train, y_train, X_test, y_test, scaler, feature_names = read_data(
    f"{file_path}{file_name}.csv", label_col=label_col)

print("===> Label Distribution")
print(f"Train: 1={y_train.sum()} / 0={len(y_train) - y_train.sum()}")
print(f"Test : 1={y_test.sum()} / 0={len(y_test) - y_test.sum()}\n")

# 2. 设定不同隐藏层结构
hidden_layers_list = [(8,), (16, 8), (32, 16), (32, 32)]

for hidden_layers in hidden_layers_list:
    print(f"--- Training MLP {hidden_layers} ---")
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers,
                        activation='relu',
                        solver='adam',
                        alpha=1e-3,
                        max_iter=1000,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=760,
                        verbose=False)
    mlp.fit(X_train, y_train)

    # 3. 评估
    for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        pred = mlp.predict(X)
        prob = mlp.predict_proba(X)[:, 1]
        print(f"{name}: F1={f1_score(y, pred):.4f} | "
              f"Acc={accuracy_score(y, pred):.4f} | "
              f"BalAcc={balanced_accuracy_score(y, pred):.4f} | "
              f"AUC={roc_auc_score(y, prob):.4f}")
        if name == "Test":
            test_metrics = {
                'F1': round(f1_score(y, pred), 4),
                'Accuracy': round(accuracy_score(y, pred), 4),
                'Balanced Accuracy': round(balanced_accuracy_score(y, pred), 4),
                'AUC': round(roc_auc_score(y, prob), 4)
            }
            write_metrics_to_csv(test_metrics, filename=file_name, model_name=f"Neural Network {hidden_layers}")

    print("n_iter =", mlp.n_iter_, "\n")
    # 写入csv(加入到最后一行)

    # 4. LIME 可解释性（仅解释前 5 条测试样本）
    # from lime.lime_tabular import LimeTabularExplainer
    # import os
    # out_dir = "lime_reports"
    # os.makedirs(out_dir, exist_ok=True)
    #
    # explainer = LimeTabularExplainer(
    #     X_train,
    #     feature_names=feature_names,
    #     class_names=["No-HD", "HeartDisease"],
    #     discretize_continuous=True)
    #
    # for i in range(min(5, len(X_test))):
    #     exp = explainer.explain_instance(
    #         X_test[i],
    #         mlp.predict_proba,
    #         num_features=10)
    #     html_path = f"{out_dir}/lime_sample{i}_{'_'.join(map(str, hidden_layers))}.html"
    #     exp.save_to_file(html_path)
    #     print(f"[LIME] saved → {html_path}")
