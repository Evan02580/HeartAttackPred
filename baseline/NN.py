from loadData import read_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


# ===================== 🚨 没有早停的过拟合版本，它会在训练集上表现优异，但是在测试集和验证集上非常糟糕 =====================
#
# # 1. 读取并检查数据
# file_path = "../datasets/heart-attack-risk-prediction-dataset.csv"
# X_train, y_train, X_val, y_val, X_test, y_test = read_data(file_path)
#
# print("===> Checking label distribution <===")
# print("Train: Positive samples =", np.sum(y_train), " Negative =", len(y_train) - np.sum(y_train))
# print("Valid: Positive samples =", np.sum(y_val), " Negative =", len(y_val) - np.sum(y_val))
# print("Test:  Positive samples =", np.sum(y_test), " Negative =", len(y_test) - np.sum(y_test))
# print("")
#
# # 2. 对特征进行标准化（Standardization）
# scaler = StandardScaler()
# scaler.fit(X_train)  # 只在训练集上fit
# X_train_scaled = scaler.transform(X_train)
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)
#
# # 3. 设置不同的隐藏层结构供测试
# hidden_layers_list = [
#     (32, 16),
#     (64, 32),
# ]
#
# for hidden_layers in hidden_layers_list:
#     # 4. 建立 MLP 模型
#     # 这里演示关闭 early_stopping, 提高 max_iter, 并可选 class_weight='balanced'
#     mlp_model = MLPClassifier(
#         hidden_layer_sizes=hidden_layers,
#         activation='relu',
#         solver='adam',
#         max_iter=1000,             # 加大迭代次数
#         random_state=760,
#         # alpha=1e-4,              # 正则化强度(默认是1e-4),可根据表现调整
#         # learning_rate_init=0.001,# 默认学习率,若仍难收敛可尝试稍微调大
#         early_stopping=False,      # 暂时关闭 early_stopping
#         # class_weight='balanced', # 如果数据极度不平衡，可以启用它
#         verbose=True               # 打印损失和迭代信息，便于观察收敛
#     )
#
#     # 5. 训练
#     mlp_model.fit(X_train_scaled, y_train)
#
#     # 打印本次迭代了多少轮
#     print(f"\n[Debug] Hidden Layers: {hidden_layers}, n_iter_ = {mlp_model.n_iter_}")
#
#     # 6. 在三套数据集上预测
#     y_train_pred = mlp_model.predict(X_train_scaled)
#     y_val_pred = mlp_model.predict(X_val_scaled)
#     y_test_pred = mlp_model.predict(X_test_scaled)
#
#     # 7. 计算并打印指标
#     train_f1  = f1_score(y_train, y_train_pred)
#     val_f1    = f1_score(y_val,   y_val_pred)
#     test_f1   = f1_score(y_test,  y_test_pred)
#
#     train_acc = accuracy_score(y_train, y_train_pred)
#     val_acc   = accuracy_score(y_val,   y_val_pred)
#     test_acc  = accuracy_score(y_test,  y_test_pred)
#
#     train_auc = roc_auc_score(y_train, mlp_model.predict_proba(X_train_scaled)[:, 1])
#     val_auc   = roc_auc_score(y_val,   mlp_model.predict_proba(X_val_scaled)[:, 1])
#     test_auc  = roc_auc_score(y_test,  mlp_model.predict_proba(X_test_scaled)[:, 1])
#
#     print(f"MLP with hidden layers = {hidden_layers}")
#     print(f"Train: F1={train_f1:.4f}, Acc={train_acc:.4f}, AUC={train_auc:.4f}")
#     print(f"Valid: F1={val_f1:.4f},   Acc={val_acc:.4f},   AUC={val_auc:.4f}")
#     print(f"Test:  F1={test_f1:.4f},  Acc={test_acc:.4f},  AUC={test_auc:.4f}\n")
#







# ===================== 🚨 早停的版本 =====================

# 1. 加载数据
file_path = "../datasets/heart-attack-risk-prediction-dataset.csv"
X_train, y_train, X_val, y_val, X_test, y_test = read_data(file_path)

# 输出标签分布
print("===> Checking label distribution <===")
print("Train: Positive =", int(np.sum(y_train)), " Negative =", len(y_train) - int(np.sum(y_train)))
print("Valid: Positive =", int(np.sum(y_val)),   " Negative =", len(y_val) - int(np.sum(y_val)))
print("Test:  Positive =", int(np.sum(y_test)),  " Negative =", len(y_test) - int(np.sum(y_test)))
print("")

# 2. 标准化特征
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 3. 定义隐藏层结构
hidden_layers_list = [
    (16,),
    (32, 16),
    (16, 8),
    (64, 32),
]

results_storage = []

for hidden_layers in hidden_layers_list:
    print(f"--- Training MLP with hidden_layers={hidden_layers} ---")

    mlp_model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=760,
        verbose=False
    )

    mlp_model.fit(X_train_scaled, y_train)

    y_train_pred = mlp_model.predict(X_train_scaled)
    y_val_pred = mlp_model.predict(X_val_scaled)
    y_test_pred = mlp_model.predict(X_test_scaled)

    result = {
        'hidden_layers': hidden_layers,
        'n_iter': mlp_model.n_iter_,
        'train_f1': f1_score(y_train, y_train_pred),
        'val_f1': f1_score(y_val, y_val_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'train_acc': accuracy_score(y_train, y_train_pred),
        'val_acc': accuracy_score(y_val, y_val_pred),
        'test_acc': accuracy_score(y_test, y_test_pred),
        'train_auc': roc_auc_score(y_train, mlp_model.predict_proba(X_train_scaled)[:, 1]),
        'val_auc': roc_auc_score(y_val, mlp_model.predict_proba(X_val_scaled)[:, 1]),
        'test_auc': roc_auc_score(y_test, mlp_model.predict_proba(X_test_scaled)[:, 1]),
    }

    results_storage.append(result)

    # 单个模型结果
    print(f"n_iter = {result['n_iter']}")
    print(f"Train: F1={result['train_f1']:.4f}, Acc={result['train_acc']:.4f}, AUC={result['train_auc']:.4f}")
    print(f"Valid: F1={result['val_f1']:.4f}, Acc={result['val_acc']:.4f}, AUC={result['val_auc']:.4f}")
    print(f"Test:  F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}, AUC={result['test_auc']:.4f}")
    print("")

# 4. 汇总所有模型结果
print("============ All MLP Configurations Summary ============")
for res in results_storage:
    hl = res['hidden_layers']
    print(f"hidden_layers={hl}, n_iter={res['n_iter']}")
    print(f"  Train: F1={res['train_f1']:.4f}, Acc={res['train_acc']:.4f}, AUC={res['train_auc']:.4f}")
    print(f"  Valid: F1={res['val_f1']:.4f}, Acc={res['val_acc']:.4f}, AUC={res['val_auc']:.4f}")
    print(f"  Test:  F1={res['test_f1']:.4f}, Acc={res['test_acc']:.4f}, AUC={res['test_auc']:.4f}")
    print("--------------------------------------------------------")
