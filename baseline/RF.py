from loadData import read_data
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# 1. Loading Data
file_path = "../datasets/heart-attack-risk-prediction-dataset.csv"
X_train, y_train, X_val, y_val, X_test, y_test = read_data(file_path)

n_estimators_list = [50, 100, 150, 200]

for n_estimators in n_estimators_list:
    # 构建随机森林模型
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=760)
    rf_model.fit(X_train, y_train)

    # 对各数据集进行预测
    y_train_pred = rf_model.predict(X_train)
    y_val_pred = rf_model.predict(X_val)
    y_test_pred = rf_model.predict(X_test)

    # 计算各指标：F1分数、准确率和 AUC
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])

    val_f1 = f1_score(y_val, y_val_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, rf_model.predict_proba(X_val)[:, 1])

    test_f1 = f1_score(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    # 输出结果，每个指标合并为一行
    print(f"Result with n_estimators = {n_estimators}:")
    print(f"Train Set Metrics: F1 Score: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}")
    print(f"Valid Set Metrics: F1 Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
    print(f"Test Set Metrics: F1 Score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}\n")