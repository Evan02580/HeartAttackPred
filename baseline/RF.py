from loadData import read_data
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Loading Data
file_num = 3

file_path = "../datasets/"
file_name = ["heart", "heart-1", "UCI-1190-11", "statlog_heart"][file_num]
label_col = ["HeartDisease", "target", "target", "target"][file_num]

def write_metrics_to_csv(metrics, filename=file_name, model_name="Logistic Regression"):
    filename = f"../results/{filename}.csv"
    df = pd.DataFrame([metrics])
    # 最前面加一列
    df.insert(0, 'Dataset', model_name)
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)


X_train, y_train, X_test, y_test = read_data(f"{file_path}{file_name}.csv", label_col=label_col)

n_estimators_list = [10, 20]

for n_estimators in n_estimators_list:
    # Construct RF
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=760)
    rf_model.fit(X_train, y_train)

    # Predict Data
    y_train_pred = rf_model.predict(X_train)
    # y_val_pred = rf_model.predict(X_val)
    y_test_pred = rf_model.predict(X_test)

    # Calculate metric
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, rf_model.predict_proba(X_train)[:, 1])

    # val_f1 = f1_score(y_val, y_val_pred)
    # val_accuracy = accuracy_score(y_val, y_val_pred)
    # val_auc = roc_auc_score(y_val, rf_model.predict_proba(X_val)[:, 1])

    test_f1 = f1_score(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    # result
    print(f"Result with n_estimators = {n_estimators}:")
    print(f"Train Set Metrics: F1 Score: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}")
    # print(f"Valid Set Metrics: F1 Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
    print(f"Test Set Metrics: \n"
          f"F1: {test_f1:.4f}\n"
          f"Accuracy: {test_accuracy:.4f}\n"
          f"Balanced Accuracy: {test_bal_acc:.4f}\n"
          f"AUC: {test_auc:.4f}\n")
    test_metrics = {
        'F1': round(test_f1, 4),
        'Accuracy': round(test_accuracy, 4),
        'Balanced Accuracy': round(test_bal_acc, 4),
        'AUC': round(test_auc, 4)
    }

    # 写入csv(加入到最后一行)
    write_metrics_to_csv(test_metrics, model_name=f"Random Forest (n = {n_estimators})")