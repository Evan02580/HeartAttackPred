from loadData import read_data
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
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

kernels = ['poly', 'rbf', 'sigmoid']

for kernel in kernels:
#2. Build SVM drictly on data set
    svm_model = SVC(probability=True, kernel=kernel, random_state=760)
    svm_model.fit(X_train, y_train)

    #3. Predict on each set to see the result
    y_train_pred = svm_model.predict(X_train)
    # y_val_pred = svm_model.predict(X_val)
    y_test_pred = svm_model.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_balacc = balanced_accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, svm_model.predict_proba(X_train)[:, 1])

    # val_f1 = f1_score(y_val, y_val_pred)
    # val_accuracy = accuracy_score(y_val, y_val_pred)
    # val_auc = roc_auc_score(y_val, svm_model.predict_proba(X_val)[:, 1])

    test_f1 = f1_score(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_balacc = balanced_accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])

    print("Result in",kernel,"kernel.")
    print(f"Train: \nF1: {train_f1:.4f}\nAccuracy: {train_accuracy:.4f}\nBalanced Accuracy: {train_balacc:.4f}\nAUC: {train_auc:.4f}")
    # print(f"\nValid Set Metrics: F1 Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
    print(f" Test: \nF1: {test_f1:.4f}\nAccuracy: {test_accuracy:.4f}\nBalanced Accuracy: {test_balacc:.4f}\nAUC: {test_auc:.4f}\n")

    test_metrics = {
        'F1': round(test_f1, 4),
        'Accuracy': round(test_accuracy, 4),
        'Balanced Accuracy': round(test_balacc, 4),
        'AUC': round(test_auc, 4)
    }
    write_metrics_to_csv(test_metrics, filename=file_name, model_name=f"SVM (kernel = {kernel})")
