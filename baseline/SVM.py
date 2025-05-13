from loadData import read_data
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

# 1. Loading Data
file_path = "../datasets/heart.csv"
X_train, y_train, X_test, y_test = read_data(file_path, label_col="HeartDisease")

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

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

