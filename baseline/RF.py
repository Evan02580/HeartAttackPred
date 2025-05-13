from loadData import read_data
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# 1. Loading Data
file_path = "../datasets/heart.csv"
X_train, y_train, X_test, y_test = read_data(file_path, label_col="HeartDisease")

n_estimators_list = [5, 10, 20, 30]

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
    test_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    # result
    print(f"Result with n_estimators = {n_estimators}:")
    print(f"Train Set Metrics: F1 Score: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}")
    # print(f"Valid Set Metrics: F1 Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
    print(f"Test Set Metrics: F1 Score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}\n")