from loadData import read_data
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


# 1. Loading Data
file_path = "../datasets/heart-attack-risk-prediction-dataset.csv"
X_train, y_train, X_val, y_val, X_test, y_test = read_data(file_path)

#2. Build SVM drictly on data set
svm_model = SVC(probability=True, random_state=760)
svm_model.fit(X_train, y_train)

#3. Predict on each set to see the result
y_train_pred = svm_model.predict(X_train)
y_val_pred = svm_model.predict(X_val)
y_test_pred = svm_model.predict(X_test)

train_f1 = f1_score(y_train, y_train_pred)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, svm_model.predict_proba(X_train)[:, 1])

val_f1 = f1_score(y_val, y_val_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_auc = roc_auc_score(y_val, svm_model.predict_proba(X_val)[:, 1])

test_f1 = f1_score(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])

print(f"\nTrain Set Metrics: F1 Score: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}, AUC: {train_auc:.4f}")
print(f"\nTrain Set Metrics: F1 Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
print(f"\nTrain Set Metrics: F1 Score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}")

