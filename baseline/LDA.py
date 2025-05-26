from loadData import read_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
import numpy as np

# Read the data using your custom read_data function
file_num = 3

file_path = "../datasets/"
file_name = ["heart", "heart-1", "UCI-1190-11", "statlog_heart"][file_num]
label_col = ["HeartDisease", "target", "target", "risk"][file_num]

X_train, y_train, X_test, y_test = read_data(f"{file_path}{file_name}.csv", label_col=label_col)

# Train the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Tune the decision threshold on the validation set based on F1 score
# proba_val = lda.predict_proba(X_val)[:, 1]
# thresholds = np.linspace(0, 1, 101)
# best_threshold = 0.5
# best_f1 = 0.0
# for thr in thresholds:
#     y_val_pred = (proba_val >= thr).astype(int)
#     current_f1 = f1_score(y_val, y_val_pred)
#     if current_f1 > best_f1:
#         best_f1 = current_f1
#         best_threshold = thr

# print("Best threshold found on validation set:", best_threshold)

# Define an evaluation function that outputs metrics in dictionary format
def evaluate_model(model, X_data, y_data, threshold):
    # Get predicted probabilities and apply the tuned threshold to obtain predictions
    y_prob = model.predict_proba(X_data)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        'F1': f1_score(y_data, y_pred),
        'Accuracy': accuracy_score(y_data, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_data, y_pred),
        'AUC': roc_auc_score(y_data, y_prob)
    }
    # 保留四位小数
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics

# Evaluate the model on the validation set
# val_metrics = evaluate_model(lda, X_val, y_val, best_thresholed)
# print("\nValidation Set Metrics:", val_metrics)

# Evaluate the model on the test set
test_metrics = evaluate_model(lda, X_test, y_test, 0.7)
print("\nTest Set Metrics:", test_metrics)

# 写入csv(加入到最后一行)
import pandas as pd
def write_metrics_to_csv(metrics, filename=file_name, model_name="LDA"):
    filename = f"../results/{filename}.csv"
    df = pd.DataFrame([metrics])
    # 最前面加一列
    df.insert(0, 'Dataset', model_name)
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)

# write_metrics_to_csv(test_metrics)