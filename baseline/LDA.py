from loadData import read_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
import numpy as np

# Read the data using your custom read_data function
file_path = "../datasets/heart-attack-risk-prediction-dataset.csv"
X_train, y_train, X_val, y_val, X_test, y_test = read_data(file_path)

# Train the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Tune the decision threshold on the validation set based on F1 score
proba_val = lda.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0, 1, 101)
best_threshold = 0.5
best_f1 = 0.0
for thr in thresholds:
    y_val_pred = (proba_val >= thr).astype(int)
    current_f1 = f1_score(y_val, y_val_pred)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thr

print("Best threshold found on validation set:", best_threshold)

# Define an evaluation function that outputs metrics in dictionary format
def evaluate_model(model, X_data, y_data, threshold):
    # Get predicted probabilities and apply the tuned threshold to obtain predictions
    y_prob = model.predict_proba(X_data)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        'F1': f1_score(y_data, y_pred),
        'Accuracy': accuracy_score(y_data, y_pred),
        'AUC': roc_auc_score(y_data, y_prob),
        'Balanced Accuracy': balanced_accuracy_score(y_data, y_pred)
    }
    return metrics

# Evaluate the model on the validation set
val_metrics = evaluate_model(lda, X_val, y_val, best_threshold)
print("\nValidation Set Metrics:", val_metrics)

# Evaluate the model on the test set
test_metrics = evaluate_model(lda, X_test, y_test, best_threshold)
print("\nTest Set Metrics:", test_metrics)
