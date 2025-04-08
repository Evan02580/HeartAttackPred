import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score

def train_rf_by_cluster(X_train, y_train, cluster_labels, n_estimators):
    cluster_models = {}
    cluster_scores = {}

    for cluster in set(cluster_labels):
        idx = (cluster_labels == cluster)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=20, random_state=760)
        clf.fit(X_train[idx], y_train[idx])

        y_pred = clf.predict(X_train[idx])
        cluster_models[cluster] = clf
        cluster_scores[cluster] = {
            'F1': f1_score(y_train[idx], y_pred),
            'Accuracy': accuracy_score(y_train[idx], y_pred),
            'AUC': roc_auc_score(y_train[idx], clf.predict_proba(X_train[idx])[:,1]),
            'Balanced Accuracy': balanced_accuracy_score(y_train[idx], y_pred)

        }

    return cluster_models, cluster_scores

def evaluate_rf_by_cluster(models, X_data, y_data, cluster_labels):
    metrics = {'F1': [], 'Accuracy': [], 'AUC': [], 'Balanced Accuracy': []}
    weights = []
    for cluster in set(cluster_labels):
        idx = (cluster_labels == cluster)
        model = models[cluster]

        y_true = y_data[idx]
        y_pred = model.predict(X_data[idx])
        y_prob = model.predict_proba(X_data[idx])[:, 1]

        metrics['F1'].append(f1_score(y_true, y_pred))
        metrics['Accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['AUC'].append(roc_auc_score(y_true, y_prob))
        metrics['Balanced Accuracy'].append(balanced_accuracy_score(y_true, y_pred))

        weights.append(len(y_true))

    # Normalize the metrics by the number of samples in each cluster
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    for m in metrics:
        metrics[m] = np.average(metrics[m], weights=weights)

    return metrics
