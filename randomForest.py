from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def train_rf_by_cluster(X_train, y_train, cluster_labels):
    cluster_models = {}
    cluster_scores = {}

    for cluster in set(cluster_labels):
        idx = (cluster_labels == cluster)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train[idx], y_train[idx])

        y_pred = clf.predict(X_train[idx])
        cluster_models[cluster] = clf
        cluster_scores[cluster] = {
            'F1': f1_score(y_train[idx], y_pred),
            'Accuracy': accuracy_score(y_train[idx], y_pred),
            'AUC': roc_auc_score(y_train[idx], clf.predict_proba(X_train[idx])[:,1])
        }

    return cluster_models, cluster_scores

def evaluate_rf_by_cluster(models, X_data, y_data, cluster_labels):
    metrics = {'F1': [], 'Accuracy': [], 'AUC': []}
    for cluster in set(cluster_labels):
        idx = (cluster_labels == cluster)
        model = models[cluster]

        y_true = y_data[idx]
        y_pred = model.predict(X_data[idx])
        y_prob = model.predict_proba(X_data[idx])[:, 1]

        metrics['F1'].append(f1_score(y_true, y_pred))
        metrics['Accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['AUC'].append(roc_auc_score(y_true, y_prob))

    return {m: sum(scores)/len(scores) for m, scores in metrics.items()}
