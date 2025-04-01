from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def find_optimal_k(data, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(data)
        score = silhouette_score(data, model.labels_)
        scores.append((k, score))
    return scores

def apply_clustering(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return labels
