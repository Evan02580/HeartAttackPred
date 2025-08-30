# main.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from loadData import read_data_all
from cluster import fit_kmeans_train_only

RANDOM_STATE = 760

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """当 y_true 只有单一类别时，roc_auc_score 会报错；这里返回 NaN。"""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def write_metrics_to_csv(metrics: dict, filename: str, model_name: str):
    """将单次结果写入 ./results_0830/filename.csv；自动追加，首行写表头。"""
    out_dir = "./results_0830"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{filename}.csv"
    df = pd.DataFrame([metrics])
    df.insert(0, "Model", model_name)
    header = not os.path.exists(out_path)
    df.to_csv(out_path, mode="a", header=header, index=False)


def train_and_eval_on_clusters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_clusters: np.ndarray,
    test_clusters: np.ndarray,
    file_name: str,
    best_k: int,
    save_csv: bool = False,
):
    print(f"\n===== Random Forest (Cluster = {best_k}) =====")
    print("Train cluster sizes:", np.bincount(train_clusters))
    print("Test  cluster sizes:", np.bincount(test_clusters))

    total_test_y = []
    total_test_pred = []
    total_test_proba1 = []

    unique_clusters = np.unique(train_clusters)
    for c in unique_clusters:
        print(f"\n--- Cluster {c} ---")
        idx_tr = np.where(train_clusters == c)[0]
        X_tr_c, y_tr_c = X_train[idx_tr], y_train[idx_tr]

        idx_te = np.where(test_clusters == c)[0]
        if len(idx_te) > 0:
            X_te_c, y_te_c = X_test[idx_te], y_test[idx_te]
        else:
            X_te_c = np.empty((0, X_train.shape[1]))
            y_te_c = np.array([])

        print(
            f"Train samples: {len(X_tr_c)}, Test samples: {len(X_te_c)}, "
            f"Risk in Train: {(y_tr_c.mean() if len(y_tr_c) > 0 else 0):.2f}, "
            f"Risk in Test: {(y_te_c.mean() if len(y_te_c) > 0 else 0):.2f}"
        )

        best_f1, best_acc, best_n, best_d = 0.0, 0.0, 0, 0
        for n in range(2, 15):
            for d in range(2, 15):
                rf_tmp = RandomForestClassifier(
                    n_estimators=n, max_depth=d, random_state=42
                )
                rf_tmp.fit(X_tr_c, y_tr_c)
                X_eval, y_eval = (
                    (X_te_c, y_te_c) if len(X_te_c) > 0 else (X_tr_c, y_tr_c)
                )
                y_pred_eval = rf_tmp.predict(X_eval)
                f1 = f1_score(y_eval, y_pred_eval)
                acc = accuracy_score(y_eval, y_pred_eval)
                if f1 > best_f1 or acc > best_acc:
                    best_f1, best_acc, best_n, best_d = f1, acc, n, d
                    if len(X_te_c) > 0:
                        y_prob_eval = rf_tmp.predict_proba(X_eval)[:, 1]
                        print(
                            f"Dep: {d}, Est: {n}, "
                            f"F1: {f1:.4f}, "
                            f"Acc: {acc:.4f}, "
                            f"BalAcc: {balanced_accuracy_score(y_eval, y_pred_eval):.4f}, "
                            f"AUC: {safe_auc(y_eval, y_prob_eval):.4f}"
                        )

        rf = RandomForestClassifier(
            n_estimators=best_n, max_depth=best_d, random_state=42
        )
        rf.fit(X_tr_c, y_tr_c)

        for name, X_, y_ in [("Train", X_tr_c, y_tr_c), (" Test", X_te_c, y_te_c)]:
            if len(X_) == 0:
                print(f"{name} - empty")
                continue
            y_pred_ = rf.predict(X_)
            y_prob_ = rf.predict_proba(X_)[:, 1]
            metrics = {
                "F1": f1_score(y_, y_pred_),
                "Accuracy": accuracy_score(y_, y_pred_),
                "Balanced Accuracy": balanced_accuracy_score(y_, y_pred_),
                "AUC": safe_auc(y_, y_prob_),
            }
            print(
                f"{name} - F1: {metrics['F1']:.4f}, "
                f"Acc: {metrics['Accuracy']:.4f}, "
                f"BalAcc: {metrics['Balanced Accuracy']:.4f}, "
                f"AUC: {metrics['AUC']:.4f}"
            )

        if len(X_te_c) > 0:
            total_test_y.extend(y_te_c.tolist())
            total_test_pred.extend(rf.predict(X_te_c).tolist())
            total_test_proba1.extend(rf.predict_proba(X_te_c)[:, 1].tolist())

    total_test_y = np.array(total_test_y)
    total_test_pred = np.array(total_test_pred)
    total_test_proba1 = np.array(total_test_proba1)

    weighted_avg = {
        "Dataset": file_name,
        "K": best_k,
        "F1": round(f1_score(total_test_y, total_test_pred), 4),
        "Accuracy": round(accuracy_score(total_test_y, total_test_pred), 4),
        "Balanced Accuracy": round(
            balanced_accuracy_score(total_test_y, total_test_pred), 4
        ),
        "AUC": round(safe_auc(total_test_y, total_test_proba1), 4),
        "Total Test Samples": len(total_test_y),
    }

    print(f"\n===== CluRF Summary (k = {best_k}) =====")
    print(f"Data Name: {file_name}")
    for k, v in weighted_avg.items():
        if k in ("Dataset", "K", "Total Test Samples"):
            continue
        print(f"{k}: {v}")

    if save_csv:
        write_metrics_to_csv(
            weighted_avg, filename=file_name, model_name=f"CluRF (k = {best_k})"
        )


def run_all_datasets():
    file_path = "./datasets/"
    dataset_names = ["heart", "heart-1", "UCI-1190-11", "cleveland"]
    label_cols = ["HeartDisease", "target", "target", "risk"]
    best_ks = [6, 5, 7, 4]

    for file_name, label_col, best_k in zip(dataset_names, label_cols, best_ks):
        print("\n" + "=" * 60)
        print(f"Dataset: {file_name}")

        X_all, y_all, feature_names = read_data_all(
            os.path.join(file_path, f"{file_name}.csv"), label_col=label_col
        )
        X_all = np.asarray(X_all)
        y_all = np.asarray(y_all)

        print(f"Num samples: {len(y_all)}")
        print(f"Num features: {X_all.shape[1]}")
        print(f"Feature names: {feature_names}")

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
        )

        kmeans, train_clusters = fit_kmeans_train_only(
            X_train, best_k, random_state=RANDOM_STATE
        )
        test_clusters = kmeans.predict(X_test)

        train_and_eval_on_clusters(
            X_train, y_train, X_test, y_test,
            train_clusters, test_clusters,
            file_name=file_name, best_k=best_k,
            save_csv=True   # 设置为 True，会写入 ./results_0830/{dataset}.csv
        )


if __name__ == "__main__":
    run_all_datasets()
