# explain_rf_shap.py
import shap, os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

def explain_rf_by_cluster(models: dict,
                          X_data: np.ndarray,
                          cluster_labels: np.ndarray,
                          feature_names,
                          save_dir="shap_plots",
                          samples_per_cluster=3):
    Path(save_dir).mkdir(exist_ok=True)

    for cid, model in tqdm(models.items(), desc="[SHAP] cluster"):
        idx = cluster_labels == cid
        if idx.sum() == 0:
            continue

        X_cluster = X_data[idx]
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_cluster)

        # -------- 1. summary plot ----------
        png_path = f"{save_dir}/cluster{cid}_summary.png"
        shap.summary_plot(shap_vals[1], X_cluster,
                          feature_names=feature_names, show=False)
        plt.title(f"Cluster {cid} – Feature Importance")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ saved {png_path}")

        # -------- 2. local force plots -------
        for i in range(min(samples_per_cluster, X_cluster.shape[0])):
            html_path = f"{save_dir}/cluster{cid}_sample{i}.html"
            force_fig = shap.force_plot(
                explainer.expected_value[1],
                shap_vals[1][i],
                X_cluster[i],
                feature_names=feature_names,
                matplotlib=False,
                show=False
            )
            shap.save_html(html_path, force_fig)   # ← 新写法
            print(f"  ↳ saved {html_path}")


