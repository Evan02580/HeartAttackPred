# cluster_profile.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cluster_profiles(X, cluster_labels, feature_names,
                          topk=8, save_dir="cluster_profiles"):
    """
    为每个 cluster 生成：
      • cluster<i>_radar.png  —— Top-k 特征均值雷达图
      • cluster<i>_means.csv  —— 所有特征均值
    """
    Path(save_dir).mkdir(exist_ok=True)
    df = pd.DataFrame(X, columns=feature_names)
    df["cluster"] = cluster_labels
    overall_mean = df[feature_names].mean()

    for cid in np.unique(cluster_labels):
        sub = df[df.cluster == cid][feature_names]
        diff = (sub.mean() - overall_mean).abs().sort_values(ascending=False)
        key_feats = diff.head(topk).index.tolist()

        # -------- 角度 ----------
        angles_label = np.linspace(0, 2*np.pi, len(key_feats), endpoint=False)
        angles_plot  = np.concatenate((angles_label, [angles_label[0]]))

        # -------- 数据 ----------
        stats = sub[key_feats].mean().values
        stats = np.concatenate((stats, [stats[0]]))

        # -------- 画图 ----------
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)

        ax.plot(angles_plot, stats, 'o-', linewidth=2)
        ax.fill(angles_plot, stats, alpha=0.25)
        ax.set_thetagrids(angles_label * 180/np.pi, key_feats)
        ax.set_title(f"Cluster {cid} – Top {topk} Feature Means")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cluster{cid}_radar.png", dpi=300)
        plt.close()

        # -------- CSV ----------
        sub.mean().to_csv(f"{save_dir}/cluster{cid}_means.csv")
        print(f"[Profile] cluster{cid} done.")
        