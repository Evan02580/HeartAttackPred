import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = "./datasets/heart-attack-risk-prediction-dataset.csv"
df = pd.read_csv(file_path).dropna()
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# 计算相关性矩阵
corr_matrix = df.corr(method='pearson')  # 默认是 Pearson，也可以改为 'spearman' 或 'kendall'

# 可视化热力图
plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .75})
plt.title("Correlation Matrix of All Variables")
plt.tight_layout()
plt.show()
