import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distribution(df, column_name):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=column_name, data=df)
    plt.title(f"Distribution of {column_name}")
    plt.xlabel(f"{column_name}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


file_path = "./datasets/heart.csv"
# file_path = "./datasets/heart-attack-risk-prediction-dataset.csv"
data = pd.read_csv(file_path).dropna()

column_names = ["HeartDisease", "Sex", "Age"]

for column in column_names:
    plot_distribution(data, column)


data['Sex'] = data['Sex'].map({'F': 0, 'M': 1})

# Calculate number of male with heart attack and female with heart attack
gender_heart_attack_counts = data.groupby(['Sex', 'HeartDisease']).size().unstack(fill_value=0)

gender_heart_attack_counts.index = ['F', 'M']
gender_heart_attack_counts.columns = ['No Heart Attack', 'Heart Attack']

print(gender_heart_attack_counts)

gender_heart_attack_counts.plot(kind='bar', stacked=True)
plt.title("Heart Attack Risk by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of People")
plt.legend(title="Heart Attack Risk")
plt.tight_layout()
plt.show()


# file_path = "./datasets/heart.csv"
# df = pd.read_csv(file_path).dropna()
# # df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
#
# sns.set(style="whitegrid")
#
# # 获取除 label 外的所有特征列
# features = df.columns.difference(['HeartDisease'])
#
# # 设置子图布局
# n_cols = 3
# n_rows = -(-len(features) // n_cols)  # 向上取整
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
#
# # 遍历每个特征画图
# for i, feature in enumerate(features):
#     row, col = divmod(i, n_cols)
#     ax = axes[row, col] if n_rows > 1 else axes[col]
#
#     sns.boxplot(
#         x='HeartDisease',
#         y=feature,
#         data=df,
#         ax=ax,
#         palette='Set2'
#     )
#     ax.set_title(f'{feature} vs HeartDisease')
#
# # 删除多余子图
# for j in range(len(features), n_rows * n_cols):
#     fig.delaxes(axes.flat[j])
#
# plt.tight_layout()
# plt.show()
