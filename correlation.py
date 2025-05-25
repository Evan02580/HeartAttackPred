import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file_path = "./datasets/statlog_heart.csv"
df = pd.read_csv(file_path).dropna()

# 归一化
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# print(df.head())
# # 2. 血压拆分为高低值
# df[['BP_High', 'BP_Low']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
#
# # 3. 删除不需要的列
# df.drop(columns=['Patient ID', 'Country', 'Blood Pressure'], inplace=True)

# 4. 类别列标签编码
from sklearn.preprocessing import LabelEncoder
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

# 计算相关性矩阵
corr_matrix = df.corr(method='pearson')  # 默认是 Pearson，也可以改为 'spearman' 或 'kendall'

# 可视化热力图
plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": .75})
plt.title("Correlation Matrix of All Variables")
plt.tight_layout()
plt.show()
