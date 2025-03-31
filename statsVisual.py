import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "./datasets/heart-attack-risk-prediction-dataset.csv"
df = pd.read_csv(file_path)

column_name = "Heart Attack Risk (Binary)"

plt.figure(figsize=(6, 4))
sns.countplot(x=column_name, data=df)
plt.title("Distribution of Heart Attack Risk (Binary)")
plt.xlabel("Heart Attack Risk (Binary)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
