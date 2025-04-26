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


file_path = "./datasets/heart-attack-risk-prediction-dataset.csv"
data = pd.read_csv(file_path).dropna()

column_names = ["Heart Attack Risk (Binary)", "Gender"]

for column in column_names:
    plot_distribution(data, column)


data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})

# Calculate number of male with heart attack and female with heart attack
gender_heart_attack_counts = data.groupby(['Gender', 'Heart Attack Risk (Binary)']).size().unstack(fill_value=0)

gender_heart_attack_counts.index = ['Female', 'Male']
gender_heart_attack_counts.columns = ['No Heart Attack', 'Heart Attack']

print(gender_heart_attack_counts)

gender_heart_attack_counts.plot(kind='bar', stacked=True)
plt.title("Heart Attack Risk by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of People")
plt.legend(title="Heart Attack Risk")
plt.tight_layout()
plt.show()


