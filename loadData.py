import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def read_data(filepath, label_col='Heart Attack Risk (Binary)'):
    df = pd.read_csv(filepath)
    df = df.dropna()
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})


    features = df.drop(columns=[label_col, "Heart Attack Risk (Text)"])
    labels = df[label_col]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 7:1:2 split
    X_train, X_temp, y_train, y_temp = train_test_split(scaled_features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test