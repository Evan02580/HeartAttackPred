import pandas as pd
from sklearn.preprocessing import StandardScaler

def read_data(filepath, label_col='Heart Attack Risk (Binary)'):
    df = pd.read_csv(filepath)
    features = df.drop(columns=[label_col, "Heart Attack Risk (Text)"])
    labels = df[label_col]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, labels
