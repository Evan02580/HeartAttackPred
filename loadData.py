import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#For feature selection
from sklearn.feature_selection import RFE, SelectKBest, f_classif, RFECV
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectFromModel

# def General_Feature_selector(method='RFE', n_features=10, estimator=None, score_func=None):
#     if method == 'RFE':
#         if estimator is None:
#             estimator = LogisticRegression(max_iter=1000)
#         selector = RFE(estimator, n_features_to_select=n_features)
#     elif method == 'RFECV':
#         if estimator is None:
#             estimator = LogisticRegression(max_iter=1000)
#         selector = RFECV(estimator, step=1, cv=5, scoring='accuracy', min_features_to_select=n_features)
#     elif method == 'SelectKBest':
#         if score_func is None:
#             score_func = f_classif
#         selector = SelectKBest(score_func=score_func, k=n_features)
#     elif method == 'Lasso':
#         if estimator is None:
#             estimator = LassoCV(cv=5, random_state=42)
#         selector = SelectFromModel(estimator)
#     else:
#         raise ValueError("Please enter: 'RFE', 'RFECV', 'SelectKBest' or 'Lasso'")
#     return selector



def read_data(filepath, label_col='Heart Attack Risk (Binary)',FS_method='RFECV',n_features=8):
    df = pd.read_csv(filepath)
    df = df.dropna()
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

    features = df.drop(columns=[label_col, "Heart Attack Risk (Text)"])
    labels = df[label_col]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 7:1:2 split
    X_train, X_temp, y_train, y_temp = train_test_split(scaled_features, labels, test_size=0.5, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)

    # Feature_selector=General_Feature_selector(method=FS_method,n_features=n_features)
    # X_train = Feature_selector.fit_transform(X_train, y_train)
    # X_val = Feature_selector.transform(X_val)
    # X_test = Feature_selector.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


"""
# load all data in one variable
def read_data_all(filepath, label_col='Heart Attack Risk (Binary)'):

    df = pd.read_csv(filepath).dropna()
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    labels = df[label_col]
    features = df.drop(columns=[label_col, "Heart Attack Risk (Text)"])
    features = StandardScaler().fit_transform(features)
    return features, labels
"""

# 读取数据并且进行特征选择 先选择特征然后再用子数据进行聚类
def read_data_all(filepath, label_col='Heart Attack Risk (Binary)', k=10):
    df = pd.read_csv(filepath).dropna()
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

    labels = df[label_col]
    features = df.drop(columns=[label_col, "Heart Attack Risk (Text)"])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # ✅ 添加特征选择 note：需要修改
    # from sklearn.feature_selection import SelectKBest, f_classif
    # selector = SelectKBest(score_func=f_classif, k=k) #进行特征选择
    # selected = selector.fit_transform(scaled, labels)

    return scaled, labels
