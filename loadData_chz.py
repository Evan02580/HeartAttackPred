# loadData.py
"""
数据读取 / 预处理工具
-------------------
核心函数
---------
read_data(filepath, label_col)
    ▶ 读入 CSV → 标签编码 → 标准化
    ▶ 返回 (X_train, y_train, X_test, y_test, scaler, feature_names)

其余 read_data_all / read_data_general 保留，方便其他脚本复用。
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# 主函数：分类任务常用（7:3 划分）
# -----------------------------------------------------------
def read_data(filepath, label_col="HeartDisease"):
    df = pd.read_csv(filepath).dropna()

    # 类别特征编码
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col])

    feature_names = df.drop(columns=[label_col]).columns.tolist()
    features = df[feature_names]
    labels   = df[label_col]

    # 标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 8:2 训练 / 测试
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_features, labels,
        test_size=0.2, random_state=42, stratify=labels
    )

    # ⚠️ 新增：返回 scaler & feature_names，供 SHAP / LIME 使用
    return X_train, y_train, X_test, y_test, scaler, feature_names


# ===========================================================
#  下面两个函数和之前一致，仅稍微整理了格式
# ===========================================================

def read_data_all(filepath, label_col="Heart Attack Risk (Binary)", k=10):
    df = pd.read_csv(filepath).dropna()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col])

    features = df.drop(columns=[label_col, "RestingECG"])  # 可按需调整
    labels   = df[label_col]

    scaled = StandardScaler().fit_transform(features)
    return scaled, labels


def read_data_general(filepath, label_col, k=10):
    df = pd.read_csv(filepath).dropna()
    exclude = {label_col, "Heart Attack Risk (Text)"}

    str_cols = [c for c in df.select_dtypes("object").columns if c not in exclude]
    if str_cols:
        df = pd.get_dummies(df, columns=str_cols, drop_first=True)

    features = df.drop(columns=[label_col, "Heart Attack Risk (Text)"], errors="ignore")
    labels   = df[label_col]
    scaled   = StandardScaler().fit_transform(features)
    return scaled, labels
