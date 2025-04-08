# Heart Attack Prediction

The project was developed for the COMPSCI 760 course

## Group Members

Bofan Wang, Yihong Pan, Yifan Zhou, Huazhe Cheng, Zhipeng Lin

## Dataset Overview

The open source dataset can be available on kaggle: [Heart Attack Risk Prediction](https://www.kaggle.com/datasets/alikalwar/heart-attack-risk-prediction-cleaned-dataset)

The dataset consists of normalized numerical values across a range of features, including:

- **People**: Age, Gender, Income...
- **Health Metrics**: Cholesterol, BMI, Blood Pressure, CK-MB, Troponin, Blood Sugar...
- **Lifestyle Indicators**: Smoking, Alcohol, Exercise, Diet, Sedentary Time, Sleep...
- **Medical History**: Diabetes, Family History, Medication Use, Previous Heart Problems...
- **Target Variable**: Heart Attack Risk (Binary) — 0 (No risk), 1 (High risk)...

## Methodologies

### 1. Cluster + RF

Use **cluster** methods (Kmeans, DBSCAN...) to form groups of people. For each group, build their own **Random Forest**.

![graphics-CluRF](./images/graphics-CluRF.png)

### 2. Cluster + (Feature Selection) + SVMs

Apply **cluster** methods. For each cluster, apply **feature selection** methods (e.g. Mutual Information, RFE). Train **SVM classifiers** for each cluster

![graphics-CluSVM](./images/graphics-CluSVM.png)

### 3. Neural Network    

Train a **Multilayer Perceptron (MLP)** with optimal layers and nodes. Predict risk probability directly with NN.

![graphics-NN](./images/graphics-NN.png)


## Experiments

The following table summarizes the performance of different models on the test set. The metrics used for evaluation include F1 score, accuracy, AUC.

|                 Model                  | F1 Score  | Accuracy  |    AUC    |
|:--------------------------------------:|:---------:|:---------:|:---------:|
|         Random Forest (n = 5)          |  *0.350*  |   0.592   |   0.542   |
|         Random Forest (n = 10)         |   0.249   |   0.632   |   0.555   |
|         Random Forest (n = 20)         |   0.242   | **0.653** |   0.559   |
|          Logistic Regression           |   0.522   |   0.355   |   0.501   |
|                  LDA                   | **0.523** |   0.356   |   0.502   |
|            SVM (kernel=rbf)            |   0.015   |   0.648   |   0.540   |
|           SVM (kernel=poly)            |   0.120   |   0.641   |   0.518   |
| Neural Network (hidden_layers=(16,0))  |   0.089   |   0.641   |   0.518   |
| Neural Network (hidden_layers=(32,16)) |   0.043   |   0.648   |   0.490   |
| Neural Network (hidden_layers=(64,32)) |   0.021   |   0.645   |   0.519   |
|       **CluRF (n = 10) (ours)**        |   0.303   |  *0.644*  | **0.608** |
|       **CluRF (n = 15) (ours)**        |   0.348   |   0.642   |  *0.594*  |

