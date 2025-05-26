# Heart Attack Prediction

The project was developed for the COMPSCI 760 course

## Group Members

Bofan Wang, Yihong Pan, Yifan Zhou, Huazhe Cheng, Zhipeng Lin

## Dataset Overview

The datasets we use can be available on open resources like Kaggle, IEEE dataset. The details of the dataset can be found in [Dataset README](./datasets/README.md).

The datasets consist of multiple values across a range of features, including:

- **People**: Age, Sex, Exercise...
- **Health Metrics**: Chest Pain Type, Resting Blood Pressure, Cholesterol, Max Heart Rate...
- **Target Variable**: HeartDisease — 0 (No risk), 1 (High risk)...

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

|                 Model                  |  F1 Score  |  Accuracy  | Balanced Accuracy |  AUC   | 
|:--------------------------------------:|:----------:|:----------:|:-----------------:|:------:|
|         Random Forest (n = 20)         |   0.8670   |   0.8533   |      0.8593       | 0.9215 |
|         Random Forest (n = 30)         |   0.8670   |   0.8533   |      0.8593       | 0.9325 |
|          Logistic Regression           |   0.7741   |   0.7717   |      0.7910       | 0.9008 |
|                  LDA                   |   0.7831   |   0.7772   |      0.7938       | 0.9006 |
|            SVM (kernel=rbf)            |   0.8815   |   0.8641   |      0.8632       | 0.9484 |
|           SVM (kernel=poly)            |   0.8442   |   0.8315   |      0.8406       | 0.8966 |
| Neural Network (hidden_layers=(32,16)) |   0.8670   |   0.8533   |      0.8593       | 0.8859 |
| Neural Network (hidden_layers=(64,32)) |   0.8670   |   0.8533   |      0.8593       | 0.9283 |
|        **CluRF (k = 4) (ours)**        |  *0.9073*  |  *0.8973*  |     *0.8966*      | 0.8966 |
|        **CluRF (k = 6) (ours)**        | **0.9223** | **0.9140** |    **0.9140**     | 0.9140 |

