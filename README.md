# 🔧 Predictive Maintenance Failure Detection Project 🚀

## 🌟 Project Overview

This project focuses on **Predictive Maintenance** for industrial machinery using **Machine Learning** 🧠. The goal is to **predict machine failures** (Target) and classify **failure types** to assist in **preventive maintenance planning**, reduce downtime ⏱️, and save operational costs 💰.

The dataset contains machine operational metrics such as **temperature, torque, rotational speed, and tool wear**, along with failure indicators. This project demonstrates a **full ML pipeline** from data cleaning to model training and evaluation.

---

## 📊 Dataset Description

| Feature                   | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `UDI`                     | Unique identifier for each machine 🆔                |
| `Product ID`              | Machine model identifier 🏭                          |
| `Type`                    | Machine type: `M` = Medium, `H` = High, `L` = Low ⚙️ |
| `Air temperature [K]`     | Ambient temperature during operation 🌡️              |
| `Process temperature [K]` | Process-specific temperature 🛠️                      |
| `Rotational speed [rpm]`  | Machine speed in revolutions per minute 🔄           |
| `Torque [Nm]`             | Torque applied to tools or shafts 🏋️                 |
| `Tool wear [min]`         | Cumulative tool wear ⏳                              |
| `Target`                  | Binary: 0 = No Failure, 1 = Failure ❌✅             |
| `Failure Type`            | Multi-label: `HDA, NF, OF, PF, RF, TWF` 🚨           |

---

## 🧹 Data Preprocessing

### 1️⃣ Handling Outliers

- Used **Interquartile Range (IQR)** method to detect and remove outliers 🕵️‍♂️.
- Outliers can bias model training, so we carefully cleaned the dataset.

### 2️⃣ Feature Scaling

- Applied **RobustScaler** to reduce the influence of remaining outliers.
- For Neural Networks, applied **MinMaxScaler** to normalize values between 0️⃣–1️⃣.

### 3️⃣ Encoding Categorical Features

- **Machine Type**: Ordinal encoding (`M=2, H=1, L=0`) 🏭 → 🔢
- **Failure Type**: One-hot encoding (`HDA, NF, OF, PF, RF, TWF`) 🧩 → avoids ordinal assumptions.

### 4️⃣ Handling Imbalanced Classes

- Failure types are extremely imbalanced ⚖️ (e.g., `Random Failures` are very rare).
- Used **SMOTE** & **RandomOverSampler** to boost minority failure classes to **~800–1500 samples** without altering non-failure samples.
- Ensured **Target column remains consistent** with oversampled failure rows.

---

## 🔍 Exploratory Data Analysis (EDA)

### 1️⃣ Numerical Features

- Boxplots 📦 & strip plots ✨ were created for numerical features against `Failure Type`.
- Checked skewness and applied **log transformation** to highly skewed features like `Rotational speed [rpm]`.

### 2️⃣ Categorical Features

- Pie charts 🥧 and count plots 📈 show distribution of failures for each type.
- Correlation heatmaps 🔥 reveal relationships between failure types and operational metrics.

### 3️⃣ Visualizations

- Tool wear distribution by **failure vs non-failure**.
- Feature comparison across **failure types** for deeper insights.

---

## 🤖 Machine Learning Approach

### 1️⃣ Problem Formulation

- **Target Prediction**: Binary classification (0 = No Failure, 1 = Failure)
- **Failure Type Prediction**: Multi-label classification for `HDA, OF, PF, RF, TWF, NF`

### 2️⃣ Models Used

- **Tree-based models** 🌳: Random Forest, XGBoost
- **Neural Networks** 🧠: Feedforward fully connected networks

### 3️⃣ Evaluation Metrics

- Accuracy ✅
- F1-score 🎯
- Confusion matrix 🔳

### 4️⃣ Data Split

- Training: 80% 🏋️‍♂️
- Validation: 10% 🔍
- Test: 10% 🧪

---

## 📈 Key Insights

1. **Tool wear** and **rotational speed** are the most predictive features ⚡.
2. Rare failure types require oversampling to prevent model bias 🏋️‍♂️.
3. **Balancing dataset** significantly improves prediction on minority classes 📊.
4. One-hot encoding of failure types enhances Neural Network performance 🧩.

---

## 🛠️ Dependencies

- Python 3.10+
- Pandas 🐼
- Numpy 🔢
- Matplotlib 📊
- Seaborn 🎨
- Scikit-learn 🏫
- Imbalanced-learn ⚖️

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```
