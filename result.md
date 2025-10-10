# ML and DL Pipeline Results and Comparison of Best Models

## 01_Data_Loading_and_Mapping

🚀 **Starting Complete ML Pipeline**

📊 **Loading and mapping data**

- Data shape: (10000, 8)
- Using `Failure_Type_Encoded` as target

**Original target distribution:**

| Failure Type             | Count | %     |
| ------------------------ | ----- | ----- |
| No Failure               | 9652  | 96.5% |
| Heat Dissipation Failure | 112   | 1.1%  |
| Overstrain Failure       | 78    | 0.8%  |
| Power Failure            | 95    | 0.9%  |
| Random Failure           | 18    | 0.2%  |
| Tool Wear Failure        | 45    | 0.4%  |

---

## 02_Scaling_and_Preprocessing

⚖️ **Scaling numerical features before split**

- Features scaled: `['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']`
- Scaling saved as `scaled.pkl`

---

## 03_Train_Test_Split

🔄 **Splitting data**

- Training set shape: (8000, 6)
- Test set shape: (2000, 6)

---

## 04_Before_Balancing_Distribution

📊 **Training class distribution BEFORE balancing:**

| Failure Type             | Count | %      |
| ------------------------ | ----- | ------ |
| No Failure               | 7,722 | 96.53% |
| Heat Dissipation Failure | 90    | 1.12%  |
| Overstrain Failure       | 62    | 0.78%  |
| Power Failure            | 76    | 0.95%  |
| Random Failure           | 14    | 0.18%  |
| Tool Wear Failure        | 36    | 0.45%  |

---

## 05_SMOTE_Balancing

🧬 **Applying SMOTE to balance failure types**

- Target samples per class: 2,500

**SMOTE sampling:**

| Failure Type             | Before | After |
| ------------------------ | ------ | ----- |
| Heat Dissipation Failure | 90     | 2,500 |
| Overstrain Failure       | 62     | 2,500 |
| Power Failure            | 76     | 2,500 |
| Random Failure           | 14     | 2,500 |
| Tool Wear Failure        | 36     | 2,500 |

---

## 06_After_Balancing_Distribution

📊 **Training class distribution AFTER SMOTE balancing:**

| Failure Type             | Count | %      |
| ------------------------ | ----- | ------ |
| No Failure               | 7,722 | 38.19% |
| Heat Dissipation Failure | 2,500 | 12.36% |
| Overstrain Failure       | 2,500 | 12.36% |
| Power Failure            | 2,500 | 12.36% |
| Random Failure           | 2,500 | 12.36% |
| Tool Wear Failure        | 2,500 | 12.36% |

---

## 07_Class_Weights

💪 **Updated class weights after balancing:**

| Failure Type             | Weight | Count |
| ------------------------ | ------ | ----- |
| Heat Dissipation Failure | 1.348  | 2,500 |
| Overstrain Failure       | 1.348  | 2,500 |
| Power Failure            | 1.348  | 2,500 |
| Random Failure           | 1.348  | 2,500 |
| Tool Wear Failure        | 1.348  | 2,500 |
| No Failure               | 0.436  | 7,722 |

---

## 08_Final_Data_Summary

✅ **Final Balanced Training Data Summary**

- Original training samples: 8,000
- Balanced training samples: 20,222
- Test samples: 2,000
- Number of classes: 6
- No Failure samples preserved: 7,722
- Failure types balanced to ~2,500 each
- New imbalance ratio: 3.1:1
- Saved as `balanced_training_data.pkl`

---

## 09_Model_Training_and_Evaluation

🔬 **Evaluating all models with class weighting**

**Summary of F1 scores (Test set):**

| Model              | Test F1 |
| ------------------ | ------- |
| RandomForest       | 0.9664  |
| GradientBoosting   | 0.9538  |
| ExtraTrees         | 0.9677  |
| SVM                | 0.8367  |
| LogisticRegression | 0.6287  |
| KNN                | 0.9257  |
| NaiveBayes         | 0.8269  |
| DecisionTree       | 0.9654  |
| MLP                | 0.9408  |
| AdaBoost           | 0.8930  |
| CatBoost           | 0.9601  |
| XGBoost            | 0.9707  |

---

## 10_Top_Models

🏆 **Top 5 Models:**

1. XGBoost: 0.9707
2. ExtraTrees: 0.9677
3. RandomForest: 0.9664
4. DecisionTree: 0.9654
5. CatBoost: 0.9601

---

## 11_Hyperparameter_Tuning

🎯 **Hyperparameter tuning for top 5 models**

**XGBoost**

- Best CV F1: 0.9918
- Test F1: 0.9729
- Best params: `{'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 300, 'subsample': 0.8}`

**ExtraTrees**

- Best CV F1: 0.9804
- Test F1: 0.9571
- Best params: `{'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 300}`

**RandomForest**

- Best CV F1: 0.9779
- Test F1: 0.9590
- Best params: `{'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}`

**DecisionTree**

- Best CV F1: 0.9629
- Test F1: 0.9487
- Best params: `{'max_depth': 20, 'min_samples_split': 5}`

**CatBoost**

- Best CV F1: 0.9776
- Test F1: 0.9627
- Best params: `{'depth': 8, 'iterations': 300, 'l2_leaf_reg': 1, 'learning_rate': 0.2}`

---

## 12_Final_Model_Saved

💾 **Saving best model: XGBoost**

- `best_model_final.pkl`
- `scaler_final.pkl`
- `pipeline_metadata.pkl`

**Class weights applied:**

- No Failure: 0.436
- Heat Dissipation Failure: 1.348
- Overstrain Failure: 1.348
- Power Failure: 1.348
- Random Failure: 1.348
- Tool Wear Failure: 1.348

---

## 12 ✅ Key Results from Analysis

- **Model Performance:**
  • Overall Accuracy: 97.15%
  • Weighted F1-Score: 97.61%
  • Tool Wear Detection: 98.59% F1-score (Excellent!)

- **Class-wise Performance:**
  • No Failure: 95.24% F1-score
  • Tool Wear: 98.59% F1-score (Best performing)
  • Overstrain: 90.00% F1-score
  • Heat Dissipation: 66.67% F1-score
  • Random Failure: 6.25% F1-score (Challenging class)
  • Power Failure: 0% F1-score (Very rare in test set)

---

## 13_Overall_Summary

🎉 **Pipeline Complete!**

- Best Model: XGBoost
- F1 Score: 0.9729

# For DeepLearning Models

- 📈 Original Failure Type Distribution:
  No Failure: 9652 (96.52%)
  Heat Dissipation Failure: 112 (1.12%)
  Power Failure: 95 (0.95%)
  Overstrain Failure: 78 (0.78%)
  Tool Wear Failure: 45 (0.45%)
  Random Failures: 18 (0.18%)

## ⚖️ Applying SMOTE balancing...

- 📊 Before SMOTE:
  Class 0: 7723
  Class 5: 86
  Class 1: 77
  Class 3: 63
  Class 2: 38
  Class 4: 13

* 🎯 SMOTE sampling strategy: {np.int64(0): 7723, np.int64(1): 2500, np.int64(2): 2500, np.int64(3): 2500, np.int64(4): 2500, np.int64(5): 2500}

📊 After SMOTE:
Class 0: 7723
Class 5: 2500
Class 1: 2500
Class 3: 2500
Class 2: 2500
Class 4: 2500

- ✅ Balanced tensor shapes: X=torch.Size([20223, 8]), y=torch.Size([20223])

---

## Trainig the MOdels Pipeline

- 🧠 Training AdvancedDNN...
  📈 Using adam...
  ✅ AdvancedDNN-adam: Acc=0.9495, F1=0.9610
  📈 Using rmsprop...
  Early stopping at epoch 28
  ✅ AdvancedDNN-rmsprop: Acc=0.6880, F1=0.7933
  📈 Using sgd_momentum...
  ✅ AdvancedDNN-sgd_momentum: Acc=0.8910, F1=0.9263

- 🧠 Training TabularLSTM...
  📈 Using adam...
  Early stopping at epoch 40
  ✅ TabularLSTM-adam: Acc=0.7255, F1=0.8210
  📈 Using rmsprop...
  Early stopping at epoch 25
  ✅ TabularLSTM-rmsprop: Acc=0.9645, F1=0.9471
  📈 Using sgd_momentum...
  Early stopping at epoch 32
  ✅ TabularLSTM-sgd_momentum: Acc=0.0130, F1=0.0003

- 🧠 Training TabularTransformer...
  📈 Using adam...
  Early stopping at epoch 29
  ✅ TabularTransformer-adam: Acc=0.9645, F1=0.9471
  📈 Using rmsprop...
  Early stopping at epoch 27
  ✅ TabularTransformer-rmsprop: Acc=0.0090, F1=0.0002
  📈 Using sgd_momentum...
  Early stopping at epoch 144
  ✅ TabularTransformer-sgd_momentum: Acc=0.8510, F1=0.9028

---

## 📊 ADVANCED RESULTS SUMMARY:

```bash
Model Optimizer Accuracy F1_Score Best_Val_F1
AdvancedDNN adam 0.9495 0.961024 0.990108
AdvancedDNN rmsprop 0.6880 0.793293 0.894393
AdvancedDNN sgd_momentum 0.8910 0.926329 0.968486
TabularLSTM adam 0.7255 0.821047 0.931085
TabularLSTM rmsprop 0.9645 0.947071 0.219527
TabularLSTM sgd_momentum 0.0130 0.000334 0.266747
TabularTransformer adam 0.9645 0.947071 0.814562
TabularTransformer rmsprop 0.0090 0.000161 0.219527
TabularTransformer sgd_momentum 0.8510 0.902812 0.941084
```

## 🏆 BEST MODEL: AdvancedDNN with adam

Test Accuracy: 0.9495
Test F1-Score: 0.9610
Best Val F1: 0.9901

---

## 🚀 Model Comparison Analysis: AdvancedDNN vs XGBoost

## 1️⃣ AdvancedDNN with Adam 🤖

**Metrics:**

- Test Accuracy: 0.9495
- Test F1-Score: 0.9610
- Best Validation F1: 0.9901 🎯

**Pros 🌟:**

1. **High Capacity** – Can model complex nonlinear relationships.
2. **Flexible Architecture** – Layers, neurons, activation functions, and optimizers can be tuned.
3. **Strong on Large Datasets** – Typically outperforms trees with enough data.
4. **Potential for Transfer Learning** – Can integrate embeddings or pre-trained features.

**Cons ⚠️:**

1. **Overfitting Risk** – Validation F1 >> Test F1.
2. **Training Complexity** – Needs GPU, more epochs, and careful tuning.
3. **Interpretability** – Hard to explain predictions.
4. **Inference Time** – Slower than tree-based models for large architectures.

**Summary ✨:** Performs well but shows signs of overfitting. Needs regularization or more data for reliable generalization.

---

## 2️⃣ XGBoost 🌲

**Metrics:**

- F1 Score: 0.9729 🏆

**Pros 🌟:**

1. **High Generalization** – Works well on small-to-medium tabular datasets.
2. **Handles Imbalanced Data** – `scale_pos_weight` and `eval_metric` help.
3. **Faster Training** – Trains faster than deep networks.
4. **Interpretability** – Feature importance, SHAP values, PDP available.
5. **Less Hyperparameter Sensitivity** – Performs well out-of-the-box.

**Cons ⚠️:**

1. **Limited Nonlinear Feature Interaction** – Might miss extremely complex patterns.
2. **Scalability Limits** – Large datasets slow down training.
3. **GPU Advantage Limited** – Less speedup than deep nets.

**Summary ✨:** Slightly better F1 and more stable. Easy to interpret and deploy.

---

## 3️⃣ Practical Recommendation 📌

| Model          | Test F1-Score | Pros 🌟                                                        | Cons ⚠️                                                     | Recommendation                                        |
| -------------- | ------------- | -------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| AdvancedDNN 🤖 | 0.9610 🎯     | High Capacity, Flexible, Potential for large data 🌟           | Overfitting, Training Complexity, Hard to Interpret ⚠️      | Explore if dataset grows or multimodal features added |
| XGBoost 🌲     | 0.9729 🏆     | High Generalization, Handles Imbalance, Fast, Interpretable 🌟 | Limited Nonlinear Interaction, Slight Scalability Issues ⚠️ | Recommended choice ✅                                 |

**Conclusion ✅:**
For current dataset: **XGBoost 🌲** is the recommended choice. AdvancedDNN 🤖 can be explored if dataset grows or multimodal features are added.

---

## 4️⃣ Comparison Table 📊

| Aspect              | AdvancedDNN 🤖                   | XGBoost 🌲            | Verdict ✅              |
| ------------------- | -------------------------------- | --------------------- | ----------------------- |
| Test F1-Score       | 0.9610 🎯                        | 0.9729 🏆             | XGBoost 🌲              |
| Overfitting         | Moderate ⚠️                      | Low ✅                | XGBoost 🌲              |
| Training Complexity | High 🔥                          | Medium ⚡             | XGBoost 🌲              |
| Inference Time      | Slower 🐢                        | Fast ⚡               | XGBoost 🌲              |
| Interpretability    | Hard 😕                          | Easy 👍               | XGBoost 🌲              |
| Scalability         | Needs more data 📈               | Efficient ⚡          | XGBoost 🌲              |
| Future Potential    | High (embeddings, multimodal) 🚀 | Limited to tabular 🌳 | DNN 🤖 if dataset grows |

---

## 5️⃣ Visualization Idea 📈

```mermaid
bar
    title Model F1 Comparison
    "AdvancedDNN 🤖": 0.961
    "XGBoost 🌲": 0.973
```

**Insight 💡:** XGBoost 🌲 is slightly better in F1-score and more stable. AdvancedDNN 🤖 shows potential for high-capacity modeling but risks overfitting.

---

## 6️⃣ Optional Next Step 🚀

- Ensemble both models to combine XGBoost stability 🌲 and AdvancedDNN capacity 🤖.
- Use a weighted average of predictions or stacking for potentially higher F1-score 🎯.
