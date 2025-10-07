

I'll create a comprehensive README.md file for your GitHub repository based on the presentation content.

# A Predictive Maintenance Approach in Manufacturing Systems via AI-based Early Failure Detection

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🎯 Project Overview

This project implements a predictive maintenance system for manufacturing equipment using machine learning techniques. The approach focuses on early failure detection to minimize downtime, reduce operational costs, and improve overall manufacturing efficiency.

The system evaluates multiple machine learning algorithms to identify the most effective model for predicting equipment failures before they occur, enabling a shift from reactive to proactive maintenance strategies.

## 🚨 Problem Statement

Manufacturing systems face several critical challenges:

- Unpredictable tool wear and system malfunctions cause unexpected machine failures
- Poor workpiece quality leads to rework, material waste, and increased production costs
- High maintenance costs represent a significant percentage of operational expenses
- Significant downtime occurs due to a reactive, "fix-it-when-it-breaks" maintenance approach

**Goal**: Identify potential failures before they occur to reduce waste, lower costs, and improve overall manufacturing efficiency.

## 🎯 Project Objectives

- Evaluate a comprehensive suite of Machine Learning models for defect detection
- Prevent idle time caused by unpredictable tool wear and system failures
- Improve workpiece quality through accurate, early failure identification
- Reduce downtime and operational costs by shifting from reactive to predictive maintenance

## 🔧 Methodology

### Data Collection & Preprocessing
- Dataset: AI4I 2020 Predictive Maintenance Dataset
- 10,000 records with features simulating real-world manufacturing scenarios
- Severely unbalanced data, with only 3.39% failures (339 instances)
- Key Features: Type, Air Temperature, Rotational Speed, Torque, and Tool Wear

### Preprocessing Techniques
- Label Encoding: Applied to the categorical 'Type' feature (L/M/H -> 0/1/2)
- MinMaxScaler: Used for numerical features to scale them to a [0, 1] range
- Data Splitting: 80% for training and 20% for testing
- Handling Imbalance: Oversampling applied only to training data

### Model Selection & Training
- Evaluated 9 baseline models including Ensemble Models, Linear Models, Instance-Based, and others
- Top 3 models (CatBoost, GradientBoosting, SVC) selected for intensive hyperparameter tuning
- GridSearchCV with 3-fold cross-validation used for hyperparameter optimization
- F1-Score (Weighted) used as the primary evaluation metric

## 📊 Key Findings

- The CatBoost model achieved the highest overall accuracy at 80.88% on unseen test data
- Gradient Boosting models and SVC consistently outperformed other algorithms
- The model excels at predicting common failures like Heat Dissipation (100% Recall) and Power Failure (100% Recall)
- Challenge: The model struggles with extremely rare failures like Tool Wear Failure
- Oversampling the minority failure classes was the most critical preprocessing step

## 📁 Repository Structure

```
predictive-maintenance-ai/
├── data/
│   ├── raw/                     # Raw dataset files
│   └── processed/               # Processed and cleaned data
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Data cleaning and preprocessing functions
│   ├── feature_engineering.py   # Feature extraction and transformation
│   ├── model_training.py        # Model training and evaluation functions
│   └── prediction.py            # Functions for making predictions
├── models/                      # Trained model files
├── results/                     # Model evaluation results and visualizations
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── main.py                      # Main script for running the pipeline
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/predictive-maintenance-ai.git
cd predictive-maintenance-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Complete Pipeline
```bash
python main.py
```

### Using Individual Components

#### Data Preprocessing
```python
from src.data_processing import preprocess_data
from src.feature_engineering import engineer_features

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess_data('data/raw/ai4i2020.csv')

# Apply feature engineering
X_train_processed = engineer_features(X_train)
X_test_processed = engineer_features(X_test)
```

#### Model Training
```python
from src.model_training import train_catboost_model

# Train the best performing model
model = train_catboost_model(X_train_processed, y_train)
```

#### Making Predictions
```python
from src.prediction import predict_failure

# Make predictions on new data
predictions = predict_failure(model, new_data)
```

## 📈 Results

The CatBoost model with the following hyperparameters achieved the best performance:
- depth: 4
- iterations: 200
- learning_rate: 0.1

Performance Metrics:
- Overall Accuracy: 80.88%
- Heat Dissipation Failure Recall: 100%
- Power Failure Recall: 100%
- Tool Wear Failure Recall: 0% (Challenge area for future work)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- AI4I 2020 Predictive Maintenance Dataset
- "A Predictive Maintenance Approach in Manufacturing Systems via AI-based Early Failure Detection" by J Anand, SRM Institute of Science and Technology

## 👤 Author

**J Anand**
- Mtech AI
- SRM Institute of Science and Technology

## 🙏 Acknowledgments

- SRM Institute of Science and Technology for providing resources and support
- The creators of the AI4I 2020 Predictive Maintenance Dataset
- The open-source community for the tools and libraries used in this project