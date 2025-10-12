import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os  # Import the 'os' library to handle file paths
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Correct mapping of failure types
LABEL_MAP = {
    0: 'Heat Dissipation',
    1: 'Overstrain',
    2: 'Power Failure',
    3: 'Tool Wear',
    4: 'Random Failure'
}

# --- DEFINE FILE PATHS ---
# Input paths
DATA_PATH = '/Users/anand/Desktop/FAI/Data file/final_preprocessed_df_cleaned.csv'
MODEL_PATH = '/Users/anand/Desktop/FAI/Model_file/xgboost_analysis_model.pkl'
SCALER_PATH = '/Users/anand/Desktop/FAI/Model_file/xgboost_analysis_scaler.pkl'

# --- MODIFICATION 1: Define the specific output path ---
OUTPUT_PATH = '/Users/anand/Desktop/FAI/ML_Pipeline'

# --- MODIFICATION 2: Create the output directory if it doesn't exist ---
# This prevents errors if the folder is not already there.
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"✅ Output files will be saved to: {OUTPUT_PATH}")


# Load data
print("\n📊 Loading data...")
df = pd.read_csv(DATA_PATH)

# Remove 'No Failure' (encoded as 5)
df = df[df['Failure_Type_Encoded'] != 5]

# Features and target
feature_cols = ['Type', 'Air_temperature_K', 'Process_temperature_K', 
                'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']

X = df[feature_cols]
y_true = df['Failure_Type_Encoded']

# Load scaler and scale features
print("🚀 Loading XGBoost model and scaler...")
scaler = joblib.load(SCALER_PATH)
X_scaled = X.copy()
num_features = ['Air_temperature_K', 'Process_temperature_K', 
                'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
X_scaled[num_features] = scaler.transform(X[num_features])

# Load model
model = joblib.load(MODEL_PATH)

# Predictions
y_pred = model.predict(X_scaled)

# Confusion matrix with all failure types
all_labels = [0, 1, 2, 3, 4]
class_names_all = [LABEL_MAP[l] for l in all_labels]

print("\n📈 Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred, labels=all_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral',
            xticklabels=class_names_all,
            yticklabels=class_names_all,
            linewidths=1.5, linecolor='black',
            cbar_kws={'label': 'Count'},
            annot_kws={"size":12, "weight":'bold', "color":"black"})
plt.title('Confusion Matrix - XGBoost Model', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
accuracy = np.trace(cm) / np.sum(cm)
plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.4f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()

# --- MODIFICATION 3: Save the plot to the specified output path ---
confusion_matrix_save_path = os.path.join(OUTPUT_PATH, 'confusion_matrix_xgboost.png')
plt.savefig(confusion_matrix_save_path, dpi=300, bbox_inches='tight')
plt.show()

# Classification report
print("\n📋 Generating classification report...")
report = classification_report(y_true, y_pred, labels=all_labels, target_names=class_names_all, digits=4, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# --- MODIFICATION 4: Save the CSV report to the specified output path ---
report_save_path = os.path.join(OUTPUT_PATH, 'classification_report_xgboost.csv')
report_df.to_csv(report_save_path)

print("\n✅ Analysis Complete!")
print("Generated files:")
print(f" - {confusion_matrix_save_path}")
print(f" - {report_save_path}")