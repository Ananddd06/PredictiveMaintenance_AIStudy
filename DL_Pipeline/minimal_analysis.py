import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# 1. Configuration & Setup
# --------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- THE ACTUAL MAPPING THE MODEL WAS TRAINED ON (REVERSE-ENGINEERED) ---
# This is the key to solving the problem.
ACTUAL_TRAINING_MAPPING = {
    'No Failure': 0,
    'Power Failure': 1,              # Was previously Heat Dissipation
    'Tool Wear Failure': 2,          # Was previously Power
    'Overstrain Failure': 3,         # This one was correct
    'Random Failures': 4,            # Was previously Tool Wear
    'Heat Dissipation Failure': 5    # Was previously Random
}

# Create the failure-only map based on the correct mapping above
FAILURE_ONLY_MAP_REVERSED = {
    1: 'Power',
    2: 'Tool Wear',
    3: 'Overstrain',
    4: 'Randoms',
    5: 'Heat Dissipation'
}

# --- Define file paths ---
DATA_PATH = '/content/preprocessed_df.csv'
MODEL_PATH = '/content/best_deep_learning_model.pkl'
SCALER_PATH = '/content/scaler.pkl'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔥 Using device: {device}")

# --------------------------------------------------------------------------
# 2. PyTorch Model & Dataset Class Definitions
# --------------------------------------------------------------------------
# (These class definitions remain unchanged)
class AdvancedDNN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.block1 = self._make_block(512, 256, dropout_rate)
        self.block2 = self._make_block(256, 128, dropout_rate)
        self.block3 = self._make_block(128, 64, dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(32, num_classes))
    def _make_block(self, i, o, d): return nn.Sequential(nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(d))
    def forward(self, x):
        x = torch.relu(self.bn1(self.input_layer(x)))
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.classifier(x)

class TabularLSTM(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, num_layers: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.feature_embedding = nn.Linear(1, 16)
        self.lstm = nn.LSTM(16, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout_rate, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 2 * input_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, num_classes))
    def forward(self, x):
        b, s = x.shape; x = self.feature_embedding(x.unsqueeze(-1)); o, _ = self.lstm(x); a, _ = self.attention(o, o, o); return self.classifier(a.reshape(b, -1))

class TabularTransformer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.feature_projection = nn.Linear(1, d_model)
        self.pos_encoding = self._create_positional_encoding(input_dim, d_model)
        e = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout_rate, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(e, num_layers=num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(128, num_classes))
    def _create_positional_encoding(self, m, d):
        p=torch.zeros(m,d);pos=torch.arange(0,m,dtype=torch.float).unsqueeze(1);div=torch.exp(torch.arange(0,d,2).float()*(-np.log(10000.0)/d));p[:,0::2]=torch.sin(pos*div);p[:,1::2]=torch.cos(pos*div);return p.unsqueeze(0)
    def forward(self, x):
        b, s = x.shape; x = self.feature_projection(x.unsqueeze(-1)); x = x + self.pos_encoding[:, :s, :].to(x.device); x = self.transformer(x); return self.classifier(x.mean(dim=1))

class TabularDataset(Dataset):
    def __init__(self, X, y): self.X=X.clone().detach().float(); self.y=y.clone().detach().long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# --------------------------------------------------------------------------
# 3. Data Loading and Preprocessing
# --------------------------------------------------------------------------
print("📊 Loading and preparing data using the CORRECT discovered mapping...")
df = pd.read_csv(DATA_PATH)
df_proc = df.drop(['UDI', 'Product ID', 'Target'], axis=1)

# --- Use the reverse-engineered mapping to encode labels ---
y_full = torch.LongTensor(df_proc['Failure Type'].map(ACTUAL_TRAINING_MAPPING).values)
X_features = df_proc.drop('Failure Type', axis=1)

scaler = joblib.load(SCALER_PATH)
num_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X_numerical_scaled = scaler.transform(X_features[num_features])
X_categorical = pd.get_dummies(X_features['Type'], prefix='Type').values
X_full = torch.FloatTensor(np.concatenate([X_numerical_scaled, X_categorical], axis=1))

# --- Create the full TEST SET for final evaluation ---
torch.manual_seed(42) # for reproducible split
indices = torch.randperm(len(X_full))
train_size = int(0.8 * len(X_full))
test_indices = indices[train_size:]
X_test_full, y_test_full = X_full[test_indices], y_full[test_indices]
print(f"  - Original test set created with {len(X_test_full)} samples.")

# --- Filter the TEST set to ONLY include failures (label != 0) ---
failure_mask = y_test_full != 0
X_test = X_test_full[failure_mask]
y_test = y_test_full[failure_mask]
print(f"  - Filtered test set for failures only: {len(X_test)} samples.")

# --------------------------------------------------------------------------
# 4. Model Loading and Prediction
# --------------------------------------------------------------------------
print("🚀 Loading Deep Learning model...")
model = joblib.load(MODEL_PATH)
model = model.to(device)
model.eval()

test_dataset = TabularDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

y_pred_list, y_true_list = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions = torch.argmax(outputs, dim=1).cpu()
        y_pred_list.extend(predictions.numpy())
        y_true_list.extend(y_batch.numpy())

y_pred = np.array(y_pred_list)
y_true = np.array(y_true_list)

# --------------------------------------------------------------------------
# 5. Analysis and Visualization (on TEST DATA Failures with correct labels)
# --------------------------------------------------------------------------
failure_labels = sorted(list(FAILURE_ONLY_MAP_REVERSED.keys()))
class_names_failures = [FAILURE_ONLY_MAP_REVERSED[l].replace(" Failure", "") for l in failure_labels]

print("📈 Generating FINAL confusion matrix for failure types on the TEST SET...")
cm = confusion_matrix(y_true, y_pred, labels=failure_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral', # Changed cmap for visibility
            xticklabels=class_names_failures,
            yticklabels=class_names_failures,
            linewidths=1.5, linecolor='black',
            cbar_kws={'label': 'Count'},
            annot_kws={"size":12, "weight":'bold', "color":"black"})
plt.title('Corrected Confusion Matrix - Deep Learning Model (TEST SET - Failures Only)', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
accuracy = np.trace(cm) / np.sum(cm)
plt.figtext(0.5, 0.02, f'True Accuracy on Failures: {accuracy:.4f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_deep_learning_FINAL_CORRECTED.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification report
print("📋 Generating FINAL classification report for failure types on the TEST SET...")
report = classification_report(y_true, y_pred, labels=failure_labels, target_names=class_names_failures, digits=4, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report_deep_learning_FINAL_CORRECTED.csv')

print("\n✅ Final Analysis Complete!")
print("Generated files:")
print(" - confusion_matrix_deep_learning_FINAL_CORRECTED.png")
print(" - classification_report_deep_learning_FINAL_CORRECTED.csv")