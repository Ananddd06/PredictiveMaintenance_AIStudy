import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# ------------------------------
# Device
# ------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔥 Using device: {device}")

# ------------------------------
# Load preprocessed data
# ------------------------------
processor = DataProcessor('/content/preprocessed_df.csv', '/content')
X_tensor, y_tensor, metadata = processor.load_and_preprocess()

# ------------------------------
# Train-test split
# ------------------------------
indices = torch.randperm(len(X_tensor))
train_size = int(0.8 * len(X_tensor))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_test = X_tensor[test_indices]
y_test = y_tensor[test_indices]

# ------------------------------
# Create test dataset & loader
# ------------------------------
test_dataset = TabularDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# ------------------------------
# Load best model
# ------------------------------
best_model = joblib.load('/content/best_deep_learning_model.pkl')
best_model = best_model.to(device)
best_model.eval()

# ------------------------------
# Predict & collect outputs
# ------------------------------
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = best_model(X_batch)
        predictions = torch.argmax(outputs, dim=1).cpu()
        
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(predictions.numpy())

# ------------------------------
# Metrics
# ------------------------------
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"\n📊 Test Accuracy: {accuracy:.4f}")
print(f"📊 Weighted F1-Score: {f1:.4f}\n")

print("📋 Classification Report:")
print(classification_report(y_true, y_pred))

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Test Set')
plt.show()
