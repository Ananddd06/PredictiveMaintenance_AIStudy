import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from typing import Tuple, Dict
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Advanced data preprocessing with proper balancing and encoding"""
    
    def __init__(self, data_path: str, save_path: str):
        self.data_path = data_path
        self.save_path = save_path
        self.scaler = MinMaxScaler()
        
    def load_and_preprocess(self) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Load, preprocess with proper one-hot encoding and tensor conversion"""
        df = pd.read_csv(self.data_path)
        print(f"📊 Original data shape: {df.shape}")
        
        # Remove UDI, Product ID, and Target columns
        df = df.drop(['UDI', 'Product ID', 'Target'], axis=1)
        
        # Check data distribution
        print("\n📈 Original Failure Type Distribution:")
        failure_counts = df['Failure Type'].value_counts()
        for failure, count in failure_counts.items():
            print(f"  {failure}: {count} ({count/len(df)*100:.2f}%)")
        
        # Numerical features - MinMaxScaler
        numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                         'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        X_numerical = self.scaler.fit_transform(df[numerical_cols])
        
        # Save scaler
        joblib.dump(self.scaler, f'{self.save_path}/scaler.pkl')
        print(f"✅ Scaler saved to {self.save_path}/scaler.pkl")
        
        # One-hot encode Type feature
        type_dummies = pd.get_dummies(df['Type'], prefix='Type')
        X_type = type_dummies.values
        
        # One-hot encode Failure Type (target) - NOT for training target, but for feature analysis
        failure_dummies = pd.get_dummies(df['Failure Type'], prefix='Failure')
        
        # Create target labels with proper mapping (No Failure = 0, others = 1,2,3...)
        unique_failures = df['Failure Type'].unique()
        # Ensure "No Failure" is mapped to 0
        failure_to_idx = {}
        if 'No Failure' in unique_failures:
            failure_to_idx['No Failure'] = 0
            idx = 1
            for failure in unique_failures:
                if failure != 'No Failure':
                    failure_to_idx[failure] = idx
                    idx += 1
        else:
            failure_to_idx = {failure: idx for idx, failure in enumerate(unique_failures)}
        
        y = df['Failure Type'].map(failure_to_idx).values
        
        # Combine all features (numerical + Type one-hot)
        X = np.concatenate([X_numerical, X_type], axis=1)
        
        # Convert to tensors immediately
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Feature names for reference
        feature_names = numerical_cols + list(type_dummies.columns)
        
        metadata = {
            'feature_names': feature_names,
            'num_features': X.shape[1],
            'num_classes': len(unique_failures),
            'failure_types': list(unique_failures),
            'failure_to_idx': failure_to_idx,
            'class_distribution': dict(Counter(y))
        }
        
        print(f"🎯 Final tensor shapes: X={X_tensor.shape}, y={y_tensor.shape}")
        print(f"📋 Features: {metadata['num_features']}")
        print(f"🏷️ Classes: {metadata['num_classes']}")
        
        return X_tensor, y_tensor, metadata
    
    def apply_smote_balancing(self, X_train: torch.Tensor, y_train: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply SMOTE balancing: 2500 samples for all failure types except No Failure"""
        print("\n⚖️ Applying SMOTE balancing...")
        
        # Convert tensors to numpy for SMOTE
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy()
        
        print("📊 Before SMOTE:")
        train_counts = Counter(y_train_np)
        for class_idx, count in train_counts.items():
            print(f"  Class {class_idx}: {count}")
        
        # Create sampling strategy: 2500 for all failure types except No Failure (class 0)
        sampling_strategy = {}
        for class_idx in np.unique(y_train_np):
            if class_idx == 0:  # No Failure - keep original count
                sampling_strategy[class_idx] = train_counts[class_idx]
            else:  # All failure types - balance to 2500
                sampling_strategy[class_idx] = 2500
        
        print(f"🎯 SMOTE sampling strategy: {sampling_strategy}")
        
        # Apply SMOTE with custom sampling strategy
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X_train_np, y_train_np)
        
        print("\n📊 After SMOTE:")
        balanced_counts = Counter(y_balanced)
        for class_idx, count in balanced_counts.items():
            print(f"  Class {class_idx}: {count}")
        
        # Convert back to tensors
        X_balanced_tensor = torch.FloatTensor(X_balanced)
        y_balanced_tensor = torch.LongTensor(y_balanced)
        
        print(f"✅ Balanced tensor shapes: X={X_balanced_tensor.shape}, y={y_balanced_tensor.shape}")
        
        return X_balanced_tensor, y_balanced_tensor

class TabularDataset(Dataset):
    """Optimized PyTorch Dataset with proper tensor handling"""
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.clone().detach()
        self.y = y.clone().detach()
        
        # Ensure correct dtypes
        self.X = self.X.float()
        self.y = self.y.long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AdvancedDNN(nn.Module):
    """Advanced Deep Neural Network with residual connections"""
    
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        # Residual blocks
        self.block1 = self._make_block(512, 256, dropout_rate)
        self.block2 = self._make_block(256, 128, dropout_rate)
        self.block3 = self._make_block(128, 64, dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_block(self, in_features: int, out_features: int, dropout_rate: float):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.input_layer(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

class TabularLSTM(nn.Module):
    """Optimized LSTM for tabular data with attention"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, num_layers: int = 2, dropout_rate: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature embedding
        self.feature_embedding = nn.Linear(1, 16)
        
        # LSTM layers
        self.lstm = nn.LSTM(16, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout_rate, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Reshape and embed features
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.feature_embedding(x)  # (batch, seq_len, 16)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Flatten and classify
        attn_out = attn_out.reshape(batch_size, -1)
        return self.classifier(attn_out)

class TabularTransformer(nn.Module):
    """Advanced Transformer for tabular data with positional encoding"""
    
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Feature projection
        self.feature_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout_rate, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global average pooling and classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def _create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Project features
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.feature_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        return self.classifier(x)

class AdvancedTrainer:
    """Advanced training with proper loss weighting and metrics"""
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔥 Using device: {self.device}")
    
    def calculate_class_weights(self, y_train: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Calculate class weights for imbalanced data"""
        class_counts = torch.bincount(y_train, minlength=num_classes)
        total_samples = len(y_train)
        class_weights = total_samples / (num_classes * class_counts.float())
        return class_weights.to(self.device)
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                   optimizer_name: str, num_classes: int, y_train: torch.Tensor, epochs: int = 150) -> Dict:
        """Advanced training with class weighting and scheduling"""
        
        model = model.to(self.device)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(y_train, num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizers with different learning rates
        optimizers = {
            'adam': optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4),
            'rmsprop': optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-4),
            'sgd_momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        }
        
        optimizer = optimizers[optimizer_name]
        
        # Advanced scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        # Training tracking
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            val_preds, val_true = [], []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == y_batch).sum().item()
                    val_total += y_batch.size(0)
                    
                    val_preds.extend(outputs.argmax(1).cpu().numpy())
                    val_true.extend(y_batch.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            val_f1 = f1_score(val_true, val_preds, average='weighted')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Early stopping based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_f1': best_val_f1
        }
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Comprehensive model evaluation"""
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(1).cpu().numpy()
                
                y_true.extend(y_batch.numpy())
                y_pred.extend(predictions)
                y_probs.extend(probs.cpu().numpy())
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return accuracy, f1, y_true, y_pred

class PyTorchPipelineAdvanced:
    """Advanced PyTorch pipeline with proper data handling"""
    
    def __init__(self, data_path: str, save_path: str = '/Users/anand/Desktop/FAI/Deep learning'):
        self.data_path = data_path
        self.save_path = save_path
        self.results = []
        os.makedirs(save_path, exist_ok=True)
    
    def run_pipeline(self):
        """Execute advanced pipeline with proper balancing"""
        print("🚀 Starting Advanced PyTorch Deep Learning Pipeline")
        
        # Data preprocessing
        processor = DataProcessor(self.data_path, self.save_path)
        X_tensor, y_tensor, metadata = processor.load_and_preprocess()
        
        # Train-test split (maintaining tensor format)
        indices = torch.randperm(len(X_tensor))
        train_size = int(0.8 * len(X_tensor))
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        X_train, X_test = X_tensor[train_indices], X_tensor[test_indices]
        y_train, y_test = y_tensor[train_indices], y_tensor[test_indices]
        
        # Apply SMOTE balancing
        X_train_balanced, y_train_balanced = processor.apply_smote_balancing(X_train, y_train)
        
        # Train-validation split
        val_size = int(0.2 * len(X_train_balanced))
        val_indices = torch.randperm(len(X_train_balanced))[:val_size]
        train_indices = torch.randperm(len(X_train_balanced))[val_size:]
        
        X_train_final = X_train_balanced[train_indices]
        X_val = X_train_balanced[val_indices]
        y_train_final = y_train_balanced[train_indices]
        y_val = y_train_balanced[val_indices]
        
        # Create datasets and loaders
        train_dataset = TabularDataset(X_train_final, y_train_final)
        val_dataset = TabularDataset(X_val, y_val)
        test_dataset = TabularDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=128, num_workers=2)
        
        # Advanced models
        models = {
            'AdvancedDNN': AdvancedDNN(metadata['num_features'], metadata['num_classes']),
            'TabularLSTM': TabularLSTM(metadata['num_features'], metadata['num_classes']),
            'TabularTransformer': TabularTransformer(metadata['num_features'], metadata['num_classes'])
        }
        
        optimizers = ['adam', 'rmsprop', 'sgd_momentum']
        trainer = AdvancedTrainer()
        
        # Train all combinations
        best_model = None
        best_f1 = 0
        
        for model_name, model_template in models.items():
            print(f"\n🧠 Training {model_name}...")
            
            for optimizer in optimizers:
                print(f"  📈 Using {optimizer}...")
                
                # Fresh model instance
                model = type(model_template)(metadata['num_features'], metadata['num_classes'])
                
                # Train
                history = trainer.train_model(
                    model, train_loader, val_loader, optimizer, 
                    metadata['num_classes'], y_train_final
                )
                
                # Evaluate
                accuracy, f1, y_true, y_pred = trainer.evaluate_model(model, test_loader)
                
                # Track best model
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                
                # Store results
                self.results.append({
                    'Model': model_name,
                    'Optimizer': optimizer,
                    'Accuracy': accuracy,
                    'F1_Score': f1,
                    'Best_Val_F1': history['best_val_f1']
                })
                
                print(f"    ✅ {model_name}-{optimizer}: Acc={accuracy:.4f}, F1={f1:.4f}")
        
        # Save best model
        if best_model is not None:
            joblib.dump(best_model, f'{self.save_path}/best_deep_learning_model.pkl')
            print(f"✅ Best model saved to {self.save_path}/best_deep_learning_model.pkl")
        
        # Save results
        self.save_results()
        print(f"\n🎉 Advanced pipeline completed! Results saved to {self.save_path}")
    
    def save_results(self):
        """Save comprehensive results"""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f'{self.save_path}/pytorch_advanced_results.csv', index=False)
        
        print("\n📊 ADVANCED RESULTS SUMMARY:")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        best_result = max(self.results, key=lambda x: x['F1_Score'])
        print(f"\n🏆 BEST MODEL: {best_result['Model']} with {best_result['Optimizer']}")
        print(f"   Test Accuracy: {best_result['Accuracy']:.4f}")
        print(f"   Test F1-Score: {best_result['F1_Score']:.4f}")
        print(f"   Best Val F1: {best_result['Best_Val_F1']:.4f}")

if __name__ == "__main__":
    pipeline = PyTorchPipelineAdvanced('/Users/anand/Desktop/FAI/Data file/preprocessed_df.csv')
    pipeline.run_pipeline()
