import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("📊 Loading and preparing data...")
    
    # Load data
    df = pd.read_csv('Data file/final_preprocessed_df_cleaned.csv')
    
    # Prepare features and target
    feature_cols = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
    X_numerical = df[feature_cols]
    X_categorical = df[['Type']] if 'Type' in df.columns else pd.DataFrame()
    y = df['Failure_Type_Encoded']
    
    # Scale numerical features
    scaler = MinMaxScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=feature_cols)
    
    # Combine features
    if not X_categorical.empty:
        X = pd.concat([X_categorical.reset_index(drop=True), X_numerical_scaled.reset_index(drop=True)], axis=1)
    else:
        X = X_numerical_scaled
    
    return X, y, scaler

def apply_smote_balancing(X_train, y_train):
    """Apply SMOTE balancing to training data"""
    print("🧬 Applying SMOTE balancing...")
    
    # Apply SMOTE only to failure classes (not No Failure)
    target_samples = 2500
    sampling_strategy = {}
    
    for class_label in [1, 2, 3, 4, 5]:  # Exclude 0 (No Failure)
        current_count = sum(y_train == class_label)
        if current_count < target_samples:
            sampling_strategy[class_label] = target_samples
    
    if sampling_strategy:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        return X_train_balanced, y_train_balanced
    
    return X_train, y_train

def train_xgboost_with_validation(X_train, y_train, X_test, y_test):
    """Train XGBoost with validation tracking"""
    print("🚀 Training XGBoost with validation tracking...")
    
    # Best parameters from hyperparameter tuning
    best_params = {
        'learning_rate': 0.2,
        'max_depth': 6,
        'n_estimators': 300,
        'subsample': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    # Create XGBoost model
    model = xgb.XGBClassifier(**best_params)
    
    # Fit with evaluation set for loss tracking
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Create and save confusion matrix plot"""
    print("📈 Creating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - XGBoost Model\n', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.4f}', 
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_xgboost.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def plot_training_validation_loss(model):
    """Plot training and validation loss curves"""
    print("📉 Creating training/validation loss curves...")
    
    # Get evaluation results
    results = model.evals_result()
    
    # Extract loss values
    train_loss = results['validation_0']['mlogloss']
    val_loss = results['validation_1']['mlogloss']
    epochs = range(1, len(train_loss) + 1)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Log Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Difference (Overfitting indicator)
    loss_diff = np.array(val_loss) - np.array(train_loss)
    ax2.plot(epochs, loss_diff, 'g-', linewidth=2)
    ax2.set_title('Validation - Training Loss\n(Overfitting Indicator)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss Difference', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add text annotations
    min_val_loss = min(val_loss)
    min_val_epoch = val_loss.index(min_val_loss) + 1
    ax1.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_epoch}',
                xy=(min_val_epoch, min_val_loss), xytext=(min_val_epoch + 50, min_val_loss + 0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('training_validation_loss_xgboost.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_loss, val_loss

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    print("🎯 Creating feature importance plot...")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                    color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.title('Feature Importance - XGBoost Model', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_classification_report(y_true, y_pred, class_names):
    """Generate and save classification report"""
    print("📋 Generating classification report...")
    
    # Generate report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 digits=4, output_dict=True)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Save to CSV
    report_df.to_csv('classification_report_xgboost.csv')
    
    # Print report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT - XGBoost Model")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    return report_df

def main():
    """Main function to run all analyses"""
    print("🏭 XGBoost Model Analysis")
    print("="*50)
    
    # Load and prepare data
    X, y, scaler = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Apply SMOTE balancing
    X_train_balanced, y_train_balanced = apply_smote_balancing(X_train, y_train)
    
    # Train model with validation tracking
    model = train_xgboost_with_validation(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Define class names
    class_names = ['No Failure', 'Heat Dissipation', 'Overstrain', 
                   'Power Failure', 'Random Failure', 'Tool Wear']
    
    feature_names = ['Type', 'Air_temperature_K', 'Process_temperature_K', 
                     'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
    
    # Generate all plots and reports
    cm = plot_confusion_matrix(y_test, y_pred, class_names)
    train_loss, val_loss = plot_training_validation_loss(model)
    plot_feature_importance(model, feature_names)
    report_df = generate_classification_report(y_test, y_pred, class_names)
    
    # Save model and scaler
    joblib.dump(model, 'xgboost_analysis_model.pkl')
    joblib.dump(scaler, 'xgboost_analysis_scaler.pkl')
    
    print("\n✅ Analysis Complete!")
    print("📁 Files Generated:")
    print("   - confusion_matrix_xgboost.png")
    print("   - training_validation_loss_xgboost.png") 
    print("   - feature_importance_xgboost.png")
    print("   - classification_report_xgboost.csv")
    print("   - xgboost_analysis_model.pkl")
    print("   - xgboost_analysis_scaler.pkl")

if __name__ == "__main__":
    main()
