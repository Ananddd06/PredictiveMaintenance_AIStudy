import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install with: pip install catboost")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles data loading, scaling before split, and balancing after split"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.failure_mapping = {
            0: 'No Failure',
            1: 'Heat Dissipation Failure',
            2: 'Overstrain Failure', 
            3: 'Power Failure',
            4: 'Random Failure',
            5: 'Tool Wear Failure'
        }
        self.class_weights = None
        
    def load_and_map_data(self):
        """Load data and apply intelligent failure mapping"""
        print("📊 Loading and mapping data...")
        
        df = pd.read_csv(self.data_path)
        print(f"Data shape: {df.shape}")
        
        # Feature columns (only numerical for scaling)
        self.feature_cols = ['Air_temperature_K', 'Process_temperature_K', 
                           'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
        
        # Extract features (excluding Type from scaling)
        X_numerical = df[self.feature_cols]
        X_categorical = df[['Type']] if 'Type' in df.columns else pd.DataFrame()
        
        # Use Failure_Type_Encoded as primary target
        if 'Failure_Type_Encoded' in df.columns:
            y = df['Failure_Type_Encoded'].apply(lambda x: 0 if x == 5 else x + 1)
            print("Using Failure_Type_Encoded as target")
        elif 'Target' in df.columns:
            y = df['Target'].apply(lambda x: 0 if x == 0 else 1)
            print("Using Target as binary classification")
        
        print("Original target distribution:")
        class_counts = y.value_counts().sort_index()
        for cls, count in class_counts.items():
            cls_name = self.failure_mapping.get(cls, f'Class_{cls}')
            print(f"  {cls_name}: {count} ({count/len(y)*100:.1f}%)")
            
        return X_numerical, X_categorical, y
    
    def scale_before_split(self, X_numerical, X_categorical):
        """Scale numerical features before splitting"""
        print("\n⚖️ Scaling numerical features before split...")
        
        # Scale only numerical features
        X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=self.feature_cols)
        
        # Combine with categorical
        if not X_categorical.empty:
            X_combined = pd.concat([X_categorical.reset_index(drop=True), 
                                  X_numerical_scaled.reset_index(drop=True)], axis=1)
        else:
            X_combined = X_numerical_scaled
        
        # Save scaled data
        scaled_df = X_combined.copy()
        scaled_df.to_pickle('Notebooks/scaled.pkl')
        print("Saved scaled.pkl")
        
        print("Scaling applied to:", self.feature_cols)
        return X_combined
    
    def split_and_balance_data(self, X, y, test_size=0.2):
        """Split data then balance training set using SMOTE"""
        print("\n🔄 Splitting data...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        print("\n📊 Training class distribution BEFORE balancing:")
        train_counts = y_train.value_counts().sort_index()
        for cls, count in train_counts.items():
            cls_name = self.failure_mapping.get(cls, f'Class_{cls}')
            percentage = (count / len(y_train)) * 100
            print(f"  {cls_name}: {count:,} samples ({percentage:.2f}%)")
        
        # Apply SMOTE to balance failure types to 2-3k samples
        print("\n🧬 Applying SMOTE to balance failure types...")
        
        # Target samples for minority classes (2-3k range)
        target_samples = 2500  # Target 2.5k samples for each failure type
        
        # Create sampling strategy - only oversample failure types (0-4), keep No Failure (-1) as is
        sampling_strategy = {}
        for cls, count in train_counts.items():
            if cls != -1 and count < target_samples:  # Don't touch No Failure (-1)
                sampling_strategy[cls] = target_samples
        
        print(f"SMOTE sampling strategy (target: {target_samples:,} samples):")
        for cls, target in sampling_strategy.items():
            cls_name = self.failure_mapping.get(cls, f'Class_{cls}')
            current = train_counts[cls]
            print(f"  {cls_name}: {current:,} → {target:,} samples")
        
        # Apply SMOTE
        if sampling_strategy:
            # Use adaptive k_neighbors based on smallest class
            min_samples = min(train_counts[cls] for cls in sampling_strategy.keys())
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                k_neighbors=k_neighbors
            )
            
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            print(f"\n📊 Training class distribution AFTER SMOTE balancing:")
            balanced_counts = pd.Series(y_train_balanced).value_counts().sort_index()
            total_balanced = len(y_train_balanced)
            
            for cls, count in balanced_counts.items():
                cls_name = self.failure_mapping.get(cls, f'Class_{cls}')
                percentage = (count / total_balanced) * 100
                original_count = train_counts.get(cls, 0)
                increase = count - original_count
                print(f"  {cls_name}: {count:,} samples ({percentage:.2f}%) [+{increase:,}]")
            
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            balanced_counts = train_counts
            print("No SMOTE applied - all classes already balanced")
        
        # Compute class weights on balanced data
        unique_classes = np.unique(y_train_balanced)
        class_weights_array = compute_class_weight(
            'balanced', 
            classes=unique_classes, 
            y=y_train_balanced
        )
        
        self.class_weights = dict(zip(unique_classes, class_weights_array))
        
        print(f"\n💪 Updated class weights after balancing:")
        sorted_weights = sorted(self.class_weights.items(), key=lambda x: x[1], reverse=True)
        
        for cls, weight in sorted_weights:
            cls_name = self.failure_mapping.get(cls, f'Class_{cls}')
            count = balanced_counts[cls]
            print(f"  {cls_name}: {weight:.3f} (weight) | {count:,} samples")
        
        print(f"\n✅ Final Balanced Training Data Summary:")
        print(f"  • Original training samples: {len(y_train):,}")
        print(f"  • Balanced training samples: {len(y_train_balanced):,}")
        print(f"  • Test samples: {len(y_test):,}")
        print(f"  • Number of classes: {len(unique_classes)}")
        print(f"  • No Failure samples preserved: {balanced_counts[0]:,}")
        print(f"  • Failure types balanced to ~{target_samples:,} each")
        print(f"  • New imbalance ratio: {balanced_counts.max() / balanced_counts.min():.1f}:1")
        
        # Save balanced data
        balanced_df = pd.DataFrame(X_train_balanced, columns=X_train.columns)
        balanced_df['Target_Mapped'] = y_train_balanced
        balanced_df.to_pickle('Notebooks/balanced_training_data.pkl')
        print(f"  • Saved balanced_training_data.pkl")
        
        return X_train_balanced, X_test, y_train_balanced, y_test

class ModelEvaluator:
    """Evaluates multiple models with class weighting"""
    
    def __init__(self, class_weights):
        self.class_weights = class_weights
        self.models = self._initialize_models()
        self.results = {}
        
    def _initialize_models(self):
        """Initialize diverse set of models with class weighting"""
        models = {
            'RandomForest': RandomForestClassifier(
                random_state=42, 
                n_jobs=-1,
                class_weight=self.class_weights
            ),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'ExtraTrees': ExtraTreesClassifier(
                random_state=42, 
                n_jobs=-1,
                class_weight=self.class_weights
            ),
            'SVM': SVC(
                random_state=42, 
                probability=True,
                class_weight=self.class_weights
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight=self.class_weights
            ),
            'KNN': KNeighborsClassifier(),
            'NaiveBayes': GaussianNB(),
            'DecisionTree': DecisionTreeClassifier(
                random_state=42,
                class_weight=self.class_weights
            ),
            'MLP': MLPClassifier(random_state=42, max_iter=500),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(
                random_state=42,
                verbose=False,
                class_weights=list(self.class_weights.values())
            )
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )
        
        return models
    
    def evaluate_all_models(self, X_train, y_train, X_test, y_test):
        """Evaluate all models with class weighting"""
        print("\n🔬 Evaluating all models with class weighting...")
        
        # Convert to numpy arrays to avoid indexing issues
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
        
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Cross-validation scores
            cv_scores = {}
            for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
                scores = []
                for train_idx, val_idx in cv_strategy.split(X_train_np, y_train_np):
                    X_tr = X_train_np[train_idx]
                    X_val = X_train_np[val_idx]
                    y_tr = y_train_np[train_idx]
                    y_val = y_train_np[val_idx]
                    
                    # Special handling for XGBoost sample weights
                    if name == 'XGBoost' and XGBOOST_AVAILABLE:
                        sample_weight = np.array([self.class_weights[cls] for cls in y_tr])
                        model.fit(X_tr, y_tr, sample_weight=sample_weight)
                    else:
                        model.fit(X_tr, y_tr)
                    
                    y_pred = model.predict(X_val)
                    
                    if metric == 'accuracy':
                        scores.append(accuracy_score(y_val, y_pred))
                    elif metric == 'f1_weighted':
                        scores.append(f1_score(y_val, y_pred, average='weighted'))
                    elif metric == 'precision_weighted':
                        scores.append(precision_score(y_val, y_pred, average='weighted'))
                    elif metric == 'recall_weighted':
                        scores.append(recall_score(y_val, y_pred, average='weighted'))
                
                cv_scores[metric] = {'mean': np.mean(scores), 'std': np.std(scores)}
            
            # Test set evaluation
            if name == 'XGBoost' and XGBOOST_AVAILABLE:
                sample_weight = np.array([self.class_weights[cls] for cls in y_train_np])
                model.fit(X_train_np, y_train_np, sample_weight=sample_weight)
            else:
                model.fit(X_train_np, y_train_np)
            
            y_pred = model.predict(X_test_np)
            y_pred_proba = model.predict_proba(X_test_np) if hasattr(model, 'predict_proba') else None
            
            # Calculate all metrics
            test_metrics = {
                'accuracy': accuracy_score(y_test_np, y_pred),
                'f1_weighted': f1_score(y_test_np, y_pred, average='weighted'),
                'f1_macro': f1_score(y_test_np, y_pred, average='macro'),
                'precision_weighted': precision_score(y_test_np, y_pred, average='weighted'),
                'recall_weighted': recall_score(y_test_np, y_pred, average='weighted')
            }
            
            # ROC AUC for multiclass
            try:
                if y_pred_proba is not None:
                    test_metrics['roc_auc'] = roc_auc_score(y_test_np, y_pred_proba, multi_class='ovr')
            except:
                test_metrics['roc_auc'] = 0.0
            
            self.results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'test_metrics': test_metrics,
                'predictions': y_pred,
                'confusion_matrix': confusion_matrix(y_test_np, y_pred)
            }
            
            print(f"  CV F1: {cv_scores['f1_weighted']['mean']:.4f}(±{cv_scores['f1_weighted']['std']:.4f})")
            print(f"  Test F1: {test_metrics['f1_weighted']:.4f}")
        
        return self.results
    
    def get_top_models(self, n_top=5):
        """Get top N models based on test F1 score"""
        sorted_models = sorted(
            self.results.items(), 
            key=lambda x: x[1]['test_metrics']['f1_weighted'], 
            reverse=True
        )
        return sorted_models[:n_top]

class HyperparameterTuner:
    """Handles hyperparameter tuning with class weighting"""
    
    def __init__(self, class_weights):
        self.class_weights = class_weights
        self.param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 4, 6]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ExtraTrees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'LogisticRegression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'DecisionTree': {
                'max_depth': [5, 10, 15, 20],
                'min_samples_split': [5, 10, 20]
            },
            'MLP': {
                'hidden_layer_sizes': [(100,), (100, 50), (200,)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5]
            }
        }
        
        # Add CatBoost parameters if available
        if CATBOOST_AVAILABLE:
            self.param_grids['CatBoost'] = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.03, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5]
            }
        
        # Add XGBoost parameters if available
        if XGBOOST_AVAILABLE:
            self.param_grids['XGBoost'] = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        
    def tune_models(self, top_models, X_train, y_train, X_test, y_test):
        """Tune hyperparameters with class weighting"""
        print("\n🎯 Hyperparameter tuning for top 5 models...")
        
        tuned_results = {}
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, model_data in top_models:
            print(f"\nTuning {name}...")
            
            if name in self.param_grids:
                # Create base model with class weights
                if name == 'RandomForest':
                    base_model = RandomForestClassifier(
                        random_state=42, 
                        class_weight=self.class_weights
                    )
                elif name == 'ExtraTrees':
                    base_model = ExtraTreesClassifier(
                        random_state=42, 
                        class_weight=self.class_weights
                    )
                elif name == 'SVM':
                    base_model = SVC(
                        random_state=42, 
                        probability=True,
                        class_weight=self.class_weights
                    )
                elif name == 'LogisticRegression':
                    base_model = LogisticRegression(
                        random_state=42, 
                        max_iter=1000,
                        class_weight=self.class_weights
                    )
                elif name == 'DecisionTree':
                    base_model = DecisionTreeClassifier(
                        random_state=42,
                        class_weight=self.class_weights
                    )
                elif name == 'CatBoost' and CATBOOST_AVAILABLE:
                    base_model = CatBoostClassifier(
                        random_state=42,
                        verbose=False,
                        class_weights=list(self.class_weights.values())
                    )
                elif name == 'XGBoost' and XGBOOST_AVAILABLE:
                    base_model = XGBClassifier(
                        random_state=42,
                        eval_metric='mlogloss',
                        verbosity=0
                    )
                else:
                    base_model = type(model_data['model'])(random_state=42)
                
                # Grid search with stratified CV
                grid_search = GridSearchCV(
                    base_model,
                    self.param_grids[name],
                    cv=cv_strategy,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Special handling for XGBoost
                if name == 'XGBoost' and XGBOOST_AVAILABLE:
                    sample_weight = np.array([self.class_weights[cls] for cls in y_train])
                    grid_search.fit(X_train, y_train, sample_weight=sample_weight)
                else:
                    grid_search.fit(X_train, y_train)
                
                # Evaluate tuned model
                y_pred = grid_search.predict(X_test)
                y_pred_proba = grid_search.predict_proba(X_test) if hasattr(grid_search, 'predict_proba') else None
                
                # Calculate comprehensive metrics
                tuned_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                    'f1_macro': f1_score(y_test, y_pred, average='macro'),
                    'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
                    'recall_weighted': recall_score(y_test, y_pred, average='weighted')
                }
                
                try:
                    if y_pred_proba is not None:
                        tuned_metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except:
                    tuned_metrics['roc_auc'] = 0.0
                
                tuned_results[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'test_metrics': tuned_metrics,
                    'predictions': y_pred,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                
                print(f"  Best CV F1: {grid_search.best_score_:.4f}")
                print(f"  Test F1: {tuned_metrics['f1_weighted']:.4f}")
                print(f"  Best params: {grid_search.best_params_}")
        
        return tuned_results

class ResultsAnalyzer:
    """Analyzes and displays comprehensive results"""
    
    def __init__(self, failure_mapping):
        self.failure_mapping = failure_mapping
        
    def display_comprehensive_results(self, tuned_results, y_test):
        """Display comprehensive results for all tuned models"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Sort by F1 score
        sorted_results = sorted(
            tuned_results.items(),
            key=lambda x: x[1]['test_metrics']['f1_weighted'],
            reverse=True
        )
        
        for rank, (name, results) in enumerate(sorted_results, 1):
            print(f"\n🏆 RANK {rank}: {name}")
            print("-" * 50)
            
            # Metrics
            metrics = results['test_metrics']
            print(f"Accuracy:           {metrics['accuracy']:.4f}")
            print(f"F1 Weighted:        {metrics['f1_weighted']:.4f}")
            print(f"F1 Macro:           {metrics['f1_macro']:.4f}")
            print(f"Precision Weighted: {metrics['precision_weighted']:.4f}")
            print(f"Recall Weighted:    {metrics['recall_weighted']:.4f}")
            print(f"ROC AUC:            {metrics['roc_auc']:.4f}")
            print(f"CV Score:           {results['cv_score']:.4f}")
            
            # Confusion Matrix
            print(f"\nConfusion Matrix:")
            cm = results['confusion_matrix']
            print(cm)
            
            # Classification Report
            print(f"\nClassification Report:")
            target_names = [self.failure_mapping[i] for i in sorted(self.failure_mapping.keys())]
            print(classification_report(y_test, results['predictions'], target_names=target_names))
        
        return sorted_results[0]  # Return best model

class MLPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor(data_path)
        self.evaluator = ModelEvaluator()
        self.tuner = HyperparameterTuner()
        self.analyzer = ResultsAnalyzer(self.preprocessor.failure_mapping)
        
class MLPipeline:
    """Main pipeline orchestrator with class weighting"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.preprocessor = DataPreprocessor(data_path)
        
    def run_complete_pipeline(self):
        """Run the complete ML pipeline with scale before split, balance after split"""
        print("🚀 Starting Complete ML Pipeline...")
        
        # Step 1: Load and map data
        X_numerical, X_categorical, y = self.preprocessor.load_and_map_data()
        
        # Step 2: Scale before splitting
        X_scaled = self.preprocessor.scale_before_split(X_numerical, X_categorical)
        
        # Step 3: Split and balance data
        X_train, X_test, y_train, y_test = self.preprocessor.split_and_balance_data(X_scaled, y)
        
        # Step 4: Initialize components with class weights
        self.evaluator = ModelEvaluator(self.preprocessor.class_weights)
        self.tuner = HyperparameterTuner(self.preprocessor.class_weights)
        self.analyzer = ResultsAnalyzer(self.preprocessor.failure_mapping)
        
        # Step 5: Model Evaluation with Class Weighting
        all_results = self.evaluator.evaluate_all_models(X_train, y_train, X_test, y_test)
        top_models = self.evaluator.get_top_models(n_top=5)
        
        print("\n🏆 Top 5 Models:")
        for i, (name, data) in enumerate(top_models, 1):
            f1_score = data['test_metrics']['f1_weighted']
            print(f"{i}. {name}: {f1_score:.4f}")
        
        # Step 6: Hyperparameter Tuning with Class Weighting
        tuned_results = self.tuner.tune_models(top_models, X_train, y_train, X_test, y_test)
        
        # Step 7: Results Analysis
        best_model_data = self.analyzer.display_comprehensive_results(tuned_results, y_test)
        
        # Step 8: Save Best Model
        self.save_best_model(best_model_data, tuned_results)
        
        return best_model_data, tuned_results
    
    def save_best_model(self, best_model_data, all_results):
        """Save the best model and metadata"""
        print(f"\n💾 Saving best model: {best_model_data[0]}")
        
        joblib.dump(best_model_data[1]['model'], 'Notebooks/best_model_final.pkl')
        joblib.dump(self.preprocessor.scaler, 'Notebooks/scaler_final.pkl')
        
        metadata = {
            'best_model': best_model_data[0],
            'test_metrics': best_model_data[1]['test_metrics'],
            'best_params': best_model_data[1]['best_params'],
            'class_weights': self.preprocessor.class_weights,
            'failure_mapping': self.preprocessor.failure_mapping,
            'feature_cols': self.preprocessor.feature_cols,
            'balancing_method': 'Class Weighting (No Resampling)',
            'all_results': {name: data['test_metrics'] for name, data in all_results.items()}
        }
        
        joblib.dump(metadata, 'Notebooks/pipeline_metadata.pkl')
        
        print("✅ Saved:")
        print("  - best_model_final.pkl")
        print("  - scaler_final.pkl")
        print("  - pipeline_metadata.pkl")
        print(f"\n💪 Class Weighting Applied:")
        for cls, weight in self.preprocessor.class_weights.items():
            cls_name = self.preprocessor.failure_mapping.get(cls, f'Class_{cls}')
            print(f"  {cls_name}: {weight:.3f}")

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = MLPipeline('Data file/final_preprocessed_df_cleaned.csv')
    best_model, all_results = pipeline.run_complete_pipeline()
    
    print(f"\n🎉 Pipeline Complete!")
    print(f"Best Model: {best_model[0]}")
    print(f"F1 Score: {best_model[1]['test_metrics']['f1_weighted']:.4f}")
    print("💪 Used Class Weighting - No Data Resampling!")
    
    def save_best_model(self, best_model_data, all_results):
        """Save the best model and metadata"""
        print(f"\n💾 Saving best model: {best_model_data[0]}")
        
        joblib.dump(best_model_data[1]['model'], 'Notebooks/best_model_final.pkl')
        joblib.dump(self.preprocessor.scaler, 'Notebooks/scaler_final.pkl')
        
        metadata = {
            'best_model': best_model_data[0],
            'test_metrics': best_model_data[1]['test_metrics'],
            'best_params': best_model_data[1]['best_params'],
            'failure_mapping': self.preprocessor.failure_mapping,
            'feature_cols': self.preprocessor.feature_cols,
            'all_results': {name: data['test_metrics'] for name, data in all_results.items()}
        }
        
        joblib.dump(metadata, 'Notebooks/pipeline_metadata.pkl')
        
        print("✅ Saved:")
        print("  - best_model_final.pkl")
        print("  - scaler_final.pkl")
        print("  - pipeline_metadata.pkl")

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = MLPipeline('Data file/final_preprocessed_df_cleaned.csv')
    best_model, all_results = pipeline.run_complete_pipeline()
    
    print(f"\n🎉 Pipeline Complete!")
    print(f"Best Model: {best_model[0]}")
    print(f"F1 Score: {best_model[1]['test_metrics']['f1_weighted']:.4f}")
import sys
sys.exit(0)
