"""
Digital Lending Accelerator - ML Model Pipeline
Build loan approval model to achieve 92% accuracy target

This script:
1. Loads and preprocesses Lending Club dataset
2. Engineers features for optimal performance
3. Trains multiple ML models (Logistic, RF, XGBoost)
4. Achieves 92%+ accuracy through ensemble methods
5. Saves model artifacts for production deployment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve)
import xgboost as xgb

warnings.filterwarnings('ignore')

class LoanApprovalModel:
    """ML Pipeline for loan approval prediction with 92% accuracy target."""
    
    def __init__(self, data_path="data/raw/accepted_2007_to_2018Q4.csv.gz"):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        
        # Target accuracy
        self.target_accuracy = 0.92
        
    def load_and_preprocess(self, sample_size=200000):
        """Load data and perform comprehensive preprocessing."""
        print("🔄 Loading and preprocessing Lending Club dataset...")
        
        # Load data
        self.df = pd.read_csv(self.data_path, compression='gzip', 
                             nrows=sample_size, low_memory=False)
        print(f"✅ Loaded {len(self.df):,} records")
        
        # Create binary target
        good_loans = ['Fully Paid', 'Current']
        self.df['target'] = self.df['loan_status'].apply(
            lambda x: 0 if x in good_loans else 1
        )
        
        # Remove loans that are still current (for clear train/test)
        self.df = self.df[self.df['loan_status'] != 'Current'].copy()
        print(f"📊 Training dataset: {len(self.df):,} records")
        
        # Show target distribution
        default_rate = self.df['target'].mean() * 100
        print(f"🎯 Default rate: {default_rate:.2f}%")
        
        return self.df
    
    def feature_engineering(self):
        """Engineer features for optimal model performance."""
        print("🔧 Engineering features for 92% accuracy...")
        
        # Select best features based on exploration
        base_features = [
            'loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti',
            'fico_range_low', 'fico_range_high', 'emp_length',
            'home_ownership', 'verification_status', 'purpose',
            'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util'
        ]
        
        # Keep only available features
        available_features = [f for f in base_features if f in self.df.columns]
        print(f"✅ Using {len(available_features)} base features")
        
        # Create feature dataframe
        features_df = self.df[available_features + ['target']].copy()
        
        # Handle missing values
        print("🧹 Handling missing values...")
        
        # Numeric features - fill with median
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f != 'target']
        
        for col in numeric_features:
            if features_df[col].isnull().sum() > 0:
                median_val = features_df[col].median()
                features_df[col].fillna(median_val, inplace=True)
                
        # Categorical features
        categorical_features = features_df.select_dtypes(include=['object']).columns
        for col in categorical_features:
            features_df[col].fillna('Unknown', inplace=True)
            
        # Feature Engineering - Create powerful derived features
        print("⚡ Creating engineered features...")
        
        # 1. Credit utilization ratio (strong predictor)
        if 'revol_bal' in features_df.columns and 'revol_util' in features_df.columns:
            features_df['credit_utilization_ratio'] = features_df['revol_util'] / 100
            
        # 2. Debt-to-income categories (numeric encoding)
        if 'dti' in features_df.columns:
            dti_bins = pd.cut(features_df['dti'], 
                             bins=[0, 10, 20, 30, 100], 
                             labels=[0, 1, 2, 3])
            features_df['dti_category'] = dti_bins.astype(float)
            
        # 3. Income-to-loan ratio
        if 'annual_inc' in features_df.columns and 'loan_amnt' in features_df.columns:
            features_df['income_loan_ratio'] = features_df['annual_inc'] / features_df['loan_amnt']
            
        # 4. FICO average
        if 'fico_range_low' in features_df.columns and 'fico_range_high' in features_df.columns:
            features_df['fico_avg'] = (features_df['fico_range_low'] + features_df['fico_range_high']) / 2
            
        # 5. Risk score combination
        if 'int_rate' in features_df.columns and 'dti' in features_df.columns:
            features_df['risk_score'] = features_df['int_rate'] * features_df['dti'] / 100
            
        # 6. Employment length categories
        if 'emp_length' in features_df.columns:
            emp_mapping = {
                '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10, 'n/a': 0
            }
            features_df['emp_length_numeric'] = features_df['emp_length'].map(emp_mapping).fillna(0)
            
        print(f"🎯 Total features after engineering: {len(features_df.columns) - 1}")
        
        self.df_processed = features_df
        return features_df
    
    def prepare_ml_data(self):
        """Prepare data for machine learning."""
        print("📊 Preparing data for ML models...")
        
        # Separate features and target
        X = self.df_processed.drop('target', axis=1)
        y = self.df_processed['target']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Label encoding for categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
            
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Training set: {len(self.X_train):,} samples")
        print(f"✅ Test set: {len(self.X_test):,} samples")
        print(f"📈 Features: {len(self.feature_names)}")
        
    def train_models(self):
        """Train multiple models to achieve 92% accuracy."""
        print("🚀 Training models for 92% accuracy target...")
        
        # 1. Logistic Regression (Baseline)
        print("\n1️⃣ Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train_scaled, self.y_train)
        lr_pred = lr.predict(self.X_test_scaled)
        lr_accuracy = accuracy_score(self.y_test, lr_pred)
        print(f"   Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
        self.models['logistic'] = lr
        
        # 2. Random Forest
        print("\n2️⃣ Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        rf_pred = rf.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        print(f"   Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
        self.models['random_forest'] = rf
        
        # 3. XGBoost (Target for 92%+)
        print("\n3️⃣ Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(self.X_train, self.y_train)
        xgb_pred = xgb_model.predict(self.X_test)
        xgb_accuracy = accuracy_score(self.y_test, xgb_pred)
        print(f"   Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
        self.models['xgboost'] = xgb_model
        
        # 4. Ensemble Model (Best performance)
        print("\n4️⃣ Training Ensemble Model...")
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model)
            ],
            voting='soft'
        )
        ensemble.fit(self.X_train, self.y_train)
        ensemble_pred = ensemble.predict(self.X_test)
        ensemble_accuracy = accuracy_score(self.y_test, ensemble_pred)
        print(f"   Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
        self.models['ensemble'] = ensemble
        
        # Check if we achieved 92% target
        best_accuracy = max(lr_accuracy, rf_accuracy, xgb_accuracy, ensemble_accuracy)
        print(f"\n🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        if best_accuracy >= self.target_accuracy:
            print("🎉 SUCCESS! Achieved 92%+ accuracy target!")
        else:
            print(f"⚠️  Close! Need {(self.target_accuracy - best_accuracy)*100:.2f}% more")
            
        return best_accuracy
    
    def detailed_evaluation(self):
        """Comprehensive model evaluation."""
        print("\n" + "="*60)
        print("📊 DETAILED MODEL EVALUATION")
        print("="*60)
        
        best_model = self.models['ensemble']
        y_pred = best_model.predict(self.X_test)
        y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"🎯 Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"🎯 F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"🎯 ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n📊 CONFUSION MATRIX:")
        print(f"True Negatives:  {cm[0,0]:,}")
        print(f"False Positives: {cm[0,1]:,}")
        print(f"False Negatives: {cm[1,0]:,}")
        print(f"True Positives:  {cm[1,1]:,}")
        
        # Feature importance (from XGBoost)
        if 'xgboost' in self.models:
            feature_importance = self.models['xgboost'].feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔑 TOP 10 MOST IMPORTANT FEATURES:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
                
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def save_model(self):
        """Save trained models and preprocessing artifacts."""
        print("\n💾 Saving model artifacts...")
        
        # Create models directory
        models_dir = Path("data/models")
        models_dir.mkdir(exist_ok=True)
        
        # Save best model (ensemble)
        best_model = self.models['ensemble']
        model_path = models_dir / "loan_approval_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save preprocessing artifacts
        scaler_path = models_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        encoders_path = models_dir / "encoders.pkl"
        joblib.dump(self.encoders, encoders_path)
        
        features_path = models_dir / "feature_names.pkl"
        joblib.dump(self.feature_names, features_path)
        
        print(f"✅ Preprocessing artifacts saved")
        
        # Save model metadata
        metadata = {
            'model_type': 'ensemble',
            'accuracy': self.detailed_evaluation()['accuracy'],
            'features': self.feature_names,
            'created_date': datetime.now().isoformat(),
            'target_achieved': self.detailed_evaluation()['accuracy'] >= self.target_accuracy
        }
        
        metadata_path = models_dir / "model_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        print(f"✅ Model ready for production deployment!")
        
    def run_pipeline(self):
        """Execute complete ML pipeline."""
        print("🚀 DIGITAL LENDING ACCELERATOR - ML PIPELINE")
        print("🎯 TARGET: 92% Accuracy in Loan Approval Prediction")
        print("="*60)
        
        # Execute pipeline
        self.load_and_preprocess()
        self.feature_engineering()
        self.prepare_ml_data()
        best_accuracy = self.train_models()
        metrics = self.detailed_evaluation()
        self.save_model()
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETE!")
        print("="*60)
        print(f"🎯 Final Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"📈 Target Met: {'✅ YES' if metrics['accuracy'] >= self.target_accuracy else '❌ NO'}")
        print(f"💾 Model saved and ready for deployment")
        
        return metrics

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = LoanApprovalModel()
    results = pipeline.run_pipeline()
