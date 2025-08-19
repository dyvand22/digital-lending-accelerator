"""
Digital Lending Accelerator - 92% Accuracy Achievement
Clean, robust ML pipeline to achieve the 92% accuracy target
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings('ignore')

def achieve_92_accuracy():
    """Main function to achieve 92% accuracy target."""
    
    print("🚀 DIGITAL LENDING ACCELERATOR - 92% ACCURACY ACHIEVEMENT")
    print("🎯 TARGET: 92% Loan Approval Prediction Accuracy")
    print("="*65)
    
    # Step 1: Load and clean data
    print("\n🔄 Step 1: Loading dataset...")
    df = pd.read_csv("data/raw/accepted_2007_to_2018Q4.csv.gz", 
                     compression='gzip', nrows=150000, low_memory=False)
    print(f"✅ Loaded {len(df):,} records")
    
    # Step 2: Filter to completed loans only
    print("\n📊 Step 2: Filtering to completed loans...")
    good_loans = ['Fully Paid']
    bad_loans = ['Charged Off', 'Default']
    df_clean = df[df['loan_status'].isin(good_loans + bad_loans)].copy()
    df_clean['target'] = df_clean['loan_status'].apply(lambda x: 0 if x in good_loans else 1)
    
    print(f"✅ Clean dataset: {len(df_clean):,} records")
    print(f"🎯 Default rate: {df_clean['target'].mean()*100:.2f}%")
    
    # Step 3: Feature engineering
    print("\n🔧 Step 3: Engineering features...")
    features = [
        'loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti',
        'fico_range_low', 'fico_range_high', 'emp_length',
        'home_ownership', 'verification_status', 'purpose',
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util'
    ]
    
    # Keep available features
    available = [f for f in features if f in df_clean.columns]
    feature_df = df_clean[available + ['target']].copy()
    print(f"✅ Using {len(available)} features")
    
    # Clean term column
    if 'term' in feature_df.columns:
        feature_df['term'] = feature_df['term'].str.extract('(\\d+)').astype(float)
    
    # Add engineered features
    if 'fico_range_low' in feature_df.columns and 'fico_range_high' in feature_df.columns:
        feature_df['fico_avg'] = (feature_df['fico_range_low'] + feature_df['fico_range_high']) / 2
    
    if 'annual_inc' in feature_df.columns and 'loan_amnt' in feature_df.columns:
        feature_df['income_loan_ratio'] = feature_df['annual_inc'] / feature_df['loan_amnt']
    
    # Step 4: Prepare data
    print("\n📊 Step 4: Preparing data...")
    X = feature_df.drop('target', axis=1)
    y = feature_df['target']
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    print(f"✅ Final features: {len(X_clean.columns)}")
    print(f"✅ Missing values: {X_clean.isnull().sum().sum()}")
    
    # Step 5: Split data
    print("\n✂️ Step 5: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Training: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Step 6: Train models
    print("\n🚀 Step 6: Training models for 92% target...")
    
    models = {}
    results = {}
    
    # Logistic Regression
    print("1️⃣ Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"   Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
    models['logistic'] = lr
    results['logistic'] = lr_acc
    
    # Random Forest
    print("2️⃣ Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10,
        min_samples_leaf=4, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"   Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    models['random_forest'] = rf
    results['random_forest'] = rf_acc
    
    # XGBoost
    print("3️⃣ XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"   Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
    models['xgboost'] = xgb_model
    results['xgboost'] = xgb_acc
    
    # Ensemble
    print("4️⃣ Ensemble Model...")
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"   Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    models['ensemble'] = ensemble
    results['ensemble'] = ensemble_acc
    
    # Step 7: Results
    best_acc = max(results.values())
    best_model_name = max(results, key=results.get)
    
    print(f"\n🎯 BEST MODEL: {best_model_name.upper()}")
    print(f"🎯 BEST ACCURACY: {best_acc:.4f} ({best_acc*100:.2f}%)")
    
    if best_acc >= 0.92:
        print("🎉 SUCCESS! 92%+ accuracy target ACHIEVED!")
        success = True
    else:
        gap = (0.92 - best_acc) * 100
        print(f"⚠️  Close! Need {gap:.2f}% more for 92% target")
        success = False
    
    # Step 8: Detailed evaluation
    print("\n" + "="*60)
    print("📊 DETAILED EVALUATION")
    print("="*60)
    
    best_pred = ensemble_pred
    accuracy = accuracy_score(y_test, best_pred)
    precision = precision_score(y_test, best_pred)
    recall = recall_score(y_test, best_pred)
    f1 = f1_score(y_test, best_pred)
    
    print(f"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"🎯 Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"🎯 F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_pred)
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"Good Loans Approved (TN):  {cm[0,0]:,}")
    print(f"Bad Loans Approved (FP):   {cm[0,1]:,}")
    print(f"Good Loans Rejected (FN):  {cm[1,0]:,}")
    print(f"Bad Loans Rejected (TP):   {cm[1,1]:,}")
    
    # Step 9: Save model
    print("\n💾 Saving model...")
    models_dir = Path("data/models")
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(models['ensemble'], models_dir / "loan_model_92.pkl")
    joblib.dump(scaler, models_dir / "scaler_92.pkl")
    
    metadata = {
        'accuracy': accuracy,
        'target_achieved': success,
        'created_date': datetime.now().isoformat()
    }
    joblib.dump(metadata, models_dir / "metadata_92.pkl")
    
    print("✅ Model saved for production!")
    
    # Final summary
    print("\n" + "="*65)
    print("🎉 PIPELINE COMPLETE!")
    print("="*65)
    print(f"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📈 92% Target Met: {'✅ YES' if success else '❌ NO'}")
    print("💾 Model ready for Flask API integration!")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'target_achieved': success
    }

if __name__ == "__main__":
    results = achieve_92_accuracy()
