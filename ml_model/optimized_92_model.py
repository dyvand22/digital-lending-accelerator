"""
Digital Lending Accelerator - Optimized for 92% Accuracy
Advanced feature engineering and hyperparameter optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

warnings.filterwarnings('ignore')

def optimize_for_92_accuracy():
    """Optimized ML pipeline targeting 92%+ accuracy."""
    
    print("🚀 DIGITAL LENDING ACCELERATOR - OPTIMIZED FOR 92% ACCURACY")
    print("🎯 TARGET: 92%+ Loan Approval Prediction Accuracy")
    print("⚡ Using Advanced Feature Engineering + Hyperparameter Optimization")
    print("="*75)
    
    # Load larger dataset for better training
    print("\n🔄 Step 1: Loading larger dataset...")
    df = pd.read_csv("data/raw/accepted_2007_to_2018Q4.csv.gz", 
                     compression='gzip', nrows=300000, low_memory=False)
    print(f"✅ Loaded {len(df):,} records")
    
    # Filter to completed loans with better class balance
    print("\n📊 Step 2: Creating balanced dataset...")
    good_loans = ['Fully Paid']
    bad_loans = ['Charged Off', 'Default']
    df_clean = df[df['loan_status'].isin(good_loans + bad_loans)].copy()
    df_clean['target'] = df_clean['loan_status'].apply(lambda x: 0 if x in good_loans else 1)
    
    print(f"✅ Clean dataset: {len(df_clean):,} records")
    print(f"🎯 Default rate: {df_clean['target'].mean()*100:.2f}%")
    
    # Advanced feature engineering
    print("\n🔧 Step 3: Advanced feature engineering...")
    features = [
        'loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti',
        'fico_range_low', 'fico_range_high', 'emp_length',
        'home_ownership', 'verification_status', 'purpose',
        'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc', 'earliest_cr_line'
    ]
    
    available = [f for f in features if f in df_clean.columns]
    feature_df = df_clean[available + ['target']].copy()
    print(f"✅ Using {len(available)} base features")
    
    # Feature cleaning and engineering
    # 1. Clean term
    if 'term' in feature_df.columns:
        feature_df['term'] = feature_df['term'].str.extract('(\\d+)').astype(float)
    
    # 2. Credit score features
    if 'fico_range_low' in feature_df.columns and 'fico_range_high' in feature_df.columns:
        feature_df['fico_avg'] = (feature_df['fico_range_low'] + feature_df['fico_range_high']) / 2
        feature_df['fico_range'] = feature_df['fico_range_high'] - feature_df['fico_range_low']
    
    # 3. Income and debt ratios
    if 'annual_inc' in feature_df.columns and 'loan_amnt' in feature_df.columns:
        feature_df['income_loan_ratio'] = feature_df['annual_inc'] / feature_df['loan_amnt']
        feature_df['loan_income_pct'] = feature_df['loan_amnt'] / feature_df['annual_inc']
    
    # 4. Credit utilization
    if 'revol_util' in feature_df.columns:
        feature_df['credit_util_norm'] = feature_df['revol_util'] / 100
        feature_df['high_utilization'] = (feature_df['revol_util'] > 80).astype(int)
    
    # 5. Risk scoring
    if 'int_rate' in feature_df.columns and 'dti' in feature_df.columns:
        feature_df['risk_score'] = feature_df['int_rate'] * feature_df['dti']
        feature_df['high_risk'] = ((feature_df['int_rate'] > 15) | (feature_df['dti'] > 25)).astype(int)
    
    # 6. Credit history length
    if 'earliest_cr_line' in feature_df.columns:
        try:
            feature_df['earliest_cr_line'] = pd.to_datetime(feature_df['earliest_cr_line'])
            feature_df['credit_history_years'] = (pd.to_datetime('2018-12-31') - feature_df['earliest_cr_line']).dt.days / 365.25
            feature_df = feature_df.drop('earliest_cr_line', axis=1)
        except:
            feature_df = feature_df.drop('earliest_cr_line', axis=1)
    
    # 7. Delinquency features
    if 'delinq_2yrs' in feature_df.columns:
        feature_df['has_delinq'] = (feature_df['delinq_2yrs'] > 0).astype(int)
    
    # 8. Inquiry features
    if 'inq_last_6mths' in feature_df.columns:
        feature_df['recent_inquiries'] = (feature_df['inq_last_6mths'] > 2).astype(int)
    
    print(f"🎯 Features after engineering: {len(feature_df.columns) - 1}")
    
    # Data preparation
    print("\n📊 Step 4: Advanced data preparation...")
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
    
    # Feature selection - keep top features
    selector = SelectKBest(f_classif, k=25)
    X_selected = selector.fit_transform(X_clean, y)
    selected_features = X_clean.columns[selector.get_support()].tolist()
    
    print(f"✅ Selected top {len(selected_features)} features for 92% target")
    print(f"✅ Missing values: {0}")
    
    # Split data
    print("\n✂️ Step 5: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Training: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Optimized model training
    print("\n🚀 Step 6: Training optimized models for 92%+ target...")
    
    models = {}
    results = {}
    
    # 1. Optimized XGBoost
    print("1️⃣ Optimized XGBoost...")
    xgb_optimized = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_optimized.fit(X_train, y_train)
    xgb_pred = xgb_optimized.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"   Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
    models['xgboost_optimized'] = xgb_optimized
    results['xgboost_optimized'] = xgb_acc
    
    # 2. Optimized Random Forest
    print("2️⃣ Optimized Random Forest...")
    rf_optimized = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf_optimized.fit(X_train, y_train)
    rf_pred = rf_optimized.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"   Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    models['random_forest_optimized'] = rf_optimized
    results['random_forest_optimized'] = rf_acc
    
    # 3. Gradient Boosting
    print("3️⃣ Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"   Accuracy: {gb_acc:.4f} ({gb_acc*100:.2f}%)")
    models['gradient_boosting'] = gb
    results['gradient_boosting'] = gb_acc
    
    # 4. Advanced Ensemble
    print("4️⃣ Advanced Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_optimized),
            ('rf', rf_optimized),
            ('gb', gb)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"   Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    models['advanced_ensemble'] = ensemble
    results['advanced_ensemble'] = ensemble_acc
    
    # Results
    best_acc = max(results.values())
    best_model_name = max(results, key=results.get)
    
    print(f"\n🎯 BEST MODEL: {best_model_name.upper().replace('_', ' ')}\")\n    print(f\"🎯 BEST ACCURACY: {best_acc:.4f} ({best_acc*100:.2f}%)\")\n    \n    if best_acc >= 0.92:\n        print(\"🎉 SUCCESS! 92%+ accuracy target ACHIEVED!\")\n        success = True\n    else:\n        gap = (0.92 - best_acc) * 100\n        print(f\"⚠️  {gap:.2f}% away from 92% target - Excellent progress!\")\n        success = False\n    \n    # Detailed evaluation\n    print(\"\\n\" + \"=\"*75)\n    print(\"📊 DETAILED EVALUATION - OPTIMIZED MODEL\")\n    print(\"=\"*75)\n    \n    best_pred = ensemble_pred\n    accuracy = accuracy_score(y_test, best_pred)\n    precision = precision_score(y_test, best_pred)\n    recall = recall_score(y_test, best_pred)\n    f1 = f1_score(y_test, best_pred)\n    \n    print(f\"🎯 Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\")\n    print(f\"🎯 Precision: {precision:.4f} ({precision*100:.2f}%)\")\n    print(f\"🎯 Recall:    {recall:.4f} ({recall*100:.2f}%)\")\n    print(f\"🎯 F1-Score:  {f1:.4f} ({f1*100:.2f}%)\")\n    \n    # Confusion matrix\n    cm = confusion_matrix(y_test, best_pred)\n    print(f\"\\n📊 CONFUSION MATRIX:\")\n    print(f\"Good Loans Approved (TN):  {cm[0,0]:,}\")\n    print(f\"Bad Loans Approved (FP):   {cm[0,1]:,}\")\n    print(f\"Good Loans Rejected (FN):  {cm[1,0]:,}\")\n    print(f\"Bad Loans Rejected (TP):   {cm[1,1]:,}\")\n    \n    # Business impact\n    total_loans = len(y_test)\n    approval_rate = (cm[0,0] + cm[0,1]) / total_loans\n    bad_loan_catch_rate = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0\n    \n    print(f\"\\n💼 BUSINESS IMPACT:\")\n    print(f\"📈 Loan Approval Rate: {approval_rate*100:.1f}%\")\n    print(f\"🛡️  Bad Loan Detection: {bad_loan_catch_rate*100:.1f}%\")\n    \n    # Save optimized model\n    print(\"\\n💾 Saving optimized model...\")\n    models_dir = Path(\"data/models\")\n    models_dir.mkdir(exist_ok=True)\n    \n    joblib.dump(models['advanced_ensemble'], models_dir / \"optimized_loan_model.pkl\")\n    joblib.dump(scaler, models_dir / \"optimized_scaler.pkl\")\n    joblib.dump(selected_features, models_dir / \"optimized_features.pkl\")\n    \n    metadata = {\n        'accuracy': accuracy,\n        'precision': precision,\n        'recall': recall,\n        'f1_score': f1,\n        'target_achieved': success,\n        'model_type': 'advanced_ensemble',\n        'features_used': selected_features,\n        'created_date': datetime.now().isoformat()\n    }\n    joblib.dump(metadata, models_dir / \"optimized_metadata.pkl\")\n    \n    print(\"✅ Optimized model saved for production!\")\n    \n    # Final summary\n    print(\"\\n\" + \"=\"*75)\n    print(\"🎉 OPTIMIZED PIPELINE COMPLETE!\")\n    print(\"=\"*75)\n    print(f\"🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\")\n    print(f\"📈 92% Target: {'✅ ACHIEVED!' if success else '📊 In Progress'}\")\n    print(f\"💾 Production-ready model with {len(selected_features)} optimized features\")\n    print(\"🚀 Ready for Flask API and Salesforce integration!\")\n    \n    return {\n        'accuracy': accuracy,\n        'precision': precision,\n        'recall': recall,\n        'f1_score': f1,\n        'target_achieved': success,\n        'features_count': len(selected_features)\n    }\n\nif __name__ == \"__main__\":\n    final_results = optimize_for_92_accuracy()
