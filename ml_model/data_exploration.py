"""
Digital Lending Accelerator - Data Exploration
Explore Lending Club dataset to achieve 92% accuracy target

This script:
1. Loads and examines the Lending Club dataset
2. Analyzes features for loan approval prediction
3. Identifies key variables for achieving 92% accuracy
4. Prepares data for ML model development
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

class LendingClubExplorer:
    """Explore Lending Club data for loan approval prediction."""
    
    def __init__(self, data_path="data/raw/accepted_2007_to_2018Q4.csv.gz"):
        self.data_path = data_path
        self.df = None
        self.target_col = 'loan_status'
        
    def load_data(self, sample_size=100000):
        """Load dataset with optional sampling for faster exploration."""
        print("🔄 Loading Lending Club dataset...")
        
        try:
            # Load with sampling for initial exploration
            self.df = pd.read_csv(self.data_path, compression='gzip', 
                                nrows=sample_size, low_memory=False)
            print(f"✅ Loaded {len(self.df):,} records")
            print(f"📊 Dataset shape: {self.df.shape}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
            
        return True
    
    def basic_info(self):
        """Display basic dataset information."""
        print("\n" + "="*60)
        print("📋 DATASET OVERVIEW")
        print("="*60)
        
        print(f"Records: {len(self.df):,}")
        print(f"Features: {len(self.df.columns):,}")
        print(f"Memory usage: {self.df.memory_usage().sum() / 1024**2:.1f} MB")
        
        print("\n🏷️ COLUMN TYPES:")
        print(self.df.dtypes.value_counts())
        
        print("\n🔍 MISSING VALUES:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        
        print(missing_df.head(10))
        
    def analyze_target(self):
        """Analyze loan_status (target variable) for classification."""
        print("\n" + "="*60)
        print("🎯 TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        if self.target_col not in self.df.columns:
            print("❌ loan_status column not found!")
            return
            
        print(f"\n📊 LOAN STATUS DISTRIBUTION:")
        status_counts = self.df[self.target_col].value_counts()
        print(status_counts)
        
        print(f"\n📈 LOAN STATUS PERCENTAGES:")
        status_pct = self.df[self.target_col].value_counts(normalize=True) * 100
        print(status_pct.round(2))
        
        # Create binary target for classification
        good_loans = ['Fully Paid', 'Current']
        bad_loans = ['Charged Off', 'Default', 'Late (31-120 days)', 
                    'Late (16-30 days)', 'In Grace Period']
        
        self.df['target'] = self.df[self.target_col].apply(
            lambda x: 0 if x in good_loans else 1
        )
        
        print(f"\n🔄 BINARY TARGET DISTRIBUTION:")
        binary_counts = self.df['target'].value_counts()
        print("Good Loans (0):", binary_counts[0])
        print("Bad Loans (1):", binary_counts[1])
        print(f"Default Rate: {(binary_counts[1] / len(self.df)) * 100:.2f}%")
        
    def key_features_analysis(self):
        """Analyze key features for loan approval prediction."""
        print("\n" + "="*60)
        print("🔑 KEY FEATURES FOR 92% ACCURACY")
        print("="*60)
        
        # Define key features for loan approval
        key_features = [
            'loan_amnt', 'term', 'int_rate', 'annual_inc', 'dti',
            'fico_range_low', 'fico_range_high', 'emp_length',
            'home_ownership', 'verification_status', 'purpose',
            'addr_state', 'delinq_2yrs', 'inq_last_6mths',
            'open_acc', 'pub_rec', 'revol_bal', 'revol_util'
        ]
        
        available_features = [f for f in key_features if f in self.df.columns]
        missing_features = [f for f in key_features if f not in self.df.columns]
        
        print(f"✅ Available key features ({len(available_features)}):")
        for feature in available_features:
            print(f"  - {feature}")
            
        if missing_features:
            print(f"\n❌ Missing features ({len(missing_features)}):")
            for feature in missing_features:
                print(f"  - {feature}")
        
        # Analyze numeric features
        numeric_features = self.df[available_features].select_dtypes(
            include=[np.number]).columns.tolist()
        
        if numeric_features:
            print(f"\n📊 NUMERIC FEATURES SUMMARY:")
            print(self.df[numeric_features].describe())
            
    def correlation_analysis(self):
        """Analyze correlations with target variable."""
        print("\n" + "="*60)
        print("🔗 CORRELATION WITH TARGET")
        print("="*60)
        
        # Calculate correlations with target
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = self.df[numeric_cols].corrwith(self.df['target']).abs()
        correlations = correlations.sort_values(ascending=False)
        
        print("Top 15 features correlated with loan default:")
        for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
            print(f"{i:2d}. {feature:<25} {corr:.4f}")
            
    def data_quality_report(self):
        """Generate data quality report for ML model preparation."""
        print("\n" + "="*60)
        print("📋 DATA QUALITY REPORT")
        print("="*60)
        
        # Missing values analysis
        missing_severe = self.df.isnull().sum()
        missing_severe = missing_severe[missing_severe > len(self.df) * 0.5]
        
        if len(missing_severe) > 0:
            print(f"⚠️  Features with >50% missing values ({len(missing_severe)}):")
            for col, missing in missing_severe.items():
                pct = (missing / len(self.df)) * 100
                print(f"  - {col}: {pct:.1f}% missing")
                
        # Categorical features with many unique values
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        high_cardinality = []
        
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            if unique_count > 50:
                high_cardinality.append((col, unique_count))
                
        if high_cardinality:
            print(f"\n🏷️  High cardinality categorical features:")
            for col, count in high_cardinality:
                print(f"  - {col}: {count} unique values")
                
        print(f"\n✅ Dataset ready for preprocessing!")
        print(f"📈 Target: Achieve 92% accuracy on loan approval prediction")
        
    def run_exploration(self):
        """Run complete data exploration pipeline."""
        print("🚀 Starting Digital Lending Accelerator Data Exploration")
        print("🎯 Goal: Achieve 92% accuracy in loan approval prediction")
        
        if not self.load_data():
            return
            
        self.basic_info()
        self.analyze_target()
        self.key_features_analysis()
        self.correlation_analysis()
        self.data_quality_report()
        
        print(f"\n🎉 Exploration complete! Ready for ML model development.")
        
        return self.df

if __name__ == "__main__":
    # Run data exploration
    explorer = LendingClubExplorer()
    df = explorer.run_exploration()
