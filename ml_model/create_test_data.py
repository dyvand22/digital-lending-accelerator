"""
Create synthetic loan data for testing the Digital Lending Accelerator
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_synthetic_loan_data():
    """Create synthetic loan data for testing."""
    
    print("🔄 Creating synthetic loan data for testing...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of samples
    n_samples = 10000
    
    # Create synthetic features
    data = {
        'loan_amnt': np.random.normal(15000, 8000, n_samples).clip(1000, 40000),
        'term': np.random.choice([36, 60], n_samples),
        'int_rate': np.random.normal(12, 4, n_samples).clip(5, 25),
        'annual_inc': np.random.lognormal(11, 0.5, n_samples).clip(20000, 200000),
        'dti': np.random.normal(15, 8, n_samples).clip(0, 40),
        'fico_range_low': np.random.normal(700, 50, n_samples).clip(300, 850),
        'fico_range_high': None,  # Will be calculated
        'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', 
                                      '4 years', '5 years', '6 years', '7 years',
                                      '8 years', '9 years', '10+ years'], n_samples),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, 
                                         p=[0.4, 0.3, 0.3]),
        'verification_status': np.random.choice(['Verified', 'Source Verified', 'Not Verified'], 
                                              n_samples, p=[0.3, 0.3, 0.4]),
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement',
                                   'major_purchase', 'auto', 'other'], n_samples,
                                  p=[0.6, 0.15, 0.1, 0.05, 0.05, 0.05]),
        'delinq_2yrs': np.random.poisson(0.3, n_samples).clip(0, 10),
        'inq_last_6mths': np.random.poisson(1, n_samples).clip(0, 10),
        'open_acc': np.random.normal(10, 5, n_samples).clip(1, 30).astype(int),
        'pub_rec': np.random.poisson(0.1, n_samples).clip(0, 5),
        'revol_bal': np.random.lognormal(8, 1, n_samples).clip(0, 100000),
        'revol_util': np.random.normal(30, 20, n_samples).clip(0, 100),
        'total_acc': np.random.normal(25, 10, n_samples).clip(1, 80).astype(int)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate fico_range_high (typically 20 points higher)
    df['fico_range_high'] = df['fico_range_low'] + 20
    df['fico_range_high'] = df['fico_range_high'].clip(300, 850)
    
    # Create loan status based on risk factors
    # Higher risk = higher chance of default
    risk_score = (
        (df['int_rate'] > 15) * 0.3 +
        (df['dti'] > 25) * 0.2 +
        (df['fico_range_low'] < 650) * 0.4 +
        (df['delinq_2yrs'] > 0) * 0.3 +
        (df['revol_util'] > 70) * 0.2 +
        np.random.normal(0, 0.1, n_samples)  # Add some randomness
    )
    
    # Convert risk score to binary outcome (0 = Fully Paid, 1 = Default)
    default_prob = 1 / (1 + np.exp(-5 * (risk_score - 0.5)))  # Sigmoid function
    defaults = np.random.random(n_samples) < default_prob
    
    df['loan_status'] = np.where(defaults, 'Charged Off', 'Fully Paid')
    
    print(f"✅ Created {len(df):,} synthetic loan records")
    print(f"📊 Default rate: {(defaults.sum() / len(df) * 100):.1f}%")
    
    # Create directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the data
    output_file = raw_dir / "accepted_2007_to_2018Q4.csv.gz"
    df.to_csv(output_file, compression='gzip', index=False)
    
    print(f"✅ Saved test data to: {output_file}")
    
    return df

if __name__ == "__main__":
    df = create_synthetic_loan_data()
    print("\n🎯 Sample data preview:")
    print(df.head())
    print(f"\n📊 Data shape: {df.shape}")
    print(f"📊 Features: {list(df.columns)}")
