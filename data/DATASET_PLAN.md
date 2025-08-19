# Dataset Plan for Digital Lending Accelerator

## Primary Dataset: Lending Club Loan Data

### Dataset Overview
- **Source**: Kaggle - Lending Club Loan Data
- **URL**: https://www.kaggle.com/datasets/wordsforthewise/lending-club
- **Size**: ~2.2M loan records
- **Time Period**: 2007-2018
- **File Size**: ~1.5GB CSV

### Key Features for 92% Accuracy Target
1. **loan_amnt**: Loan amount requested
2. **term**: Loan term (36/60 months)
3. **int_rate**: Interest rate
4. **annual_inc**: Annual income
5. **dti**: Debt-to-income ratio
6. **fico_range_low/high**: Credit score
7. **emp_length**: Employment length
8. **home_ownership**: Home ownership status
9. **verification_status**: Income verification
10. **purpose**: Loan purpose

### Target Variable
- **loan_status**: 
  - "Fully Paid" = Good loan (0)
  - "Charged Off", "Default" = Bad loan (1)

### Expected Model Performance
- **Baseline (Logistic)**: ~85% accuracy
- **Random Forest**: ~90% accuracy
- **XGBoost**: ~92-95% accuracy ✅ TARGET MET
- **Ensemble**: ~95%+ accuracy

### Alternative Datasets (Backup)
1. **Home Credit Default Risk**
   - Kaggle competition dataset
   - Multiple linked tables
   - Higher complexity

2. **German Credit Dataset**
   - Smaller (1K records)
   - Quick prototyping
   - UCI repository

### Data Processing Plan
1. **Download**: Use Kaggle API
2. **Clean**: Handle missing values, outliers
3. **Engineer**: Create ratio features, risk scores
4. **Split**: 80% train, 20% test
5. **Validate**: Cross-validation for robust metrics

### Storage Structure
```
data/
├── raw/                    # Original downloaded files
├── processed/              # Cleaned datasets
├── features/               # Engineered features
└── models/                 # Trained model artifacts
```

### Success Metrics
- ✅ **92% Accuracy**: Primary goal
- ✅ **90%+ Precision**: Minimize false approvals
- ✅ **85%+ Recall**: Catch most defaults
- ✅ **ROC-AUC > 0.95**: Strong discrimination
