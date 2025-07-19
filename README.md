# Housing Price Prediction: Preprocessing Impact Demo

This repository contains a demonstration project showcasing how advanced preprocessing techniques can significantly improve machine learning model performance. The project compares two approaches to housing price prediction using the Sberbank Housing dataset.

## 🎯 Project Overview

The goal of this project is to demonstrate the dramatic impact that proper data preprocessing can have on model performance. We compare:

- **Baseline Model** (`base.ipynb`/`base.py`): Minimal preprocessing approach
- **Advanced Model** (`advanced.ipynb`/`advanced.py`): Comprehensive preprocessing pipeline

**Key Result**: The advanced preprocessing approach achieved a **~27% reduction in RMSE** compared to the baseline model.

## 📊 Dataset

The project uses the **Sberbank Housing Dataset**, which contains real estate data from Moscow, Russia. The dataset includes:

- **Target Variable**: `price_doc` (house price in rubles)
- **Features**: 20+ variables including square footage, location data, building characteristics, and proximity to amenities
- **Size**: ~27,000 records after preprocessing

### Data Files
- `Data/sberbank_housing.csv` - Main dataset (preprocessed subset)
- `Data/train.csv` - Original training data
- `Data/test.csv` - Original test data

## 🏗️ Project Structure

```
untappedEnergy/
├── base.ipynb              # Baseline model with minimal preprocessing
├── base.py                 # Baseline model script
├── advanced.ipynb          # Advanced model with comprehensive preprocessing
├── advanced.py             # Advanced model script
├── main.py                 # Complete preprocessing pipeline
├── baseline_rmse.txt       # Baseline model RMSE result
├── presentation.pdf        # Project presentation
└── Data/
    ├── sberbank_housing.csv
    ├── train.csv
    └── test.csv
```

## 🔬 Methodology

### Baseline Approach (`base.ipynb`/`base.py`)
- **Minimal preprocessing**: Only basic cleaning and log transformation of target
- **Feature selection**: Uses only complete numeric features
- **Model**: Ridge regression with minimal hyperparameter tuning
- **Limitations**: Ignores categorical data, cannot handle missing values effectively

### Advanced Approach (`advanced.ipynb`/`advanced.py`)
- **Comprehensive preprocessing pipeline**:
  - **Data cleaning**: Outlier removal, duplicate elimination
  - **Missing value imputation**: Hybrid approach (KNN for numeric, mode for categorical)
  - **Feature engineering**: Living efficiency, room size, building age, floor ratio
  - **Categorical encoding**: Smart encoding based on cardinality
  - **Scaling**: StandardScaler for all features
- **Model**: Ridge regression with optimized hyperparameters

## 📈 Results

| Model | RMSE (Dollars) | Features Used | Preprocessing Level |
|-------|----------------|---------------|-------------------|
| Baseline | $4,595,983 | ~10 features | Minimal |
| Advanced | ~$3,355,000 | All features | Comprehensive |

**Improvement**: ~27% reduction in RMSE through advanced preprocessing

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
```

### Running the Models

1. **Baseline Model**:
   ```bash
   python base.py
   ```

2. **Advanced Model**:
   ```bash
   python advanced.py
   ```

3. **Jupyter Notebooks**:
   - Open `base.ipynb` for baseline approach
   - Open `advanced.ipynb` for advanced approach

## 🔧 Key Preprocessing Techniques Demonstrated

### 1. Missing Value Imputation
- **KNN Imputation**: For numeric features using 5 nearest neighbors
- **Mode Imputation**: For categorical features
- **Domain-specific imputation**: Living space ratios based on neighborhood

### 2. Feature Engineering
- **Living Efficiency**: `life_sq / full_sq`
- **Average Room Size**: `full_sq / num_room`
- **Building Age**: `transaction_year - build_year`
- **Floor Ratio**: `floor / max_floor`
- **Log Transformations**: For skewed features

### 3. Categorical Encoding
- **Frequency Encoding**: For high-cardinality features (sub_area)
- **One-Hot Encoding**: For low-cardinality features
- **Ordinal Encoding**: For medium-cardinality features

### 4. Data Cleaning
- **Outlier Removal**: Based on domain knowledge
- **Duplicate Elimination**: Remove exact duplicates
- **Data Type Conversion**: Proper numeric types

## 📊 Key Insights

1. **Feature Engineering Impact**: Domain-specific features like living efficiency and building age significantly improve predictions
2. **Missing Value Strategy**: Sophisticated imputation methods preserve data relationships
3. **Categorical Data**: Proper encoding of categorical variables captures important market information
4. **Data Quality**: Cleaning outliers and duplicates improves model stability

## 🎓 Learning Objectives

This project demonstrates:
- The importance of thorough data preprocessing
- How domain knowledge can guide feature engineering
- The impact of missing value strategies
- Effective categorical variable handling
- The relationship between data quality and model performance

## 📝 Files Description

- **`main.py`**: Complete preprocessing pipeline with detailed comments
- **`base.py`**: Minimal preprocessing baseline model
- **`advanced.py`**: Advanced preprocessing with comprehensive pipeline
- **`baseline_rmse.txt`**: Baseline model RMSE result for comparison
- **`presentation.pdf`**: Project presentation slides

## 🤝 Contributing

This is a demonstration project. Feel free to experiment with different preprocessing techniques or models to further improve performance.

## 📄 License

This project is for educational and demonstration purposes.

---

**Note**: This project demonstrates the significant impact that proper data preprocessing can have on machine learning model performance, with a focus on practical, domain-specific techniques rather than complex algorithms. 