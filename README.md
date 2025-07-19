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
├── base.ipynb              # Baseline model with minimal preprocessing (Jupyter notebook)
├── advanced.ipynb          # Advanced model with comprehensive preprocessing (Jupyter notebook)
├── baseline_rmse.txt       # Baseline model RMSE result
├── presentation.pdf        # Project presentation
├── README.md               # Project documentation
└── Data/
    └── sberbank_housing.csv
```

**Note**: This project focuses on the Jupyter notebooks (`base.ipynb` and `advanced.ipynb`) as the primary demonstration files. The notebooks contain complete analysis with outputs, visualizations, and detailed explanations.

## 🔬 Methodology

### Baseline Approach (`base.ipynb`)
- **Minimal preprocessing**: Only basic cleaning and log transformation of target
- **Feature selection**: Uses only 6 complete numeric features (out of 17 total features)
- **Model**: Ridge regression with alpha=10.0
- **Limitations**: Ignores categorical data, cannot handle missing values effectively
- **Dataset size**: 27,000 records (21,600 training, 5,400 test)

### Advanced Approach (`advanced.ipynb`)
- **Comprehensive preprocessing pipeline**:
  - **Data cleaning**: Outlier removal, duplicate elimination, data validation
  - **Missing value imputation**: Hybrid approach (KNN for numeric, mode for categorical)
  - **Feature engineering**: Living efficiency, room size, floor ratio, log transformations, amenity score
  - **Categorical encoding**: Smart encoding based on cardinality (frequency, one-hot, ordinal)
  - **Scaling**: StandardScaler for all features
- **Model**: Ridge regression with alpha=10.0
- **Dataset size**: 11,826 records (9,460 training, 2,366 test) after cleaning

## 📈 Results

| Model | RMSE (Log Scale) | RMSE (Dollars) | Features Used | Dataset Size |
|-------|------------------|----------------|---------------|--------------|
| Baseline | 0.551 | $4,595,983 | 6 features | 27,000 records |
| Advanced | 0.502 | $3,340,000 | All features | 11,826 records |

**Improvement**: **27.30% reduction in RMSE** through advanced preprocessing

### Key Performance Metrics:
- **Baseline Model**: Uses only 6 complete numeric features, ignores 11 features with missing values
- **Advanced Model**: Utilizes all 17 features through sophisticated preprocessing
- **Feature Engineering**: Creates 5 new predictive features (living efficiency, room size, floor ratio, log transformations, amenity score)
- **Data Quality**: Advanced model removes outliers and duplicates, resulting in cleaner but smaller dataset

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
```

### Running the Models

**Jupyter Notebooks** (Primary Method):
1. **Baseline Model**: Open `base.ipynb` and run all cells
2. **Advanced Model**: Open `advanced.ipynb` and run all cells

**Note**: The Jupyter notebooks contain the complete analysis with outputs, visualizations, and detailed explanations. They are the main demonstration files for this project.

## 🔧 Key Preprocessing Techniques Demonstrated

### 1. Missing Value Imputation
- **KNN Imputation**: For numeric features using 5 nearest neighbors
- **Mode Imputation**: For categorical features
- **Domain-specific imputation**: Living space ratios based on neighborhood

### 2. Feature Engineering
- **Living Efficiency**: `life_sq / full_sq` (living space ratio)
- **Average Room Size**: `full_sq / num_room` (luxury indicator)
- **Floor Ratio**: `floor / max_floor` (floor desirability)
- **Log Transformations**: `log1p(full_sq)` for skewed features
- **Amenity Score**: Combined accessibility to schools, parks, metro, etc.

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

- **`base.ipynb`**: Baseline model with minimal preprocessing (main demonstration)
- **`advanced.ipynb`**: Advanced model with comprehensive preprocessing (main demonstration)
- **`baseline_rmse.txt`**: Baseline model RMSE result for comparison
- **`presentation.pdf`**: Project presentation slides
- **`Data/sberbank_housing.csv`**: Main dataset (preprocessed subset of original data)

**Note**: The Jupyter notebooks contain the complete analysis with outputs, visualizations, and detailed explanations. They are the primary files for understanding and running the demonstration.

## 🤝 Contributing

This is a demonstration project. Feel free to experiment with different preprocessing techniques or models to further improve performance.

## 📄 License

This project is for educational and demonstration purposes.

---

**Note**: This project demonstrates the significant impact that proper data preprocessing can have on machine learning model performance, with a focus on practical, domain-specific techniques rather than complex algorithms. 