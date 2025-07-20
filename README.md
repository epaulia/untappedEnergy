# Housing Price Prediction: Preprocessing Impact Demo

This repository contains a demonstration project showcasing how better preprocessing techniques can improve machine learning model performance. The project compares two approaches to housing price prediction using the Sberbank Housing dataset.

## ⚠️ Important Disclaimer

**This project is NOT about achieving the lowest possible prediction error.** The absolute RMSE values (in rubles) are high because:

- **Focus**: The goal is to demonstrate the **relative improvement** (27.30%) achieved through better preprocessing
- **Methodology**: Both models use the same simple Ridge regression algorithm
- **Purpose**: Show that systematic data preprocessing can deliver improvements even with basic models
- **Learning Objective**: Emphasize that clean data and proper preprocessing often trump complex algorithms

**Key Takeaway**: The 27.30% improvement demonstrates that investing in data quality and preprocessing can deliver immediate benefits to any ML project.

## 🎯 Project Overview

The goal of this project is to demonstrate the impact that proper data preprocessing can have on model performance. We compare:

- **Baseline Model** (`base.ipynb`): Minimal preprocessing approach
- **Improved Model** (`advanced.ipynb`): Better preprocessing pipeline

**Key Result**: The improved preprocessing approach achieved a **27.30% reduction in RMSE** compared to the baseline model.

### Business Context
This case study addresses a critical industry challenge: **70% of AI project failures are due to data quality issues**, resulting in an estimated **$5T annual revenue loss globally**. This project demonstrates how strategic data preprocessing can deliver immediate, substantial improvements in AI project success rates.

## 📊 Dataset

The project uses the **Sberbank Housing Dataset**, which contains real estate data from Moscow, Russia. The dataset includes:

- **Target Variable**: `price_doc` (house price in rubles)
- **Features**: 20 variables including square footage, location data, building characteristics, and proximity to amenities
- **Size**: ~27,000 records after stripping the dataset down (for compute optimization reasons)

### Data Files
- `Data/sberbank_housing.csv` - Main dataset

### Initial Data State
The raw dataset exhibited typical real-world data quality issues:
- **Missing values**: Up to 50% in some features (life_sq: 5,537, build_year: 12,869, state: 13,067)
- **Inconsistent data types**: Numeric values stored as strings
- **No proper normalization**: Features on different scales
- **Outliers and noise**: In target variable and feature distributions
- **Categorical variables**: Requiring proper encoding strategies

## 🏗️ Project Structure

```
untappedEnergy/
├── base.ipynb              # Baseline model with minimal preprocessing (Jupyter notebook)
├── advanced.ipynb          # Improved model with better preprocessing (Jupyter notebook)
├── requirements.txt        # Python package dependencies
├── baseline_rmse.txt       # Baseline model RMSE result
├── presentation.pdf        # Project presentation
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
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

### Improved Approach (`advanced.ipynb`)
- **Better preprocessing pipeline**:
  - **Data cleaning**: Outlier removal, duplicate elimination, data validation
  - **Missing value imputation**: Hybrid approach (KNN for numeric, mode for categorical)
  - **Feature engineering**: Living efficiency, room size, floor ratio, log transformations, amenity score
  - **Categorical encoding**: Smart encoding based on cardinality (frequency, one-hot, ordinal)
  - **Scaling**: StandardScaler for all features
- **Model**: Ridge regression with alpha=10.0
- **Dataset size**: 11,826 records (9,460 training, 2,366 test) after cleaning

## 📈 Results

| Model | RMSE (Log Scale) | RMSE (Rubles) | Features Used |
|-------|------------------|----------------|---------------|
| Baseline | 0.551 | 4,595,983 | 6 features |
| Improved | 0.502 | 3,340,000 | 17 features |

**Improvement**: **27.30% reduction in RMSE** through better preprocessing

### Key Performance Metrics:
- **Baseline Model**: Uses only 6 complete numeric features, ignores 11 features with missing values
- **Improved Model**: Utilizes all 17 features through better preprocessing
- **Feature Engineering**: Creates 5 new predictive features (living efficiency, room size, floor ratio, log transformations, amenity score)
- **Data Quality**: Improved model removes outliers and duplicates, resulting in cleaner but smaller dataset

### Business Impact
For a housing market application:
- **Baseline RMSE**: ₽5.9M average prediction error
- **Improved RMSE**: ₽4.3M average prediction error
- **Practical benefit**: ₽1.6M more accurate predictions on average

This improvement translates to:
- More accurate property valuations
- Better investment decisions
- Reduced financial risk in real estate transactions

## 🚀 Getting Started

### Prerequisites

**Option 1: Install from requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Install packages individually**
```bash
pip install pandas numpy scikit-learn matplotlib scipy jupyter ipykernel
```

### Required Packages
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **matplotlib**: Data visualization
- **scipy**: Statistical functions
- **jupyter**: Jupyter notebook environment
- **ipykernel**: Python kernel for Jupyter

### Running the Models

**Jupyter Notebooks** (Primary Method):
1. **Baseline Model**: Open `base.ipynb` and run all cells
2. **Improved Model**: Open `advanced.ipynb` and run all cells

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

## 💡 Key Takeaways

1. **Clean data trumps complex algorithms**: 27.30% improvement through preprocessing alone
2. **Domain knowledge guides feature engineering**: Understanding housing markets led to effective feature creation
3. **Pipeline design matters**: Proper architecture prevents common pitfalls like data leakage
4. **Categorical data is valuable**: Smart encoding preserves important information
5. **ROI is immediate**: High-impact improvements with reasonable time investment (~4 hours for comprehensive pipeline)
6. **Preprocessing ROI is exceptional**: Time investment vs. performance gain ratio is very high

**Note**: The Jupyter notebooks contain the complete analysis with outputs, visualizations, and detailed explanations. They are the primary files for understanding and running the demonstration.

## 🤝 Contributing

This is a demonstration project.

---

**Presented by**: This case study was presented by GroupLabs at the "AI-Ready or AI-Hopeful?" workshop for UntappedEnergy, demonstrating practical approaches to improving AI project success rates through systematic data quality practices. 