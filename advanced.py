import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

print("="*60)
print("ADVANCED PREPROCESSING MODEL")
print("="*60)

# Load data
house_data = pd.read_csv("Data/sberbank_housing.csv", index_col=0, low_memory=True)

print(f"Original dataset shape: {house_data.shape}")
print(f"Missing values per column:\n{house_data.isnull().sum()}")

# ============================================================================
# BASIC CLEANING (before split)
# ============================================================================

# Clean column names
house_data.columns = [c.lower().strip().replace(" ", "_") for c in house_data.columns]

# Drop non-predictive columns
house_data = house_data.drop(columns=["timestamp", "id"])

# Let's look at boxplots

columns_to_inspect = house_data.select_dtypes(include=[np.number]).columns

# Let's look at histograms
nrows=3
ncols=6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, columns_to_inspect):
    
    ax.hist(house_data[col], bins=50)
    ax.set_title(col)
    
for ax in axes_flat[len(columns_to_inspect):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

# Remove obvious errors and impossible values
house_data = house_data[house_data["full_sq"] <= 1200]  # Remove extreme outliers for full square footage
house_data = house_data[house_data["life_sq"] <= 1000] # Remove extreme outliers for living space square footage
house_data = house_data[house_data["kitch_sq"] <= 1000] # Remove extreme outliers for living space square footage
house_data = house_data[house_data["state"] < 10]
house_data = house_data[house_data["num_room"] <= 15]   # Remove impossible room counts
house_data = house_data[house_data["build_year"] > 1800]  # Remove impossible years
house_data = house_data.drop_duplicates()               # Remove duplicates

# Apply log transform to target (prices are heavily right-skewed)
house_data["price_doc"] = np.log1p(house_data["price_doc"])

print(f"After basic cleaning: {house_data.shape}")

# Let's re-check the histograms
nrows=3
ncols=6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, columns_to_inspect):
    
    ax.hist(house_data[col], bins=50)
    ax.set_title(col)
    
for ax in axes_flat[len(columns_to_inspect):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

# ============================================================================
# SPLIT DATA FIRST
# ============================================================================
X = house_data.drop("price_doc", axis=1)
y = house_data["price_doc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================================
# FOCUSED PRACTICAL PREPROCESSING
# ============================================================================

class HybridImputer(BaseEstimator, TransformerMixin):
    """KNN for numeric, mode for categorical"""
    
    def __init__(self):
        self.knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
        self.modes = {}
        
    def fit(self, X, y=None):
        # Handle categorical first
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            self.modes[col] = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown'
        
        # Fit KNN on numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.knn_imputer.fit(X[numeric_cols])
        
        return self
    
    def transform(self, X):
        X_work = X.copy()
        
        # Fill categorical with modes first
        for col, mode_val in self.modes.items():
            if col in X_work.columns:
                X_work[col] = X_work[col].fillna(mode_val)
        
        # KNN imputation for numeric
        numeric_cols = X_work.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_work[numeric_cols] = self.knn_imputer.transform(X_work[numeric_cols])
        
        return X_work

class SelectiveFeatureEngineer(BaseEstimator, TransformerMixin):
    """Only the most impactful feature engineering"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_work = X.copy()
        epsilon = 1e-5
        
        # Only add features that are likely to be very predictive
        
        # 1. Living space efficiency (strong predictor of price per sqft)
        if 'life_sq' in X_work.columns and 'full_sq' in X_work.columns:
            X_work['living_efficiency'] = X_work['life_sq'] / (X_work['full_sq'] + epsilon)
        
        # 2. Room size (bigger rooms = luxury)
        if 'full_sq' in X_work.columns and 'num_room' in X_work.columns:
            X_work['avg_room_size'] = X_work['full_sq'] / (X_work['num_room'] + epsilon)
        
        # 3. Building age (major price factor)
        if 'build_year' in X_work.columns and 'trans_year' in X_work.columns:
            X_work['building_age'] = X_work['trans_year'] - X_work['build_year']
        
        # 4. Floor desirability (middle floors often preferred)
        if 'floor' in X_work.columns and 'max_floor' in X_work.columns:
            X_work['floor_ratio'] = X_work['floor'] / (X_work['max_floor'] + epsilon)
        
        # 5. Overall area (log transform to handle skewness)
        if 'full_sq' in X_work.columns:
            X_work['log_full_sq'] = np.log1p(X_work['full_sq'])
        
        # That's it - no more features to avoid overfitting
        return X_work

class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Effective categorical encoding without overcomplication"""
    
    def __init__(self):
        self.encodings = {}
        
    def fit(self, X, y=None):
        cat_cols = X.select_dtypes(include=['object']).columns
        
        for col in cat_cols:
            unique_vals = X[col].unique()
            
            if len(unique_vals) > 5:
                # Frequency encode sub_area (market activity indicator)
                freq_map = X[col].value_counts().to_dict()
                self.encodings[col] = ('frequency', freq_map)
            elif len(unique_vals) <= 5:
                # One-hot for low cardinality
                self.encodings[col] = ('onehot', unique_vals)
            else:
                # Ordinal for medium cardinality
                encoding_map = {val: i for i, val in enumerate(unique_vals)}
                self.encodings[col] = ('ordinal', encoding_map)
        
        return self
    
    def transform(self, X):
        X_work = X.copy()
        
        for col, (method, mapping) in self.encodings.items():
            if col not in X_work.columns:
                continue
                
            if method == 'frequency':
                X_work[col] = X_work[col].map(mapping).fillna(1)
            
            elif method == 'ordinal':
                X_work[col] = X_work[col].map(mapping).fillna(-1)
            
            elif method == 'onehot':
                # Simple one-hot encoding
                for val in mapping:
                    X_work[f"{col}_{val}"] = (X_work[col] == val).astype(int)
                X_work = X_work.drop(columns=[col])
        
        return X_work

# ============================================================================
# FOCUSED PREPROCESSING PIPELINE
# ============================================================================

# Much simpler, focused preprocessing
focused_preprocessor = Pipeline([
    ('imputer', HybridImputer()),
    ('feature_engineer', SelectiveFeatureEngineer()), 
    ('encoder', SmartCategoricalEncoder()),
    ('scaler', StandardScaler()) # Standardize all features after encoding
])

# ============================================================================
# MODEL PIPELINE 
# ============================================================================

# Create model pipeline
model_pipeline = Pipeline([
    ('preprocessor', focused_preprocessor),
    ('regressor', Ridge(alpha=1.0, random_state=42))  # Less regularization since fewer features
])

# ============================================================================
# FIT THE PREPROCESSING PIPELINE
# ============================================================================

print("Fitting advanced preprocessing pipeline...")
model_pipeline.fit(X_train, y_train)

# Make predictions
test_pred = model_pipeline.predict(X_test)

# Calculate metrics
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_r2 = r2_score(y_test, test_pred)

# Identify column types for reporting
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print("\nADVANCED PREPROCESSING MODEL:")

print(f"   Features used: {len(numeric_cols) + len(categorical_cols)} (all features)")
print(f"   Test RMSE: {test_rmse:.3f} (log scale)")
print(f"   Test R²: {test_r2:.3f}")

# Convert back to dollar scale for interpretation
test_pred_dollars = np.expm1(test_pred)
y_test_dollars = np.expm1(y_test)

test_rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, test_pred_dollars))

print(f"   Test RMSE in dollars: ${test_rmse_dollars:,.2f}")

print(f"\n   Additional features utilized: {len(categorical_cols)} categorical")
print(f"   Missing value handling: {X_train.isnull().sum().sum()} missing values imputed")

print("\n" + "="*60)
print("ADVANCED PREPROCESSING BENEFITS:")
print("="*60)
print("✓ Handles missing values (median/mode imputation)")
print("✓ Converts categorical data to numeric (smart encoding)")
print("✓ Creates new predictive features (feature engineering)")
print("✓ Standardizes feature scales") 
print("✓ Uses ALL available features (not just complete numeric)")
print("✓ Prevents data leakage (fit only on training data)")
print("✓ Robust to new data (handles unseen categories)")
print("✓ Focused feature engineering (avoids overfitting)")