import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("BASELINE MODEL - MINIMAL PREPROCESSING")
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

house_data["price_doc"].hist()

# Apply log transform to target (prices are heavily right-skewed)
house_data["price_doc"] = np.log1p(house_data["price_doc"])

# Drop non-predictive columns
house_data = house_data.drop(columns=["timestamp", "id"])

print(f"After basic cleaning: {house_data.shape}")

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
# BASELINE MODEL - MINIMAL PREPROCESSING
# ============================================================================

print("\nBASELINE MODEL (Minimal preprocessing):")

# Only use numeric columns with no missing values for baseline
numeric_complete = X_train.select_dtypes(include=[np.number]).dropna(axis=1)
baseline_model = Ridge(alpha=10.0, random_state=42)
baseline_model.fit(numeric_complete, y_train)

numeric_complete_test = X_test[numeric_complete.columns]
baseline_pred = baseline_model.predict(numeric_complete_test)

# Calculate metrics on log scale
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

# Convert back to dollar scale for interpretation
baseline_pred_dollars = np.expm1(baseline_pred)
y_test_dollars = np.expm1(y_test)
baseline_rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, baseline_pred_dollars))

print(f"   Features used: {len(numeric_complete.columns)} (only complete numeric)")
print(f"   RMSE: {baseline_rmse:.3f} (log scale)")
print(f"   RMSE in dollars: ${baseline_rmse_dollars:,.2f}")

print(f"\nFeatures used in baseline model:")
print(f"{list(numeric_complete.columns)}")

print("\n" + "="*60)
print("BASELINE MODEL LIMITATIONS:")
print("="*60)
print("✗ Only uses complete numeric features")
print("✗ Ignores categorical information")
print("✗ Cannot handle missing values")
print("✗ No feature engineering")
print("✗ Limited feature set")
print(f"✗ Uses only {len(numeric_complete.columns)} out of {X_train.shape[1]} available features")