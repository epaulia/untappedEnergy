import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# train = pd.read_csv(r"Data\train.csv")

# train = train[["id", "timestamp", "price_doc", "full_sq", "life_sq", "kitch_sq", "floor", "max_floor", "build_year", "num_room", "state", "product_type", "ecology", "sub_area", "raion_popul", "kindergarten_km", "school_km", "park_km", "railroad_km", "metro_min_walk"]].iloc[:27000,:]

#train.to_csv(r"Data\sberbank_housing.csv")

house_data = pd.read_csv(
    r"Data\sberbank_housing.csv",  
    index_col = 0, 
    low_memory=True
    )

#housekeeping
house_data.columns = [c.lower().strip().replace(" ", "_") for c in house_data.columns] 

# dtypes check
print(house_data.dtypes)

# Let's see some basic info
print(house_data.dtypes)

#check NA's
print(house_data.isna().sum())

# Let's address any column changes for the columns with no NA's. Changing types with NA's in the column can open au room forsome errors

# Let's convert date string into a proper datetime type
house_data["timestamp"] = pd.to_datetime(house_data["timestamp"])

# Let's convert some float columns to int, which are logically more appropriate
house_data["build_year"] = house_data["build_year"].astype("Int64")
house_data["floor"] = house_data["floor"].astype("Int64")
house_data["max_floor"] = house_data["max_floor"].astype("Int64")
house_data["num_room"] = house_data["num_room"].astype("Int64")
house_data["state"] = house_data["state"].astype("Int64")

# Let's decrease decimals number to 3/trimming down precision for some km distance columns to ease up further calculations (e.g. for gradieant decscent). Also check consistency for km/meters and min/sec
house_data["kindergarten_km"] = np.floor(house_data["kindergarten_km"] * 1000)/1000

house_data["school_km"] = np.floor(house_data["school_km"] * 1000)/1000

house_data["park_km"] = np.floor(house_data["park_km"] * 1000)/1000

house_data["railroad_km"] = np.floor(house_data["railroad_km"] * 1000)/1000

# Let's do the same with metro min walk column
house_data["metro_min_walk"] = np.floor(house_data["metro_min_walk"] * 1000)/1000

# Let's address the NA's now. 
na_cols = [c for c in house_data.columns if house_data[c].isna().sum()>0]

na_cols_threshold = [c for c in house_data.columns if ((house_data[c].isna().sum()/len(house_data))>=0.15)]

na_cols_diff = [c for c in na_cols if c not in na_cols_threshold]

# Let's drop the na's in the columns where na's are not a signifact portion of the column values

house_data.dropna(subset=na_cols_diff, inplace = True)

print(f"Columns, where we need to address NA's: {na_cols_threshold}")

# For life_sq column - let's assign the values for NA's based on the living area ration for the sub_area where life_sq is not NULL accordingly

# Identify all neighborhoods with missing living space sizes
# and compare them to neighborhoods with at least one known living space size.
# If any neighborhood only appears in the missing set and never in the known set,
# we will need a fallback imputation strategy for those cases.

# 1) Gather the set of all sub_areas with at least one known living space size
no_na = set(house_data[~house_data["life_sq"].isna()]["sub_area"].values)

# 2) Gather the set of all sub_areas where living space size is missing
na =    set(house_data[house_data["life_sq"].isna()]["sub_area"].values)

# 3) Find any neighborhood that has no known living space measurements at all
missing_subs = na - no_na

if missing_subs:
    print("Sub_areas with all life_sq missing:", missing_subs)
else:
    print("Every sub_area with missing life_sq also has at least one non‐missing example—safe to proceed.")

# Let's see what sub_area we have where life_sq is NA 
sub_w_nas = house_data[house_data["life_sq"].isna()]["sub_area"].value_counts().index.to_list()

# Let's filter our df to those sub_area values as well as filter out any 0's for the full_sq since it'll go to the denominator in our ratio calculation
masked_temp = house_data[house_data["sub_area"].isin(sub_w_nas) & house_data["full_sq"].ne(0)].dropna(subset=["life_sq","full_sq"])

# Now, let's calculate additional column with the ratio values (life sq) / (full sq)
masked_temp["ratio"] = masked_temp["life_sq"] / masked_temp["full_sq"]

# Let's get the ratio's median for every repeating sub_area
ratio_by_sub = masked_temp.groupby("sub_area")["ratio"].median()

# Now, let's filter out initial house_data df to only where there are NA's in the life_sq column, to fill them in
mask_to_fill = house_data["life_sq"].isna() & house_data["sub_area"].isin(ratio_by_sub.index.to_list())

# let's fill the NAs in, by multiplying full_sq value by our calculated ratio, and rounding it to the nearest 0
house_data.loc[mask_to_fill, "life_sq"] = (
    (house_data.loc[mask_to_fill, "full_sq"] 
     * house_data.loc[mask_to_fill, "sub_area"].map(ratio_by_sub))
    .round(0)
)

# Let's apply the same logic for kitch_sq

# Identify all neighborhoods with missing kitchen sizes
# and compare them to neighborhoods with at least one known kitchen size.
# If any neighborhood only appears in the missing set and never in the known set,
# we will need a fallback imputation strategy for those cases.

# 1) Gather the set of all sub_areas with at least one known kitchen size
no_na = set(house_data[~house_data["kitch_sq"].isna()]["sub_area"].values)

# 2) Gather the set of all sub_areas where kitchen size is missing
na =    set(house_data[house_data["kitch_sq"].isna()]["sub_area"].values)

# 3) Find any neighborhood that has no known kitchen measurements at all
missing_subs = na - no_na

if missing_subs:
    print("Sub_areas with all kitch_sq missing:", missing_subs)
else:
    print("Every sub_area with missing kitch_sq also has at least one non‐missing example—safe to proceed.")

sub_kitch_w_nas = house_data[house_data["kitch_sq"].isna()]["sub_area"].value_counts().index.to_list()

masked_kich = house_data[house_data["sub_area"].isin(sub_kitch_w_nas) & house_data["full_sq"].ne(0)].dropna(subset=["kitch_sq","full_sq"])

masked_kich["ratio"] =  masked_kich["kitch_sq"] / masked_kich["full_sq"]

ratio_kitch_by_sub = masked_kich.groupby("sub_area")["ratio"].median()

mask_kitch_to_fill = house_data["kitch_sq"].isna() & house_data["sub_area"].isin(ratio_kitch_by_sub.index.to_list())

house_data.loc[mask_kitch_to_fill, "kitch_sq"] = (
    house_data.loc[mask_kitch_to_fill, "full_sq"]
    * house_data.loc[mask_kitch_to_fill, "sub_area"].map(ratio_kitch_by_sub)
).round(0)

# Let's proceed to the max_floor

# For max_floor, it seems like there is no real other column with appropriate pedictive nature to fill in the NAs. Let's use KNNImputer

num_features = [f for f in house_data.columns if is_numeric_dtype(house_data[f]) and f not in ["id", "price_doc"]]

df_numeric = house_data[num_features]

# Instantiate the imputer
#   n_neighbors=5: each missing value is filled by the weighted average of its 5 “nearest” apartments in feature‐space.
#   weights="distance": closer apartments have more say. Euclidean distance
imputer = KNNImputer(n_neighbors=5, weights="distance")

# Fit & transform
imputed_array = imputer.fit_transform(df_numeric)

# Pull the imputed max_floor back into your DataFrame
# The columns stay in the same order as `features`
imputed_max_floor = np.round(imputed_array[:, num_features.index("max_floor")]).astype(int)

house_data["max_floor"] = imputed_max_floor

# Let's proceed to the next one
na_cols = [c for c in house_data.columns if house_data[c].isna().sum()>0]
print(f"Let's proceed to {na_cols[0]}")

# Let's do the same for build_year NAs

house_data["build_year"] = np.round(imputed_array[:, num_features.index("build_year")]).astype(int)

# We can apply similar logic to the life_sq imputation for num_room and state, but slightly more complex.
# For hese 2 let's group by multiple external features, that might be representative in regards to num_room and state.
# Let's group by life_sq, kitch_sq, product_type, ecology, sub_area, raion_popul, all the km variables and metro_min_walk

external_features = ["life_sq", "sub_area", "raion_popul", "ecology"]

# 1) Compute medians on the non‐missing subset
medians = (
    house_data
      .loc[house_data["num_room"].notna(), external_features + ["num_room"]]
      .groupby(external_features)["num_room"]
      .median()
      .rename("median_num_room")
      .astype(int)
      .reset_index()
)

house_data = house_data.merge(
    medians, 
    on=external_features, 
    how="left"
    )

house_data["num_room"] = house_data["num_room"].fillna(house_data["median_num_room"])

# After imputing the NA bu the medians grouped by out 4 external columns, we still have ~1400 NAs for the num_room column. It's much better than >9000, even tho stillnot ideal. Since we have over 26,000 records, dropping just slightly over 1000 (~4%) of NA rows is not too bad. Let's do it and proceed

house_data = house_data.dropna(subset=["num_room"]).drop(columns=["median_num_room"])

# Let's perform similar operation with state column, assigning build_year, sub_area, ecology and product_type to our external features to which to group by

external_features = ["build_year", "ecology"]

# Compute medians on the non‐missing subset
medians = (
    house_data
      .loc[house_data["state"].notna(), external_features + ["state"]]
      .groupby(external_features)["state"]
      .median()
      .rename("median_state")
      .astype(int)
      .reset_index()
)

house_data = house_data.merge(
    medians, 
    on=external_features, 
    how="left"
    )

house_data["state"] = house_data["state"].fillna(house_data["median_state"])

# After the imputation we still have ~2400 NAs, which is still below out threshold of 15% we decided on in the beginning, so we'll drop the remaining NAs

house_data = house_data.dropna(subset=["state"]).drop(columns=["median_state"])

# Let's deal with non-numeric columns now, since we need all the columns to be numeric for our future Regression model. We have product_type, ecology, sub_area being string columns

print(f"Unique values for product_type: {house_data['product_type'].unique()}")

# We have just 2 values, so "binary" OHE is completely fine here

house_data = pd.get_dummies(
    house_data, 
    columns=["product_type"], 
    drop_first=True, 
    dtype=int).rename(columns={"product_type_OwnerOccupier":"product_type"})

print(f"Unique values for ecology: {house_data['ecology'].unique()}")

# We can see that the story with ecology is quite similar, with some "no data" values. Even though it literally says "no data" - it can hold some data in itself - let's keep it

house_data = pd.get_dummies(
    house_data, 
    columns=["ecology"], 
    drop_first=True,
    dtype=int
    )
house_data.rename(columns={"ecology_no data":"ecology_no_data"}, inplace = True)

# And let's look into sub_area
print(f"Unique values for sub_area: {len(house_data['sub_area'].unique())}")

# We have too many unique string values for simple OHE to work. Let's use Frequency Encoding - data leakage-fre, meaningfull (market activity), easy to interpret

frequencies = house_data["sub_area"].value_counts()
house_data["sub_area"] = house_data["sub_area"].map(frequencies)

# Now we have all columns being a numeric type and no NAs in the datset

print(house_data.dtypes)
print(house_data.isna().sum())
print(house_data.shape)

# Let's quickly check if there any duplicates in the dataset
house_data.duplicated().sum() # no duplicates - moving on

# Let's proceed with checking features for outliers and some concearning patterns.
columns_to_inspect = [c for c in house_data.columns if c not in ["id", "timestamp", "price_doc"]]

# Let's look at boxplots

nrows=4
ncols=5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, columns_to_inspect):
    
    ax.boxplot(house_data[col])
    ax.set_title(col)
    
for ax in axes_flat[len(columns_to_inspect):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

# Let's look at histograms

nrows=4
ncols=5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
axes_flat = axes.flatten()

for ax, col in zip(axes_flat, columns_to_inspect):
    
    ax.hist(house_data[col], bins=50)
    ax.set_title(col)
    
for ax in axes_flat[len(columns_to_inspect):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()

# Ok, we can see the same story pretty much as we have seen with the boxplots. Let's address the issues

# we can see that for build_year we have a couplem fo some extremely large values, let's try to catch them

house_data[house_data["build_year"] > house_data["build_year"].mean()]

# Ok, row index 8439 has a wrong year format - it should be just 2005, let's fix it
house_data.loc[8439, "build_year"] = 2005

# The other one indexed 13562 just doesn't make sense - let's drop it
house_data = house_data.drop(13562)

#Let's re-check the histogram
house_data["build_year"].hist()

# We can see that some build_year numbers just don't make sense - they are below 1800. Let's assign NAs to anything less than 1950 and then impute the feature

house_data[house_data["build_year"] < 1950] = pd.NA

build_year_med_by_area = (
    house_data
      .loc[house_data["build_year"].notna()]
      .groupby("sub_area")["build_year"]
      .median()
)

house_data["build_year"] = house_data["build_year"].fillna(
    house_data["sub_area"].map(build_year_med_by_area)
)

# There arestill some NAs after the imputation - let's drop those
house_data.dropna(subset=["build_year"], inplace=True)

#Let's re-check the histogram
house_data["build_year"].hist()

# Good - build year in a better shape now, let's move on to full_sq, life_sq and kitch_sq
house_data[house_data["full_sq"] > 1000]

# For full_sq, we can see that there is just one vlaue with extremely big number - over 5000 sq ft, let's drop it

house_data.drop(house_data[house_data["full_sq"] > 1000].index, inplace=True)

house_data["full_sq"].hist()
# Still very highly right-skewed, we'll adress it on the scaling stage

house_data[house_data["life_sq"] > 1000]
# Same story with life_sq

house_data.drop(house_data[house_data["life_sq"] > 1000].index, inplace=True)
house_data["life_sq"].hist()

# And kitch_sq
house_data[house_data["kitch_sq"] > 1000]

house_data.drop(house_data[house_data["kitch_sq"] > 1000].index, inplace=True)
house_data["kitch_sq"].hist()

# floor and max_floor now. Both features show some extreme values, even tjo they seem possible.Let's check values > 1.5*IqR

Q1, Q3 = house_data["floor"].quantile([0.25, 0.75])

IQR = Q3 - Q1

lower, upper = max(Q1 - 1.5*IQR, 0), Q3 + 1.5*IQR

floor_outliers = house_data[(house_data["floor"] < lower) | (house_data["floor"] > upper)]
floor_outliers["build_year"]
floor_outliers["floor"].max()

# Thse outlier values are totally fine, however sicne there are not as many of them , they'll be considered as "outliers" statistically - we can confidently apply log transfrom on the floor column (good for highly right-skewed values too). Also, floors of 0s might be possible - let's add epsilon to avoid log transform errors.
# epsilon = 1e-5

# house_data["log1p_floor"] = np.log1p(house_data["floor"] + epsilon)

# max_floor seems to follow exact same logic as floor variable

# num_room looks like it might have some extreme number of rooms for a few entries (> 15)
house_data[house_data["num_room"]>15]

# Let's drop these
house_data.drop(house_data[house_data["num_room"]>15].index, inplace=True)
house_data["num_room"].hist()

# State variable looks like having some too out-of-range values too (>30)
house_data[house_data["state"]>15]

# Yes, it looks like the state column is on the scale 0-5 and we have one entry with a value of 33 - let's drop it
house_data.drop(house_data[house_data["state"]>15].index,inplace=True)

#kindergarden_km and school_km look overly right-skewed, while railroad_km and metro_min_walk look highly right-skewed, but perhaps logically those extreme values might make sense - we'll transform later on and re-check
house_data[house_data["kindergarten_km"]>15][["kindergarten_km", "sub_area"]]

# Some values with some "rural" areas. It's just ~40 of them, s let's just drop them for similicity-sake
house_data.drop(house_data[house_data["kindergarten_km"]>15].index, inplace=True)

house_data[house_data["school_km"]>15] #only 15 entries, let's drop them too

house_data.drop(house_data[house_data["school_km"]>15].index, inplace=True)

# Looks like the sanity check is complete. Let's proceed to feature eng now, before we get to the scaling part.

# quick correlation matrix
corr = house_data[columns_to_inspect + ["price_doc"]].corr()
sns.heatmap(corr, vmin=-1, vmax=1, center=0)
plt.show()

# full_sq and life_sq are almost fully interchangeable, let's create a ratio instead. let's also drop "id" column

# Let's derive aan interaction term full_sq * num_room before dropping full_sq. Could proxy for average room size
house_data["full_sq_num_room"] = house_data["full_sq"]*house_data["num_room"]

house_data["life_full_ratio"] = house_data["life_sq"] / house_data["full_sq"]
house_data.drop(columns=["full_sq", "life_sq", "id"], inplace=True)

# also, let's extract year, month, dayofweek, and day from the timestamp column
house_data["trans_year"] = house_data["timestamp"].dt.year
house_data["trans_month"] = house_data["timestamp"].dt.month
house_data["trans_dayofweek"] = house_data["timestamp"].dt.dayofweek
house_data["trans_day"] = house_data["timestamp"].dt.day
house_data.drop(columns=["timestamp"], inplace=True)

# Let's proceed to scaling. Let's look at the distributions again

columns_to_inspect = [c for c in house_data.columns if c != "price_doc"]

nrows=4
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

# quick look at the target variable
house_data["price_doc"].hist()

# log transform looks reasonable here
epsilon = 1e-5
house_data["price_doc"] = np.log1p(house_data["price_doc"] + epsilon)

# Looks like we would want to apply log transform for kitch_sq, floor and max_floor, num_room, all the km variables, and metro_min_walk.
# We can perfrom standard scaling (mean=0, std=1) on build_year, trans_year, trans_month, trans_dayofweek, trans_day
skewed_feats = ["kitch_sq", "floor", "max_floor", "num_room", "kindergarten_km", "school_km", "park_km", "railroad_km", "metro_min_walk"]
gauss_feats = ["build_year", "trans_year", "trans_month", "trans_dayofweek", "trans_day"]

# Establish preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("gauss", StandardScaler(), gauss_feats),
        ("skew", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, validate=True)),
            ("stds", StandardScaler())
        ]), skewed_feats),
        # No transformer for binary flags—they stay as-is
    ],
    remainder="passthrough"  # keep the ecology_* and product_type columns as-is
)

features = [c for c in house_data.columns if c != "price_doc"]

X = house_data[features]
y = house_data["price_doc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # Create the full model pipeline for baselinemodel
# model_pipeline = Pipeline([
#     ("pre", preprocessor),
#     ("reg", LinearRegression())
# ])

# # Fit & evaluate. baseline model

# model_pipeline.fit(X_train, y_train)

# # calc rmse on test
# y_pred = model_pipeline.predict(X_test)
# rmse_baseline = np.sqrt(mean_squared_error(y_pred, y_test))

# Fit RF
pipe = Pipeline([
    ("pre", preprocessor),
    ("reg", RandomForestRegressor(n_jobs=-1, random_state=0))
])

param_grid = {
    "reg__n_estimators": [300, 500, 700],
    #"reg__max_depth": [20, 30]
    }

grid_rf = GridSearchCV(
    pipe, 
    param_grid, 
    cv=5, 
    scoring="neg_root_mean_squared_error", 
    refit=True
)
grid_rf.fit(X, y)

best_params = grid_rf.best_params_
best_rmse  = -grid_rf.best_score_

print("\nGRID SEARCH RF RESULTS")
print(f" Best parameters: {best_params}")
print(f" CV RMSE    : {best_rmse:.3f}")

best_pipe = grid_rf.best_estimator_
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))