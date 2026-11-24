"""Task 3: Car Price Prediction"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.makedirs("images", exist_ok=True)
SEED = 42

# ---------------- 1. Load & clean ----------------
df = pd.read_csv("car_data.csv")
print("Shape:", df.shape, "| Missing:", int(df.isna().sum().sum()), "| Duplicates:", df.duplicated().sum())
# exact duplicate rows removed
df = df.drop_duplicates().reset_index(drop=True)   
print("After removing duplicates:", df.shape)
print(df.describe().round(2).T)

# ---------------- 2. Feature engineering ----------------
# Car age is more meaningful than raw year (reference year = newest car in the data + 1)
REF_YEAR = df["Year"].max() + 1
df["Car_Age"] = REF_YEAR - df["Year"]
# Price drop vs. a new car: how much value the car has kept (used price / new price)
df["Price_Retention"] = df["Selling_Price"] / df["Present_Price"]   
# EDA only, NOT used as feature (uses target)
# Car_Name has 98 unique models for ~300 rows -> too sparse to encode reliably, so it is dropped.
# Present_Price and Driven_kms are heavily right-skewed (a few extreme cars).
# Log-transforming them stops those outliers from dominating linear models.
df["Log_Present_Price"] = np.log1p(df["Present_Price"])
df["Log_Driven_kms"] = np.log1p(df["Driven_kms"])
features = ["Car_Age", "Log_Present_Price", "Log_Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner"]
num = ["Car_Age", "Log_Present_Price", "Log_Driven_kms", "Owner"]
cat = ["Fuel_Type", "Selling_type", "Transmission"]
X, y = df[features], df["Selling_Price"]

# ---------------- 3. EDA plots ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["Selling_Price"], kde=True, ax=ax[0]); ax[0].set_title("Selling price (lakhs)")
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Fuel_Type", ax=ax[1]); ax[1].set_title("Selling vs present price")
sns.scatterplot(data=df, x="Car_Age", y="Selling_Price", hue="Transmission", ax=ax[2]); ax[2].set_title("Selling price vs car age")
plt.tight_layout(); plt.savefig("images/eda_overview.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(ax, cat):
    sns.boxplot(data=df, x=c, y="Selling_Price", ax=a); a.set_title(f"Price by {c}")
plt.tight_layout(); plt.savefig("images/price_by_category.png", dpi=120); plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(df[["Selling_Price", "Present_Price", "Car_Age", "Driven_kms", "Owner"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig("images/correlation.png", dpi=120); plt.close()

# ---------------- 4. Split ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

def make(model, log_target=False):
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
    pipe = Pipeline([("pre", pre), ("model", model)])
    if log_target:  # train on log(price) to tame skew, predictions are converted back automatically
        return TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe

models = {
    "Linear Regression": make(LinearRegression()),
    "Linear Regression (log target)": make(LinearRegression(), True),
    "Ridge (log target)": make(Ridge(alpha=1.0), True),
    "Random Forest": make(RandomForestRegressor(n_estimators=300, random_state=SEED)),
    "Gradient Boosting": make(GradientBoostingRegressor(random_state=SEED)),
}

# ---------------- 5. Compare with 5-fold CV, then test ----------------
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)  # repeated CV: more stable than one split on ~300 rows
rows = []
for name, m in models.items():
    r2_cv = cross_val_score(m, X_train, y_train, cv=cv, scoring="r2")
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    rows.append((name, r2_cv.mean(), r2_cv.std(), r2_score(y_test, p),
                 mean_absolute_error(y_test, p), np.sqrt(mean_squared_error(y_test, p))))
res = pd.DataFrame(rows, columns=["Model", "CV R2 (mean)", "CV R2 (std)", "Test R2", "Test MAE", "Test RMSE"]).round(3)
print("\n", res.to_string(index=False))
res.to_csv("model_comparison.csv", index=False)

# ---------------- 6. Final model ----------------
best_name = res.sort_values("CV R2 (mean)", ascending=False).iloc[0]["Model"]
best = models[best_name]
pred = best.predict(X_test)
print(f"\nSelected model: {best_name}")
print(f"Test R2 = {r2_score(y_test, pred):.3f} | MAE = {mean_absolute_error(y_test, pred):.3f} lakhs | "
      f"RMSE = {np.sqrt(mean_squared_error(y_test, pred)):.3f} lakhs")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(y_test, pred, alpha=0.7)
lim = [0, max(y_test.max(), pred.max()) + 1]
ax[0].plot(lim, lim, "r--"); ax[0].set(xlabel="Actual price (lakhs)", ylabel="Predicted price (lakhs)", title=f"Actual vs predicted - {best_name}")
ax[1].scatter(pred, y_test - pred, alpha=0.7); ax[1].axhline(0, color="r", ls="--")
ax[1].set(xlabel="Predicted price (lakhs)", ylabel="Residual", title="Residuals")
plt.tight_layout(); plt.savefig("images/actual_vs_predicted.png", dpi=120); plt.close()

# Permutation importance on raw input features (works for any model, test set)
pi = permutation_importance(best, X_test, y_test, n_repeats=20, random_state=SEED, scoring="r2")
imp = pd.Series(pi.importances_mean, index=features).sort_values()
imp.plot.barh(figsize=(7, 4), title="Permutation importance (drop in R2)")
plt.tight_layout(); plt.savefig("images/feature_importance.png", dpi=120); plt.close()
print("\nFeature importance (drop in test R2 when shuffled):\n", imp.sort_values(ascending=False).round(3))

# ---------------- 7. Predict a new car ----------------
new = pd.DataFrame([{"Car_Age": 5, "Log_Present_Price": np.log1p(8.0), "Log_Driven_kms": np.log1p(40000), "Fuel_Type": "Petrol",
                     "Selling_type": "Dealer", "Transmission": "Manual", "Owner": 0}])
print(f"\nExample: 5-yr-old petrol manual, new price 8 lakhs, 40,000 km -> predicted {best.predict(new)[0]:.2f} lakhs")
