import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("data/output", exist_ok=True)

df = pd.read_csv("data/cleaned_data.csv")

# convert student_group and category to dummy variables
# drop_first avoids the dummy variable trap
group_dummies = pd.get_dummies(df["student_group"], prefix="group", drop_first=True)
category_dummies = pd.get_dummies(df["category"], prefix="cat", drop_first=True)

# use the two prior years of attendance and all three years of student counts as features
# the goal is to predict the 2021-2022 attendance rate
feature_cols = [
    "2020_2021_attendance_rate",
    "2019_2020_attendance_rate",
    "2021_2022_student_count",
    "2020_2021_student_count",
    "2019_2020_student_count",
]

X = pd.concat([df[feature_cols], group_dummies, category_dummies], axis=1)
y = df["2021_2022_attendance_rate"]

# split into training and test sets - 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# train the model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on the test set
y_pred = model.predict(X_test)

# evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Results:")
print(f"  R2:   {r2:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")

# look at which features had the biggest impact
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})
coef_df["abs_coef"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns="abs_coef").reset_index(drop=True)

print(f"\nIntercept: {model.intercept_:.4f}")
print("\nTop 15 features by coefficient size:")
print(coef_df.head(15).to_string(index=False))

coef_df.to_csv("data/output/coefficients.csv", index=False)

# plot actual vs predicted attendance rates
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.4, s=16, color="steelblue", label="Predictions")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
plt.xlabel("Actual attendance rate (normalized)")
plt.ylabel("Predicted attendance rate (normalized)")
plt.title(f"Actual vs. Predicted Attendance\nR2 = {r2:.4f}  |  MAE = {mae:.4f}  |  RMSE = {rmse:.4f}")
plt.legend()
plt.tight_layout()
plt.savefig("data/output/actual_vs_predicted.png", dpi=150)
plt.close()
print("\nSaved: data/output/actual_vs_predicted.png")

# bar chart of the top 15 coefficients
top15 = coef_df.head(15).copy()
colors = ["#d62728" if c < 0 else "#1f77b4" for c in top15["coefficient"]]

plt.figure(figsize=(10, 6))
plt.barh(top15["feature"][::-1], top15["coefficient"][::-1], color=colors[::-1])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coefficient")
plt.title("Top 15 Feature Coefficients\n(blue = positive, red = negative effect on attendance)")
plt.tight_layout()
plt.savefig("data/output/top_coefficients.png", dpi=150)
plt.close()
print("Saved: data/output/top_coefficients.png")

# save the test set predictions to a csv
results = X_test.copy()
results["actual_attendance"] = y_test.values
results["predicted_attendance"] = y_pred
results["residual"] = y_test.values - y_pred
results["district_name"] = df.loc[X_test.index, "district_name"].values
results["student_group"] = df.loc[X_test.index, "student_group"].values

results.to_csv("data/output/predictions.csv", index=False)
print("Saved: data/output/predictions.csv")
