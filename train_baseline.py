import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("production_sweep.csv", dtype={"airfoil": str})
shape_df = pd.read_csv("shape_features.csv", dtype={"airfoil": str})

df = df.merge(shape_df, on="airfoil", how="left")

df["log_re"] = np.log10(df["reynolds"])

feature_cols = ["alpha", "log_re", "max_thickness", "thickness_loc", "max_camber", "camber_loc"]
target_cols = ["CL", "CD", "CM"]

X = df[feature_cols].values
y = df[target_cols].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["airfoil"]))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

train_airfoils = set(df["airfoil"].values[train_idx])
test_airfoils = set(df["airfoil"].values[test_idx])

x_scaler = StandardScaler().fit(X_train)
X_train_scaled = x_scaler.transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_scaler = StandardScaler().fit(y_train)
y_train_scaled = y_scaler.transform(y_train)

model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", max_iter=2000, early_stopping=True, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_scaler.inverse_transform(y_pred_scaled)

for i, name in enumerate(target_cols):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f"{name}: R^2 = {r2:.4f}, MAE = {mae:.5f}")