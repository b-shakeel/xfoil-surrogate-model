import numpy as np
import pandas as pd
import torch, joblib
from sklearn.metrics import r2_score, mean_absolute_error
from model import AirfoilNet

df = pd.read_csv("production_sweep.csv", dtype={"airfoil": str})
shape_df = pd.read_csv("shape_features.csv", dtype={"airfoil": str})
df = df.merge(shape_df, on="airfoil", how="left")
df["log_re"] = np.log10(df["reynolds"])

test_airfoils = pd.read_csv("test_airfoils.csv", dtype={"airfoil": str})["airfoil"]
test_df = df[df["airfoil"].isin(test_airfoils)]

feature_cols = ["alpha", "log_re", "max_thickness", "thickness_loc", "max_camber", "camber_loc"]
target_cols = ["CL", "CD", "CM"]

model = AirfoilNet()
model.load_state_dict(torch.load("airfoil_net.pt"))
model.eval()

scalers = joblib.load("scalers.joblib")
x_scaler = scalers["x_scaler"]
y_scaler = scalers["yscaler"]

X_test = test_df[feature_cols].values
y_test = test_df[target_cols].values

X_test_tensor = torch.tensor(x_scaler.transform(X_test), dtype=torch.float32)
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()
y_pred = y_scaler.inverse_transform(y_pred_scaled)

for i, name in enumerate(target_cols):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f"{name}: R^2 = {r2:.4f}, MAE = {mae:.5f}")