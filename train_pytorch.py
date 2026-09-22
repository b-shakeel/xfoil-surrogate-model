import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch, copy, joblib
import torch.nn as nn
from model import AirfoilNet
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('production_sweep.csv', dtype={"airfoil": str})
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

x_scaler = StandardScaler().fit(X_train)
X_train_scaled = x_scaler.transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler().fit(y_train)
y_train_scaled = y_scaler.transform(y_train)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

torch.manual_seed(42)

model = AirfoilNet()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 1000

y_test_scaled = y_scaler.transform(y_test)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

train_losses = []
test_losses = []

best_test_loss = float('inf')
best_model_state = None
patience = 100
epochs_without_improvement = 0

for epoch in range(n_epochs):
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor)
        test_losses.append(test_loss.item())
        model.train()
    if test_loss.item() < best_test_loss:
        best_test_loss = test_loss.item()
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}")
        break


    if epoch % 100 == 0:
        print(f"Epoch {epoch}: train_loss: {loss.item():.4f}, test_loss: {test_loss.item():.4f}")

model.load_state_dict(best_model_state)
torch.save(best_model_state, "airfoil_net.pt")
joblib.dump({"x_scaler": x_scaler, "yscaler": y_scaler}, "scalers.joblib")
test_airfoils = df["airfoil"].iloc[test_idx].unique()
pd.DataFrame({"airfoil": test_airfoils}).to_csv("test_airfoils.csv", index=False)

model.eval()
with torch.no_grad():
    y_pred_scaled_tensor = model(X_test_tensor)
y_pred_scaled = y_pred_scaled_tensor.numpy()
y_pred = y_scaler.inverse_transform(y_pred_scaled)

for i, name in enumerate(target_cols):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f"{name}: R^2 = {r2:.4f}, MAE = {mae:.5f}")
print(f"Best model at epoch {best_epoch} with test loss: {best_test_loss:.4f}")