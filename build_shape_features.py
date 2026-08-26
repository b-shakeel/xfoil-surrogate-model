import pandas as pd
import os
from geometry_features import naca4_coordinates, read_selig_dat, compute_shape_features

df = pd.read_csv("production_sweep.csv", dtype={"airfoil": str})

unique_airfoils = df[["airfoil", "source"]].drop_duplicates()

uiuc_dir = "uiuc_airfoils/coord_seligFmt"

shape_rows = []
for _, row in unique_airfoils.iterrows():
    airfoil = row["airfoil"]
    source = row["source"]

    if source == "naca_parametric":
        xu, yu, xl, yl = naca4_coordinates(airfoil)
    else:
        filepath = os.path.join(uiuc_dir, airfoil + ".dat")
        xu, yu, xl, yl = read_selig_dat(filepath)

    features = compute_shape_features(xu, yu, xl, yl)
    features["airfoil"] = airfoil
    shape_rows.append(features)

shape_df = pd.DataFrame(shape_rows)
print(shape_df.head())

shape_df.to_csv("shape_features.csv", index=False)