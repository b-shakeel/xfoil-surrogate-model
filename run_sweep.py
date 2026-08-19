import pandas as pd
import os
import random
from xfoil_wrapper import run_polar

def _append_polar_to_csv(polar, csv_path, **metadata):
    """Appends one XFOIL run's polar data to a CSV, tagging each row w/ metadata. Writes header once when the file is created, appends after that."""
    df = pd.DataFrame(polar) #columns alpha, CL, CD, CDp, CM
    for key, value in metadata.items():
        df[key] = value

    file_exists = os.path.exists(csv_path)
    if file_exists:
        existing_cols = pd.read_csv(csv_path, nrows=0).columns.to_list()
        df = df[existing_cols]
    else:
        cols = list(metadata.keys()) + ["alpha", "CL", "CD", "CDp", "CM"]
        df = df[cols]

    df.to_csv(csv_path, mode="a", header=not file_exists, index=False)

def sample_airfoils(airfoil_dir, n, seed=42):
    """Randomly samples n airfoil .dat files from airfoil_dir. Fixed seed"""
    all_files = [f for f in os.listdir(airfoil_dir) if f.endswith(".dat")]
    random.seed(seed)
    sampled = random.sample(all_files, n)
    return [os.path.join(airfoil_dir, f) for f in sampled]

def naca_4digit_grid(camber_vals=(0,2,4,6), position_vals=(2,4,6), thickness_vals=(8,12,15,18)):
    """Builds a systematic grid of 4-digit NACA codes, crossing every camber x position x thickness combination. Camber=0 pairs only w/ position=0."""
    codes = []
    for m in camber_vals:
        if m == 0:
            for xx in thickness_vals:
                codes.append(f"00{xx:02d}")
        else:
            for p in position_vals:
                for xx in thickness_vals:
                    codes.append(f"{m}{p}{xx:02d}")
    return codes

def run_sweep(airfoils, reynolds_list, csv_path, alpha_start=-5, alpha_end=15, alpha_step=0.5, mach=0.0, naca=True, n_iter=100, timeout=60, source=None):
    """Runs run_polar for every (airfoil, reynolds) combo and appends each successful result to csv_path. failed runs logged & skipped"""
    if source is None:
        source = "naca_parametric" if naca else "uiuc"
    for airfoil in airfoils:
        airfoil_name = os.path.splitext(os.path.basename(airfoil))[0]
        for reynolds in reynolds_list:
            print(f"Running {airfoil_name} at Re={reynolds}...")
            try:
                polar = run_polar(airfoil, alpha_start, alpha_end, alpha_step, reynolds, mach=mach, n_iter=n_iter, timeout=timeout, naca=naca)
            except RuntimeError as e:
                print(f"  FAILED: {e}")
                continue
            _append_polar_to_csv(polar, csv_path, airfoil=airfoil_name, reynolds=reynolds, mach=mach, source=source)
            print(f" OK: {len(polar['alpha'])} points converged")

if __name__ == "__main__":
    reynolds_list = [50000, 100000, 200000, 500000, 1000000]
    naca_codes = naca_4digit_grid()
    run_sweep(naca_codes, reynolds_list, "production_sweep.csv", naca=True, n_iter=150, timeout=90)
    results = pd.read_csv("production_sweep.csv", dtype={"airfoil": str})
    print(results.shape)
    print(results["airfoil"].nunique(), "unique airfoils total")