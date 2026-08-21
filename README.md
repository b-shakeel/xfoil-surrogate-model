# XFOIL Surrogate Model

A machine learning pipeline that predicts airfoil aerodynamic performance without running full XFOIL simulations. Built to enable rapid screening of wing designs before committing to high-fidelity analysis.

## Why this project

Running XFOIL (or any CFD software) might be fast for a one-off calculation, but running it across hundreds of geometries and flow conditions is slow. I wanted to see if a neural network trained on a bunch of XFOIL results could be used to predict performance instantly, basically a surrogate model, something used in the real world (check out [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) for a fully-fledged, production example).

I'm currently an aerospace engineering student, and this started as a summer project to build something that targeted aerospace, automation, and ML, all areas I want to specialize in. Instead of using an existing wrapper for XFOIL, I custom built one because I wanted to understand every step of the process.

## Status

**In progress** — Phase 4 of 5.

- [x] **Phase 1: Environment setup** — got XFOIL compiled and running on macOS (Apple Silicon) via gfortran/gcc and XQuartz. The Python code should run fine on Windows/Linux machines, but the XFOIL setup will be different (haven't tested this myself yet).
- [x] **Phase 2: Python wrapper** — `xfoil_wrapper.py` runs both NACA 4-digit and `.dat` coordinate files, with timeout protection and handling of partial convergence failures
- [x] **Phase 3: Automation loop** — `run_sweep.py` and `download_uiuc_data.py` generate a full dataset covering both real UIUC airfoils and a systematic NACA 4-digit parametric grid
- [ ] **Phase 4: Neural network** *(current)* — building & training a surrogate model (scikit-learn → PyTorch) on the generated dataset
- [ ] **Phase 5: Validation & analysis** — comparing surrogate predictions against held-out XFOIL runs, writeup

## Results so far

Running the full sweep produced `production_sweep.csv`:
- **17,977 rows** across 9 columns (`airfoil, reynolds, mach, alpha, CL, CD, CDp, CM, source`)
- **113 unique airfoils** total, drawn from two sources (NACA & UIUC, tracked via the `source` column):
  - **73 real UIUC airfoils** (random sample, seed=42) → 10,948 rows
  - **40 NACA 4-digit airfoils**, systematically varied across camber, camber position, and thickness → 7,029 rows
- **Zero NaNs**, verified correct dtypes throughout

## How it works

1. **`xfoil_wrapper.py`** drives XFOIL via subprocess, feeding it command sequences to load an airfoil (NACA 4-digit or a `.dat` coordinate file), set flow conditions (Reynolds number, Mach number, angle of attack), and run a polar sweep.
2. **`download_uiuc_data.py`** pulls the full UIUC Selig-format airfoil coordinate database (~1,650 `.dat` files) as a zip and extracts it locally to `uiuc_airfoils/coord_seligFmt/`.
3. **`run_sweep.py`** generates the full dataset by running an automated sweep across both the sampled UIUC airfoils and a systematic NACA grid, at multiple Reynolds numbers. Failures are logged and skipped rather than treated as fatal, and results are written incrementally (one run at a time), so an interrupted sweep doesn't lose completed work.
4. Each individual XFOIL run happens in an isolated temp directory (`tempfile.TemporaryDirectory`) to avoid XFOIL's polar-file append behavior contaminating results across runs.
5. *(Upcoming)* Training a neural network on `production_sweep.csv` to predict lift/drag polars directly from airfoil geometry and flow conditions.

## Notable engineering details

Some things I ran into while building this that are worth pointing out:

- **XFOIL's menu system only works if you're in the right state.** It's a Fortran-based command-line kind of tool, and commands only work if you're in the right submenu. It fails silently if you're not. For example, `QUIT` doesn't work directly from the `.OPERv` submenu, you have to send a blank line to back out first. Debugging this meant a lot of trial and error watching XFOIL's actual output instead of trusting that a command *should* have worked.
- **Long file paths led to errors on macOS.** XFOIL's Fortran backend uses fixed-length string buffers, and macOS's longer temp directory paths were getting cut mid-string, corrupting file reads. Fixed it by running XFOIL with `cwd` set to a short working directory and using relative filenames only instead of full paths.
- **A CSV bug corrupted part of the dataset.** After adding NACA airfoil support to `run_sweep.py`, the code that appends new rows assumed columns were in the same order instead of checking the header. Some rows got written into the wrong columns without any error being raised. Only noticed after some values didn't make any sense. Fixed it by filtering back to the original UIUC-only data and re-running the NACA sweep with a fix in place.
- **Tolerates partial convergence errors.** Some airfoils (like `goe590` at high Reynolds numbers) consistently failed to converge, no matter how many iterations. The sweep logs things like this and keeps moving instead of crashing. Ensures a multi-hour run won't crash because of a bad airfoil.
- Fixed random seed (42) for reproducible airfoil sampling.
- Uses UIUC Selig-format coordinate database (`coord_seligFmt`) to avoid errors from format mismatches.

## Stack

- **Simulation:** XFOIL (compiled from source, [christophe-david/XFOIL_compilation](https://github.com/christophe-david/XFOIL_compilation))
- **Automation & data:** Python, pandas, numpy, subprocess
- **ML (upcoming):** scikit-learn, PyTorch
- **Data source:** [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html)

## Running it

```bash
# One-time: download the UIUC airfoil coordinate database
python3 download_uiuc_data.py

# Run the full sweep (generates production_sweep.csv)
python3 run_sweep.py
```

The repo already includes `production_sweep.csv` and the downloaded airfoil data, so you can skip straight to inspecting results or building on top of them without re-running the sweep.