# XFOIL Surrogate Model

A machine learning pipeline that predicts airfoil aerodynamic performance without running full XFOIL simulations. Built to enable rapid screening of wing designs before committing to high-fidelity analysis.

## Why this project

Running XFOIL (or any CFD software) might be fast for a one-off calculation, but running it across hundreds of geometries and flow conditions is slow. I wanted to see if a neural network trained on a bunch of XFOIL results could be used to predict performance instantly, basically a surrogate model, something used in the real world (check out [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) for a fully-fledged, production example).

I'm currently an aerospace engineering student, and this started as a summer project to build something that targeted aerospace, automation, and ML, all areas I want to specialize in. Instead of using an existing wrapper for XFOIL, I custom built one because I wanted to understand every step of the process.

## Status

**In progress** — Phase 5 of 5.

- [x] **Phase 1: Environment setup:** Got XFOIL compiled and running on macOS (Apple Silicon) via gfortran/gcc and XQuartz. The Python code should run fine on Windows/Linux machines, but the XFOIL setup will be different (haven't tested this myself yet).
- [x] **Phase 2: Python wrapper:** `xfoil_wrapper.py` runs both NACA 4-digit and `.dat` coordinate files, with timeout protection and handling of partial convergence failures
- [x] **Phase 3: Automation loop:** `run_sweep.py` and `download_uiuc_data.py` generate a full dataset covering both real UIUC airfoils and a systematic NACA 4-digit parametric grid
- [x] **Phase 4: Neural network:** Building & training a surrogate model (scikit-learn → PyTorch) on the generated dataset
- [ ] **Phase 5: Validation & analysis** *(current)***:** Comparing surrogate predictions against held-out XFOIL runs, writeup & documentation

## Results so far

Running the full sweep produced `production_sweep.csv`:
- **17,977 rows** across 9 columns (`airfoil, reynolds, mach, alpha, CL, CD, CDp, CM, source`)
- **113 unique airfoils** total, drawn from two sources (NACA & UIUC, tracked via the `source` column):
  - **73 real UIUC airfoils** (random sample, seed=42) → 10,948 rows
  - **40 NACA 4-digit airfoils**, systematically varied across camber, camber position, and thickness → 7,029 rows
- **Zero NaNs**, verified correct dtypes throughout

### Model results

Both models predict CL, CD, and CM from 6 inputs: angle of attack, log10(Reynolds), and 4 shape descriptors (max thickness, max camber, and their chordwise locations), computed identically for NACA and UIUC airfoils from actual coordinates, not trusted from airfoil name/code. Mach was dropped as an input (constant at 0 across the whole dataset); CDp was dropped as a target (a component of CD, not independently useful).

Train/test split is **grouped by airfoil** (90 train / 23 test, scikit-learn's `GroupShuffleSplit`), not by row. Otherwise the same airfoil at a nearby angle of attack could appear in both sets, letting the model "cheat" by memorizing shapes instead of learning general geometry-to-performance relationships.

| Target | sklearn MLP baseline | PyTorch (checkpointed) |
|---|---|---|
| CL | R² = 0.866 | R² = 0.925 |
| CD | R² = 0.823 | R² = 0.852 |
| CM | R² = 0.653 | R² = 0.830 |

Results are on airfoils the model never saw during training.

## How it works

1. **`xfoil_wrapper.py`** drives XFOIL via subprocess, feeding it command sequences to load an airfoil (NACA 4-digit or a `.dat` coordinate file), set flow conditions (Reynolds number, Mach number, angle of attack), and run a polar sweep.
2. **`download_uiuc_data.py`** pulls the full UIUC Selig-format airfoil coordinate database (~1,650 `.dat` files) as a zip and extracts it locally to `uiuc_airfoils/coord_seligFmt/`.
3. **`run_sweep.py`** generates the full dataset by running an automated sweep across both the sampled UIUC airfoils and a systematic NACA grid, at multiple Reynolds numbers. Failures are logged and skipped rather than treated as fatal, and results are written incrementally (one run at a time), so an interrupted sweep doesn't lose completed work.
4. Each individual XFOIL run happens in an isolated temp directory (`tempfile.TemporaryDirectory`) to avoid XFOIL's polar-file append behavior contaminating results across runs.
5. **`geometry_features.py`** computes 4 shape descriptors (max thickness, max camber, and their chordwise locations) from airfoil coordinates. An analytic formula for NACA 4-digit codes, parsed `.dat` files for UIUC using the same downstream feature-extraction logic for both, so geometry enters the model consistently regardless of source.
6. **`build_shape_features.py`** runs that extraction once per unique airfoil and saves `shape_features.csv`, kept separate from the XFOIL sweep so shape features can be recomputed without rerunning simulations.
7. **`train_baseline.py`** merges shape features into the XFOIL results and trains a scikit-learn `MLPRegressor` baseline.
8. **`train_pytorch.py`** trains the same problem in PyTorch with a manually-written training loop. Initially trained for a fixed 1000 epochs with no safeguards, it became clear from tracking train vs. test loss that the model overfits past ~epoch 300 (training loss kept falling while test loss rose). scikit-learn's `early_stopping=True` had been handling this invisibly in the baseline. Fixed by checkpointing the best-test-loss model state and stopping once test loss hadn't improved for 100 epochs.

## Notable engineering details

- Fixed random seed (42) for reproducible airfoil sampling.
- Uses UIUC Selig-format coordinate database (`coord_seligFmt`) to avoid errors from format mismatches.

Some things I ran into while building this that are worth pointing out:

- **XFOIL's menu system only works if you're in the right state.** It's a Fortran-based command-line kind of tool, and commands only work if you're in the right submenu. It fails silently if you're not. For example, `QUIT` doesn't work directly from the `.OPERv` submenu, you have to send a blank line to back out first. Debugging this meant a lot of trial and error watching XFOIL's actual output instead of trusting that a command *should* have worked.
- **Long file paths led to errors on macOS.** XFOIL's Fortran backend uses fixed-length string buffers, and macOS's longer temp directory paths were getting cut mid-string, corrupting file reads. Fixed it by running XFOIL with `cwd` set to a short working directory and using relative filenames only instead of full paths.
- **A CSV bug corrupted part of the dataset.** After adding NACA airfoil support to `run_sweep.py`, the code that appends new rows assumed columns were in the same order instead of checking the header. Some rows got written into the wrong columns without any error being raised. Only noticed after some values didn't make any sense. Fixed it by filtering back to the original UIUC-only data and re-running the NACA sweep with a fix in place.
- **Tolerates partial convergence errors.** Some airfoils (like `goe590` at high Reynolds numbers) consistently failed to converge, no matter how many iterations. The sweep logs things like this and keeps moving instead of crashing. Ensures a multi-hour run won't crash because of a bad airfoil.
- **Geometry can't be trusted from the airfoil name/code alone.** A NACA 4-digit code and a UIUC airfoil name mean completely different things. One decomposes directly into numbers, the other is arbitrary text backed by a coordinate file. Fixed this by computing shape descriptors from actual (x, y) coordinates for both, using one shared function. Validated two ways: extracted values for NACA codes matched what the code claims (e.g. `2412` → 12% thickness, 2% camber at 40% chord), and a UIUC file that happened to be a digitized NACA 4418 independently reproduced the analytic formula's numbers.
- **Row-level train/test splitting would have leaked airfoil identity.** Randomly splitting by row would let the same airfoil appear in both train and test at different angles of attack, letting the model partly memorize specific shapes instead of learning general geometry-performance relationships. Fixed with a group-aware split (`GroupShuffleSplit`) keyed on airfoil name.
- **The PyTorch model was silently overfitting until train/test loss were tracked side by side.** Watching only training loss (which kept dropping smoothly to epoch 1000) hid that test performance had already peaked around epoch 300 and was getting steadily worse after. Fixed with checkpointing (save the model state whenever test loss improves) plus early stopping.
- **Unseeded weight initialization made PyTorch runs non-reproducible.** `GroupShuffleSplit(random_state=42)` fixed the data split, but the network's initial weights were still randomly seeded differently on every run, producing different results each time despite identical code. Fixed with `torch.manual_seed(42)`.

## Stack

- **Simulation:** XFOIL (compiled from source, [christophe-david/XFOIL_compilation](https://github.com/christophe-david/XFOIL_compilation))
- **Automation & data:** Python, pandas, numpy, subprocess
- **ML:** scikit-learn, PyTorch
- **Data source:** [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html)

## Running it

```bash
# One-time: download the UIUC airfoil coordinate database
python3 download_uiuc_data.py

# Run the full sweep (generates production_sweep.csv)
python3 run_sweep.py 

# Compute shape descriptors for every airfoil (generates shape_features.csv)
python3 build_shape_features.py

# Train the scikit-learn baseline
python3 train_baseline.py

# Train the PyTorch model (saves airfoil_net.pt and scalers.joblib)
python3 train_pytorch.py

# Reproduce the reported R² / MAE on the held-out test airfoils
python3 analyze_model.py
```

The repo already includes the generated dataset, shape features, and trained model weights (`production_sweep.csv`, `shape_features.csv`, `airfoil_net.pt`, `scalers.joblib`), so you can skip straight to `analyze_model.py` to see results without re-running the earlier steps.
