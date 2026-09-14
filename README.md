# thermal-conductivity-prediction

Machine learning surrogate models for the effective thermal conductivity of a
two-phase composite. Input features are the phase conductivities `K1`, `K2`
and the phase-2 volume fraction `V2`; the target `Keff-Num` comes from
numerical simulation (`Data-Keff-Num.csv`, a full 21 x 21 x 10 grid).

## Setup

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

## Usage

Evaluate one model (metrics: R2, RMSE, MAE, MAPE, max relative error):

```bash
python main.py --model svr              # single fixed 80/20 split
python main.py --model svr --cv random  # shuffled 10-fold CV (interpolation)
python main.py --model svr --cv v2      # leave-one-V2-out CV (extrapolation)
```

Models: `linear`, `ridge`, `lasso`, `svr`, `rf`, `xgb`, `mlp`. Pass
`--results PATH` to save metrics (upserted, one row per model and hyperparameter
set); omit it to run without saving. For CV the per-fold metrics are written
next to the summary with a `_folds` suffix (`results.csv` -> `results_folds.csv`).

Run several models in one evaluation mode from a YAML config, with per-model
hyperparameter overrides:

```bash
python main.py --config configs/baseline.yaml
```

The config sets the evaluation mode, an optional results file, and the models
with their hyperparameters; see [configs/baseline.yaml](configs/baseline.yaml).
Defaults for everything (paths, constants, per-model hyperparameters) live in
[config.py](config.py); a config file only lists what it overrides. Generated
`results*.csv`, `tuning.csv`, `models/`, and `figures/` are git-ignored.

Search hyperparameters for `svr`, `mlp` and `rf` under leave-one-V2-out; the
full ranking goes to `tuning.csv`:

```bash
python tune.py
python tune.py --models svr --metric max_rel_err
```

Winners go into `DEFAULT_PARAMS` in [config.py](config.py) by hand.

Predict Keff for new inputs (trains on the full dataset, cached in `models/`;
`--retrain` after changing hyperparameters or data):

```bash
python predict.py --k1 0.0018 --k2 0.0012 --v2 0.3
python predict.py --input new_points.csv --output predictions.csv
```

Explain the SVR model with SHAP (plots written to `figures/`):

```bash
python shap_analysis.py
```

## Results

SVR performs best in every evaluation (single split test R2 = 0.99996,
leave-one-V2-out mean R2 = 0.9989). Tree ensembles degrade on the
extrapolation folds; SHAP attributes ~72% of the prediction to `K1`.
