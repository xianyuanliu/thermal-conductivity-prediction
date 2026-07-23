# thermal-conductivity-prediction

Machine learning surrogate models for the effective thermal conductivity of a
two-phase composite. Input features are the phase conductivities `K1`, `K2`
and the phase-2 volume fraction `V2`; the target `Keff-Num` comes from
numerical simulation (`Data-Keff-Num.csv`, a full 21 x 21 x 10 grid).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Evaluate a model (metrics: R2, RMSE, MAE, MAPE, max relative error):

```bash
python main.py --model svr              # single fixed 80/20 split
python main.py --model svr --cv random  # shuffled 10-fold CV (interpolation)
python main.py --model svr --cv v2      # leave-one-V2-out CV (extrapolation)
```

Models: `linear`, `ridge`, `lasso`, `svr`, `rf`, `xgb`, `mlp`. Results are
upserted into `results.csv` (single split) and `results_cv.csv` plus
`results_cv_folds.csv` (CV), one row per configuration.

Predict Keff for new inputs (trains on the full dataset, cached in `models/`):

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
