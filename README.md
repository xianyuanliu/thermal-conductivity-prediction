# Thermal Conductivity Prediction

Machine learning surrogate models predicting the effective conductivity `Keff` of a two-phase composite from the conductivities of the two phases (`K1`, `K2`) and the volume fraction of phase 2 (`V2`).

Each task keeps its data in `data/<task>/`, its config in `configs/<task>.yaml` and its outputs (metrics, figures, cached models, predictions) in `outputs/<task>/`:

- `concrete_thermal`: thermal conductivity of concrete.
- `soil_electrical`: electrical conductivity of soil mixtures.

## Data

The `concrete_thermal` dataset is hosted on Zenodo: [10.5281/zenodo.23325199](https://doi.org/10.5281/zenodo.23325199). Download both files into `data/concrete_thermal/`:

- `thermal_conductivity_data.csv`: numerical simulation data (`K1`, `K2`, `V2`, `Keff-Num`) on a full 21 x 21 x 10 grid (4410 samples).
- `blind_validation_data.csv`: held-out data (`K1`, `K2`, `V2`) for blind prediction (40 samples).

The `soil_electrical` dataset goes in `data/soil_electrical/electrical_conductivity_data.csv` (`K1`, `K2`, `V2`, `Keff`) on the same 21 x 21 x 10 grid.

## Installation

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python tune.py                                                              # hyperparameter search (defaults in config.py already tuned)
python main.py --model svr --cv v2                                          # evaluate one model (leave-one-V2-out CV)
python main.py --config configs/concrete_thermal.yaml                       # evaluate several models from a config
python print.py --input outputs/concrete_thermal/results_random.csv         # bar charts of CV results
python shap_analysis.py                                                     # SHAP plots
python predict.py --input data/concrete_thermal/blind_validation_data.csv   # predict Keff
```

Without arguments the scripts run `concrete_thermal`. To run another task, pass its config (or `--data` and `--target`); outputs then go to `outputs/<task>/`, named after the data folder:

```bash
python main.py --config configs/soil_electrical.yaml
python shap_analysis.py --config configs/soil_electrical.yaml
python predict.py --config configs/soil_electrical.yaml --k1 0.1 --k2 1e-6 --v2 0.6
python tune.py --data data/soil_electrical/electrical_conductivity_data.csv --target Keff
```

Pass `--output` to write elsewhere.

Three evaluation modes are available:

- `single`: one fixed 80/20 train/test split.
- `random`: 10-fold cross-validation (CV) with shuffled folds.
- `v2`: leave-one-V2-out CV. Each fold holds out all samples at one V2 value, so the model is tested on volume fractions it has not seen during training.

By default every fold uses the hyperparameters in `config.py`, which `tune.py` selected on the full dataset. Add `--tune` to `main.py` to select `svr`, `mlp` and `rf` hyperparameters inside each fold from its training part only: a validation set is held out from the training part (one random V2 value for `v2`, a random 20% otherwise), the best candidate on it is refitted on the whole training part, and the test part is used once. Tuned results are saved next to the untuned ones with a `_tuned` suffix.

Models: linear regression (`linear`), ridge regression (`ridge`), lasso regression (`lasso`), support vector regression (`svr`), random forest (`rf`), XGBoost (`xgb`), multi-layer perceptron (`mlp`). See [config.py](config.py) for details.

## Citation

Citation details will be added upon publication.

## License

[MIT](LICENSE)
