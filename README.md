# Thermal Conductivity Prediction

Machine learning surrogate models predicting the effective thermal conductivity `Keff` of a two-phase composite from the conductivities of the two phases (`K1`, `K2`) and the volume fraction of phase 2 (`V2`).

## Data

The dataset is hosted on Zenodo: [10.5281/zenodo.23325199](https://doi.org/10.5281/zenodo.23325199). Download both files into the repository root:

- `thermal_conductivity_data.csv`: numerical simulation data (`K1`, `K2`, `V2`, `Keff-Num`) on a full 21 x 21 x 10 grid (4410 samples).
- `blind_validation_data.csv`: held-out data (`K1`, `K2`, `V2`) for blind prediction (40 samples).

## Installation

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --model svr --cv v2                # evaluate one model (leave-one-V2-out CV)
python main.py --config configs/baseline.yaml     # evaluate several models from a config
python print.py --results results_random.csv      # bar charts of CV results in figures/
python tune.py                                    # hyperparameter search
python predict.py --input blind_validation_data.csv --output predictions.csv   # predict Keff
python shap_analysis.py                           # SHAP plots in figures/
```

Models: linear regression (`linear`), ridge regression (`ridge`), lasso regression (`lasso`), support vector regression (`svr`), random forest (`rf`), XGBoost (`xgb`), multi-layer perceptron (`mlp`). See [config.py](config.py) for details.

## Citation

Citation details will be added upon publication.

## License

[MIT](LICENSE)
