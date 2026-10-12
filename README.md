# Thermal Conductivity Prediction

Machine learning surrogate models predicting the effective thermal conductivity `Keff` of a two-phase composite from the conductivities of the two phases (`K1`, `K2`) and the volume fraction of phase 2 (`V2`).

## Data

- `Data-Keff-Num.csv`: numerical simulation data (`K1`, `K2`, `V2`, `Keff-Num`) on a full 21 x 21 x 10 grid (4410 samples).
- `blind_validation.csv`: held-out data (`K1`, `K2`, `V2`) for blind prediction.

## Installation

Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --model svr --cv v2                # evaluate one model (leave-one-V2-out CV)
python main.py --config configs/baseline.yaml     # evaluate several models from a config
python tune.py                                    # hyperparameter search
python predict.py --input blind_validation.csv --output predictions.csv   # predict Keff
python shap_analysis.py                           # SHAP plots in figures/
```

Models: linear regression (`linear`), ridge regression (`ridge`), lasso regression (`lasso`), support vector regression (`svr`), random forest (`rf`), XGBoost (`xgb`), multi-layer perceptron (`mlp`), and the tabular foundation models TabPFN (`tabpfn`), TabFM (`tabfm`) and TabICL (`tabicl`). See [config.py](config.py) for details.

> **License note:** TabFM automatically downloads pretrained weights that upstream licenses only for non-commercial, non-production use; see [TabFM's pretrained-weights license notice](https://github.com/google-research/tabfm#license-notice-for-pretrained-weights).

## Citation

Citation details will be added upon publication.

## License

[MIT](LICENSE)
