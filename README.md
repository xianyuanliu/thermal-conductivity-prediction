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

The tabular foundation models (`tabpfn`, `tabfm`, `tabicl`) are optional and pull in PyTorch. To use them, install instead:

```bash
pip install -r requirements-fm.txt
```

## Usage

```bash
python main.py --model svr --cv v2                # evaluate one model (leave-one-V2-out CV)
python main.py --config configs/baseline.yaml     # evaluate several models from a config
python tune.py                                    # hyperparameter search
python predict.py --input blind_validation_data.csv --output predictions.csv   # predict Keff
python shap_analysis.py                           # SHAP plots in figures/
```

Models: linear regression (`linear`), ridge regression (`ridge`), lasso regression (`lasso`), support vector regression (`svr`), random forest (`rf`), XGBoost (`xgb`), multi-layer perceptron (`mlp`), and the tabular foundation models TabPFN (`tabpfn`), TabFM (`tabfm`) and TabICL (`tabicl`). See [config.py](config.py) for details.

> **License notes:** The foundation models download pretrained weights whose licenses are more restrictive than this repository's.
>
> - **TabPFN:** the weights are licensed for non-commercial use only, and downloading them requires accepting the license with a Prior Labs account. On first use, TabPFN opens a browser to log in and caches the token. Without a browser (servers, CI), accept the license at [ux.priorlabs.ai](https://ux.priorlabs.ai), copy an API key, and set the `TABPFN_TOKEN` environment variable. See [Prior Labs' gated-model guide](https://docs.priorlabs.ai/how-to-access-gated-models).
> - **TabFM:** the weights are licensed only for non-commercial, non-production use; see [TabFM's pretrained-weights license notice](https://github.com/google-research/tabfm#license-notice-for-pretrained-weights). TabFM is therefore available for evaluation in `main.py` only, not in `predict.py`.

## Citation

Citation details will be added upon publication.

## License

[MIT](LICENSE)
