# Random-forest quasar classifier

## Purpose

This component-level classifier removes star-like light curves before the more
expensive lens-pair stage. It produces a probability and binary prediction for
each `lensComponentSourceId`.

## Saved models

| File | Training domain |
|---|---|
| `models/random_forest_catch22.joblib` | Observed Gaia quasars and stars |
| `models/random_forest_catch22_synth.joblib` | Synthetic quasars and observed Gaia stars |

The synthetic variant broadens the simulated quasar behaviours represented at
training time. Both variants can be run together; disagreement is useful
diagnostic information.

## Inputs and preprocessing

Required canonical columns:

- `source_id`
- `lensComponentSourceId`
- `epoch_obs_jd`
- `flux_obs`

The inference pipeline groups measurements into time windows, rejects
components below the configured minimum number of measurements and converts
each light curve into Catch22 features. The saved joblib bundle contains the
trained estimator, expected feature order and relevant preprocessing
configuration.

## Training summary

- approximate dataset size: 18,000 Gaia objects;
- approximate class composition: 70% quasars, 30% stars;
- group-aware train/validation/test split: 70% / 15% / 15%;
- model selection compared RF, XGBoost, Extra Trees, KNN and logistic
  regression;
- reported random-forest balanced accuracy: approximately 0.967.

Balanced accuracy is

\[
\operatorname{BAcc}=\frac12\left(
\frac{TP}{TP+FN}+\frac{TN}{TN+FP}
\right).
\]

## Usage

Use `notebooks/04_quasar_inference.ipynb` or:

```bash
python Utility/Inference.py data/cleaned_lightcurves.csv \
  --domain both \
  --include-probabilities \
  --output results/quasar_predictions.csv
```

## Interpretation and limitations

The thresholded output is a candidate-selection tool. Performance depends on
the Gaia cadence, cleaning rules and training distribution. Recalibrate and
revalidate before applying the model to another survey or substantially
different preprocessing.

## Principal reference

Lubba et al. (2019), *catch22: CAnonical Time-series CHaracteristics*.
