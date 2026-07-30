# Reproducible workflows

## 1. Preprocess Gaia light curves

Use `notebooks/02_preprocessing.ipynb`. It documents binning, rolling robust
outlier scores, SNR filtering and every threshold before calling
`Utility.Pretraitement.pretraitement`.

The function returns cleaned measurements, a per-source report, removed points,
a summary dictionary and the paths of the written files.

## 2. Detect quasar-like components

Use `notebooks/04_quasar_inference.ipynb` or:

```bash
python Utility/Inference.py data/cleaned_lightcurves.csv \
  --domain both \
  --include-probabilities \
  --add-final-pred \
  --output results/quasar_predictions.csv
```

`real` uses models trained on observed quasars/stars. `synthetic` uses the
synthetic-quasar variant. `both` keeps both sets of predictions.

## 3. Score candidate component pairs

Random forest:

```bash
python Utility/RF.py data/cleaned_lightcurves.csv \
  --model models/lens22_rf.joblib \
  --output results/rf_pair_predictions.csv
```

Neural network:

```bash
python Utility/LensNN.py data/cleaned_lightcurves.csv \
  --model models/resTCN.pt \
  --output results/nn_pair_predictions.csv
```

Both outputs contain `sourceID`, `compA`, `compB` and `Proba`.

## 4. Estimate a time delay

Start with the interpolation baseline in
`notebooks/07_time_delay_interpolation.ipynb`. Then run the adaptive P-spline
method in `notebooks/08_time_delay_pspline.ipynb`.

The recommended validation order is:

1. run the `quick` profile to verify IDs and overlap;
2. inspect the delay/LOO/overlap profiles;
3. run the `standard` profile;
4. propagate `flux_obs_error` with at least 200–300 draws;
5. inspect the full sample distribution for multiple modes;
6. use `high_precision` only for selected systems.

Do not reduce a clearly multimodal distribution to a single symmetric standard
deviation. Report the sample distribution and at least Q16, median and Q84;
mode-conditional summaries may be added after a documented mode-selection
procedure.

## 5. Batch processing

The pair table is a CSV with:

```text
source_id,component_a,component_b
```

Run:

```bash
python -m Utility.time_delay_pspline data/cleaned_lightcurves.csv \
  --pairs configs/time_delay_system_pairs.csv \
  --profile standard \
  --mc-samples 300 \
  --output-dir results/batch_time_delays
```

The batch runner checkpoints summary files after every pair and records failures
without stopping subsequent systems.
