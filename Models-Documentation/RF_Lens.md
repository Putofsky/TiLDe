# Random-forest lens-pair classifier

## Purpose

This model scores every component pair inside a candidate source after the
quasar-selection stage. The output probability measures similarity to the
training examples of lensed quasar pairs.

## Saved model

`models/lens22_rf.joblib`

The bundle stores the trained estimator, exact feature order and configuration.

## Inputs

The canonical light-curve table requires:

- `source_id`
- `lensComponentSourceId`
- `epoch_obs_jd`
- `flux_obs`
- `flux_obs_error`

`flag_outlier` is optional.

## Feature families

`Utility/RF.py` reconstructs the same features used during training:

- robust per-curve location and scale;
- point count, cadence, span and overlap descriptors;
- distribution moments and polynomial shape summaries;
- derivative summaries;
- delay-grid correlation features;
- microlensing-resistant polynomial residual scores for degrees 0–4;
- pair-level asymmetry and scale comparisons.

Each source with at least two valid components is expanded into every unordered
pair. Failed pairs may be retained with `Proba = NaN` for auditability.

## Usage

Use `notebooks/05_lens_random_forest.ipynb` or:

```bash
python Utility/RF.py data/cleaned_lightcurves.csv \
  --model models/lens22_rf.joblib \
  --output results/rf_pair_predictions.csv
```

Output columns are exactly:

```text
sourceID,compA,compB,Proba
```

## Limitations

The classifier ranks pairs; it does not estimate a publishable time delay and
does not confirm lensing. Feature distributions must be checked after changes
to binning, normalisation, cadence or survey.
