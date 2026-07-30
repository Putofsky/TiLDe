# Data contract

## Canonical light-curve table

All reusable workflows accept a CSV with one measurement per row.

| Column | Type | Required | Meaning |
|---|---|---:|---|
| `source_id` | string / integer | yes | Lens-system or field identifier. |
| `lensComponentSourceId` | string / integer | yes | Gaia component identifier within the system. |
| `epoch_obs_jd` | float | yes | Observation epoch in Julian days. |
| `flux_obs` | float | yes | Observed flux. |
| `flux_obs_error` | float, positive | preprocessing/time delay | Measurement uncertainty on `flux_obs`. |
| `flag_outlier` | boolean | optional | Marks measurements rejected by preprocessing. |

Gaia IDs must not be converted to floating-point numbers. They can exceed the
integer precision available in spreadsheet applications. The command-line
time-delay readers therefore load and write IDs as strings.

## Minimum validation before analysis

```python
from pathlib import Path
import pandas as pd

path = Path("data/cleaned_lightcurves.csv")
df = pd.read_csv(
    path,
    dtype={
        "source_id": "string",
        "lensComponentSourceId": "string",
    },
)

required = {
    "source_id",
    "lensComponentSourceId",
    "epoch_obs_jd",
    "flux_obs",
    "flux_obs_error",
}
missing = required.difference(df.columns)
assert not missing, f"Missing columns: {sorted(missing)}"
assert (pd.to_numeric(df["flux_obs_error"], errors="coerce") > 0).all()
```

## Recommended directory names

| Purpose | Suggested path |
|---|---|
| Raw Gaia extraction | `data/raw_lightcurves.csv` |
| Preprocessed table | `data/cleaned_lightcurves.csv` |
| Quasar predictions | `results/quasar_predictions.csv` |
| Lens RF predictions | `results/rf_pair_predictions.csv` |
| Lens NN predictions | `results/nn_pair_predictions.csv` |
| Time-delay outputs | `results/time_delay_*` |

No input data are embedded in the Python source. Paths are supplied explicitly
from notebooks or the command line.
