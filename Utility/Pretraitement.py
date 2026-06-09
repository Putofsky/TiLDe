from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_SOURCE_COL = "source_id"
DEFAULT_COMPONENT_COL = "lensComponentSourceId"
DEFAULT_Y_COL = "flux_obs"
DEFAULT_YERR_COL = "flux_obs_error"


# ============================================================
# BASIC DATA UTILS
# ============================================================

def read_lightcurve_data(data: Any) -> pd.DataFrame:
    """
    Accepts:
    - CSV path
    - pandas DataFrame
    - nested dict {source_id: {lensComponentSourceId: DataFrame}}

    Returns a DataFrame.
    """
    if isinstance(data, (str, Path)):
        return pd.read_csv(data, low_memory=False)

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, dict):
        parts = []

        for source_id, comps in data.items():
            if not isinstance(comps, dict):
                continue

            for component_id, component_df in comps.items():
                tmp = pd.DataFrame(component_df).copy()

                if DEFAULT_SOURCE_COL not in tmp.columns:
                    tmp[DEFAULT_SOURCE_COL] = source_id

                if DEFAULT_COMPONENT_COL not in tmp.columns:
                    tmp[DEFAULT_COMPONENT_COL] = component_id

                parts.append(tmp)

        if parts:
            return pd.concat(parts, ignore_index=True)

    raise ValueError("data must be a CSV path, pandas DataFrame, or nested dict")


def guess_time_col(df: pd.DataFrame) -> str:
    candidates = [
        "jd_time",
        "jdTime",
        "JD_TIME",
        "jd",
        "JD",
        "mjd",
        "MJD",
        "julian_date",
        "JulianDate",
        "julianDate",
        "time",
        "Time",
        "timestamp",
        "Timestamp",
        "datetime",
        "Datetime",
        "date",
        "Date",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        name = str(col).lower()
        if "jd" in name or "julian" in name or "time" in name or "date" in name:
            return col

    raise ValueError(f"No time column found. Columns are: {list(df.columns)}")


def _time_to_days(series: pd.Series) -> np.ndarray:
    """
    Converts numeric time or datetime-like time to float days.

    If time is already numeric, it is kept as numeric days.
    For JD or MJD this is fine because both are day-based.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() >= max(3, int(0.5 * len(series))):
        return numeric.to_numpy(dtype=float)

    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.view("int64").to_numpy(dtype=float) / 1e9 / 86400.0


def _days_to_output_time(
    days: np.ndarray,
    original_series: pd.Series,
) -> np.ndarray | pd.Series:
    """
    Converts internal day values back to the style of the original time column.
    Numeric time stays numeric.
    Datetime-like time returns UTC datetime.
    """
    numeric = pd.to_numeric(original_series, errors="coerce")

    if numeric.notna().sum() >= max(3, int(0.5 * len(original_series))):
        return days

    return pd.to_datetime(days * 86400.0, unit="s", utc=True)


def _safe_group_cols(
    df: pd.DataFrame,
    source_col: str,
    component_col: str | None,
) -> list[str]:
    cols = []

    if source_col in df.columns:
        cols.append(source_col)

    if component_col is not None and component_col in df.columns:
        cols.append(component_col)

    if not cols:
        raise ValueError(
            f"No valid group columns found. Tried source_col={source_col!r}, "
            f"component_col={component_col!r}."
        )

    return cols


# ============================================================
# 1. BI-DAILY AGGREGATION
# ============================================================

def aggregate_bi_daily(
    data: Any,
    *,
    time_col: str | None = None,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    source_col: str = DEFAULT_SOURCE_COL,
    component_col: str | None = DEFAULT_COMPONENT_COL,
    bin_days: float = 2.0,
    error_mode: str = "mean",
    keep_other_cols: bool = True,
) -> pd.DataFrame:
    """
    Aggregates each light curve every 2 days.

    For each source/component/bin:
    - time is the mean time
    - y_col is the mean flux
    - yerr_col is the mean error by default

    error_mode:
    - "mean": mean of errors
    - "quadrature": sqrt(sum(error^2)) / n
    - "sem": standard error of the mean flux
    """

    df = read_lightcurve_data(data)

    if time_col is None:
        time_col = guess_time_col(df)

    required = [time_col, y_col]
    if yerr_col in df.columns:
        required.append(yerr_col)

    group_cols = _safe_group_cols(df, source_col, component_col)
    required += group_cols

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    data_df = df.copy()
    data_df["_time_days"] = _time_to_days(data_df[time_col])
    data_df[y_col] = pd.to_numeric(data_df[y_col], errors="coerce")

    if yerr_col in data_df.columns:
        data_df[yerr_col] = pd.to_numeric(data_df[yerr_col], errors="coerce")
    else:
        data_df[yerr_col] = np.nan

    data_df = data_df.dropna(subset=["_time_days", y_col])
    data_df = data_df.sort_values(group_cols + ["_time_days"])

    if data_df.empty:
        raise ValueError("No valid data after converting time and flux columns.")

    origin = np.nanmin(data_df["_time_days"].to_numpy(dtype=float))
    data_df["_bin_index"] = np.floor((data_df["_time_days"] - origin) / bin_days).astype(int)

    rows = []

    for keys, sub in data_df.groupby(group_cols + ["_bin_index"], sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_values = dict(zip(group_cols + ["_bin_index"], keys))

        y = sub[y_col].to_numpy(dtype=float)
        err = sub[yerr_col].to_numpy(dtype=float)
        finite_err = np.isfinite(err)

        if error_mode == "mean":
            if finite_err.any():
                new_err = float(np.nanmean(err))
            else:
                new_err = np.nan

        elif error_mode == "quadrature":
            if finite_err.any():
                new_err = float(np.sqrt(np.nansum(err[finite_err] ** 2)) / finite_err.sum())
            else:
                new_err = np.nan

        elif error_mode == "sem":
            if len(y) > 1:
                new_err = float(np.nanstd(y, ddof=1) / np.sqrt(np.isfinite(y).sum()))
            else:
                new_err = float(err[finite_err][0]) if finite_err.any() else np.nan

        else:
            raise ValueError("error_mode must be 'mean', 'quadrature', or 'sem'.")

        row = {}
        for col in group_cols:
            row[col] = sub[col].iloc[0]

        mean_time_days = float(np.nanmean(sub["_time_days"]))
        row[time_col] = _days_to_output_time(np.array([mean_time_days]), df[time_col])[0]
        row[y_col] = float(np.nanmean(y))
        row[yerr_col] = new_err
        row["n_points_in_bin"] = int(len(sub))
        row["pre_bin_index"] = int(key_values["_bin_index"])
        row["pre_bin_start_day"] = float(origin + key_values["_bin_index"] * bin_days)
        row["pre_bin_end_day"] = float(origin + (key_values["_bin_index"] + 1) * bin_days)

        if keep_other_cols:
            for col in sub.columns:
                if col.startswith("_"):
                    continue
                if col in row:
                    continue
                if col in [time_col, y_col, yerr_col]:
                    continue

                vals = sub[col].dropna().unique()
                if len(vals) == 1:
                    row[col] = vals[0]

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(group_cols + [time_col]).reset_index(drop=True)

    return out


# ============================================================
# POLYNOMIAL FITTING
# ============================================================

def _design_matrix(x_scaled: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x_scaled, N=degree + 1, increasing=False)


def _weighted_polyfit(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None,
    degree: int,
    extra_weights: np.ndarray | None = None,
) -> dict:
    """
    Weighted polynomial fit with scaled x for numerical stability.
    Returns a model dict.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if yerr is None:
        yerr = np.ones_like(y, dtype=float)
    else:
        yerr = np.asarray(yerr, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)

    if yerr is not None:
        finite &= np.isfinite(yerr) & (yerr > 0)

    if extra_weights is not None:
        extra_weights = np.asarray(extra_weights, dtype=float)
        finite &= np.isfinite(extra_weights) & (extra_weights > 0)

    x_fit = x[finite]
    y_fit = y[finite]
    err_fit = yerr[finite]

    if extra_weights is None:
        ew_fit = np.ones_like(y_fit)
    else:
        ew_fit = extra_weights[finite]

    if len(x_fit) < degree + 1:
        raise ValueError(
            f"Not enough points for degree={degree}. "
            f"Need at least {degree + 1}, got {len(x_fit)}."
        )

    x0 = float(np.nanmedian(x_fit))
    x_scale = float(np.nanstd(x_fit))

    if not np.isfinite(x_scale) or x_scale <= 0:
        x_scale = 1.0

    x_scaled = (x_fit - x0) / x_scale

    # np.polyfit minimizes sum((w * residual)^2)
    w = np.sqrt(ew_fit) / err_fit
    coeff = np.polyfit(x_scaled, y_fit, deg=degree, w=w)

    yhat_fit = np.polyval(coeff, x_scaled)
    residual_fit = y_fit - yhat_fit

    wrss = float(np.nansum((residual_fit / err_fit) ** 2))
    wrmse = float(np.sqrt(np.nanmean((residual_fit / err_fit) ** 2)))
    rmse = float(np.sqrt(np.nanmean(residual_fit ** 2)))

    return {
        "coeff": coeff,
        "degree": degree,
        "x0": x0,
        "x_scale": x_scale,
        "n_fit": int(len(x_fit)),
        "wrss": wrss,
        "wrmse": wrmse,
        "rmse": rmse,
    }


def _poly_predict(model: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xs = (x - model["x0"]) / model["x_scale"]
    return np.polyval(model["coeff"], xs)


def _robust_sigma(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) < 3:
        return np.nan

    med = np.nanmedian(residuals)
    mad = np.nanmedian(np.abs(residuals - med))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(residuals)

    return float(sigma)


def fit_one_polynomial(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    degree: int = 3,
) -> dict:
    """
    Fits one polynomial to one light curve.
    """
    x = _time_to_days(df[time_col])
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    if yerr_col in df.columns:
        yerr = pd.to_numeric(df[yerr_col], errors="coerce").to_numpy(dtype=float)
    else:
        yerr = np.ones_like(y)

    yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, np.nanmedian(yerr[np.isfinite(yerr) & (yerr > 0)]))

    if not np.isfinite(yerr).any():
        yerr = np.ones_like(y)

    model = _weighted_polyfit(x, y, yerr, degree)
    yhat = _poly_predict(model, x)
    residual = y - yhat

    model["x"] = x
    model["y"] = y
    model["yerr"] = yerr
    model["yhat"] = yhat
    model["residual"] = residual

    return model


def fit_two_polynomials_em(
    df: pd.DataFrame,
    *,
    time_col: str,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    degree: int = 3,
    max_iter: int = 100,
    tol: float = 1e-5,
    n_init: int = 5,
    random_state: int = 0,
) -> dict:
    """
    EM algorithm for a mixture of 2 polynomial curves.

    It tries to explain the same source/component with two polynomial tracks.
    If this improves the error a lot compared to one polynomial, the source is suspicious.
    """
    rng = np.random.default_rng(random_state)

    x = _time_to_days(df[time_col])
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)

    if yerr_col in df.columns:
        yerr = pd.to_numeric(df[yerr_col], errors="coerce").to_numpy(dtype=float)
    else:
        yerr = np.ones_like(y)

    good_err = np.isfinite(yerr) & (yerr > 0)

    if good_err.any():
        fallback_err = np.nanmedian(yerr[good_err])
    else:
        fallback_err = 1.0

    yerr = np.where(good_err, yerr, fallback_err)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    x = x[finite]
    y = y[finite]
    yerr = yerr[finite]

    if len(x) < 2 * (degree + 1):
        one = _weighted_polyfit(x, y, yerr, degree)
        yhat = _poly_predict(one, x)
        return {
            "success": False,
            "reason": "not_enough_points_for_two_polynomials",
            "x": x,
            "y": y,
            "yerr": yerr,
            "single_model": one,
            "models": [one, one],
            "pi": np.array([0.5, 0.5]),
            "responsibilities": np.column_stack([np.ones_like(y), np.zeros_like(y)]),
            "labels": np.zeros_like(y, dtype=int),
            "yhat_best": yhat,
            "wrmse_two": one["wrmse"],
            "rmse_two": one["rmse"],
            "log_likelihood": np.nan,
        }

    single = _weighted_polyfit(x, y, yerr, degree)
    single_yhat = _poly_predict(single, x)
    single_resid = y - single_yhat

    init_responsibilities = []

    # Init 1: split by residual sign
    r1 = np.zeros((len(y), 2), dtype=float)
    r1[:, 0] = single_resid <= np.nanmedian(single_resid)
    r1[:, 1] = 1.0 - r1[:, 0]
    init_responsibilities.append(r1)

    # Init 2: split by y median
    r2 = np.zeros((len(y), 2), dtype=float)
    r2[:, 0] = y <= np.nanmedian(y)
    r2[:, 1] = 1.0 - r2[:, 0]
    init_responsibilities.append(r2)

    # Init 3: split by time median
    r3 = np.zeros((len(y), 2), dtype=float)
    r3[:, 0] = x <= np.nanmedian(x)
    r3[:, 1] = 1.0 - r3[:, 0]
    init_responsibilities.append(r3)

    # Random soft starts
    for _ in range(max(0, n_init - len(init_responsibilities))):
        r = rng.uniform(0.1, 0.9, size=(len(y), 2))
        r = r / r.sum(axis=1, keepdims=True)
        init_responsibilities.append(r)

    best = None

    for init_r in init_responsibilities:
        resp = init_r.copy()
        prev_ll = -np.inf

        for iteration in range(max_iter):
            models = []
            pi = np.clip(resp.mean(axis=0), 1e-6, 1.0)
            pi = pi / pi.sum()

            # M-step
            failed = False
            for k in range(2):
                try:
                    model_k = _weighted_polyfit(
                        x,
                        y,
                        yerr,
                        degree,
                        extra_weights=np.clip(resp[:, k], 1e-8, 1.0),
                    )
                except Exception:
                    failed = True
                    break

                models.append(model_k)

            if failed:
                break

            # E-step
            logp = np.zeros((len(y), 2), dtype=float)

            for k in range(2):
                yhat_k = _poly_predict(models[k], x)
                resid_k = y - yhat_k

                logp[:, k] = (
                    np.log(pi[k])
                    - 0.5 * (resid_k / yerr) ** 2
                    - np.log(yerr)
                )

            max_logp = np.max(logp, axis=1, keepdims=True)
            prob = np.exp(logp - max_logp)
            resp = prob / prob.sum(axis=1, keepdims=True)

            ll = float(np.sum(max_logp[:, 0] + np.log(prob.sum(axis=1))))

            if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol:
                break

            prev_ll = ll

        if failed:
            continue

        labels = np.argmax(resp, axis=1)
        yhat_all = np.column_stack([_poly_predict(models[k], x) for k in range(2)])
        yhat_best = yhat_all[np.arange(len(y)), labels]
        resid_best = y - yhat_best

        wrmse_two = float(np.sqrt(np.nanmean((resid_best / yerr) ** 2)))
        rmse_two = float(np.sqrt(np.nanmean(resid_best ** 2)))

        candidate = {
            "success": True,
            "reason": "ok",
            "x": x,
            "y": y,
            "yerr": yerr,
            "single_model": single,
            "models": models,
            "pi": pi,
            "responsibilities": resp,
            "labels": labels,
            "yhat_best": yhat_best,
            "wrmse_two": wrmse_two,
            "rmse_two": rmse_two,
            "log_likelihood": prev_ll,
        }

        if best is None or candidate["log_likelihood"] > best["log_likelihood"]:
            best = candidate

    if best is None:
        yhat = _poly_predict(single, x)
        return {
            "success": False,
            "reason": "em_failed",
            "x": x,
            "y": y,
            "yerr": yerr,
            "single_model": single,
            "models": [single, single],
            "pi": np.array([0.5, 0.5]),
            "responsibilities": np.column_stack([np.ones_like(y), np.zeros_like(y)]),
            "labels": np.zeros_like(y, dtype=int),
            "yhat_best": yhat,
            "wrmse_two": single["wrmse"],
            "rmse_two": single["rmse"],
            "log_likelihood": np.nan,
        }

    return best


# ============================================================
# SOURCE CLEANING
# ============================================================

def _make_group_key(row_or_tuple: Any) -> str:
    if isinstance(row_or_tuple, tuple):
        return " | ".join(str(x) for x in row_or_tuple)
    return str(row_or_tuple)


def _get_group_title(keys: Any, group_cols: list[str]) -> str:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return ", ".join(f"{col}={val}" for col, val in zip(group_cols, keys))


def _plot_group_removal(
    sub: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    yerr_col: str,
    title: str,
    reason: str,
    one_model: dict | None = None,
    em_model: dict | None = None,
    removed_point_mask: np.ndarray | None = None,
):
    x = _time_to_days(sub[time_col])
    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)

    if yerr_col in sub.columns:
        yerr = pd.to_numeric(sub[yerr_col], errors="coerce").to_numpy(dtype=float)
    else:
        yerr = np.full_like(y, np.nan)

    order = np.argsort(x)
    x_sorted = x[order]

    fig, ax = plt.subplots(figsize=(12, 5))

    has_err = np.isfinite(yerr).any()

    if removed_point_mask is None:
        removed_point_mask = np.zeros(len(sub), dtype=bool)

    kept = ~removed_point_mask

    if has_err:
        ax.errorbar(
            x[kept],
            y[kept],
            yerr=yerr[kept],
            fmt="o",
            markersize=4,
            capsize=2,
            alpha=0.75,
            label="kept points",
        )
    else:
        ax.plot(
            x[kept],
            y[kept],
            "o",
            markersize=4,
            alpha=0.75,
            label="kept points",
        )

    if removed_point_mask.any():
        if has_err:
            ax.errorbar(
                x[removed_point_mask],
                y[removed_point_mask],
                yerr=yerr[removed_point_mask],
                fmt="x",
                markersize=7,
                capsize=2,
                alpha=0.95,
                label="removed 3σ points",
            )
        else:
            ax.plot(
                x[removed_point_mask],
                y[removed_point_mask],
                "x",
                markersize=7,
                alpha=0.95,
                label="removed 3σ points",
            )

    if one_model is not None:
        ax.plot(
            x_sorted,
            _poly_predict(one_model, x_sorted),
            linewidth=2,
            label="1 polynomial",
        )

    if em_model is not None and em_model.get("models") is not None:
        for k, model in enumerate(em_model["models"]):
            ax.plot(
                x_sorted,
                _poly_predict(model, x_sorted),
                linewidth=2,
                linestyle="--",
                label=f"EM polynomial {k + 1}",
            )

    ax.set_title(f"{title}\nREMOVED: {reason}")
    ax.set_xlabel(f"{time_col} converted to numeric days")
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def preprocess_lightcurves(
    data: Any,
    *,
    time_col: str | None = None,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    source_col: str = DEFAULT_SOURCE_COL,
    component_col: str | None = DEFAULT_COMPONENT_COL,

    # Step 1: bi-daily aggregation
    do_bi_daily: bool = True,
    bin_days: float = 2.0,
    error_mode: str = "mean",

    # Step 2: polynomial + EM
    poly_degree: int = 3,
    em_max_iter: int = 100,
    em_tol: float = 1e-5,
    em_n_init: int = 5,

    # Remove source if two-polynomial EM improves error too much
    em_improvement_threshold: float = 0.35,
    em_ratio_threshold: float = 1.50,

    # Step 3: for non-removed sources, remove 3σ points
    sigma_clip: float = 3.0,

    # Step 4: remove source with less than 20 points after preprocessing
    min_points_after: int = 20,

    # Display
    show_removed_plots: bool = True,
    verbose: bool = True,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline.

    Steps:
    1. Aggregate every 2 days.
    2. Fit one polynomial.
    3. Fit EM with two polynomials.
    4. Remove whole source/component if EM improves the error too much.
    5. Else remove 3σ outlier points.
    6. Remove source/component with less than 20 points after preprocessing.
    7. Show every removed source and why, with plots.

    Returns:
    - clean_df
    - source_report
    - removed_points_df
    """

    df0 = read_lightcurve_data(data)

    if time_col is None:
        time_col = guess_time_col(df0)

    if do_bi_daily:
        df = aggregate_bi_daily(
            df0,
            time_col=time_col,
            y_col=y_col,
            yerr_col=yerr_col,
            source_col=source_col,
            component_col=component_col,
            bin_days=bin_days,
            error_mode=error_mode,
        )
    else:
        df = df0.copy()

    group_cols = _safe_group_cols(df, source_col, component_col)

    df = df.copy()
    df["_pre_keep"] = True
    df["_pre_removed_reason"] = ""

    source_reports = []
    removed_points = []

    for group_i, (keys, sub) in enumerate(df.groupby(group_cols, sort=False)):
        sub = sub.copy()
        idx = sub.index.to_numpy()
        group_title = _get_group_title(keys, group_cols)

        n_after_aggregation = len(sub)

        if n_after_aggregation < poly_degree + 2:
            df.loc[idx, "_pre_keep"] = False
            df.loc[idx, "_pre_removed_reason"] = "too_few_points_for_polynomial_fit"

            source_reports.append({
                "group": group_title,
                "removed": True,
                "reason": "too_few_points_for_polynomial_fit",
                "n_after_aggregation": n_after_aggregation,
                "n_after_sigma_clip": 0,
                "n_final": 0,
                "wrmse_one": np.nan,
                "wrmse_two": np.nan,
                "em_improvement": np.nan,
                "em_ratio": np.nan,
                "n_sigma_removed_points": 0,
            })

            if verbose:
                print(f"REMOVED SOURCE: {group_title} | reason=too_few_points_for_polynomial_fit")

            if show_removed_plots:
                _plot_group_removal(
                    sub,
                    time_col=time_col,
                    y_col=y_col,
                    yerr_col=yerr_col,
                    title=group_title,
                    reason="too few points for polynomial fit",
                )

            continue

        try:
            one_model = fit_one_polynomial(
                sub,
                time_col=time_col,
                y_col=y_col,
                yerr_col=yerr_col,
                degree=poly_degree,
            )

            em_model = fit_two_polynomials_em(
                sub,
                time_col=time_col,
                y_col=y_col,
                yerr_col=yerr_col,
                degree=poly_degree,
                max_iter=em_max_iter,
                tol=em_tol,
                n_init=em_n_init,
                random_state=random_state + group_i,
            )

            wrmse_one = float(one_model["wrmse"])
            wrmse_two = float(em_model["wrmse_two"])

            if np.isfinite(wrmse_one) and wrmse_one > 0 and np.isfinite(wrmse_two):
                em_improvement = (wrmse_one - wrmse_two) / wrmse_one
                em_ratio = wrmse_one / max(wrmse_two, 1e-12)
            else:
                em_improvement = np.nan
                em_ratio = np.nan

            remove_by_em = (
                np.isfinite(em_improvement)
                and np.isfinite(em_ratio)
                and (
                    em_improvement >= em_improvement_threshold
                    or em_ratio >= em_ratio_threshold
                )
            )

            if remove_by_em:
                reason = (
                    f"two_polynomial_EM_improves_too_much "
                    f"(improvement={em_improvement:.3f}, ratio={em_ratio:.3f})"
                )

                df.loc[idx, "_pre_keep"] = False
                df.loc[idx, "_pre_removed_reason"] = reason

                source_reports.append({
                    "group": group_title,
                    "removed": True,
                    "reason": reason,
                    "n_after_aggregation": n_after_aggregation,
                    "n_after_sigma_clip": 0,
                    "n_final": 0,
                    "wrmse_one": wrmse_one,
                    "wrmse_two": wrmse_two,
                    "em_improvement": em_improvement,
                    "em_ratio": em_ratio,
                    "n_sigma_removed_points": 0,
                })

                if verbose:
                    print(f"REMOVED SOURCE: {group_title} | {reason}")

                if show_removed_plots:
                    _plot_group_removal(
                        sub,
                        time_col=time_col,
                        y_col=y_col,
                        yerr_col=yerr_col,
                        title=group_title,
                        reason=reason,
                        one_model=one_model,
                        em_model=em_model,
                    )

                continue

            # If EM did not improve a lot, remove 3 sigma points using the 1-poly residuals.
            x = one_model["x"]
            y = one_model["y"]
            yerr = one_model["yerr"]
            residual = one_model["residual"]

            robust = _robust_sigma(residual)

            if not np.isfinite(robust) or robust <= 0:
                robust = np.nanstd(residual)

            if not np.isfinite(robust) or robust <= 0:
                robust = 1.0

            scale = np.sqrt(robust ** 2 + yerr ** 2)
            z = np.abs(residual) / scale
            remove_point = z > sigma_clip

            # Match one_model arrays to sub rows. fit_one_polynomial keeps the same row order after conversion.
            if len(remove_point) == len(idx):
                removed_idx = idx[remove_point]
            else:
                removed_idx = np.array([], dtype=int)
                remove_point = np.zeros(len(idx), dtype=bool)

            if len(removed_idx) > 0:
                df.loc[removed_idx, "_pre_keep"] = False
                df.loc[removed_idx, "_pre_removed_reason"] = f"{sigma_clip:g}sigma_point_outlier"

                tmp_removed = sub.loc[removed_idx].copy()
                tmp_removed["_pre_removed_reason"] = f"{sigma_clip:g}sigma_point_outlier"
                tmp_removed["_pre_z_score"] = z[remove_point]
                removed_points.append(tmp_removed)

                if verbose:
                    print(
                        f"POINTS REMOVED: {group_title} | "
                        f"{len(removed_idx)} point(s) above {sigma_clip:g} sigma"
                    )

                if show_removed_plots:
                    _plot_group_removal(
                        sub,
                        time_col=time_col,
                        y_col=y_col,
                        yerr_col=yerr_col,
                        title=group_title,
                        reason=f"{len(removed_idx)} point(s) removed above {sigma_clip:g} sigma",
                        one_model=one_model,
                        em_model=None,
                        removed_point_mask=remove_point,
                    )

            n_after_sigma = int(len(idx) - len(removed_idx))

            source_reports.append({
                "group": group_title,
                "removed": False,
                "reason": "kept_after_EM_then_sigma_clip",
                "n_after_aggregation": n_after_aggregation,
                "n_after_sigma_clip": n_after_sigma,
                "n_final": n_after_sigma,
                "wrmse_one": wrmse_one,
                "wrmse_two": wrmse_two,
                "em_improvement": em_improvement,
                "em_ratio": em_ratio,
                "n_sigma_removed_points": int(len(removed_idx)),
            })

        except Exception as exc:
            reason = f"fit_failed: {type(exc).__name__}: {exc}"

            df.loc[idx, "_pre_keep"] = False
            df.loc[idx, "_pre_removed_reason"] = reason

            source_reports.append({
                "group": group_title,
                "removed": True,
                "reason": reason,
                "n_after_aggregation": n_after_aggregation,
                "n_after_sigma_clip": 0,
                "n_final": 0,
                "wrmse_one": np.nan,
                "wrmse_two": np.nan,
                "em_improvement": np.nan,
                "em_ratio": np.nan,
                "n_sigma_removed_points": 0,
            })

            if verbose:
                print(f"REMOVED SOURCE: {group_title} | {reason}")

            if show_removed_plots:
                _plot_group_removal(
                    sub,
                    time_col=time_col,
                    y_col=y_col,
                    yerr_col=yerr_col,
                    title=group_title,
                    reason=reason,
                )

    # Remove groups with fewer than min_points_after after all preprocessing.
    current_keep = df["_pre_keep"].copy()

    for keys, sub in df[current_keep].groupby(group_cols, sort=False):
        idx = sub.index.to_numpy()
        group_title = _get_group_title(keys, group_cols)

        if len(sub) < min_points_after:
            reason = f"less_than_{min_points_after}_points_after_preprocessing"

            df.loc[idx, "_pre_keep"] = False
            df.loc[idx, "_pre_removed_reason"] = reason

            source_reports.append({
                "group": group_title,
                "removed": True,
                "reason": reason,
                "n_after_aggregation": np.nan,
                "n_after_sigma_clip": len(sub),
                "n_final": 0,
                "wrmse_one": np.nan,
                "wrmse_two": np.nan,
                "em_improvement": np.nan,
                "em_ratio": np.nan,
                "n_sigma_removed_points": np.nan,
            })

            if verbose:
                print(f"REMOVED SOURCE: {group_title} | {reason}")

            if show_removed_plots:
                _plot_group_removal(
                    sub,
                    time_col=time_col,
                    y_col=y_col,
                    yerr_col=yerr_col,
                    title=group_title,
                    reason=reason,
                )

    clean_df = df[df["_pre_keep"]].copy()
    removed_source_df = df[~df["_pre_keep"]].copy()

    if removed_points:
        removed_points_df = pd.concat(removed_points, ignore_index=True)
    else:
        removed_points_df = pd.DataFrame()

    source_report = pd.DataFrame(source_reports)

    clean_df = clean_df.drop(columns=["_pre_keep"], errors="ignore")
    removed_source_df = removed_source_df.drop(columns=["_pre_keep"], errors="ignore")

    if verbose:
        print("\n========== PREPROCESSING SUMMARY ==========")
        print(f"Initial rows: {len(df0)}")
        print(f"After bi-daily aggregation rows: {len(df)}")
        print(f"Final clean rows: {len(clean_df)}")
        print(f"Removed rows: {len(removed_source_df)}")
        print(f"Removed 3sigma points: {len(removed_points_df)}")
        print("==========================================\n")

    return clean_df.reset_index(drop=True), source_report, removed_points_df.reset_index(drop=True)


# ============================================================
# SMALL HELPER TO SAVE OUTPUTS
# ============================================================

def save_preprocessing_outputs(
    clean_df: pd.DataFrame,
    source_report: pd.DataFrame,
    removed_points_df: pd.DataFrame,
    *,
    clean_path: str = "clean_lightcurves.csv",
    report_path: str = "preprocessing_report.csv",
    removed_points_path: str = "removed_points.csv",
):
    clean_df.to_csv(clean_path, index=False)
    source_report.to_csv(report_path, index=False)
    removed_points_df.to_csv(removed_points_path, index=False)

    print(f"Saved clean data to: {clean_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved removed points to: {removed_points_path}")