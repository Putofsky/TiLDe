from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_SOURCE_COL = "source_id"
DEFAULT_COMPONENT_COL = "lensComponentSourceId"
DEFAULT_Y_COL = "flux_obs"
DEFAULT_YERR_COL = "flux_obs_error"


# ============================================================
# BASIC UTILS
# ============================================================

def read_lightcurve_data(data: Any) -> pd.DataFrame:
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
        "epoch_obs_jd",
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
        if "jd" in name or "julian" in name or "time" in name or "date" in name or "epoch" in name:
            return col

    raise ValueError(f"No time column found. Columns are: {list(df.columns)}")


def _time_to_days(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() >= max(3, int(0.5 * len(series))):
        return numeric.to_numpy(dtype=float)

    dt = pd.to_datetime(series, utc=True, errors="coerce")
    arr = dt.astype("int64").to_numpy(dtype=float)

    nat_value = np.iinfo("int64").min
    arr[arr == nat_value] = np.nan

    return arr / 1e9 / 86400.0


def _days_to_output_time(days: np.ndarray, original_series: pd.Series):
    numeric = pd.to_numeric(original_series, errors="coerce")

    if numeric.notna().sum() >= max(3, int(0.5 * len(original_series))):
        return days

    return pd.to_datetime(days * 86400.0, unit="s", utc=True)


def _safe_group_cols(
    df: pd.DataFrame,
    source_col: str,
    component_col: Optional[str],
) -> List[str]:
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


def _get_group_title(keys: Any, group_cols: List[str]) -> str:
    if not isinstance(keys, tuple):
        keys = (keys,)

    return ", ".join(f"{col}={val}" for col, val in zip(group_cols, keys))


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        return np.nan

    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    sigma = 1.4826 * mad

    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(values)

    return float(sigma)


def _safe_yerr(yerr: np.ndarray) -> np.ndarray:
    yerr = np.asarray(yerr, dtype=float)

    good = np.isfinite(yerr) & (yerr > 0)

    if good.any():
        fallback = np.nanmedian(yerr[good])
    else:
        fallback = 1.0

    return np.where(good, yerr, fallback)


# ============================================================
# FIRST X SOURCES
# ============================================================

def limit_first_sources(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    max_sources: Optional[int] = 2000,
    source_selection: str = "first",
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    unique_groups = df[group_cols].drop_duplicates().reset_index(drop=True)
    unique_groups["_source_rank"] = np.arange(len(unique_groups))

    total_sources = len(unique_groups)

    if max_sources is None:
        selected_groups = unique_groups.copy()

    else:
        max_sources = int(max_sources)

        if max_sources <= 0:
            raise ValueError("max_sources must be positive or None.")

        if source_selection == "first":
            selected_groups = unique_groups.head(max_sources).copy()

        elif source_selection == "random":
            selected_groups = (
                unique_groups
                .sample(
                    n=min(max_sources, total_sources),
                    random_state=random_state,
                )
                .sort_values("_source_rank")
                .copy()
            )

        else:
            raise ValueError("source_selection must be 'first' or 'random'.")

    limited = df.merge(
        selected_groups[group_cols + ["_source_rank"]],
        on=group_cols,
        how="inner",
    )

    limited = limited.sort_values("_source_rank").reset_index(drop=True)

    info = pd.DataFrame({
        "total_sources_available": [total_sources],
        "selected_sources": [len(selected_groups)],
        "max_sources": [max_sources],
        "source_selection": [source_selection],
    })

    return limited, info


# ============================================================
# BI-DAILY AGGREGATION
# ============================================================

def aggregate_bi_daily(
    data: Any,
    *,
    time_col: Optional[str] = None,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    source_col: str = DEFAULT_SOURCE_COL,
    component_col: Optional[str] = DEFAULT_COMPONENT_COL,
    bin_days: float = 2.0,
    error_mode: str = "mean",
    keep_other_cols: bool = True,
) -> pd.DataFrame:

    df = read_lightcurve_data(data)

    if time_col is None:
        time_col = guess_time_col(df)

    group_cols = _safe_group_cols(df, source_col, component_col)

    required = group_cols + [time_col, y_col]

    if yerr_col in df.columns:
        required.append(yerr_col)

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
            new_err = float(np.nanmean(err)) if finite_err.any() else np.nan

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

        if "_source_rank" in sub.columns:
            row["_source_rank"] = sub["_source_rank"].iloc[0]

        mean_time_days = float(np.nanmean(sub["_time_days"]))

        row[time_col] = _days_to_output_time(
            np.array([mean_time_days]),
            df[time_col],
        )[0]

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
# SNR
# ============================================================

def compute_snr_metrics(
    sub: pd.DataFrame,
    *,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    signal_mode: str = "amplitude",
) -> Dict[str, Any]:

    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)

    if yerr_col in sub.columns:
        yerr = pd.to_numeric(sub[yerr_col], errors="coerce").to_numpy(dtype=float)
    else:
        yerr = np.full_like(y, np.nan)

    y = y[np.isfinite(y)]
    yerr = yerr[np.isfinite(yerr) & (yerr > 0)]

    if len(y) < 3:
        return {
            "signal": np.nan,
            "noise": np.nan,
            "snr": np.nan,
            "noise_signal_ratio": np.nan,
            "snr_reason": "not_enough_flux_points",
        }

    if len(yerr) == 0:
        return {
            "signal": np.nan,
            "noise": np.nan,
            "snr": np.nan,
            "noise_signal_ratio": np.nan,
            "snr_reason": "no_valid_error_values",
        }

    if signal_mode == "amplitude":
        q05, q95 = np.nanpercentile(y, [5, 95])
        signal = 0.5 * (q95 - q05)

    elif signal_mode == "std":
        signal = np.nanstd(y)

    elif signal_mode == "robust_std":
        signal = _robust_sigma(y)

    else:
        raise ValueError("signal_mode must be 'amplitude', 'std', or 'robust_std'.")

    noise = np.nanmedian(yerr)

    if not np.isfinite(signal) or signal <= 0:
        snr = np.nan
        nsr = np.inf
        reason = "invalid_or_zero_signal"

    elif not np.isfinite(noise) or noise <= 0:
        snr = np.nan
        nsr = np.nan
        reason = "invalid_or_zero_noise"

    else:
        snr = signal / noise
        nsr = noise / signal
        reason = "ok"

    return {
        "signal": float(signal) if np.isfinite(signal) else signal,
        "noise": float(noise) if np.isfinite(noise) else noise,
        "snr": float(snr) if np.isfinite(snr) else snr,
        "noise_signal_ratio": float(nsr) if np.isfinite(nsr) else nsr,
        "snr_reason": reason,
    }


# ============================================================
# KALMAN SMOOTHER
# ============================================================

def _estimate_process_var(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    q_scale: float = 0.30,
) -> float:
    """
    Automatic process variance for constant-velocity Kalman model.

    Higher q_scale:
        follows faster variations.

    Lower q_scale:
        smoother model.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]

    if len(dt) == 0:
        median_dt = 1.0
    else:
        median_dt = float(np.nanmedian(dt))

    y_sigma = _robust_sigma(y)

    if not np.isfinite(y_sigma) or y_sigma <= 0:
        y_sigma = np.nanstd(y)

    if not np.isfinite(y_sigma) or y_sigma <= 0:
        y_sigma = np.nanmedian(yerr)

    if not np.isfinite(y_sigma) or y_sigma <= 0:
        y_sigma = 1.0

    median_dt = max(median_dt, 1e-6)

    # Want typical process position std over median_dt to be q_scale * y_sigma.
    q = 3.0 * (q_scale * y_sigma) ** 2 / (median_dt ** 3)

    return float(max(q, 1e-12))


def kalman_smoother_constant_velocity(
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    process_var: Optional[float] = None,
    process_var_scale: float = 0.30,
    model_floor: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Irregular-time Kalman filter + RTS smoother.

    State:
        x = [level, slope]

    Observation:
        y = level + noise

    This model is flexible enough for light curves but more stable than
    polynomial/DRW/Perona-Malik for preprocessing.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = _safe_yerr(np.asarray(yerr, dtype=float))

    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)

    if good.sum() < 4:
        raise ValueError("Not enough valid points for Kalman smoother.")

    original_positions = np.arange(len(t))[good]

    t_good = t[good]
    y_good = y[good]
    yerr_good = yerr[good]

    order = np.argsort(t_good)

    t_sorted = t_good[order]
    y_sorted = y_good[order]
    yerr_sorted = yerr_good[order]
    original_sorted_positions = original_positions[order]

    n = len(y_sorted)

    if process_var is None:
        q = _estimate_process_var(
            t_sorted,
            y_sorted,
            yerr_sorted,
            q_scale=process_var_scale,
        )
    else:
        q = float(process_var)

    q = max(q, 1e-12)

    if model_floor is None:
        model_floor_use = 0.5 * float(np.nanmedian(yerr_sorted))
    else:
        model_floor_use = float(model_floor)

    model_floor_use = max(model_floor_use, 1e-12)

    R = yerr_sorted ** 2 + model_floor_use ** 2

    H = np.array([[1.0, 0.0]])

    x_filt = np.zeros((n, 2), dtype=float)
    P_filt = np.zeros((n, 2, 2), dtype=float)
    x_pred = np.zeros((n, 2), dtype=float)
    P_pred = np.zeros((n, 2, 2), dtype=float)
    F_list = np.zeros((n, 2, 2), dtype=float)

    y_med = float(np.nanmedian(y_sorted))
    y_sig = _robust_sigma(y_sorted)

    if not np.isfinite(y_sig) or y_sig <= 0:
        y_sig = np.nanstd(y_sorted)

    if not np.isfinite(y_sig) or y_sig <= 0:
        y_sig = max(float(np.nanmedian(yerr_sorted)), 1.0)

    dt_all = np.diff(t_sorted)
    dt_good = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
    median_dt = float(np.nanmedian(dt_good)) if len(dt_good) > 0 else 1.0
    median_dt = max(median_dt, 1e-6)

    x_prev = np.array([y_med, 0.0], dtype=float)
    P_prev = np.array([
        [100.0 * y_sig ** 2, 0.0],
        [0.0, 100.0 * y_sig ** 2 / median_dt ** 2],
    ])

    for k in range(n):
        if k == 0:
            F = np.eye(2)
            Q = np.zeros((2, 2), dtype=float)
        else:
            dt = float(t_sorted[k] - t_sorted[k - 1])
            dt = max(dt, 0.0)

            F = np.array([
                [1.0, dt],
                [0.0, 1.0],
            ])

            Q = q * np.array([
                [dt ** 3 / 3.0, dt ** 2 / 2.0],
                [dt ** 2 / 2.0, dt],
            ])

        F_list[k] = F

        xp = F @ x_prev
        Pp = F @ P_prev @ F.T + Q

        x_pred[k] = xp
        P_pred[k] = Pp

        innovation = y_sorted[k] - float(H @ xp)
        S = float(H @ Pp @ H.T + R[k])
        S = max(S, 1e-12)

        K = (Pp @ H.T) / S

        xf = xp + (K[:, 0] * innovation)
        Pf = (np.eye(2) - K @ H) @ Pp

        x_filt[k] = xf
        P_filt[k] = Pf

        x_prev = xf
        P_prev = Pf

    # RTS smoother
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for k in range(n - 2, -1, -1):
        F_next = F_list[k + 1]
        Pp_next = P_pred[k + 1]

        try:
            C = P_filt[k] @ F_next.T @ np.linalg.inv(Pp_next)
        except np.linalg.LinAlgError:
            C = P_filt[k] @ F_next.T @ np.linalg.pinv(Pp_next)

        x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C @ (P_smooth[k + 1] - Pp_next) @ C.T

    f_sorted = x_smooth[:, 0]
    slope_sorted = x_smooth[:, 1]

    residual_sorted = y_sorted - f_sorted

    intrinsic_sigma = _robust_sigma(residual_sorted)

    if not np.isfinite(intrinsic_sigma) or intrinsic_sigma <= 0:
        intrinsic_sigma = np.nanstd(residual_sorted)

    if not np.isfinite(intrinsic_sigma) or intrinsic_sigma <= 0:
        intrinsic_sigma = float(np.nanmedian(yerr_sorted))

    total_sigma_sorted = np.sqrt(yerr_sorted ** 2 + intrinsic_sigma ** 2)
    z_sorted = residual_sorted / np.maximum(total_sigma_sorted, 1e-12)

    f_full = np.full(len(t), np.nan)
    z_full = np.full(len(t), np.nan)
    residual_full = np.full(len(t), np.nan)

    f_full[original_sorted_positions] = f_sorted
    z_full[original_sorted_positions] = z_sorted
    residual_full[original_sorted_positions] = residual_sorted

    reduced_chi2 = float(np.nanmean(z_sorted ** 2))
    rmse = float(np.sqrt(np.nanmean(residual_sorted ** 2)))

    return {
        "success": True,
        "t_sorted": t_sorted,
        "y_sorted": y_sorted,
        "yerr_sorted": yerr_sorted,
        "f_sorted": f_sorted,
        "slope_sorted": slope_sorted,
        "z_sorted": z_sorted,
        "f_full": f_full,
        "z_full": z_full,
        "residual_full": residual_full,
        "intrinsic_sigma": float(intrinsic_sigma),
        "reduced_chi2": reduced_chi2,
        "rmse": rmse,
        "process_var": q,
        "process_var_scale": process_var_scale,
        "model_floor": model_floor_use,
    }


def _predict_from_kalman_model(
    x_all: np.ndarray,
    x_train_sorted: np.ndarray,
    f_train_sorted: np.ndarray,
) -> np.ndarray:
    good = (
        np.isfinite(x_all)
        & np.isfinite(x_train_sorted).all()
        & np.isfinite(f_train_sorted).all()
    )

    if len(x_train_sorted) < 2:
        return np.full(len(x_all), np.nan)

    order = np.argsort(x_train_sorted)
    xt = x_train_sorted[order]
    ft = f_train_sorted[order]

    valid = np.isfinite(xt) & np.isfinite(ft)

    if valid.sum() < 2:
        return np.full(len(x_all), np.nan)

    return np.interp(x_all, xt[valid], ft[valid])


# ============================================================
# KALMANSAC
# ============================================================

def kalmansac_fit(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    process_var: Optional[float] = None,
    process_var_scale: float = 0.30,
    model_floor: Optional[float] = None,
    n_trials: int = 80,
    sample_fraction: float = 0.65,
    consensus_sigma: float = 4.0,
    refine_iter: int = 3,
    min_inliers: int = 10,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    KALMANSAC = Kalman smoother + RANSAC consensus.

    Why it is useful:
    - Outliers do not control the model, because each trial fits only a subset.
    - The final model is selected by maximum consensus.
    - Then it is refined on the consensus inliers.

    Output:
    - inlier_mask
    - outlier_mask
    - f_all
    - z_all
    - model
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = _safe_yerr(np.asarray(yerr, dtype=float))

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)

    if finite.sum() < 4:
        return {
            "success": False,
            "reason": "not_enough_valid_points_for_kalmansac",
            "inlier_mask": finite.copy(),
            "outlier_mask": ~finite,
            "f_all": np.full(len(x), np.nan),
            "z_all": np.full(len(x), np.nan),
            "model": None,
        }

    valid_positions = np.where(finite)[0]
    N = len(valid_positions)

    sample_size = int(np.ceil(sample_fraction * N))
    sample_size = max(4, min(sample_size, N))

    best_score = -np.inf
    best_inlier_mask_valid = None
    best_model = None
    best_f_valid = None
    best_z_valid = None

    x_valid = x[finite]
    y_valid = y[finite]
    yerr_valid = yerr[finite]

    trial_masks = []

    # Always try all-data model once.
    trial_masks.append(np.ones(N, dtype=bool))

    for _ in range(int(n_trials)):
        mask = np.zeros(N, dtype=bool)
        chosen = rng.choice(N, size=sample_size, replace=False)
        mask[chosen] = True
        trial_masks.append(mask)

    for trial_mask in trial_masks:
        if trial_mask.sum() < 4:
            continue

        try:
            model = kalman_smoother_constant_velocity(
                x_valid[trial_mask],
                y_valid[trial_mask],
                yerr_valid[trial_mask],
                process_var=process_var,
                process_var_scale=process_var_scale,
                model_floor=model_floor,
            )

            f_valid = _predict_from_kalman_model(
                x_valid,
                model["t_sorted"],
                model["f_sorted"],
            )

            residual = y_valid - f_valid

            intrinsic_sigma = _robust_sigma(residual[trial_mask])

            if not np.isfinite(intrinsic_sigma) or intrinsic_sigma <= 0:
                intrinsic_sigma = model["intrinsic_sigma"]

            if not np.isfinite(intrinsic_sigma) or intrinsic_sigma <= 0:
                intrinsic_sigma = np.nanmedian(yerr_valid)

            total_sigma = np.sqrt(yerr_valid ** 2 + intrinsic_sigma ** 2)
            z_valid = residual / np.maximum(total_sigma, 1e-12)

            inlier_mask_valid = np.isfinite(z_valid) & (np.abs(z_valid) <= consensus_sigma)

            n_inliers = int(inlier_mask_valid.sum())

            if n_inliers < max(4, min_inliers):
                continue

            median_abs_z = float(np.nanmedian(np.abs(z_valid[inlier_mask_valid])))
            robust_loss = float(np.nanmean(np.minimum(z_valid ** 2, consensus_sigma ** 2)))

            score = n_inliers - 0.05 * median_abs_z - 0.01 * robust_loss

            if score > best_score:
                best_score = score
                best_inlier_mask_valid = inlier_mask_valid.copy()
                best_model = model
                best_f_valid = f_valid.copy()
                best_z_valid = z_valid.copy()

        except Exception:
            continue

    if best_inlier_mask_valid is None:
        return {
            "success": False,
            "reason": "no_valid_kalmansac_consensus",
            "inlier_mask": finite.copy(),
            "outlier_mask": ~finite,
            "f_all": np.full(len(x), np.nan),
            "z_all": np.full(len(x), np.nan),
            "model": None,
        }

    inlier_mask_valid = best_inlier_mask_valid.copy()

    # Refinement on consensus inliers.
    for _ in range(int(refine_iter)):
        if inlier_mask_valid.sum() < 4:
            break

        try:
            model = kalman_smoother_constant_velocity(
                x_valid[inlier_mask_valid],
                y_valid[inlier_mask_valid],
                yerr_valid[inlier_mask_valid],
                process_var=process_var,
                process_var_scale=process_var_scale,
                model_floor=model_floor,
            )

            f_valid = _predict_from_kalman_model(
                x_valid,
                model["t_sorted"],
                model["f_sorted"],
            )

            residual = y_valid - f_valid

            intrinsic_sigma = _robust_sigma(residual[inlier_mask_valid])

            if not np.isfinite(intrinsic_sigma) or intrinsic_sigma <= 0:
                intrinsic_sigma = model["intrinsic_sigma"]

            if not np.isfinite(intrinsic_sigma) or intrinsic_sigma <= 0:
                intrinsic_sigma = np.nanmedian(yerr_valid)

            total_sigma = np.sqrt(yerr_valid ** 2 + intrinsic_sigma ** 2)
            z_valid = residual / np.maximum(total_sigma, 1e-12)

            new_inlier_mask_valid = np.isfinite(z_valid) & (np.abs(z_valid) <= consensus_sigma)

            if np.array_equal(new_inlier_mask_valid, inlier_mask_valid):
                best_model = model
                best_f_valid = f_valid
                best_z_valid = z_valid
                break

            inlier_mask_valid = new_inlier_mask_valid
            best_model = model
            best_f_valid = f_valid
            best_z_valid = z_valid

        except Exception:
            break

    inlier_mask_full = np.zeros(len(x), dtype=bool)
    outlier_mask_full = np.zeros(len(x), dtype=bool)
    f_all = np.full(len(x), np.nan)
    z_all = np.full(len(x), np.nan)

    inlier_mask_full[valid_positions] = inlier_mask_valid
    outlier_mask_full[valid_positions] = ~inlier_mask_valid
    outlier_mask_full[~finite] = True

    f_all[valid_positions] = best_f_valid
    z_all[valid_positions] = best_z_valid

    n_valid = int(finite.sum())
    n_inliers = int(inlier_mask_full.sum())
    n_outliers = int(outlier_mask_full[finite].sum())

    outlier_fraction = n_outliers / max(n_valid, 1)

    return {
        "success": True,
        "reason": "ok",
        "inlier_mask": inlier_mask_full,
        "outlier_mask": outlier_mask_full,
        "f_all": f_all,
        "z_all": z_all,
        "model": best_model,
        "n_valid": n_valid,
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "outlier_fraction": outlier_fraction,
        "score": best_score,
    }


def clean_source_with_kalmansac(
    sub: pd.DataFrame,
    *,
    time_col: str,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,

    process_var: Optional[float] = None,
    process_var_scale: float = 0.30,
    model_floor: Optional[float] = None,

    n_trials: int = 80,
    sample_fraction: float = 0.65,
    consensus_sigma: float = 4.0,
    refine_iter: int = 3,
    min_inliers: int = 10,
    random_state: int = 0,
) -> Dict[str, Any]:

    x_all = _time_to_days(sub[time_col])
    y_all = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)

    if yerr_col in sub.columns:
        yerr_all = pd.to_numeric(sub[yerr_col], errors="coerce").to_numpy(dtype=float)
    else:
        yerr_all = np.ones_like(y_all)

    yerr_all = _safe_yerr(yerr_all)

    result = kalmansac_fit(
        x_all,
        y_all,
        yerr_all,
        process_var=process_var,
        process_var_scale=process_var_scale,
        model_floor=model_floor,
        n_trials=n_trials,
        sample_fraction=sample_fraction,
        consensus_sigma=consensus_sigma,
        refine_iter=refine_iter,
        min_inliers=min_inliers,
        random_state=random_state,
    )

    return result


# ============================================================
# PLOTTING
# ============================================================

def _plot_kalmansac_result(
    sub: pd.DataFrame,
    *,
    time_col: str,
    y_col: str,
    yerr_col: str,
    title: str,
    reason: str,
    result: Optional[Dict[str, Any]] = None,
    removed_mask: Optional[np.ndarray] = None,
):
    x = _time_to_days(sub[time_col])
    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)

    if yerr_col in sub.columns:
        yerr = pd.to_numeric(sub[yerr_col], errors="coerce").to_numpy(dtype=float)
    else:
        yerr = np.full_like(y, np.nan)

    if removed_mask is None:
        removed_mask = np.zeros(len(sub), dtype=bool)

    removed_mask = np.asarray(removed_mask, dtype=bool)

    if len(removed_mask) != len(sub):
        removed_mask = np.zeros(len(sub), dtype=bool)

    kept = ~removed_mask

    fig, ax = plt.subplots(figsize=(12, 5))

    has_err = np.isfinite(yerr).any()

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

        if removed_mask.any():
            ax.errorbar(
                x[removed_mask],
                y[removed_mask],
                yerr=yerr[removed_mask],
                fmt="x",
                markersize=7,
                capsize=2,
                alpha=0.95,
                label="removed points",
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

        if removed_mask.any():
            ax.plot(
                x[removed_mask],
                y[removed_mask],
                "x",
                markersize=7,
                alpha=0.95,
                label="removed points",
            )

    if result is not None and result.get("success", False):
        f_all = result["f_all"]
        order = np.argsort(x)

        ax.plot(
            x[order],
            f_all[order],
            linewidth=2,
            label="KALMANSAC fit",
        )

    ax.set_title(f"{title}\n{reason}")
    ax.set_xlabel(f"{time_col} converted to numeric days")
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def preprocess_lightcurves_kalmansac(
    data: Any,
    *,
    time_col: Optional[str] = None,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    source_col: str = DEFAULT_SOURCE_COL,
    component_col: Optional[str] = DEFAULT_COMPONENT_COL,

    # First X sources
    max_sources: Optional[int] = 2000,
    source_selection: str = "first",

    # Bi-daily aggregation
    do_bi_daily: bool = True,
    bin_days: float = 2.0,
    error_mode: str = "mean",

    # SNR
    do_snr_filter: bool = True,
    snr_signal_mode: str = "amplitude",
    min_snr: Optional[float] = 1.0,
    max_noise_signal_ratio: Optional[float] = None,
    remove_if_snr_nan: bool = False,

    # KALMANSAC
    process_var: Optional[float] = None,
    process_var_scale: float = 0.30,
    model_floor: Optional[float] = None,
    n_trials: int = 80,
    sample_fraction: float = 0.65,
    consensus_sigma: float = 4.0,
    refine_iter: int = 3,
    min_inliers: int = 10,

    # Source-level removal
    min_points_after: int = 20,
    max_outlier_fraction: float = 0.50,

    # Display
    show_removed_plots: bool = True,
    max_plots: Optional[int] = 50,
    verbose: bool = True,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df0_all = read_lightcurve_data(data)

    if time_col is None:
        time_col = guess_time_col(df0_all)

    group_cols = _safe_group_cols(df0_all, source_col, component_col)

    df0, source_limit_info = limit_first_sources(
        df0_all,
        group_cols=group_cols,
        max_sources=max_sources,
        source_selection=source_selection,
        random_state=random_state,
    )

    if verbose:
        print("\n========== SOURCE LIMIT ==========")
        print(f"Total source/component groups available: {source_limit_info['total_sources_available'].iloc[0]}")
        print(f"Selected source/component groups: {source_limit_info['selected_sources'].iloc[0]}")
        print(f"max_sources: {max_sources}")
        print(f"source_selection: {source_selection}")
        print("==================================\n")

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
    plot_count = 0

    for group_i, (keys, sub) in enumerate(df.groupby(group_cols, sort=False)):
        sub = sub.copy()
        idx = sub.index.to_numpy()
        group_title = _get_group_title(keys, group_cols)

        n_after_aggregation = len(sub)

        snr_metrics = compute_snr_metrics(
            sub,
            y_col=y_col,
            yerr_col=yerr_col,
            signal_mode=snr_signal_mode,
        )

        signal = snr_metrics["signal"]
        noise = snr_metrics["noise"]
        snr = snr_metrics["snr"]
        nsr = snr_metrics["noise_signal_ratio"]
        snr_reason = snr_metrics["snr_reason"]

        # SNR filter
        remove_by_snr = False
        snr_remove_reason = ""

        if do_snr_filter:
            if remove_if_snr_nan and not np.isfinite(snr):
                remove_by_snr = True
                snr_remove_reason = f"bad_snr_nan_or_invalid ({snr_reason})"

            if min_snr is not None and np.isfinite(snr) and snr < min_snr:
                remove_by_snr = True
                snr_remove_reason = f"bad_snr_low (snr={snr:.3f} < min_snr={min_snr})"

            if (
                max_noise_signal_ratio is not None
                and np.isfinite(nsr)
                and nsr > max_noise_signal_ratio
            ):
                remove_by_snr = True
                snr_remove_reason = (
                    f"bad_noise_signal_ratio "
                    f"(nsr={nsr:.3f} > max_nsr={max_noise_signal_ratio})"
                )

        if remove_by_snr:
            df.loc[idx, "_pre_keep"] = False
            df.loc[idx, "_pre_removed_reason"] = snr_remove_reason

            source_reports.append({
                "group": group_title,
                "removed": True,
                "reason": snr_remove_reason,
                "n_after_aggregation": n_after_aggregation,
                "n_final": 0,
                "n_removed_points": 0,
                "outlier_fraction": np.nan,
                "signal": signal,
                "noise": noise,
                "snr": snr,
                "noise_signal_ratio": nsr,
                "snr_reason": snr_reason,
                "kalmansac_success": np.nan,
                "kalmansac_score": np.nan,
                "process_var": np.nan,
                "reduced_chi2": np.nan,
                "intrinsic_sigma": np.nan,
                "rmse": np.nan,
            })

            if verbose:
                print(f"REMOVED SOURCE: {group_title} | {snr_remove_reason}")

            if show_removed_plots and (max_plots is None or plot_count < max_plots):
                _plot_kalmansac_result(
                    sub,
                    time_col=time_col,
                    y_col=y_col,
                    yerr_col=yerr_col,
                    title=group_title,
                    reason=snr_remove_reason,
                )
                plot_count += 1

            continue

        if n_after_aggregation < 4:
            reason = "too_few_points_for_kalmansac"

            df.loc[idx, "_pre_keep"] = False
            df.loc[idx, "_pre_removed_reason"] = reason

            source_reports.append({
                "group": group_title,
                "removed": True,
                "reason": reason,
                "n_after_aggregation": n_after_aggregation,
                "n_final": 0,
                "n_removed_points": 0,
                "outlier_fraction": np.nan,
                "signal": signal,
                "noise": noise,
                "snr": snr,
                "noise_signal_ratio": nsr,
                "snr_reason": snr_reason,
                "kalmansac_success": False,
                "kalmansac_score": np.nan,
                "process_var": np.nan,
                "reduced_chi2": np.nan,
                "intrinsic_sigma": np.nan,
                "rmse": np.nan,
            })

            if verbose:
                print(f"REMOVED SOURCE: {group_title} | {reason}")

            continue

        try:
            result = clean_source_with_kalmansac(
                sub,
                time_col=time_col,
                y_col=y_col,
                yerr_col=yerr_col,
                process_var=process_var,
                process_var_scale=process_var_scale,
                model_floor=model_floor,
                n_trials=n_trials,
                sample_fraction=sample_fraction,
                consensus_sigma=consensus_sigma,
                refine_iter=refine_iter,
                min_inliers=min_inliers,
                random_state=random_state + group_i,
            )

            if not result["success"]:
                reason = f"kalmansac_failed: {result['reason']}"

                df.loc[idx, "_pre_keep"] = False
                df.loc[idx, "_pre_removed_reason"] = reason

                source_reports.append({
                    "group": group_title,
                    "removed": True,
                    "reason": reason,
                    "n_after_aggregation": n_after_aggregation,
                    "n_final": 0,
                    "n_removed_points": 0,
                    "outlier_fraction": np.nan,
                    "signal": signal,
                    "noise": noise,
                    "snr": snr,
                    "noise_signal_ratio": nsr,
                    "snr_reason": snr_reason,
                    "kalmansac_success": False,
                    "kalmansac_score": np.nan,
                    "process_var": np.nan,
                    "reduced_chi2": np.nan,
                    "intrinsic_sigma": np.nan,
                    "rmse": np.nan,
                })

                if verbose:
                    print(f"REMOVED SOURCE: {group_title} | {reason}")

                continue

            removed_mask = result["outlier_mask"]
            removed_idx = idx[removed_mask]

            if len(removed_idx) > 0:
                df.loc[removed_idx, "_pre_keep"] = False
                df.loc[removed_idx, "_pre_removed_reason"] = f"kalmansac_{consensus_sigma:g}sigma_outlier"

                tmp_removed = sub.loc[removed_idx].copy()
                tmp_removed["_pre_removed_reason"] = f"kalmansac_{consensus_sigma:g}sigma_outlier"
                tmp_removed["_pre_z_score"] = result["z_all"][removed_mask]
                tmp_removed["_pre_kalmansac_fit"] = result["f_all"][removed_mask]
                removed_points.append(tmp_removed)

            remove_whole_source = False
            source_reason = "kept_after_kalmansac_cleaning"

            if result["n_inliers"] < min_points_after:
                remove_whole_source = True
                source_reason = f"less_than_{min_points_after}_points_after_preprocessing"

            elif result["outlier_fraction"] > max_outlier_fraction:
                remove_whole_source = True
                source_reason = (
                    f"too_many_kalmansac_outliers "
                    f"(outlier_fraction={result['outlier_fraction']:.3f} > {max_outlier_fraction})"
                )

            if remove_whole_source:
                df.loc[idx, "_pre_keep"] = False
                df.loc[idx, "_pre_removed_reason"] = source_reason

                if verbose:
                    print(f"REMOVED SOURCE: {group_title} | {source_reason}")

                if show_removed_plots and (max_plots is None or plot_count < max_plots):
                    _plot_kalmansac_result(
                        sub,
                        time_col=time_col,
                        y_col=y_col,
                        yerr_col=yerr_col,
                        title=group_title,
                        reason=source_reason,
                        result=result,
                        removed_mask=removed_mask,
                    )
                    plot_count += 1

            else:
                if verbose and len(removed_idx) > 0:
                    print(
                        f"POINTS REMOVED: {group_title} | "
                        f"{len(removed_idx)} point(s), "
                        f"outlier_fraction={result['outlier_fraction']:.3f}"
                    )

                if (
                    show_removed_plots
                    and len(removed_idx) > 0
                    and (max_plots is None or plot_count < max_plots)
                ):
                    _plot_kalmansac_result(
                        sub,
                        time_col=time_col,
                        y_col=y_col,
                        yerr_col=yerr_col,
                        title=group_title,
                        reason=f"{len(removed_idx)} KALMANSAC outlier point(s)",
                        result=result,
                        removed_mask=removed_mask,
                    )
                    plot_count += 1

            model = result["model"]

            source_reports.append({
                "group": group_title,
                "removed": bool(remove_whole_source),
                "reason": source_reason,
                "n_after_aggregation": n_after_aggregation,
                "n_final": 0 if remove_whole_source else result["n_inliers"],
                "n_removed_points": result["n_outliers"],
                "outlier_fraction": result["outlier_fraction"],
                "signal": signal,
                "noise": noise,
                "snr": snr,
                "noise_signal_ratio": nsr,
                "snr_reason": snr_reason,
                "kalmansac_success": True,
                "kalmansac_score": result["score"],
                "process_var": model["process_var"] if model is not None else np.nan,
                "reduced_chi2": model["reduced_chi2"] if model is not None else np.nan,
                "intrinsic_sigma": model["intrinsic_sigma"] if model is not None else np.nan,
                "rmse": model["rmse"] if model is not None else np.nan,
            })

        except Exception as exc:
            reason = f"kalmansac_cleaning_failed: {type(exc).__name__}: {exc}"

            df.loc[idx, "_pre_keep"] = False
            df.loc[idx, "_pre_removed_reason"] = reason

            source_reports.append({
                "group": group_title,
                "removed": True,
                "reason": reason,
                "n_after_aggregation": n_after_aggregation,
                "n_final": 0,
                "n_removed_points": 0,
                "outlier_fraction": np.nan,
                "signal": signal,
                "noise": noise,
                "snr": snr,
                "noise_signal_ratio": nsr,
                "snr_reason": snr_reason,
                "kalmansac_success": False,
                "kalmansac_score": np.nan,
                "process_var": np.nan,
                "reduced_chi2": np.nan,
                "intrinsic_sigma": np.nan,
                "rmse": np.nan,
            })

            if verbose:
                print(f"REMOVED SOURCE: {group_title} | {reason}")

    clean_df = df[df["_pre_keep"]].copy()
    removed_rows_df = df[~df["_pre_keep"]].copy()

    if removed_points:
        removed_points_df = pd.concat(removed_points, ignore_index=True)
    else:
        removed_points_df = pd.DataFrame()

    source_report = pd.DataFrame(source_reports)

    clean_df = clean_df.drop(columns=["_pre_keep"], errors="ignore")
    removed_rows_df = removed_rows_df.drop(columns=["_pre_keep"], errors="ignore")

    if verbose:
        print("\n========== KALMANSAC PREPROCESSING SUMMARY ==========")
        print(f"Initial rows in full input: {len(df0_all)}")
        print(f"Rows selected by max_sources: {len(df0)}")
        print(f"Rows after bi-daily aggregation: {len(df)}")
        print(f"Final clean rows: {len(clean_df)}")
        print(f"Removed rows: {len(removed_rows_df)}")
        print(f"Removed individual points: {len(removed_points_df)}")

        if len(source_report) > 0:
            print("\nRemoved source/component groups by reason:")
            print(source_report[source_report["removed"] == True]["reason"].value_counts())

        print("====================================================\n")

    return clean_df.reset_index(drop=True), source_report, removed_points_df.reset_index(drop=True)


# Alias so old notebook call works.
preprocess_lightcurves = preprocess_lightcurves_kalmansac


# ============================================================
# SAVE OUTPUTS
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