"""
Pretraitement.py
================

Reusable preprocessing for EOLENS-style light curves.

Main function
-------------
    pretraitement(data, ...)

It applies the same methodology used in `PreProcessingTest.ipynb`:
1. optional first-N source/component selection,
2. per source/component binning using ((epoch_obs_jd - min_epoch) // bin_days),
3. adaptive rolling median with hybrid MAD + clipped-L2 scale, excluding the tested point,
4. SNR source/component filtering inside this module,
5. local CSV export of the cleaned data and local reports.

No download is triggered by this file. Outputs are saved only to the local paths
passed through `output_dir`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_SOURCE_COL = "source_id"
DEFAULT_COMPONENT_COL = "lensComponentSourceId"
DEFAULT_TIME_COL = "epoch_obs_jd"
DEFAULT_Y_COL = "flux_obs"
DEFAULT_YERR_COL = "flux_obs_error"


@dataclass
class PretraitementResult:
    """Container returned by `pretraitement`."""

    clean_df: pd.DataFrame
    source_report: pd.DataFrame
    removed_points_df: pd.DataFrame
    summary: Dict[str, Any]
    clean_csv_path: Path
    report_csv_path: Path
    removed_points_csv_path: Path
    summary_txt_path: Path


def read_lightcurve_data(data: Any) -> pd.DataFrame:
    """Read a CSV path, copy a DataFrame, or flatten a nested dict."""
    if isinstance(data, (str, Path)):
        return pd.read_csv(data, low_memory=False)

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, dict):
        parts: List[pd.DataFrame] = []
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
    """Guess the time column, with `epoch_obs_jd` preferred."""
    candidates = [
        DEFAULT_TIME_COL,
        "jd_time", "jdTime", "JD_TIME", "jd", "JD",
        "mjd", "MJD", "julian_date", "JulianDate", "julianDate",
        "time", "Time", "timestamp", "Timestamp", "datetime", "Datetime",
        "date", "Date",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        name = str(col).lower()
        if any(token in name for token in ["jd", "julian", "time", "date", "epoch"]):
            return col

    raise ValueError(f"No time column found. Available columns: {list(df.columns)}")


def _safe_group_cols(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SOURCE_COL,
    component_col: Optional[str] = DEFAULT_COMPONENT_COL,
) -> List[str]:
    cols: List[str] = []
    if source_col and source_col in df.columns:
        cols.append(source_col)
    if component_col and component_col in df.columns:
        cols.append(component_col)
    if not cols:
        raise ValueError(
            f"No group columns found. Tried source_col={source_col!r}, "
            f"component_col={component_col!r}."
        )
    return cols


def _group_title(keys: Any, group_cols: List[str]) -> str:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return ", ".join(f"{col}={val}" for col, val in zip(group_cols, keys))


def limit_first_sources(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    max_sources: Optional[int] = None,
    source_selection: str = "first",
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep all groups, the first N groups, or a random N groups."""
    unique_groups = df[group_cols].drop_duplicates().reset_index(drop=True)
    unique_groups["_source_rank"] = np.arange(len(unique_groups))
    total_sources = len(unique_groups)

    if max_sources is None:
        selected = unique_groups.copy()
    else:
        max_sources = int(max_sources)
        if max_sources <= 0:
            raise ValueError("max_sources must be positive or None")
        if source_selection == "first":
            selected = unique_groups.head(max_sources).copy()
        elif source_selection == "random":
            selected = (
                unique_groups
                .sample(n=min(max_sources, total_sources), random_state=random_state)
                .sort_values("_source_rank")
                .copy()
            )
        else:
            raise ValueError("source_selection must be 'first' or 'random'")

    out = df.merge(selected[group_cols + ["_source_rank"]], on=group_cols, how="inner")
    out = out.sort_values("_source_rank").reset_index(drop=True)

    info = pd.DataFrame({
        "total_source_component_groups_available": [total_sources],
        "selected_source_component_groups": [len(selected)],
        "max_sources": [max_sources],
        "source_selection": [source_selection],
    })
    return out, info


def aggregate_by_notebook_bins(
    data: Any,
    *,
    time_col: Optional[str] = None,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    source_col: str = DEFAULT_SOURCE_COL,
    component_col: Optional[str] = DEFAULT_COMPONENT_COL,
    bin_days: float = 4.0,
    error_mode: str = "mean",
    keep_other_cols: bool = True,
) -> pd.DataFrame:
    """
    Aggregate each source/component using the notebook bin formula.

    In the notebook, for each component:
        bin = ((epoch_obs_jd - epoch_obs_jd.min()) // 4).astype(int)
    """
    df = read_lightcurve_data(data)
    if time_col is None:
        time_col = guess_time_col(df)

    group_cols = _safe_group_cols(df, source_col, component_col)
    missing = [c for c in group_cols + [time_col, y_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    if yerr_col in d.columns:
        d[yerr_col] = pd.to_numeric(d[yerr_col], errors="coerce")
    else:
        d[yerr_col] = np.nan

    d = d.dropna(subset=[time_col, y_col]).sort_values(group_cols + [time_col])
    if d.empty:
        raise ValueError("No valid rows remain after converting time/flux columns")

    rows: List[Dict[str, Any]] = []

    for _, sub in d.groupby(group_cols, sort=False):
        sub = sub.sort_values(time_col).copy()
        group_min = float(sub[time_col].min())
        sub["_pre_bin_index"] = np.floor((sub[time_col] - group_min) / float(bin_days)).astype(int)

        for bin_index, b in sub.groupby("_pre_bin_index", sort=False):
            row: Dict[str, Any] = {}
            for col in group_cols:
                row[col] = b[col].iloc[0]
            if "_source_rank" in b.columns:
                row["_source_rank"] = b["_source_rank"].iloc[0]

            row[time_col] = float(b[time_col].mean())
            row[y_col] = float(b[y_col].mean())

            good_err = b[yerr_col].to_numpy(dtype=float)
            good_err = good_err[np.isfinite(good_err) & (good_err > 0)]

            if error_mode == "mean":
                row[yerr_col] = float(np.nanmean(good_err)) if len(good_err) else np.nan
            elif error_mode == "quadrature":
                row[yerr_col] = float(np.sqrt(np.nansum(good_err ** 2)) / len(good_err)) if len(good_err) else np.nan
            elif error_mode == "sem":
                y = b[y_col].to_numpy(dtype=float)
                row[yerr_col] = (
                    float(np.nanstd(y, ddof=1) / np.sqrt(np.isfinite(y).sum()))
                    if len(y) > 1
                    else (float(good_err[0]) if len(good_err) else np.nan)
                )
            else:
                raise ValueError("error_mode must be 'mean', 'quadrature', or 'sem'")

            row["n_points_in_bin"] = int(len(b))
            row["pre_bin_index"] = int(bin_index)
            row["pre_bin_start_day"] = float(group_min + int(bin_index) * float(bin_days))
            row["pre_bin_end_day"] = float(group_min + (int(bin_index) + 1) * float(bin_days))

            if keep_other_cols:
                for col in b.columns:
                    if col.startswith("_") or col in row or col in [time_col, y_col, yerr_col]:
                        continue
                    vals = b[col].dropna().unique()
                    if len(vals) == 1:
                        row[col] = vals[0]

            rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(group_cols + [time_col]).reset_index(drop=True)


def robust_hybrid_scale(values: Iterable[float], *, alpha: float = 0.7, c: float = 2.5) -> Tuple[float, float]:
    """
    Same robust scale used in the notebook.

    sigma = alpha * MAD_sigma + (1 - alpha) * clipped_L2_sigma
    """
    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan

    med = float(np.nanmedian(v))
    residuals = v - med
    mad = float(np.nanmedian(np.abs(residuals)))
    sigma_mad = 1.4826 * mad

    if not np.isfinite(sigma_mad) or sigma_mad <= 0:
        return med, 0.0

    clipped = np.clip(residuals, -float(c) * sigma_mad, float(c) * sigma_mad)
    sigma_l2_clip = float(np.sqrt(np.nanmean(clipped ** 2)))
    sigma = float(float(alpha) * sigma_mad + (1.0 - float(alpha)) * sigma_l2_clip)
    return med, sigma


def adaptive_rolling_hybrid_scores(
    sub: pd.DataFrame,
    *,
    y_col: str = DEFAULT_Y_COL,
    time_col: str = DEFAULT_TIME_COL,
    half_window: int = 5,
    alpha: float = 0.7,
    c: float = 2.5,
    exclude_center: bool = True,
) -> pd.DataFrame:
    """Return rolling median, hybrid sigma, bands, and local score for one group."""
    if len(sub) == 0:
        return sub.copy()

    d = sub.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.sort_values(time_col).reset_index(drop=False).rename(columns={"index": "_original_index"})

    n = len(d)
    half_window = int(half_window)
    if half_window < 1:
        raise ValueError("half_window must be >= 1")

    window = 2 * half_window + 1
    rolling_median: List[float] = []
    rolling_sigma: List[float] = []
    n_neighbors: List[int] = []

    for i in range(n):
        if i < half_window:
            idx = np.arange(0, min(window, n))
        elif i >= n - half_window:
            idx = np.arange(max(0, n - window), n)
        else:
            idx = np.arange(i - half_window, i + half_window + 1)

        if exclude_center:
            idx = idx[idx != i]

        values = d[y_col].iloc[idx].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        med, sigma = robust_hybrid_scale(values, alpha=alpha, c=c)

        rolling_median.append(med)
        rolling_sigma.append(sigma)
        n_neighbors.append(int(len(values)))

    d["pre_rolling_median"] = rolling_median
    d["pre_rolling_sigma_hybrid"] = rolling_sigma
    d["pre_n_neighbors_used"] = n_neighbors

    denom = d["pre_rolling_sigma_hybrid"].to_numpy(dtype=float)
    resid = d[y_col].to_numpy(dtype=float) - d["pre_rolling_median"].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = resid / denom

    z[~np.isfinite(denom) | (denom <= 0)] = np.nan
    d["pre_local_z"] = z

    return d


def compute_snr_metrics(
    sub: pd.DataFrame,
    *,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    signal_mode: str = "amplitude",
) -> Dict[str, Any]:
    """
    Compute source/component SNR.

    Default signal is the robust amplitude:
        0.5 * (P95(flux) - P5(flux))

    Noise is median positive `flux_obs_error`.
    """
    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]

    if yerr_col in sub.columns:
        yerr = pd.to_numeric(sub[yerr_col], errors="coerce").to_numpy(dtype=float)
    else:
        yerr = np.full(len(sub), np.nan)

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
        med = np.nanmedian(y)
        signal = 1.4826 * np.nanmedian(np.abs(y - med))
    else:
        raise ValueError("signal_mode must be 'amplitude', 'std', or 'robust_std'")

    noise = np.nanmedian(yerr)

    if not np.isfinite(signal) or signal <= 0:
        return {
            "signal": float(signal) if np.isfinite(signal) else np.nan,
            "noise": float(noise) if np.isfinite(noise) else np.nan,
            "snr": np.nan,
            "noise_signal_ratio": np.inf,
            "snr_reason": "invalid_or_zero_signal",
        }

    if not np.isfinite(noise) or noise <= 0:
        return {
            "signal": float(signal),
            "noise": float(noise) if np.isfinite(noise) else np.nan,
            "snr": np.nan,
            "noise_signal_ratio": np.nan,
            "snr_reason": "invalid_or_zero_noise",
        }

    snr = float(signal / noise)
    nsr = float(noise / signal)

    return {
        "signal": float(signal),
        "noise": float(noise),
        "snr": snr,
        "noise_signal_ratio": nsr,
        "snr_reason": "ok",
    }


def _should_remove_by_snr(
    snr_metrics: Dict[str, Any],
    *,
    min_snr: Optional[float],
    max_noise_signal_ratio: Optional[float],
    remove_if_snr_nan: bool,
) -> Tuple[bool, str]:
    snr = snr_metrics.get("snr", np.nan)
    nsr = snr_metrics.get("noise_signal_ratio", np.nan)
    reason = snr_metrics.get("snr_reason", "unknown")

    if remove_if_snr_nan and not np.isfinite(snr):
        return True, f"bad_snr_nan_or_invalid ({reason})"

    if min_snr is not None and np.isfinite(snr) and snr < float(min_snr):
        return True, f"bad_snr_low (snr={snr:.3f} < min_snr={float(min_snr):.3f})"

    if max_noise_signal_ratio is not None and np.isfinite(nsr) and nsr > float(max_noise_signal_ratio):
        return True, f"bad_noise_signal_ratio (nsr={nsr:.3f} > max_nsr={float(max_noise_signal_ratio):.3f})"

    return False, ""


def pretraitement(
    data: Any,
    *,
    output_dir: str | Path = "pretraitement_outputs",
    clean_filename: str = "cleaned_lightcurves.csv",
    report_filename: str = "pretraitement_report.csv",
    removed_points_filename: str = "removed_points.csv",
    summary_filename: str = "pretraitement_summary.txt",

    # Columns
    time_col: Optional[str] = None,
    y_col: str = DEFAULT_Y_COL,
    yerr_col: str = DEFAULT_YERR_COL,
    source_col: str = DEFAULT_SOURCE_COL,
    component_col: Optional[str] = DEFAULT_COMPONENT_COL,

    # Source/component selection
    max_sources: Optional[int] = None,
    source_selection: str = "first",
    random_state: int = 0,

    # Notebook binning
    do_binning: bool = True,
    bin_days: float = 4.0,
    error_mode: str = "mean",

    # Notebook adaptive rolling hybrid bands
    half_window: int = 5,
    alpha: float = 0.7,
    c: float = 2.5,
    exclude_center: bool = True,
    sigma_threshold: float = 4.0,

    # SNR filtering inside Pretraitement.py
    do_snr_filter: bool = True,
    snr_signal_mode: str = "amplitude",
    min_snr: Optional[float] = 0.8,
    max_noise_signal_ratio: Optional[float] = None,
    remove_if_snr_nan: bool = False,

    # Source/component removal rules
    min_points_before: int = 3,
    min_points_after: int = 20,
    max_outlier_fraction: float = 0.50,

    # Output behavior
    save: bool = True,
    verbose: bool = True,
) -> PretraitementResult:
    """
    Clean light curves and save a local cleaned CSV.

    Parameters are chosen to match `PreProcessingTest.ipynb` by default:
    - `bin_days=4.0` because the notebook uses `// 4`,
    - `half_window=5`, `alpha=0.7`, `c=2.5`, `exclude_center=True`,
    - outliers are points with |flux - rolling_median| > sigma_threshold * hybrid_sigma.
    """
    df_all = read_lightcurve_data(data)

    if time_col is None:
        time_col = guess_time_col(df_all)

    group_cols = _safe_group_cols(df_all, source_col, component_col)
    missing = [c for c in group_cols + [time_col, y_col] if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df_all.columns)}")

    initial_rows = int(len(df_all))
    initial_groups = int(df_all[group_cols].drop_duplicates().shape[0])

    selected_df, source_limit_info = limit_first_sources(
        df_all,
        group_cols=group_cols,
        max_sources=max_sources,
        source_selection=source_selection,
        random_state=random_state,
    )

    selected_rows = int(len(selected_df))
    selected_groups = int(source_limit_info["selected_source_component_groups"].iloc[0])

    if do_binning:
        work_df = aggregate_by_notebook_bins(
            selected_df,
            time_col=time_col,
            y_col=y_col,
            yerr_col=yerr_col,
            source_col=source_col,
            component_col=component_col,
            bin_days=bin_days,
            error_mode=error_mode,
        )
    else:
        work_df = selected_df.copy()
        work_df[time_col] = pd.to_numeric(work_df[time_col], errors="coerce")
        work_df[y_col] = pd.to_numeric(work_df[y_col], errors="coerce")

        if yerr_col in work_df.columns:
            work_df[yerr_col] = pd.to_numeric(work_df[yerr_col], errors="coerce")
        else:
            work_df[yerr_col] = np.nan

        work_df = (
            work_df
            .dropna(subset=[time_col, y_col])
            .sort_values(group_cols + [time_col])
            .reset_index(drop=True)
        )

    group_cols = _safe_group_cols(work_df, source_col, component_col)

    work_df = work_df.copy()
    work_df["_pre_keep"] = True
    work_df["_pre_removed_reason"] = ""

    reports: List[Dict[str, Any]] = []
    removed_points: List[pd.DataFrame] = []

    for keys, sub in work_df.groupby(group_cols, sort=False):
        group_name = _group_title(keys, group_cols)
        idx = sub.index.to_numpy()
        n_after_binning = int(len(sub))

        snr_metrics = compute_snr_metrics(
            sub,
            y_col=y_col,
            yerr_col=yerr_col,
            signal_mode=snr_signal_mode,
        )

        remove_by_snr, snr_remove_reason = _should_remove_by_snr(
            snr_metrics,
            min_snr=min_snr if do_snr_filter else None,
            max_noise_signal_ratio=max_noise_signal_ratio if do_snr_filter else None,
            remove_if_snr_nan=remove_if_snr_nan if do_snr_filter else False,
        )

        base_report = {
            "group": group_name,
            "n_after_binning": n_after_binning,
            "signal": snr_metrics["signal"],
            "noise": snr_metrics["noise"],
            "snr": snr_metrics["snr"],
            "noise_signal_ratio": snr_metrics["noise_signal_ratio"],
            "snr_reason": snr_metrics["snr_reason"],
        }

        if remove_by_snr:
            work_df.loc[idx, "_pre_keep"] = False
            work_df.loc[idx, "_pre_removed_reason"] = snr_remove_reason

            reports.append({
                **base_report,
                "removed_source": True,
                "source_reason": snr_remove_reason,
                "n_removed_points": 0,
                "n_final": 0,
                "outlier_fraction": np.nan,
                "max_abs_local_z": np.nan,
            })
            continue

        if n_after_binning < int(min_points_before):
            reason = f"less_than_{int(min_points_before)}_points_before_cleaning"
            work_df.loc[idx, "_pre_keep"] = False
            work_df.loc[idx, "_pre_removed_reason"] = reason

            reports.append({
                **base_report,
                "removed_source": True,
                "source_reason": reason,
                "n_removed_points": 0,
                "n_final": 0,
                "outlier_fraction": np.nan,
                "max_abs_local_z": np.nan,
            })
            continue

        scored = adaptive_rolling_hybrid_scores(
            sub,
            y_col=y_col,
            time_col=time_col,
            half_window=half_window,
            alpha=alpha,
            c=c,
            exclude_center=exclude_center,
        )

        original_idx = scored["_original_index"].to_numpy()
        local_z = scored["pre_local_z"].to_numpy(dtype=float)

        outlier_mask = np.isfinite(local_z) & (np.abs(local_z) > float(sigma_threshold))
        removed_idx = original_idx[outlier_mask]

        if len(removed_idx) > 0:
            work_df.loc[removed_idx, "_pre_keep"] = False
            work_df.loc[removed_idx, "_pre_removed_reason"] = "adaptive_rolling_hybrid_outlier"

            rm = scored.loc[outlier_mask].copy()
            rm["_pre_removed_reason"] = "adaptive_rolling_hybrid_outlier"
            removed_points.append(rm.drop(columns=["_original_index"], errors="ignore"))

        n_removed = int(outlier_mask.sum())
        n_final = int(n_after_binning - n_removed)
        outlier_fraction = float(n_removed / max(n_after_binning, 1))

        source_reason = "kept_after_adaptive_rolling_hybrid_cleaning"
        remove_source = False

        if n_final < int(min_points_after):
            source_reason = f"less_than_{int(min_points_after)}_points_after_cleaning"
            remove_source = True
        elif outlier_fraction > float(max_outlier_fraction):
            source_reason = (
                f"too_many_outliers "
                f"(outlier_fraction={outlier_fraction:.3f} > {float(max_outlier_fraction):.3f})"
            )
            remove_source = True

        if remove_source:
            work_df.loc[idx, "_pre_keep"] = False
            work_df.loc[idx, "_pre_removed_reason"] = source_reason
            n_final = 0

        reports.append({
            **base_report,
            "removed_source": bool(remove_source),
            "source_reason": source_reason,
            "n_removed_points": n_removed,
            "n_final": n_final,
            "outlier_fraction": outlier_fraction,
            "max_abs_local_z": float(np.nanmax(np.abs(local_z))) if np.isfinite(local_z).any() else np.nan,
        })

    clean_df = (
        work_df[work_df["_pre_keep"]]
        .copy()
        .drop(columns=["_pre_keep", "_pre_removed_reason"], errors="ignore")
    )

    removed_rows_df = (
        work_df[~work_df["_pre_keep"]]
        .copy()
        .drop(columns=["_pre_keep"], errors="ignore")
    )

    source_report = pd.DataFrame(reports)
    removed_points_df = pd.concat(removed_points, ignore_index=True) if removed_points else pd.DataFrame()

    n_removed_rows = int(len(removed_rows_df))
    n_removed_groups = int(source_report["removed_source"].sum()) if len(source_report) else 0
    n_clean_groups = int(clean_df[group_cols].drop_duplicates().shape[0]) if len(clean_df) else 0

    summary: Dict[str, Any] = {
        "input_rows": initial_rows,
        "input_source_component_groups": initial_groups,
        "selected_rows": selected_rows,
        "selected_source_component_groups": selected_groups,
        "rows_after_binning": int(len(work_df)),
        "clean_rows": int(len(clean_df)),
        "removed_rows": n_removed_rows,
        "clean_source_component_groups": n_clean_groups,
        "removed_source_component_groups": n_removed_groups,
        "removed_individual_outlier_points": int(len(removed_points_df)),
        "bin_days": float(bin_days) if do_binning else None,
        "half_window": int(half_window),
        "sigma_threshold": float(sigma_threshold),
        "snr_filter_enabled": bool(do_snr_filter),
        "min_snr": min_snr,
        "max_noise_signal_ratio": max_noise_signal_ratio,
    }

    output_dir = Path(output_dir)
    clean_csv_path = output_dir / clean_filename
    report_csv_path = output_dir / report_filename
    removed_points_csv_path = output_dir / removed_points_filename
    summary_txt_path = output_dir / summary_filename

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

        clean_df.to_csv(clean_csv_path, index=False)
        source_report.to_csv(report_csv_path, index=False)
        removed_points_df.to_csv(removed_points_csv_path, index=False)

        lines = ["PRETRAITEMENT SUMMARY", "====================", ""]
        for key, value in summary.items():
            lines.append(f"{key}: {value}")

        if len(source_report):
            lines.extend(["", "Removed source/component groups by reason:"])
            counts = source_report.loc[source_report["removed_source"], "source_reason"].value_counts()
            if len(counts):
                for reason, count in counts.items():
                    lines.append(f"- {reason}: {count}")
            else:
                lines.append("- none")

        summary_txt_path.write_text("\n".join(lines), encoding="utf-8")

    if verbose:
        print("\n========== PRETRAITEMENT SUMMARY ==========")
        for key, value in summary.items():
            print(f"{key}: {value}")

        if len(source_report):
            print("\nRemoved source/component groups by reason:")
            counts = source_report.loc[source_report["removed_source"], "source_reason"].value_counts()
            print(counts if len(counts) else "none")

        if save:
            print("\nLocal outputs saved to:")
            print(f"clean csv: {clean_csv_path}")
            print(f"report csv: {report_csv_path}")
            print(f"removed points csv: {removed_points_csv_path}")
            print(f"summary txt: {summary_txt_path}")

        print("===========================================\n")

    return PretraitementResult(
        clean_df=clean_df.reset_index(drop=True),
        source_report=source_report.reset_index(drop=True),
        removed_points_df=removed_points_df.reset_index(drop=True),
        summary=summary,
        clean_csv_path=clean_csv_path,
        report_csv_path=report_csv_path,
        removed_points_csv_path=removed_points_csv_path,
        summary_txt_path=summary_txt_path,
    )


# Backward-compatible aliases.
run_pretraitement = pretraitement
preprocess_lightcurves = pretraitement
preprocess_lightcurves_notebook_method = pretraitement


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean light curves and save local CSV outputs.")
    parser.add_argument("input_csv", help="Input CSV path")
    parser.add_argument("--output-dir", default="pretraitement_outputs", help="Local output directory")
    parser.add_argument("--max-sources", type=int, default=None, help="Optional first-N source/component groups")
    parser.add_argument("--min-snr", type=float, default=0.8, help="Minimum source/component SNR")
    parser.add_argument("--sigma-threshold", type=float, default=4.0, help="Adaptive rolling hybrid outlier threshold")

    args = parser.parse_args()

    pretraitement(
        args.input_csv,
        output_dir=args.output_dir,
        max_sources=args.max_sources,
        min_snr=args.min_snr,
        sigma_threshold=args.sigma_threshold,
        verbose=True,
    )