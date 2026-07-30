"""Visual diagnostics for Gaia lens-system light curves.

The public plotting functions accept a CSV path, DataFrame or nested component
dictionary.  Smoothing and Gaussian-process curves are display aids only; they
do not alter the scientific input table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


DEFAULT_FIELD_COL = "source_id"
DEFAULT_COMPONENT_COL = "lensComponentSourceId"
DEFAULT_TIMEZONE = "Europe/Paris"


# ============================================================
# DATA LOADING
# ============================================================

def _read_data(data: Any) -> pd.DataFrame:
    """
    Accepts:
    - CSV path
    - pandas DataFrame
    - nested dict {source_id: {lensComponentSourceId: DataFrame}}
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

                if DEFAULT_FIELD_COL not in tmp.columns:
                    tmp[DEFAULT_FIELD_COL] = source_id

                if DEFAULT_COMPONENT_COL not in tmp.columns:
                    tmp[DEFAULT_COMPONENT_COL] = component_id

                parts.append(tmp)

        if parts:
            return pd.concat(parts, ignore_index=True)

    raise ValueError("data must be a CSV path, pandas DataFrame, or nested dict")


def _guess_time_col(df: pd.DataFrame) -> str:
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


def _tzinfo(timezone: str):
    try:
        return ZoneInfo(timezone)
    except Exception:
        return None


def _jd_to_local_time(series: pd.Series, timezone: str) -> pd.Series:
    """
    Converts JD or MJD to local datetime.

    JD -> Unix seconds:
        unix_seconds = (JD - 2440587.5) * 86400

    MJD is detected around 30000-100000:
        JD = MJD + 2400000.5
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().any():
        jd = numeric.copy()

        # MJD usually around 50_000 - 70_000.
        # JD usually around 2_400_000+.
        mjd_mask = jd.between(30000, 100000)
        jd.loc[mjd_mask] = jd.loc[mjd_mask] + 2400000.5

        unix_seconds = (jd - 2440587.5) * 86400.0
        utc_time = pd.to_datetime(unix_seconds, unit="s", utc=True, errors="coerce")
        return utc_time.dt.tz_convert(timezone)

    utc_time = pd.to_datetime(series, utc=True, errors="coerce")
    return utc_time.dt.tz_convert(timezone)


def _to_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _downsample(sub: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    if max_points is None:
        return sub

    if len(sub) <= max_points:
        return sub

    idx = np.linspace(0, len(sub) - 1, max_points).astype(int)
    idx = np.unique(idx)

    return sub.iloc[idx].copy()


# ============================================================
# Y MODES
# ============================================================

def _apply_y_mode(y: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    """
    Returns:
        transformed_y, y_error_divisor

    Modes:
    - raw / none
    - centered
    - normalized / zscore
    - minmax
    - first=0
    - first=1
    """
    mode = str(mode).lower()
    out = y.astype(float).copy()

    finite = np.isfinite(out)

    if not finite.any():
        return out, 1.0

    yf = out[finite]

    if mode in ["raw", "none"]:
        return out, 1.0

    if mode == "centered":
        out[finite] = yf - np.nanmean(yf)
        return out, 1.0

    if mode in ["normalized", "normalize", "normalization", "zscore"]:
        std = np.nanstd(yf)
        mean = np.nanmean(yf)

        if std > 0:
            out[finite] = (yf - mean) / std
            return out, std

        return out, 1.0

    if mode == "minmax":
        ymin = np.nanmin(yf)
        ymax = np.nanmax(yf)
        scale = ymax - ymin

        if scale > 0:
            out[finite] = (yf - ymin) / scale
            return out, scale

        return out, 1.0

    if mode == "first=0":
        out[finite] = yf - yf[0]
        return out, 1.0

    if mode == "first=1":
        first = yf[0]

        if first != 0:
            out[finite] = yf / first
            return out, abs(first)

        return out, 1.0

    raise ValueError(
        "y_mode must be one of: "
        "'raw', 'centered', 'normalized', 'zscore', 'minmax', 'first=0', 'first=1'"
    )


# ============================================================
# TIME UTILS
# ============================================================

def _datetime_to_days(x_datetime: pd.Series | np.ndarray) -> np.ndarray:
    x = pd.to_datetime(x_datetime)
    return mdates.date2num(x)


def _days_to_datetime(x_days: np.ndarray, timezone: str):
    return mdates.num2date(x_days, tz=_tzinfo(timezone))


# ============================================================
# SMOOTHING
# ============================================================

def _smooth_curve(
    x_days: np.ndarray,
    y: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple rolling mean smoothing.
    """
    if window <= 1 or len(y) < 3:
        return x_days, y

    order = np.argsort(x_days)

    x_sorted = x_days[order]
    y_sorted = y[order]

    y_smooth = (
        pd.Series(y_sorted)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    return x_sorted, y_smooth


# ============================================================
# PREPARE DATA
# ============================================================

def _prepare_filtered_data(
    data: Any,
    lens_component_ids: list[str | int] | str | int,
    *,
    source_id: str | int | None,
    y_col: str,
    yerr_col: str,
    time_col: str | None,
    source_col: str,
    component_col: str,
    timezone: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    df = _read_data(data)

    if component_col not in df.columns:
        raise ValueError(f"Missing component_col={component_col!r}. Columns: {list(df.columns)}")

    if source_id is not None and source_col not in df.columns:
        raise ValueError(f"Missing source_col={source_col!r}. Columns: {list(df.columns)}")

    if y_col not in df.columns:
        raise ValueError(f"Missing y_col={y_col!r}. Columns: {list(df.columns)}")

    if time_col is None:
        time_col = _guess_time_col(df)

    if time_col not in df.columns:
        raise ValueError(f"Missing time_col={time_col!r}. Columns: {list(df.columns)}")

    component_ids = [str(x) for x in _to_list(lens_component_ids)]

    if len(component_ids) not in [1, 2]:
        raise ValueError("Give exactly 1 or 2 lensComponentSourceId.")

    df = df.copy()

    df["_component_key"] = df[component_col].astype(str)
    df["_time_local"] = _jd_to_local_time(df[time_col], timezone)
    df["_y_raw"] = pd.to_numeric(df[y_col], errors="coerce")

    if yerr_col in df.columns:
        df["_yerr_raw"] = pd.to_numeric(df[yerr_col], errors="coerce")
    else:
        df["_yerr_raw"] = np.nan

    df = df[df["_component_key"].isin(component_ids)]

    if source_id is not None:
        df = df[df[source_col].astype(str) == str(source_id)]

    df = df.dropna(subset=["_time_local", "_y_raw"])
    df = df.sort_values(["_component_key", "_time_local"])

    if df.empty:
        raise ValueError(
            "No data after filtering. Check lens_component_ids, source_id, y_col, and time_col."
        )

    return df, time_col, component_ids


# ============================================================
# MAIN PLOT FUNCTION
# ============================================================

def plot_lens_components_matplotlib(
    data: Any,
    lens_component_ids: list[str | int] | str | int,
    *,
    source_id: str | int | None = None,

    # Columns
    y_col: str = "flux_obs",
    yerr_col: str = "flux_obs_error",
    time_col: str | None = None,
    source_col: str = DEFAULT_FIELD_COL,
    component_col: str = DEFAULT_COMPONENT_COL,
    timezone: str = DEFAULT_TIMEZONE,

    # Y mode
    # "raw", "centered", "normalized", "zscore", "minmax", "first=0", "first=1"
    y_mode: str = "raw",

    # Display
    show_points: bool = True,
    show_smooth: bool = False,
    show_gp: bool = False,
    show_errorbars: bool = True,

    # Smooth curve
    smooth_window: int = 7,

    # Gaussian Process
    gp_points: int = 400,
    gp_length_scale_days: float = 180.0,
    gp_alpha_floor: float = 1e-6,

    # Important:
    # If GP is too flat, decrease: 0.1 -> 0.05 -> 0.02
    # If GP is too unstable, increase: 0.05 -> 0.1 -> 0.3
    # Avoid very tiny values like 0.001.
    gp_error_scale: float = 0.05,

    gp_show_uncertainty: bool = True,
    gp_fixed_kernel: bool = True,
    gp_suppress_warnings: bool = True,

    # Time delay in days
    time_delay_days: float = 0.0,
    time_delay_by_component: dict[str, float] | None = None,

    # Per-component Y corrections
    scale_by_component: dict[str, float] | None = None,
    offset_by_component: dict[str, float] | None = None,

    # Speed
    max_points_per_curve: int | None = None,

    # Matplotlib style
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    marker: str = "o",
    markersize: float = 4,
    linewidth: float = 2,
    alpha_points: float = 0.8,
    alpha_smooth: float = 0.95,
    alpha_gp: float = 0.95,
    uncertainty_alpha: float = 0.18,
    grid: bool = True,
    legend: bool = True,
    date_format: str = "%Y-%m-%d\n%H:%M",
    rotate_xticks: int = 0,

    # Existing axis
    ax=None,
):
    """
    Matplotlib classical plot for 1 or 2 lensComponentSourceId.

    Can show:
    - points
    - error bars with flux_obs_error
    - smooth rolling curve
    - stable Gaussian Process using flux_obs_error
    - time delay on one curve
    """

    df, resolved_time_col, component_ids = _prepare_filtered_data(
        data,
        lens_component_ids,
        source_id=source_id,
        y_col=y_col,
        yerr_col=yerr_col,
        time_col=time_col,
        source_col=source_col,
        component_col=component_col,
        timezone=timezone,
    )

    time_delay_by_component = time_delay_by_component or {}
    scale_by_component = scale_by_component or {}
    offset_by_component = offset_by_component or {}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for component_id in component_ids:
        sub = df[df["_component_key"] == component_id].copy()

        if sub.empty:
            print(f"Warning: no data for component {component_id}")
            continue

        sub = _downsample(sub, max_points_per_curve)

        delay_days = float(time_delay_by_component.get(component_id, time_delay_days))
        scale = float(scale_by_component.get(component_id, 1.0))
        offset = float(offset_by_component.get(component_id, 0.0))

        x = sub["_time_local"] + pd.to_timedelta(delay_days, unit="D")

        y_raw = sub["_y_raw"].to_numpy(dtype=float)
        y, y_error_divisor = _apply_y_mode(y_raw, y_mode)

        yerr = sub["_yerr_raw"].to_numpy(dtype=float)
        yerr = yerr / y_error_divisor

        y = y * scale + offset
        yerr = np.abs(yerr * scale)

        label_base = f"component {component_id}"

        if delay_days != 0:
            label_base += f" | delay={delay_days:g} d"

        # ----------------------------------------------------
        # POINTS
        # ----------------------------------------------------
        if show_points:
            has_error = np.isfinite(yerr).any()

            if show_errorbars and has_error:
                ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=marker,
                    markersize=markersize,
                    alpha=alpha_points,
                    capsize=2,
                    elinewidth=0.8,
                    linewidth=0.8,
                    label=label_base + " points",
                )
            else:
                ax.plot(
                    x,
                    y,
                    marker,
                    markersize=markersize,
                    alpha=alpha_points,
                    linestyle="None",
                    label=label_base + " points",
                )

        # ----------------------------------------------------
        # SMOOTH CURVE
        # ----------------------------------------------------
        if show_smooth:
            x_days = _datetime_to_days(x)

            x_smooth_days, y_smooth = _smooth_curve(
                x_days,
                y,
                window=smooth_window,
            )

            ax.plot(
                _days_to_datetime(x_smooth_days, timezone),
                y_smooth,
                linewidth=linewidth,
                alpha=alpha_smooth,
                label=label_base + f" smooth w={smooth_window}",
            )

        # ----------------------------------------------------
        # GAUSSIAN PROCESS
        # ----------------------------------------------------
        if show_gp:
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import ConstantKernel, RBF
                from sklearn.exceptions import ConvergenceWarning
            except ImportError as exc:
                raise ImportError(
                    "Gaussian Process requires scikit-learn. "
                    "Install with: pip install scikit-learn"
                ) from exc

            x_days = _datetime_to_days(x)

            finite = np.isfinite(x_days) & np.isfinite(y)

            use_yerr_for_gp = show_errorbars and np.isfinite(yerr).any()

            if use_yerr_for_gp:
                finite = finite & np.isfinite(yerr)

            x_train = x_days[finite]
            y_train = y[finite]

            if len(x_train) < 3:
                print(f"Warning: not enough valid points for GP component {component_id}")
                continue

            # Center X for numerical stability
            x0 = np.nanmin(x_train)
            X_train = (x_train - x0).reshape(-1, 1)

            # Center Y for numerical stability
            y_mean = np.nanmean(y_train)
            y_train_centered = y_train - y_mean

            # Use flux_obs_error as GP alpha.
            # gp_error_scale avoids GP becoming too flat because of huge errors.
            if use_yerr_for_gp:
                alpha = np.maximum((yerr[finite] * gp_error_scale) ** 2, gp_alpha_floor)
            else:
                alpha = gp_alpha_floor

            y_var = np.nanvar(y_train_centered)

            if not np.isfinite(y_var) or y_var <= 0:
                y_var = 1.0

            if gp_fixed_kernel:
                # Stable GP: fixed length scale, no optimizer.
                kernel = ConstantKernel(
                    y_var,
                    constant_value_bounds="fixed",
                ) * RBF(
                    length_scale=gp_length_scale_days,
                    length_scale_bounds="fixed",
                )

                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    normalize_y=False,
                    optimizer=None,
                    random_state=0,
                )

            else:
                # More automatic but less stable.
                kernel = ConstantKernel(
                    y_var,
                    constant_value_bounds=(1e-8, 1e8),
                ) * RBF(
                    length_scale=gp_length_scale_days,
                    length_scale_bounds=(1e-2, 1e5),
                )

                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    normalize_y=False,
                    n_restarts_optimizer=5,
                    random_state=0,
                )

            if gp_suppress_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gp.fit(X_train, y_train_centered)
            else:
                gp.fit(X_train, y_train_centered)

            x_grid_days = np.linspace(
                np.nanmin(x_train),
                np.nanmax(x_train),
                gp_points,
            )

            X_grid = (x_grid_days - x0).reshape(-1, 1)

            y_gp_centered, y_gp_std = gp.predict(X_grid, return_std=True)
            y_gp = y_gp_centered + y_mean

            x_grid_dates = _days_to_datetime(x_grid_days, timezone)

            ax.plot(
                x_grid_dates,
                y_gp,
                linewidth=linewidth + 0.8,
                alpha=alpha_gp,
                label=label_base + " GP",
            )

            if gp_show_uncertainty:
                ax.fill_between(
                    x_grid_dates,
                    y_gp - y_gp_std,
                    y_gp + y_gp_std,
                    alpha=uncertainty_alpha,
                    label=label_base + " GP ±1σ",
                )

    if title is None:
        title = f"{y_col} vs time | mode={y_mode}"

    if xlabel is None:
        xlabel = f"Local time ({timezone})"

    if ylabel is None:
        ylabel = y_col
        if y_mode not in ["raw", "none"]:
            ylabel += f" ({y_mode})"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(date_format, tz=_tzinfo(timezone))
    )

    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

    if legend:
        ax.legend()

    fig.tight_layout()

    return fig, ax


# ============================================================
# GP CONVENIENCE FUNCTION
# ============================================================

def plot_lens_components_gp(
    data: Any,
    lens_component_ids: list[str | int] | str | int,
    *,
    source_id: str | int | None = None,

    y_col: str = "flux_obs",
    yerr_col: str = "flux_obs_error",
    time_col: str | None = None,
    source_col: str = DEFAULT_FIELD_COL,
    component_col: str = DEFAULT_COMPONENT_COL,
    timezone: str = DEFAULT_TIMEZONE,

    y_mode: str = "centered",

    time_delay_days: float = 0.0,
    time_delay_by_component: dict[str, float] | None = None,

    gp_points: int = 500,
    gp_length_scale_days: float = 180.0,
    gp_error_scale: float = 0.05,
    gp_show_uncertainty: bool = True,
    gp_fixed_kernel: bool = True,

    show_points: bool = True,
    show_errorbars: bool = True,

    max_points_per_curve: int | None = None,

    figsize: tuple[float, float] = (13, 6),
    title: str = "Gaussian Process using flux_obs_error",
):
    """
    Shortcut for Gaussian Process plot.

    This uses:
    - show_gp=True
    - show_smooth=False
    - stable fixed-kernel GP by default
    """

    return plot_lens_components_matplotlib(
        data,
        lens_component_ids,
        source_id=source_id,
        y_col=y_col,
        yerr_col=yerr_col,
        time_col=time_col,
        source_col=source_col,
        component_col=component_col,
        timezone=timezone,
        y_mode=y_mode,
        show_points=show_points,
        show_smooth=False,
        show_gp=True,
        show_errorbars=show_errorbars,
        gp_points=gp_points,
        gp_length_scale_days=gp_length_scale_days,
        gp_error_scale=gp_error_scale,
        gp_show_uncertainty=gp_show_uncertainty,
        gp_fixed_kernel=gp_fixed_kernel,
        time_delay_days=time_delay_days,
        time_delay_by_component=time_delay_by_component,
        max_points_per_curve=max_points_per_curve,
        figsize=figsize,
        title=title,
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_lens_components(
    data: Any,
    *,
    source_id: str | int | None = None,
    source_col: str = DEFAULT_FIELD_COL,
    component_col: str = DEFAULT_COMPONENT_COL,
) -> list[str]:
    df = _read_data(data)

    if component_col not in df.columns:
        raise ValueError(f"Missing component_col={component_col!r}. Columns: {list(df.columns)}")

    if source_id is not None:
        if source_col not in df.columns:
            raise ValueError(f"Missing source_col={source_col!r}. Columns: {list(df.columns)}")

        df = df[df[source_col].astype(str) == str(source_id)]

    return sorted(df[component_col].astype(str).dropna().unique().tolist())


def list_sources(
    data: Any,
    *,
    source_col: str = DEFAULT_FIELD_COL,
) -> list[str]:
    df = _read_data(data)

    if source_col not in df.columns:
        raise ValueError(f"Missing source_col={source_col!r}. Columns: {list(df.columns)}")

    return sorted(df[source_col].astype(str).dropna().unique().tolist())


def list_numeric_columns(data: Any) -> list[str]:
    df = _read_data(data)

    cols = []

    for col in df.columns:
        test = pd.to_numeric(df[col], errors="coerce")

        if test.notna().any():
            cols.append(col)

    return cols
