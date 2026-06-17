"""
RF.py
=====

Random-Forest inference for all available component pairs in a light-curve CSV.

Input expected columns by default:
    source_id, lensComponentSourceId, epoch_obs_jd, flux_obs, flux_obs_error

Output CSV columns only:
    sourceID, compA, compB, Proba

Example:
    python RF.py pretraitement_outputs/cleaned_lightcurves.csv \
        --model models/lens22_rf.joblib \
        --output rf_pair_predictions.csv
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, fields
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

@dataclass
class CFG:
    source_col: str = "source_id"
    id_col: str = "lensComponentSourceId"
    time_col: str = "epoch_obs_jd"
    flux_col: str = "flux_obs"
    err_col: str = "flux_obs_error"
    outlier_col: str = "flag_outlier"

    min_points_source: int = 25
    max_curves_for_bank: Optional[int] = None

    T_days: float = 2000.0
    max_abs_delay: float = 805.0
    n_min: int = 25
    n_max: int = 40
    max_len: int = 213
    input_dim: int = 6

    pos_fraction: float = 0.5
    strong_microlensing_prob: float = 0.45
    hard_negative_prob: float = 0.65

    n_delay_grid: int = 161
    min_overlap_for_corr: int = 5

    train_pairs: int = 10000
    val_pairs: int = 1000
    augment_views_train: int = 1
    augment_views_val: int = 1

    rf_n_estimators: int = 500
    rf_min_samples_leaf: int = 2
    rf_max_depth: Optional[int] = None
    rf_max_features: str = "sqrt"
    rf_n_jobs: int = -1


def cfg_from_bundle(bundle: Any) -> CFG:
    """Load CFG defaults and override them with the cfg saved in the joblib bundle."""
    cfg = CFG()
    if isinstance(bundle, dict) and "cfg" in bundle:
        saved = bundle["cfg"]
        if isinstance(saved, dict):
            for k, v in saved.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        else:
            for f in fields(CFG):
                if hasattr(saved, f.name):
                    setattr(cfg, f.name, getattr(saved, f.name))
    return cfg


def fix_sklearn_joblib_compatibility(model: Any) -> Any:
    """Small guard for loading a joblib created with a slightly different sklearn version."""
    # sklearn 1.8 renamed/changed some SimpleImputer internals. Older pickles may
    # miss _fill_dtype when loaded in newer sklearn. This makes predict_proba work.
    try:
        steps = getattr(model, "steps", None)
        estimators = [step for _, step in steps] if steps is not None else [model]
        for est in estimators:
            if est.__class__.__name__ == "SimpleImputer" and not hasattr(est, "_fill_dtype"):
                if hasattr(est, "_fit_dtype"):
                    est._fill_dtype = est._fit_dtype
                else:
                    est._fill_dtype = np.float64
    except Exception:
        pass
    return model


# ============================================================
# Basic robust utilities copied from the training/inference notebook
# ============================================================

PAIR_STATS_NAMES = [
    "pair_median_delta",
    "pair_log_scale_ratio",
    "pair_nA_token",
    "pair_nB_token",
    "pair_overlap_raw",
    "pair_span_diff",
]

MIC_POLY_DEGREES = (0, 1, 2, 3, 4)


def finite_or(x: Any, default: float = 0.0) -> float:
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def corr_np(a: Iterable[float], b: Iterable[float], min_n: int = 3) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < min_n:
        return 0.0
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return finite_or(np.corrcoef(a, b)[0, 1])


def robust_mad(x: Iterable[float]) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return finite_or(1.4826 * np.median(np.abs(x - med)))


def robust_scale(x: Iterable[float], eps: float = 1e-6) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        q75, q25 = np.percentile(x, [75, 25])
        scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < eps:
        scale = np.std(x)
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return float(med), float(scale)


def robust_z(x: Iterable[float], eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    med = np.nanmedian(x)
    mad = robust_mad(x)
    if not np.isfinite(mad) or mad < eps:
        mad = np.nanstd(x)
    if not np.isfinite(mad) or mad < eps:
        mad = 1.0
    return (x - med) / mad


def weighted_median(x: Iterable[float], w: Optional[Iterable[float]] = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    if w is None:
        return float(np.median(x))
    w = np.asarray(w, dtype=np.float64)
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    return float(x[np.searchsorted(cw, cutoff)])


# ============================================================
# Packing real curves exactly like the notebook inference
# ============================================================

def pack_curve(t: Iterable[float], y: Iterable[float], sigma: Iterable[float], cfg: CFG):
    """
    Convert sparse curve into padded tokens:
    [t_norm, log_dt_norm, flux_norm, err_norm, sin_time, cos_time]
    """
    t = np.asarray(t, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)

    good = np.isfinite(t) & np.isfinite(y)
    t, y, sigma = t[good], y[good], sigma[good]
    if len(t) == 0:
        x = np.zeros((cfg.max_len, cfg.input_dim), dtype=np.float32)
        mask = np.zeros((cfg.max_len,), dtype=np.float32)
        return x, mask, {"median": 0.0, "scale": 1.0, "n": 0, "t_min": 0.0, "t_max": 0.0}

    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nanmedian(np.abs(y - np.nanmedian(y))) * 0.05 + 1e-3)

    order = np.argsort(t)
    t, y, sigma = t[order], y[order], sigma[order]

    med, scale = robust_scale(y)
    y_norm = (y - med) / scale
    sig_norm = np.clip(sigma / scale, 1e-4, 5.0)

    n = min(len(t), int(cfg.max_len))
    t, y_norm, sig_norm = t[:n], y_norm[:n], sig_norm[:n]

    dt = np.diff(t, prepend=t[0])
    dt_norm = np.log1p(dt) / np.log1p(cfg.T_days)

    t_norm = t / cfg.T_days
    sin_t = np.sin(2 * np.pi * t_norm)
    cos_t = np.cos(2 * np.pi * t_norm)

    x = np.zeros((cfg.max_len, cfg.input_dim), dtype=np.float32)
    mask = np.zeros((cfg.max_len,), dtype=np.float32)

    feats = np.stack([t_norm, dt_norm, y_norm, sig_norm, sin_t, cos_t], axis=-1).astype(np.float32)
    x[:n] = feats
    mask[:n] = 1.0

    return x, mask, {
        "median": float(med),
        "scale": float(scale),
        "n": int(n),
        "t_min": float(t.min()),
        "t_max": float(t.max()),
    }


def extract_curve(x: np.ndarray, mask: np.ndarray, cfg: CFG):
    x = np.asarray(x, dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    xx = x[mask]
    if len(xx) == 0:
        return np.array([]), np.array([]), np.array([])
    t = xx[:, 0].astype(np.float64) * cfg.T_days
    y = xx[:, 2].astype(np.float64)
    e = xx[:, 3].astype(np.float64)
    order = np.argsort(t)
    return t[order], y[order], e[order]


# ============================================================
# Feature engineering from the RF notebook
# ============================================================

def polyfit_values(t, y, deg: int = 2):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(y) < deg + 3 or np.std(t) < 1e-8:
        return np.full_like(y, np.nanmedian(y) if len(y) else 0.0), np.zeros(deg + 1)
    tn = (t - np.mean(t)) / (np.std(t) + 1e-8)
    try:
        coef = np.polyfit(tn, y, deg=deg)
        fit = np.polyval(coef, tn)
        return fit, coef
    except Exception:
        return np.full_like(y, np.nanmedian(y) if len(y) else 0.0), np.zeros(deg + 1)


def detrend_poly(t, y, deg: int = 2):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(y) < deg + 3 or np.std(t) < 1e-8:
        return y - np.nanmedian(y) if len(y) else y
    fit, _ = polyfit_values(t, y, deg=deg)
    return y - fit


def derivative_series(t, y):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(t) < 3:
        return np.array([]), np.array([])
    dt = np.diff(t)
    dy = np.diff(y)
    m = np.isfinite(dt) & np.isfinite(dy) & (dt > 1e-5)
    if m.sum() < 2:
        return np.array([]), np.array([])
    tm = 0.5 * (t[:-1] + t[1:])
    d = dy / np.sqrt(dt + 1.0)
    return tm[m], d[m]


def moment_skew_kurt(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0, 0.0
    mu = np.mean(x)
    sd = np.std(x)
    if sd < 1e-8:
        return 0.0, 0.0
    z = (x - mu) / sd
    return finite_or(np.mean(z ** 3)), finite_or(np.mean(z ** 4) - 3.0)


def polynomial_shape_features(prefix, t, y, max_deg: int = 4):
    feats = {}
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    var_y = np.var(y) + 1e-8
    for deg in range(1, max_deg + 1):
        if len(y) < deg + 4:
            feats[f"{prefix}_poly{deg}_explained_frac"] = 0.0
            feats[f"{prefix}_poly{deg}_resid_mad"] = 0.0
            feats[f"{prefix}_poly{deg}_coef_l2"] = 0.0
            continue
        fit, coef = polyfit_values(t, y, deg=deg)
        resid = y - fit
        feats[f"{prefix}_poly{deg}_explained_frac"] = finite_or(1.0 - np.var(resid) / var_y)
        feats[f"{prefix}_poly{deg}_resid_mad"] = robust_mad(resid)
        feats[f"{prefix}_poly{deg}_coef_l2"] = finite_or(np.sqrt(np.sum(np.asarray(coef, dtype=float) ** 2)))
    return feats


def curve_features(prefix, t, y, e):
    feats = {}
    n = len(y)
    feats[f"{prefix}_n"] = n

    if n == 0:
        return feats

    gaps = np.diff(t) if n >= 2 else np.array([0.0])
    q05, q25, q50, q75, q95 = np.percentile(y, [5, 25, 50, 75, 95])
    skew, kurt = moment_skew_kurt(y)
    resid1 = detrend_poly(t, y, deg=1)
    resid2 = detrend_poly(t, y, deg=2)
    resid3 = detrend_poly(t, y, deg=3)
    _, dy = derivative_series(t, y)

    if n >= 3 and np.std(t) > 1e-8:
        tn = (t - t.mean()) / (t.std() + 1e-8)
        try:
            slope = np.polyfit(tn, y, deg=1)[0]
        except Exception:
            slope = 0.0
        try:
            quad = np.polyfit(tn, y, deg=2)[0]
        except Exception:
            quad = 0.0
        try:
            cubic = np.polyfit(tn, y, deg=3)[0]
        except Exception:
            cubic = 0.0
    else:
        slope, quad, cubic = 0.0, 0.0, 0.0

    feats.update({
        f"{prefix}_span": finite_or(t.max() - t.min()),
        f"{prefix}_t_start": finite_or(t.min()),
        f"{prefix}_t_end": finite_or(t.max()),
        f"{prefix}_gap_mean": finite_or(np.mean(gaps)),
        f"{prefix}_gap_median": finite_or(np.median(gaps)),
        f"{prefix}_gap_std": finite_or(np.std(gaps)),
        f"{prefix}_gap_max": finite_or(np.max(gaps)),
        f"{prefix}_gap_iqr": finite_or(np.percentile(gaps, 75) - np.percentile(gaps, 25)),
        f"{prefix}_y_mean": finite_or(np.mean(y)),
        f"{prefix}_y_std": finite_or(np.std(y)),
        f"{prefix}_y_median": finite_or(q50),
        f"{prefix}_y_mad": robust_mad(y),
        f"{prefix}_y_iqr": finite_or(q75 - q25),
        f"{prefix}_y_amp_q90": finite_or(q95 - q05),
        f"{prefix}_y_min": finite_or(np.min(y)),
        f"{prefix}_y_max": finite_or(np.max(y)),
        f"{prefix}_y_skew": skew,
        f"{prefix}_y_kurt": kurt,
        f"{prefix}_trend_slope": finite_or(slope),
        f"{prefix}_trend_quad": finite_or(quad),
        f"{prefix}_trend_cubic": finite_or(cubic),
        f"{prefix}_resid1_std": finite_or(np.std(resid1)),
        f"{prefix}_resid2_std": finite_or(np.std(resid2)),
        f"{prefix}_resid3_std": finite_or(np.std(resid3)),
        f"{prefix}_resid2_over_raw_std": finite_or(np.std(resid2) / (np.std(y) + 1e-6)),
        f"{prefix}_resid3_over_raw_std": finite_or(np.std(resid3) / (np.std(y) + 1e-6)),
        f"{prefix}_roughness": finite_or(np.std(np.diff(resid2)) if len(resid2) >= 2 else 0.0),
        f"{prefix}_roughness3": finite_or(np.std(np.diff(resid3)) if len(resid3) >= 2 else 0.0),
        f"{prefix}_acf1": corr_np(y[:-1], y[1:]) if n >= 4 else 0.0,
        f"{prefix}_dy_std": finite_or(np.std(dy) if len(dy) else 0.0),
        f"{prefix}_dy_mad": robust_mad(dy) if len(dy) else 0.0,
        f"{prefix}_err_median": finite_or(np.median(e)),
        f"{prefix}_err_mean": finite_or(np.mean(e)),
    })
    feats.update(polynomial_shape_features(prefix, t, y, max_deg=4))
    return feats


def aligned_overlap_values(tA, yA, tB, yB, tau, min_overlap: int = 8, normalize: bool = False):
    if len(tA) < min_overlap or len(tB) < min_overlap:
        return None

    tB_shift = tB - tau
    lo = max(np.min(tA), np.min(tB_shift))
    hi = min(np.max(tA), np.max(tB_shift))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    mA = (tA >= lo) & (tA <= hi)
    n = int(mA.sum())
    if n < min_overlap:
        return None

    tt = tA[mA]
    yyA = yA[mA]
    yyB = np.interp(tt, tB_shift, yB)
    if normalize:
        yyA = robust_z(yyA)
        yyB = robust_z(yyB)
    return tt, yyA, yyB, n


def interpolate_corr_at_delay(tA, yA, tB, yB, tau, min_overlap: int = 8):
    ov = aligned_overlap_values(tA, yA, tB, yB, tau, min_overlap=min_overlap, normalize=False)
    if ov is None:
        return 0.0, 3.0, 0
    _, yyA, yyB, n = ov
    c = corr_np(yyA, yyB, min_n=min_overlap)
    a = robust_z(yyA)
    b = robust_z(yyB)
    mae = finite_or(np.median(np.abs(a - b)), default=3.0)
    return c, mae, n


def microlens_poly_scores_at_delay(tA, yA, tB, yB, tau, cfg: CFG, degrees=MIC_POLY_DEGREES):
    out = {}
    for deg in degrees:
        out[deg] = {
            "score": -1.0,
            "corr": 0.0,
            "mae": 3.0,
            "resid_mad": 3.0,
            "improve": 0.0,
            "var_improve": 0.0,
            "coef_l2": 0.0,
            "slope_abs": 0.0,
            "curv_abs": 0.0,
            "cubic_abs": 0.0,
            "n": 0.0,
            "n_frac": 0.0,
        }

    ov = aligned_overlap_values(tA, yA, tB, yB, tau, min_overlap=cfg.min_overlap_for_corr, normalize=True)
    if ov is None:
        return out

    tt, a, b, n = ov
    n_frac = np.clip(n / max(min(len(tA), len(tB)), 1), 0.0, 1.0)

    diff = a - b
    base_mad = robust_mad(diff)
    base_std = np.std(diff) + 1e-8

    for deg in degrees:
        if len(diff) < deg + 4 or np.std(tt) < 1e-8:
            fit = np.full_like(diff, np.median(diff))
            coef = np.zeros(deg + 1)
        else:
            fit, coef = polyfit_values(tt, diff, deg=deg)

        b_corr = b + fit
        resid = a - b_corr

        corr_corr = corr_np(a, b_corr, min_n=cfg.min_overlap_for_corr)
        mae_corr = finite_or(np.median(np.abs(resid)), default=3.0)
        resid_mad = robust_mad(resid)
        resid_std = np.std(resid)
        improve = finite_or((base_mad - resid_mad) / (base_mad + 1e-6))
        var_improve = finite_or(1.0 - (resid_std ** 2 + 1e-8) / (base_std ** 2 + 1e-8))
        coef = np.asarray(coef, dtype=np.float64)

        slope_abs = abs(coef[-2]) if deg >= 1 and len(coef) >= 2 else 0.0
        curv_abs = abs(coef[-3]) if deg >= 2 and len(coef) >= 3 else 0.0
        cubic_abs = abs(coef[-4]) if deg >= 3 and len(coef) >= 4 else 0.0

        score = (
            0.55 * corr_corr
            + 0.25 * n_frac
            + 0.25 * np.clip(improve, -1.0, 1.0)
            + 0.15 * np.clip(var_improve, -1.0, 1.0)
            - 0.10 * np.clip(mae_corr, 0.0, 5.0)
        )

        out[deg] = {
            "score": finite_or(score, default=-1.0),
            "corr": finite_or(corr_corr),
            "mae": finite_or(mae_corr, default=3.0),
            "resid_mad": finite_or(resid_mad, default=3.0),
            "improve": finite_or(improve),
            "var_improve": finite_or(var_improve),
            "coef_l2": finite_or(np.sqrt(np.sum(coef ** 2))),
            "slope_abs": finite_or(slope_abs),
            "curv_abs": finite_or(curv_abs),
            "cubic_abs": finite_or(cubic_abs),
            "n": finite_or(n),
            "n_frac": finite_or(n_frac),
        }
    return out


def add_array_summary_features(feats: Dict[str, float], arr, delay_grid, name: str) -> int:
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0 or not np.any(np.isfinite(arr)):
        feats[f"{name}_max"] = -1.0
        feats[f"{name}_tau"] = 0.0
        feats[f"{name}_abs_tau"] = 0.0
        feats[f"{name}_zero"] = 0.0
        feats[f"{name}_mean"] = 0.0
        feats[f"{name}_std"] = 0.0
        feats[f"{name}_p95"] = 0.0
        feats[f"{name}_frac_above_03"] = 0.0
        feats[f"{name}_frac_above_05"] = 0.0
        feats[f"{name}_sharp_top1_top5"] = 0.0
        return 0

    arr_safe = np.where(np.isfinite(arr), arr, -1.0)
    idx = int(np.argmax(arr_safe))
    top = np.sort(arr_safe)[::-1]

    feats[f"{name}_max"] = finite_or(arr_safe[idx])
    feats[f"{name}_tau"] = finite_or(delay_grid[idx])
    feats[f"{name}_abs_tau"] = abs(finite_or(delay_grid[idx]))
    feats[f"{name}_zero"] = finite_or(arr_safe[np.argmin(np.abs(delay_grid))])
    feats[f"{name}_mean"] = finite_or(np.mean(arr_safe))
    feats[f"{name}_std"] = finite_or(np.std(arr_safe))
    feats[f"{name}_p95"] = finite_or(np.percentile(arr_safe, 95))
    feats[f"{name}_frac_above_03"] = finite_or(np.mean(arr_safe > 0.30))
    feats[f"{name}_frac_above_05"] = finite_or(np.mean(arr_safe > 0.50))
    feats[f"{name}_sharp_top1_top5"] = finite_or(top[0] - np.mean(top[1:6])) if len(top) > 6 else 0.0
    return idx


def delay_scan_features(tA, yA, tB, yB, cfg: CFG):
    feats = {}
    delay_grid = np.linspace(-cfg.max_abs_delay, cfg.max_abs_delay, int(cfg.n_delay_grid))

    rA = detrend_poly(tA, yA, deg=2)
    rB = detrend_poly(tB, yB, deg=2)
    rA3 = detrend_poly(tA, yA, deg=3)
    rB3 = detrend_poly(tB, yB, deg=3)
    tAd, dA = derivative_series(tA, yA)
    tBd, dB = derivative_series(tB, yB)

    raw_corr, res_corr, res3_corr, der_corr = [], [], [], []
    raw_mae, res_mae, n_ov = [], [], []

    ml_arrays = {
        deg: {k: [] for k in [
            "score", "corr", "mae", "resid_mad", "improve", "var_improve",
            "coef_l2", "slope_abs", "curv_abs", "cubic_abs", "n_frac"
        ]}
        for deg in MIC_POLY_DEGREES
    }

    for tau in delay_grid:
        c_raw, mae_raw, n = interpolate_corr_at_delay(tA, yA, tB, yB, tau, min_overlap=cfg.min_overlap_for_corr)
        c_res, mae_res, _ = interpolate_corr_at_delay(tA, rA, tB, rB, tau, min_overlap=cfg.min_overlap_for_corr)
        c_res3, _, _ = interpolate_corr_at_delay(tA, rA3, tB, rB3, tau, min_overlap=cfg.min_overlap_for_corr)

        if len(tAd) >= max(5, cfg.min_overlap_for_corr - 3) and len(tBd) >= max(5, cfg.min_overlap_for_corr - 3):
            c_der, _, _ = interpolate_corr_at_delay(tAd, dA, tBd, dB, tau, min_overlap=max(5, cfg.min_overlap_for_corr - 3))
        else:
            c_der = 0.0

        ml = microlens_poly_scores_at_delay(tA, yA, tB, yB, tau, cfg, degrees=MIC_POLY_DEGREES)
        for deg in MIC_POLY_DEGREES:
            for k in ml_arrays[deg].keys():
                ml_arrays[deg][k].append(ml[deg].get(k, 0.0))

        raw_corr.append(c_raw)
        res_corr.append(c_res)
        res3_corr.append(c_res3)
        der_corr.append(c_der)
        raw_mae.append(mae_raw)
        res_mae.append(mae_res)
        n_ov.append(n)

    raw_corr = np.asarray(raw_corr, dtype=np.float64)
    res_corr = np.asarray(res_corr, dtype=np.float64)
    res3_corr = np.asarray(res3_corr, dtype=np.float64)
    der_corr = np.asarray(der_corr, dtype=np.float64)
    raw_mae = np.asarray(raw_mae, dtype=np.float64)
    res_mae = np.asarray(res_mae, dtype=np.float64)
    n_ov = np.asarray(n_ov, dtype=np.float64)

    n_norm = np.clip(n_ov / max(min(len(tA), len(tB)), 1), 0, 1)
    combined = (
        0.30 * raw_corr
        + 0.35 * res_corr
        + 0.15 * res3_corr
        + 0.20 * der_corr
        + 0.05 * n_norm
        - 0.04 * np.clip(res_mae, 0, 5)
    )
    combined = np.where(n_ov >= cfg.min_overlap_for_corr, combined, -1.0)

    idx_raw = add_array_summary_features(feats, raw_corr, delay_grid, "scan_raw")
    idx_res = add_array_summary_features(feats, res_corr, delay_grid, "scan_resid")
    idx_res3 = add_array_summary_features(feats, res3_corr, delay_grid, "scan_resid3")
    idx_der = add_array_summary_features(feats, der_corr, delay_grid, "scan_deriv")
    idx_comb = add_array_summary_features(feats, combined, delay_grid, "scan_combined")

    top = np.sort(combined)[::-1]
    feats["scan_combined_top3_mean"] = finite_or(np.mean(top[:3]))
    feats["scan_combined_top5_mean"] = finite_or(np.mean(top[:5]))
    feats["scan_combined_range"] = finite_or(np.max(combined) - np.min(combined))

    feats["scan_best_tau_combined"] = finite_or(delay_grid[idx_comb])
    feats["scan_best_abs_tau_combined"] = abs(finite_or(delay_grid[idx_comb]))
    feats["scan_raw_at_best"] = finite_or(raw_corr[idx_comb])
    feats["scan_resid_at_best"] = finite_or(res_corr[idx_comb])
    feats["scan_resid3_at_best"] = finite_or(res3_corr[idx_comb])
    feats["scan_deriv_at_best"] = finite_or(der_corr[idx_comb])
    feats["scan_n_overlap_at_best"] = finite_or(n_ov[idx_comb])
    feats["scan_n_overlap_frac_at_best"] = finite_or(n_norm[idx_comb])
    feats["scan_raw_mae_at_best"] = finite_or(raw_mae[idx_comb])
    feats["scan_resid_mae_at_best"] = finite_or(res_mae[idx_comb])

    feats["scan_tau_raw_resid_gap"] = abs(finite_or(delay_grid[idx_raw] - delay_grid[idx_res]))
    feats["scan_tau_raw_resid3_gap"] = abs(finite_or(delay_grid[idx_raw] - delay_grid[idx_res3]))
    feats["scan_tau_raw_deriv_gap"] = abs(finite_or(delay_grid[idx_raw] - delay_grid[idx_der]))
    feats["scan_tau_resid_deriv_gap"] = abs(finite_or(delay_grid[idx_res] - delay_grid[idx_der]))

    for deg in MIC_POLY_DEGREES:
        score_arr = np.asarray(ml_arrays[deg]["score"], dtype=np.float64)
        idx_ml = add_array_summary_features(feats, score_arr, delay_grid, f"ml_poly{deg}_score")
        for k in ["corr", "mae", "resid_mad", "improve", "var_improve", "coef_l2", "slope_abs", "curv_abs", "cubic_abs", "n_frac"]:
            arr = np.asarray(ml_arrays[deg][k], dtype=np.float64)
            feats[f"ml_poly{deg}_{k}_at_best"] = finite_or(arr[idx_ml]) if len(arr) else 0.0
            feats[f"ml_poly{deg}_{k}_mean"] = finite_or(np.mean(arr)) if len(arr) else 0.0
            feats[f"ml_poly{deg}_{k}_p95"] = finite_or(np.percentile(arr, 95)) if len(arr) else 0.0

        for k in ["score", "corr", "mae", "resid_mad", "improve", "var_improve", "coef_l2", "slope_abs", "curv_abs", "cubic_abs"]:
            arr = np.asarray(ml_arrays[deg][k], dtype=np.float64)
            feats[f"ml_poly{deg}_{k}_at_best_combined_tau"] = finite_or(arr[idx_comb]) if len(arr) else 0.0

    for deg in [1, 2, 3, 4]:
        feats[f"ml_poly{deg}_score_gain_vs_poly0"] = (
            feats.get(f"ml_poly{deg}_score_max", -1.0) - feats.get("ml_poly0_score_max", -1.0)
        )
        feats[f"ml_poly{deg}_resid_gain_vs_poly0"] = (
            feats.get("ml_poly0_resid_mad_at_best", 3.0) - feats.get(f"ml_poly{deg}_resid_mad_at_best", 3.0)
        )

    return feats


def cadence_pair_features(tA, tB, cfg: CFG):
    feats = {}
    gA = np.diff(tA) if len(tA) >= 2 else np.array([0.0])
    gB = np.diff(tB) if len(tB) >= 2 else np.array([0.0])

    feats["cadence_n_diff_abs"] = abs(len(tA) - len(tB))
    feats["cadence_span_overlap"] = max(0.0, min(tA.max(), tB.max()) - max(tA.min(), tB.min())) / cfg.T_days
    feats["cadence_gap_median_diff_abs"] = abs(finite_or(np.median(gA)) - finite_or(np.median(gB)))
    feats["cadence_gap_std_diff_abs"] = abs(finite_or(np.std(gA)) - finite_or(np.std(gB)))
    feats["cadence_gap_max_diff_abs"] = abs(finite_or(np.max(gA)) - finite_or(np.max(gB)))

    if len(tA) and len(tB):
        dist = np.min(np.abs(tA[:, None] - tB[None, :]), axis=1)
        feats["cadence_nn_time_median"] = finite_or(np.median(dist))
        feats["cadence_nn_time_p90"] = finite_or(np.percentile(dist, 90))
        feats["cadence_nn_close_2d_frac"] = finite_or(np.mean(dist <= 2.0))
        feats["cadence_nn_close_10d_frac"] = finite_or(np.mean(dist <= 10.0))
    else:
        feats["cadence_nn_time_median"] = 999.0
        feats["cadence_nn_time_p90"] = 999.0
        feats["cadence_nn_close_2d_frac"] = 0.0
        feats["cadence_nn_close_10d_frac"] = 0.0
    return feats


def extract_pair_features(view: Dict[str, np.ndarray], cfg: CFG) -> Dict[str, float]:
    tA, yA, eA = extract_curve(view["A_x"], view["A_mask"], cfg)
    tB, yB, eB = extract_curve(view["B_x"], view["B_mask"], cfg)

    feats: Dict[str, float] = {}

    for name, val in zip(PAIR_STATS_NAMES, view["pair_stats"]):
        feats[name] = finite_or(val)

    fa = curve_features("A", tA, yA, eA)
    fb = curve_features("B", tB, yB, eB)
    feats.update(fa)
    feats.update(fb)

    common_suffixes = sorted(
        set(k[2:] for k in fa.keys() if k.startswith("A_"))
        & set(k[2:] for k in fb.keys() if k.startswith("B_"))
    )
    for suf in common_suffixes:
        a = finite_or(fa[f"A_{suf}"])
        b = finite_or(fb[f"B_{suf}"])
        feats[f"diff_abs_{suf}"] = abs(a - b)
        feats[f"diff_signed_{suf}"] = b - a
        feats[f"ratio_log_{suf}"] = finite_or(np.log((abs(b) + 1e-3) / (abs(a) + 1e-3)))

    feats.update(cadence_pair_features(tA, tB, cfg))
    feats.update(delay_scan_features(tA, yA, tB, yB, cfg))
    return feats


# ============================================================
# Real-data pair inference
# ============================================================

def harmonize_columns(df: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """Accept common source/component column variants and rename them to cfg names."""
    df = df.copy()

    source_aliases = [cfg.source_col, "source_id", "sourceID", "SourceID", "sourceId", "sourceId_1"]
    comp_aliases = [cfg.id_col, "lensComponentSourceId", "component_id", "componentId", "comp_id", "comp"]
    time_aliases = [cfg.time_col, "epoch_obs_jd", "jd", "JD", "mjd", "MJD", "time"]
    flux_aliases = [cfg.flux_col, "flux_obs", "flux", "Flux"]
    err_aliases = [cfg.err_col, "flux_obs_error", "flux_error", "err", "error", "sigma"]

    def first_existing(names: List[str]) -> Optional[str]:
        for name in names:
            if name in df.columns:
                return name
        return None

    mapping = {}
    src = first_existing(source_aliases)
    comp = first_existing(comp_aliases)
    time = first_existing(time_aliases)
    flux = first_existing(flux_aliases)
    err = first_existing(err_aliases)

    if src and src != cfg.source_col:
        mapping[src] = cfg.source_col
    if comp and comp != cfg.id_col:
        mapping[comp] = cfg.id_col
    if time and time != cfg.time_col:
        mapping[time] = cfg.time_col
    if flux and flux != cfg.flux_col:
        mapping[flux] = cfg.flux_col
    if err and err != cfg.err_col:
        mapping[err] = cfg.err_col

    if mapping:
        df = df.rename(columns=mapping)

    required = [cfg.source_col, cfg.id_col, cfg.time_col, cfg.flux_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after harmonization: {missing}. Available columns: {list(df.columns)}")

    if cfg.err_col not in df.columns:
        df[cfg.err_col] = np.nan

    for col in [cfg.time_col, cfg.flux_col, cfg.err_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[cfg.source_col, cfg.id_col, cfg.time_col, cfg.flux_col]).copy()
    return df


def make_real_pair_view_from_df(df_source: pd.DataFrame, comp_a: Any, comp_b: Any, cfg: CFG):
    """Transform two real components into the view expected by extract_pair_features."""
    gA = df_source[df_source[cfg.id_col] == comp_a].sort_values(cfg.time_col)
    gB = df_source[df_source[cfg.id_col] == comp_b].sort_values(cfg.time_col)

    if len(gA) < 5 or len(gB) < 5:
        raise ValueError(f"Not enough points: A={len(gA)}, B={len(gB)}")

    tA_raw = gA[cfg.time_col].to_numpy(np.float64)
    tB_raw = gB[cfg.time_col].to_numpy(np.float64)
    yA = gA[cfg.flux_col].to_numpy(np.float64)
    yB = gB[cfg.flux_col].to_numpy(np.float64)

    if cfg.err_col in gA.columns and cfg.err_col in gB.columns:
        eA = gA[cfg.err_col].to_numpy(np.float64)
        eB = gB[cfg.err_col].to_numpy(np.float64)
    else:
        eA = np.full(len(gA), np.nanstd(yA) * 0.05 + 1e-3)
        eB = np.full(len(gB), np.nanstd(yB) * 0.05 + 1e-3)

    if not np.any(np.isfinite(eA) & (eA > 0)):
        eA = np.full(len(gA), np.nanstd(yA) * 0.05 + 1e-3)
    if not np.any(np.isfinite(eB) & (eB > 0)):
        eB = np.full(len(gB), np.nanstd(yB) * 0.05 + 1e-3)

    t0 = min(np.min(tA_raw), np.min(tB_raw))
    t1 = max(np.max(tA_raw), np.max(tB_raw))
    span_real = max(t1 - t0, 1.0)

    # Rescale to [0, cfg.T_days] as in training/inference notebook.
    tA = (tA_raw - t0) / span_real * cfg.T_days
    tB = (tB_raw - t0) / span_real * cfg.T_days

    xA, mA, infoA = pack_curve(tA, yA, eA, cfg)
    xB, mB, infoB = pack_curve(tB, yB, eB, cfg)

    overlap = max(0.0, min(infoA["t_max"], infoB["t_max"]) - max(infoA["t_min"], infoB["t_min"])) / cfg.T_days
    spanA = (infoA["t_max"] - infoA["t_min"]) / cfg.T_days
    spanB = (infoB["t_max"] - infoB["t_min"]) / cfg.T_days

    pair_stats = np.array([
        (infoB["median"] - infoA["median"]) / 3.0,
        np.log((infoB["scale"] + 1e-3) / (infoA["scale"] + 1e-3)),
        infoA["n"] / cfg.max_len,
        infoB["n"] / cfg.max_len,
        overlap,
        spanB - spanA,
    ], dtype=np.float32)

    return {
        "A_x": xA,
        "A_mask": mA,
        "B_x": xB,
        "B_mask": mB,
        "pair_stats": pair_stats,
        "tau": np.float32(np.nan),
        "mic_strength": np.float32(np.nan),
        "span_real_days": float(span_real),
    }


def pair_features_from_source(df_source: pd.DataFrame, comp_a: Any, comp_b: Any, cfg: CFG, feature_cols: List[str]) -> Dict[str, float]:
    view = make_real_pair_view_from_df(df_source, comp_a, comp_b, cfg)
    feats_all = extract_pair_features(view, cfg)
    return {k: feats_all.get(k, np.nan) for k in feature_cols}


def iter_all_component_pairs(df: pd.DataFrame, cfg: CFG):
    """Yield source_id, comp_a, comp_b, df_source for every available pair in every source."""
    df = df.sort_values([cfg.source_col, cfg.id_col, cfg.time_col])
    for sid, g in df.groupby(cfg.source_col, sort=False):
        comps = list(pd.Series(g[cfg.id_col].dropna().unique()).sort_values())
        if len(comps) < 2:
            continue
        for comp_a, comp_b in combinations(comps, 2):
            yield sid, comp_a, comp_b, g


def _flush_predictions(
    model: Any,
    feature_cols: List[str],
    pending_meta: List[Tuple[Any, Any, Any]],
    pending_feats: List[Dict[str, float]],
    output_csv: Path,
    write_header: bool,
) -> bool:
    if not pending_meta:
        return write_header

    X = pd.DataFrame(pending_feats, columns=feature_cols)
    proba = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "sourceID": [m[0] for m in pending_meta],
        "compA": [m[1] for m in pending_meta],
        "compB": [m[2] for m in pending_meta],
        "Proba": proba.astype(float),
    })
    out.to_csv(output_csv, mode="a", header=write_header, index=False)
    return False


def run_rf_inference(
    input_csv: str | Path,
    *,
    model_path: str | Path = "models/lens22_rf.joblib",
    output_csv: str | Path = "rf_pair_predictions.csv",
    batch_size: int = 64,
    keep_failed_pairs: bool = True,
    max_pairs: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run RF inference for every component pair available in every source.

    Parameters
    ----------
    input_csv:
        CSV with cleaned light curves.
    model_path:
        Path to lens22_rf.joblib.
    output_csv:
        Output CSV. It contains only sourceID, compA, compB, Proba.
    keep_failed_pairs:
        If True, pairs that cannot be scored, usually because one component has <5 points,
        are written with Proba = NaN so the output still contains every available pair.
    max_pairs:
        Optional debug limit.
    """
    input_csv = Path(input_csv)
    model_path = Path(model_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_cols = list(bundle.get("feature_cols", []))
        cfg = cfg_from_bundle(bundle)
    else:
        model = bundle
        cfg = CFG()
        feature_cols = list(getattr(model, "feature_names_in_", []))

    model = fix_sklearn_joblib_compatibility(model)

    if not feature_cols:
        raise ValueError("No feature_cols found in the model bundle. Use the saved lens22_rf.joblib bundle.")

    df = pd.read_csv(input_csv, low_memory=False)
    df = harmonize_columns(df, cfg)

    # Start a fresh output file with exactly these columns.
    pd.DataFrame(columns=["sourceID", "compA", "compB", "Proba"]).to_csv(output_csv, index=False)
    write_header = False

    pending_meta: List[Tuple[Any, Any, Any]] = []
    pending_feats: List[Dict[str, float]] = []
    n_pairs = 0
    n_scored = 0
    n_failed = 0

    for sid, comp_a, comp_b, df_source in iter_all_component_pairs(df, cfg):
        if max_pairs is not None and n_pairs >= max_pairs:
            break

        n_pairs += 1
        try:
            feats = pair_features_from_source(df_source, comp_a, comp_b, cfg, feature_cols)
            pending_meta.append((sid, comp_a, comp_b))
            pending_feats.append(feats)

            if len(pending_meta) >= int(batch_size):
                _flush_predictions(model, feature_cols, pending_meta, pending_feats, output_csv, write_header)
                n_scored += len(pending_meta)
                pending_meta.clear()
                pending_feats.clear()

        except Exception:
            n_failed += 1
            if keep_failed_pairs:
                fail_row = pd.DataFrame({
                    "sourceID": [sid],
                    "compA": [comp_a],
                    "compB": [comp_b],
                    "Proba": [np.nan],
                })
                fail_row.to_csv(output_csv, mode="a", header=False, index=False)

        if verbose and n_pairs % 50 == 0:
            print(f"Processed {n_pairs} pairs | scored={n_scored + len(pending_meta)} | failed={n_failed}")

    if pending_meta:
        _flush_predictions(model, feature_cols, pending_meta, pending_feats, output_csv, write_header)
        n_scored += len(pending_meta)
        pending_meta.clear()
        pending_feats.clear()

    result = pd.read_csv(output_csv)

    if verbose:
        print("\n========== RF INFERENCE SUMMARY ==========")
        print(f"Input CSV: {input_csv}")
        print(f"Model: {model_path}")
        print(f"Pairs available: {n_pairs}")
        print(f"Pairs scored by RF: {n_scored}")
        print(f"Pairs failed: {n_failed}")
        print(f"Output CSV: {output_csv}")
        print("Columns: sourceID, compA, compB, Proba")
        print("==========================================\n")

    return result


# Backward-compatible aliases.
infer_rf = run_rf_inference
rf_inference = run_rf_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF lens-pair inference on every component pair in a CSV.")
    parser.add_argument("input_csv", help="Input light-curve CSV")
    parser.add_argument("--model", default="models/lens22_rf.joblib", help="Path to lens22_rf.joblib")
    parser.add_argument("--output", default="rf_pair_predictions.csv", help="Output prediction CSV")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for model.predict_proba")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional debug limit")
    parser.add_argument("--drop-failed-pairs", action="store_true", help="Do not write rows with Proba=NaN for failed pairs")
    args = parser.parse_args()

    run_rf_inference(
        input_csv=args.input_csv,
        model_path=args.model,
        output_csv=args.output,
        batch_size=args.batch_size,
        keep_failed_pairs=not args.drop_failed_pairs,
        max_pairs=args.max_pairs,
        verbose=True,
    )
