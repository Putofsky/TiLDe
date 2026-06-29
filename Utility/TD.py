"""
time_delay_ot_pipeline_v2.py

Global-shift Optimal Transport time-delay estimator for light-curve pairs.

Expected CSV columns:
    source_id, lensComponentSourceId, epoch_obs_jd, flux_obs, flux_obs_error

Core rule:
    The delay is ONE global shift for the whole B curve:

        t_B_shifted = t_B - Delta

    No point-by-point delays are fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


@dataclass
class OTDelayConfig:
    input_csv: str
    output_dir: str = "td_ot_outputs"
    output_csv: str = "time_delay_results.csv"

    source_col: str = "source_id"
    component_col: str = "lensComponentSourceId"
    time_col: str = "epoch_obs_jd"
    flux_col: str = "flux_obs"
    flux_err_col: str = "flux_obs_error"
    calibrated_col: str = "isFluxCalibrated"

    only_calibrated: bool = False
    min_points_per_curve: int = 8
    max_pairs: Optional[int] = None

    # Important: center and normalize each component separately.
    flux_normalization: str = "per_curve"  # "per_curve", "joint", "none"
    robust_center_scale: bool = True
    sigma_floor_norm: float = 0.05
    clip_normalized_flux: Optional[float] = 8.0
    subtract_time_origin_per_source: bool = True

    use_uncertainty_weights: bool = True
    use_density_weights: bool = True
    density_bin_days: float = 60.0
    density_beta: float = 0.5
    weight_clip_quantile: float = 0.95
    min_weight: float = 1e-12

    # If None, estimated from cadence/span.
    time_scale_days: Optional[float] = None
    flux_scale: float = 1.0
    lambda_time: float = 1.0
    lambda_flux: float = 1.0
    sigma_int_norm: float = 0.10

    # Keep None by default. Hard gates can produce flat all-penalty curves.
    time_gate_days: Optional[float] = None
    gate_penalty: float = 1e4
    cost_clip: float = 1e4

    sinkhorn_reg: float = 0.05
    sinkhorn_max_iter: int = 300
    sinkhorn_tol: float = 1e-7
    overlap_penalty_weight: float = 2.0

    max_delay_days: float = 700.0
    delay_bounds: Optional[Tuple[float, float]] = None
    n_delay_grid: int = 241
    refine_delay: bool = True
    refine_window_grid_steps: float = 3.0

    run_mcmc: bool = True
    mcmc_steps: int = 1200
    mcmc_burn: int = 300
    mcmc_proposal_days: float = 8.0
    mcmc_temperature: Optional[float] = None
    mcmc_use_interpolated_profile: bool = True
    random_seed: int = 123

    make_plots: bool = True
    show_plots: bool = False
    save_plots: bool = True
    max_plots: Optional[int] = 50
    plot_cost_matrix: bool = True
    dpi: int = 150

    verbose: bool = True


def _robust_median(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(np.median(x))


def _robust_scale(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.std(x)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = max(np.ptp(x), 1.0)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return float(scale)


def _clean_curve(g: pd.DataFrame, cfg: OTDelayConfig) -> Dict[str, np.ndarray]:
    t = pd.to_numeric(g[cfg.time_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(g[cfg.flux_col], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(g[cfg.flux_err_col], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    t, y, s = t[ok], y[ok], s[ok]

    order = np.argsort(t)
    return {
        "t": t[order],
        "y_raw": y[order],
        "sigma_raw": s[order],
    }


def _normalize_one_curve(lc: Dict[str, np.ndarray], cfg: OTDelayConfig):
    out = {k: np.array(v, copy=True) for k, v in lc.items()}
    y = out["y_raw"]
    sig = out["sigma_raw"]

    if cfg.robust_center_scale:
        center = _robust_median(y)
        scale = _robust_scale(y)
    else:
        center = float(np.nanmean(y))
        scale = float(np.nanstd(y))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0

    out["center"] = center
    out["scale"] = scale
    out["y"] = (y - center) / scale
    out["sigma"] = np.maximum(sig / scale, cfg.sigma_floor_norm)

    if cfg.clip_normalized_flux is not None:
        c = float(cfg.clip_normalized_flux)
        out["y"] = np.clip(out["y"], -c, c)

    return out


def _normalize_pair(lc_a_raw, lc_b_raw, cfg: OTDelayConfig):
    if cfg.flux_normalization == "per_curve":
        return _normalize_one_curve(lc_a_raw, cfg), _normalize_one_curve(lc_b_raw, cfg)

    a = {k: np.array(v, copy=True) for k, v in lc_a_raw.items()}
    b = {k: np.array(v, copy=True) for k, v in lc_b_raw.items()}

    if cfg.flux_normalization == "joint":
        y_all = np.concatenate([a["y_raw"], b["y_raw"]])
        center = _robust_median(y_all)
        scale = _robust_scale(y_all)

        for out in (a, b):
            out["center"] = center
            out["scale"] = scale
            out["y"] = (out["y_raw"] - center) / scale
            out["sigma"] = np.maximum(out["sigma_raw"] / scale, cfg.sigma_floor_norm)
            if cfg.clip_normalized_flux is not None:
                out["y"] = np.clip(out["y"], -cfg.clip_normalized_flux, cfg.clip_normalized_flux)

        return a, b

    if cfg.flux_normalization == "none":
        for out in (a, b):
            out["center"] = 0.0
            out["scale"] = 1.0
            out["y"] = out["y_raw"].astype(float)
            out["sigma"] = np.maximum(out["sigma_raw"].astype(float), cfg.sigma_floor_norm)
        return a, b

    raise ValueError("flux_normalization must be 'per_curve', 'joint', or 'none'.")


def _auto_time_scale(lc_a, lc_b):
    t = np.sort(np.concatenate([lc_a["t"], lc_b["t"]]))
    span = float(np.nanmax(t) - np.nanmin(t)) if len(t) else 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    med_cad = float(np.median(dt)) if len(dt) else max(span / 40.0, 1.0)
    return float(max(10.0, 3.0 * med_cad, 0.04 * span))


def _curve_weights(t, sigma, cfg: OTDelayConfig):
    n = len(t)
    if n == 0:
        return np.array([], dtype=float)

    if cfg.use_uncertainty_weights:
        q = 1.0 / np.maximum(sigma**2, cfg.sigma_floor_norm**2)
        if 0 < cfg.weight_clip_quantile < 1:
            finite = q[np.isfinite(q)]
            if len(finite):
                cap = np.quantile(finite, cfg.weight_clip_quantile)
                q = np.minimum(q, cap)
    else:
        q = np.ones(n)

    q = np.where(np.isfinite(q) & (q > 0), q, 1.0)

    if not cfg.use_density_weights:
        w = q / q.sum()
        w = np.maximum(w, cfg.min_weight)
        return w / w.sum()

    t0 = np.nanmin(t)
    bins = np.floor((t - t0) / cfg.density_bin_days).astype(int)
    unique_bins = np.unique(bins)

    bin_mass_raw = {}
    for bk in unique_bins:
        idx = np.where(bins == bk)[0]
        quality = float(np.sum(q[idx]))
        quality = max(quality, float(len(idx)))
        bin_mass_raw[bk] = quality ** cfg.density_beta

    total = float(np.sum(list(bin_mass_raw.values())))
    if not np.isfinite(total) or total <= 0:
        return np.ones(n) / n

    w = np.zeros(n)
    for bk in unique_bins:
        idx = np.where(bins == bk)[0]
        local = q[idx]
        local_sum = float(np.sum(local))
        if local_sum <= 0:
            local_w = np.ones(len(idx)) / len(idx)
        else:
            local_w = local / local_sum
        w[idx] = (bin_mass_raw[bk] / total) * local_w

    w = np.maximum(w, cfg.min_weight)
    return w / w.sum()


def _overlap_fraction(t_a, t_b_shift):
    a0, a1 = float(np.min(t_a)), float(np.max(t_a))
    b0, b1 = float(np.min(t_b_shift)), float(np.max(t_b_shift))
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(min(a1 - a0, b1 - b0), 1e-12)
    return float(np.clip(overlap / denom, 0.0, 1.0))


def cost_matrix_for_delay(lc_a, lc_b, delta: float, cfg: OTDelayConfig, time_scale=None):
    if time_scale is None:
        time_scale = cfg.time_scale_days if cfg.time_scale_days is not None else _auto_time_scale(lc_a, lc_b)

    t_a = lc_a["t"]
    y_a = lc_a["y"]
    s_a = lc_a["sigma"]

    # Critical point: global rigid shift of the whole B curve.
    t_b_shift = lc_b["t"] - float(delta)
    y_b = lc_b["y"]
    s_b = lc_b["sigma"]

    dt = t_a[:, None] - t_b_shift[None, :]
    dy = y_a[:, None] - y_b[None, :]

    c_t = (dt / max(time_scale, 1e-12)) ** 2

    var_y = s_a[:, None] ** 2 + s_b[None, :] ** 2 + cfg.sigma_int_norm**2
    c_y = (dy / max(cfg.flux_scale, 1e-12)) ** 2 / np.maximum(var_y, 1e-12)

    C = cfg.lambda_time * c_t + cfg.lambda_flux * c_y

    if cfg.time_gate_days is not None:
        C = np.where(np.abs(dt) <= cfg.time_gate_days, C, cfg.gate_penalty)

    C = np.where(np.isfinite(C), C, cfg.gate_penalty)
    return np.clip(C, 0.0, cfg.cost_clip)


def _sinkhorn_cost(C, p, q, cfg: OTDelayConfig):
    p = np.maximum(np.asarray(p, dtype=float), 1e-300)
    q = np.maximum(np.asarray(q, dtype=float), 1e-300)
    p = p / p.sum()
    q = q / q.sum()

    eps = max(float(cfg.sinkhorn_reg), 1e-12)

    logK = -C / eps
    logp = np.log(p)
    logq = np.log(q)

    u = np.zeros_like(logp)
    v = np.zeros_like(logq)

    for _ in range(cfg.sinkhorn_max_iter):
        old_u = u.copy()
        u = logp - logsumexp(logK + v[None, :], axis=1)
        v = logq - logsumexp(logK + u[:, None], axis=0)

        if np.max(np.abs(u - old_u)) < cfg.sinkhorn_tol:
            break

    log_gamma = logK + u[:, None] + v[None, :]
    gamma = np.exp(log_gamma)

    return float(np.sum(gamma * C))


def ot_cost_for_delay(lc_a, lc_b, delta, cfg: OTDelayConfig, p=None, q=None, time_scale=None):
    if p is None:
        p = _curve_weights(lc_a["t"], lc_a["sigma"], cfg)
    if q is None:
        q = _curve_weights(lc_b["t"], lc_b["sigma"], cfg)
    if time_scale is None:
        time_scale = cfg.time_scale_days if cfg.time_scale_days is not None else _auto_time_scale(lc_a, lc_b)

    C = cost_matrix_for_delay(lc_a, lc_b, delta, cfg, time_scale=time_scale)
    cost = _sinkhorn_cost(C, p, q, cfg)

    if cfg.overlap_penalty_weight > 0:
        t_b_shift = lc_b["t"] - float(delta)
        ov = _overlap_fraction(lc_a["t"], t_b_shift)
        cost += cfg.overlap_penalty_weight * (1.0 - ov) ** 2

    return float(cost)


def _delay_bounds(cfg: OTDelayConfig):
    if cfg.delay_bounds is not None:
        return float(cfg.delay_bounds[0]), float(cfg.delay_bounds[1])
    return -float(cfg.max_delay_days), float(cfg.max_delay_days)


def _profile_temperature(costs):
    costs = np.asarray(costs, dtype=float)
    finite = costs[np.isfinite(costs)]
    if len(finite) < 3:
        return 1.0

    dc = finite - np.min(finite)
    positive = dc[dc > 1e-12]

    if len(positive) == 0:
        return 1.0

    T = np.percentile(positive, 20)

    if not np.isfinite(T) or T <= 1e-12:
        T = np.std(finite)
    if not np.isfinite(T) or T <= 1e-12:
        T = 1.0

    return float(T)


def _mcmc_delta_from_profile(grid, costs, start_delta, cfg: OTDelayConfig, eval_cost=None):
    rng = np.random.default_rng(cfg.random_seed)

    bounds = (float(grid[0]), float(grid[-1]))
    cmin = float(np.nanmin(costs))

    T = cfg.mcmc_temperature if cfg.mcmc_temperature is not None else _profile_temperature(costs)
    T = max(float(T), 1e-12)

    def cost_at(d):
        if eval_cost is not None and not cfg.mcmc_use_interpolated_profile:
            return float(eval_cost(d))
        return float(np.interp(d, grid, costs, left=np.inf, right=np.inf))

    def logp(d):
        if d < bounds[0] or d > bounds[1]:
            return -np.inf
        c = cost_at(d)
        if not np.isfinite(c):
            return -np.inf
        return -0.5 * (c - cmin) / T

    cur = float(np.clip(start_delta, bounds[0], bounds[1]))
    cur_lp = logp(cur)
    step = float(cfg.mcmc_proposal_days)

    samples = []
    accepts = 0

    for k in range(cfg.mcmc_steps):
        prop = cur + rng.normal(0.0, step)

        if prop < bounds[0]:
            prop = bounds[0] + (bounds[0] - prop)
        if prop > bounds[1]:
            prop = bounds[1] - (prop - bounds[1])

        prop = float(np.clip(prop, bounds[0], bounds[1]))
        prop_lp = logp(prop)

        if np.log(rng.random()) < prop_lp - cur_lp:
            cur, cur_lp = prop, prop_lp
            accepts += 1

        if k >= cfg.mcmc_burn:
            samples.append(cur)

        if k > 0 and k < cfg.mcmc_burn and k % 100 == 0:
            acc = accepts / k
            if acc < 0.15:
                step *= 0.7
            elif acc > 0.6:
                step *= 1.3

    return np.asarray(samples, dtype=float)


def estimate_delay_pair(lc_a_raw, lc_b_raw, cfg: OTDelayConfig):
    lc_a, lc_b = _normalize_pair(lc_a_raw, lc_b_raw, cfg)

    if len(lc_a["t"]) < cfg.min_points_per_curve or len(lc_b["t"]) < cfg.min_points_per_curve:
        raise ValueError("Not enough points in one or both curves.")

    bounds = _delay_bounds(cfg)
    grid = np.linspace(bounds[0], bounds[1], int(cfg.n_delay_grid))

    p = _curve_weights(lc_a["t"], lc_a["sigma"], cfg)
    q = _curve_weights(lc_b["t"], lc_b["sigma"], cfg)
    time_scale = cfg.time_scale_days if cfg.time_scale_days is not None else _auto_time_scale(lc_a, lc_b)

    def obj(delta):
        return ot_cost_for_delay(lc_a, lc_b, delta, cfg, p=p, q=q, time_scale=time_scale)

    costs = np.array([obj(d) for d in grid], dtype=float)

    if not np.any(np.isfinite(costs)):
        raise RuntimeError("All OT costs are non-finite.")

    idx = int(np.nanargmin(costs))
    best_grid = float(grid[idx])
    best_delta = best_grid
    best_cost = float(costs[idx])

    if cfg.refine_delay and len(grid) >= 5:
        step = float(grid[1] - grid[0])
        lo = max(bounds[0], best_grid - cfg.refine_window_grid_steps * step)
        hi = min(bounds[1], best_grid + cfg.refine_window_grid_steps * step)

        if hi > lo:
            opt = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 0.05})

            if opt.success and np.isfinite(opt.fun):
                best_delta = float(opt.x)
                best_cost = float(opt.fun)

    samples = None
    q16 = q50 = q84 = np.nan
    uncertainty = np.nan
    delay_report = best_delta

    if cfg.run_mcmc:
        samples = _mcmc_delta_from_profile(
            grid=grid,
            costs=costs,
            start_delta=best_delta,
            cfg=cfg,
            eval_cost=obj,
        )

        if len(samples) > 10:
            q16, q50, q84 = np.percentile(samples, [16, 50, 84])
            uncertainty = 0.5 * (q84 - q16)
            delay_report = float(q50)

    C_best = cost_matrix_for_delay(lc_a, lc_b, delay_report, cfg, time_scale=time_scale)

    return {
        "delay": float(delay_report),
        "delay_best": float(best_delta),
        "uncertainty": float(uncertainty) if np.isfinite(uncertainty) else np.nan,
        "q16": float(q16) if np.isfinite(q16) else np.nan,
        "q50": float(q50) if np.isfinite(q50) else np.nan,
        "q84": float(q84) if np.isfinite(q84) else np.nan,
        "best_cost": float(best_cost),
        "grid": grid,
        "costs": costs,
        "samples": samples,
        "lc_a_norm": lc_a,
        "lc_b_norm": lc_b,
        "cost_matrix_best": C_best,
        "time_scale": float(time_scale),
    }


def plot_pair_result(result, source_id, comp_a, comp_b, cfg: OTDelayConfig, outpath: Optional[Path] = None):
    lc_a = result["lc_a_norm"]
    lc_b = result["lc_b_norm"]

    delta = float(result["delay"])
    grid = np.asarray(result["grid"], dtype=float)
    costs = np.asarray(result["costs"], dtype=float)
    C = np.asarray(result["cost_matrix_best"], dtype=float)

    nrows = 3 if cfg.plot_cost_matrix else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 10 if nrows == 3 else 7), constrained_layout=True)

    if nrows == 2:
        ax_lc, ax_prof = axes
        ax_map = None
    else:
        ax_lc, ax_prof, ax_map = axes

    ax_lc.errorbar(
        lc_a["t"],
        lc_a["y"],
        yerr=lc_a["sigma"],
        fmt="o",
        ms=4,
        alpha=0.8,
        label=f"A: {comp_a}",
    )

    ax_lc.errorbar(
        lc_b["t"] - delta,
        lc_b["y"],
        yerr=lc_b["sigma"],
        fmt="s",
        ms=4,
        alpha=0.8,
        label=f"B shifted globally by Δ={delta:.2f} d: {comp_b}",
    )

    ax_lc.set_title(f"source_id={source_id} | {comp_a} vs {comp_b}")
    ax_lc.set_xlabel("Epoch in A frame: B uses t_B - Δ")
    ax_lc.set_ylabel("Centered / normalized flux")
    ax_lc.grid(True, alpha=0.3)
    ax_lc.legend()

    dc = costs - np.nanmin(costs)

    ax_prof.plot(grid, dc, lw=1.8)
    ax_prof.axvline(delta, linestyle="--", lw=1.3, label=f"Δ={delta:.2f} d")

    if np.isfinite(result.get("q16", np.nan)) and np.isfinite(result.get("q84", np.nan)):
        ax_prof.axvspan(float(result["q16"]), float(result["q84"]), alpha=0.25, label="16–84% MCMC")

    ax_prof.set_xlabel("Global delay Δ [days]")
    ax_prof.set_ylabel("OT cost - minimum")
    ax_prof.set_title("Global-shift OT cost profile")
    ax_prof.grid(True, alpha=0.3)
    ax_prof.legend()

    if ax_map is not None:
        im = ax_map.imshow(np.log1p(C), origin="lower", aspect="auto", interpolation="nearest")
        ax_map.set_title(f"Cost map at Δ={delta:.2f} d: log(1 + C_ij)")
        ax_map.set_xlabel("B observation index after global shift")
        ax_map.set_ylabel("A observation index")

        cb = fig.colorbar(im, ax=ax_map)
        cb.set_label("log(1 + pairwise cost)")

    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=cfg.dpi)

    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)


def run_time_delay_pipeline(cfg: OTDelayConfig) -> pd.DataFrame:
    input_path = Path(cfg.input_csv)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"

    df = pd.read_csv(input_path)

    required = [
        cfg.source_col,
        cfg.component_col,
        cfg.time_col,
        cfg.flux_col,
        cfg.flux_err_col,
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if cfg.only_calibrated and cfg.calibrated_col in df.columns:
        df = df[df[cfg.calibrated_col].astype(bool)].copy()

    for c in [cfg.time_col, cfg.flux_col, cfg.flux_err_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).copy()

    rows: List[Dict[str, Any]] = []
    n_pairs = 0
    n_plots = 0

    for source_id, g_source in df.groupby(cfg.source_col, sort=False):
        g_source = g_source.copy()

        if cfg.subtract_time_origin_per_source:
            g_source[cfg.time_col] = g_source[cfg.time_col] - g_source[cfg.time_col].min()

        comps = list(g_source[cfg.component_col].dropna().unique())

        if len(comps) < 2:
            continue

        curves = {}

        for comp in comps:
            lc = _clean_curve(g_source[g_source[cfg.component_col] == comp], cfg)

            if len(lc["t"]) >= cfg.min_points_per_curve:
                curves[comp] = lc

        valid_comps = list(curves.keys())

        if len(valid_comps) < 2:
            continue

        for comp_a, comp_b in combinations(valid_comps, 2):
            if cfg.max_pairs is not None and n_pairs >= cfg.max_pairs:
                break

            row = {
                "source_id": source_id,
                "comp_A": comp_a,
                "comp_B": comp_b,
                "TimeDelay": np.nan,
                "Incertie": np.nan,
                "q16": np.nan,
                "q50": np.nan,
                "q84": np.nan,
                "best_delay": np.nan,
                "best_cost": np.nan,
                "n_A": len(curves[comp_a]["t"]),
                "n_B": len(curves[comp_b]["t"]),
                "status": "ok",
                "plot_path": "",
            }

            try:
                res = estimate_delay_pair(curves[comp_a], curves[comp_b], cfg)

                row.update(
                    {
                        "TimeDelay": res["delay"],
                        "Incertie": res["uncertainty"],
                        "q16": res["q16"],
                        "q50": res["q50"],
                        "q84": res["q84"],
                        "best_delay": res["delay_best"],
                        "best_cost": res["best_cost"],
                        "time_scale": res["time_scale"],
                    }
                )

                if cfg.make_plots and (cfg.max_plots is None or n_plots < cfg.max_plots):
                    safe_source = str(source_id).replace("/", "_")
                    safe_a = str(comp_a).replace("/", "_")
                    safe_b = str(comp_b).replace("/", "_")

                    outpath = plots_dir / f"source_{safe_source}__{safe_a}_vs_{safe_b}.png"

                    plot_pair_result(
                        res,
                        source_id,
                        comp_a,
                        comp_b,
                        cfg,
                        outpath if cfg.save_plots else None,
                    )

                    row["plot_path"] = str(outpath)
                    n_plots += 1

            except Exception as e:
                row["status"] = f"failed: {type(e).__name__}: {e}"

            rows.append(row)
            n_pairs += 1

            if cfg.verbose and n_pairs % 25 == 0:
                print(f"Processed {n_pairs} pairs...")

        if cfg.max_pairs is not None and n_pairs >= cfg.max_pairs:
            break

    out = pd.DataFrame(rows)

    first = ["source_id", "comp_A", "comp_B", "TimeDelay", "Incertie"]
    other = [c for c in out.columns if c not in first]
    out = out[first + other]

    output_csv = Path(cfg.output_csv)

    if not output_csv.is_absolute():
        output_csv = output_dir / output_csv

    out.to_csv(output_csv, index=False)

    if cfg.verbose:
        print(f"Saved CSV: {output_csv}")

        if cfg.make_plots and cfg.save_plots:
            print(f"Saved plots: {plots_dir}")

        print(f"Done. Number of pair estimates: {len(out)}")

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Global-shift OT time-delay estimator.")

    parser.add_argument("input_csv")
    parser.add_argument("--output-dir", default="td_ot_outputs")
    parser.add_argument("--output-csv", default="time_delay_results.csv")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-plots", type=int, default=50)
    parser.add_argument("--max-delay-days", type=float, default=700.0)
    parser.add_argument("--n-delay-grid", type=int, default=241)
    parser.add_argument("--no-mcmc", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    cfg = OTDelayConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        max_pairs=args.max_pairs,
        max_plots=args.max_plots,
        max_delay_days=args.max_delay_days,
        n_delay_grid=args.n_delay_grid,
        run_mcmc=not args.no_mcmc,
        make_plots=not args.no_plots,
    )

    run_time_delay_pipeline(cfg)