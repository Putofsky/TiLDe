"""
TD_candidates.py

Wrapper around Utility.TD to run global-shift OT time-delay estimates only on
candidate systems/pairs from a candidate CSV.

Candidate CSV expected columns:
    sourceID, compA, compB, Proba

Output columns:
    Source ID, comp A, comp B, Proba, TD, incertitude
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _key_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _ordered_pair(a: Any, b: Any) -> Tuple[str, str]:
    a, b = str(a).strip(), str(b).strip()
    return (a, b) if a <= b else (b, a)


def _clean_curve_from_df(g: pd.DataFrame, cfg: Any) -> Dict[str, np.ndarray]:
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


def _read_candidates(
    candidates_csv: str | Path,
    source_col: str = "sourceID",
    comp_a_col: str = "compA",
    comp_b_col: str = "compB",
    proba_col: str = "Proba",
    cdda_col: Optional[str] = "Cdda",
    only_cdda: bool = False,
    min_proba: Optional[float] = None,
) -> pd.DataFrame:
    cand = pd.read_csv(candidates_csv)

    required = [source_col, comp_a_col, comp_b_col]
    missing = [c for c in required if c not in cand.columns]
    if missing:
        raise ValueError(
            f"Missing candidate columns: {missing}. "
            f"Available: {cand.columns.tolist()}"
        )

    if proba_col not in cand.columns:
        cand[proba_col] = np.nan

    if only_cdda and cdda_col is not None and cdda_col in cand.columns:
        cand = cand[cand[cdda_col].astype(bool)].copy()

    cand["_source_key"] = _key_series(cand[source_col])
    cand["_compA_key"] = _key_series(cand[comp_a_col])
    cand["_compB_key"] = _key_series(cand[comp_b_col])
    cand["_proba"] = pd.to_numeric(cand[proba_col], errors="coerce")

    if min_proba is not None:
        cand = cand[cand["_proba"].fillna(-np.inf) >= float(min_proba)].copy()

    cand = cand[cand["_compA_key"] != cand["_compB_key"]].copy()

    pair_keys = cand.apply(
        lambda r: _ordered_pair(r["_compA_key"], r["_compB_key"]),
        axis=1,
    )
    cand["_pair_a"] = [p[0] for p in pair_keys]
    cand["_pair_b"] = [p[1] for p in pair_keys]

    cand = (
        cand.sort_values("_proba", ascending=False, na_position="last")
        .drop_duplicates(["_source_key", "_pair_a", "_pair_b"], keep="first")
        .reset_index(drop=True)
    )

    return cand


def _build_pair_table(
    df_lc: pd.DataFrame,
    cand: pd.DataFrame,
    cfg: Any,
    candidate_mode: str = "pairs",
) -> pd.DataFrame:
    """
    candidate_mode:
        'pairs'
            test exactly compA-compB from the candidate CSV.

        'candidate_components'
            for each source, take all unique components appearing in compA/compB
            and test all combinations.

        'all_components'
            for each candidate source, test all components present in the
            light-curve CSV.
    """

    candidate_mode = candidate_mode.lower().strip()

    if candidate_mode not in {"pairs", "candidate_components", "all_components"}:
        raise ValueError(
            "candidate_mode must be 'pairs', 'candidate_components', or 'all_components'."
        )

    proba_lookup = {}
    source_max_proba = {}

    for _, r in cand.iterrows():
        a, b = _ordered_pair(r["_compA_key"], r["_compB_key"])
        src = r["_source_key"]
        p = r["_proba"]
        proba_lookup[(src, a, b)] = float(p) if pd.notna(p) else np.nan

    for src, g in cand.groupby("_source_key"):
        vals = pd.to_numeric(g["_proba"], errors="coerce")
        source_max_proba[src] = float(vals.max()) if vals.notna().any() else np.nan

    rows = []

    if candidate_mode == "pairs":
        for _, r in cand.iterrows():
            a, b = _ordered_pair(r["_compA_key"], r["_compB_key"])
            rows.append(
                {
                    "source_key": r["_source_key"],
                    "comp_A_key": a,
                    "comp_B_key": b,
                    "Proba": float(r["_proba"]) if pd.notna(r["_proba"]) else np.nan,
                }
            )

    elif candidate_mode == "candidate_components":
        for src, g in cand.groupby("_source_key"):
            comps = sorted(set(g["_compA_key"]).union(set(g["_compB_key"])))

            for a, b in combinations(comps, 2):
                aa, bb = _ordered_pair(a, b)

                rows.append(
                    {
                        "source_key": src,
                        "comp_A_key": aa,
                        "comp_B_key": bb,
                        "Proba": proba_lookup.get(
                            (src, aa, bb),
                            source_max_proba.get(src, np.nan),
                        ),
                    }
                )

    else:
        lc_sources = set(df_lc["_source_key"].unique())

        for src in sorted(set(cand["_source_key"]).intersection(lc_sources)):
            comps = sorted(
                df_lc.loc[df_lc["_source_key"] == src, "_comp_key"]
                .dropna()
                .unique()
            )

            for a, b in combinations(comps, 2):
                aa, bb = _ordered_pair(a, b)

                rows.append(
                    {
                        "source_key": src,
                        "comp_A_key": aa,
                        "comp_B_key": bb,
                        "Proba": proba_lookup.get(
                            (src, aa, bb),
                            source_max_proba.get(src, np.nan),
                        ),
                    }
                )

    pairs = pd.DataFrame(rows)

    if len(pairs) == 0:
        return pairs

    pairs = pairs.drop_duplicates(
        ["source_key", "comp_A_key", "comp_B_key"]
    ).reset_index(drop=True)

    return pairs


def run_candidate_time_delay_pipeline(
    cfg: Any,
    candidates_csv: str | Path,
    output_csv: Optional[str | Path] = None,
    candidate_source_col: str = "sourceID",
    candidate_comp_a_col: str = "compA",
    candidate_comp_b_col: str = "compB",
    candidate_proba_col: str = "Proba",
    candidate_cdda_col: Optional[str] = "Cdda",
    only_cdda: bool = False,
    min_proba: Optional[float] = None,
    candidate_mode: str = "pairs",
    add_quality_flags: bool = True,
) -> pd.DataFrame:
    import Utility.TD as TD

    input_csv = Path(cfg.input_csv)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = out_dir / "candidate_plots"

    df = pd.read_csv(input_csv)

    cand = _read_candidates(
        candidates_csv=candidates_csv,
        source_col=candidate_source_col,
        comp_a_col=candidate_comp_a_col,
        comp_b_col=candidate_comp_b_col,
        proba_col=candidate_proba_col,
        cdda_col=candidate_cdda_col,
        only_cdda=only_cdda,
        min_proba=min_proba,
    )

    required_lc_cols = [
        cfg.source_col,
        cfg.component_col,
        cfg.time_col,
        cfg.flux_col,
        cfg.flux_err_col,
    ]

    missing = [c for c in required_lc_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing light-curve columns: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    if (
        getattr(cfg, "only_calibrated", False)
        and hasattr(cfg, "calibrated_col")
        and cfg.calibrated_col in df.columns
    ):
        df = df[df[cfg.calibrated_col].astype(bool)].copy()

    for c in [cfg.time_col, cfg.flux_col, cfg.flux_err_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required_lc_cols).copy()

    df["_source_key"] = _key_series(df[cfg.source_col])
    df["_comp_key"] = _key_series(df[cfg.component_col])

    candidate_sources = set(cand["_source_key"].unique())
    df = df[df["_source_key"].isin(candidate_sources)].copy()

    pair_table = _build_pair_table(
        df_lc=df,
        cand=cand,
        cfg=cfg,
        candidate_mode=candidate_mode,
    )

    if getattr(cfg, "verbose", True):
        print(f"Candidate rows: {len(cand)}")
        print(f"Candidate sources: {len(candidate_sources)}")
        print(f"Pairs to test: {len(pair_table)}")

    rows = []
    n_done = 0
    n_plots = 0

    max_pairs = getattr(cfg, "max_pairs", None)
    max_plots = getattr(cfg, "max_plots", None)
    min_points = getattr(cfg, "min_points_per_curve", 8)

    for _, pr in pair_table.iterrows():
        if max_pairs is not None and n_done >= int(max_pairs):
            break

        src = pr["source_key"]
        comp_a = pr["comp_A_key"]
        comp_b = pr["comp_B_key"]
        proba = pr["Proba"]

        row = {
            "Source ID": src,
            "comp A": comp_a,
            "comp B": comp_b,
            "Proba": proba,
            "TD": np.nan,
            "incertitude": np.nan,
            "q16": np.nan,
            "q50": np.nan,
            "q84": np.nan,
            "best_delay": np.nan,
            "best_cost": np.nan,
            "n_A": 0,
            "n_B": 0,
            "quality_flag": "not_run",
            "status": "ok",
            "plot_path": "",
        }

        try:
            gsrc = df[df["_source_key"] == src].copy()

            if len(gsrc) == 0:
                raise ValueError("source not found in light-curve CSV")

            if getattr(cfg, "subtract_time_origin_per_source", False):
                gsrc[cfg.time_col] = gsrc[cfg.time_col] - gsrc[cfg.time_col].min()

            ga = gsrc[gsrc["_comp_key"] == comp_a]
            gb = gsrc[gsrc["_comp_key"] == comp_b]

            if len(ga) == 0:
                raise ValueError(f"comp A not found for source: {comp_a}")

            if len(gb) == 0:
                raise ValueError(f"comp B not found for source: {comp_b}")

            lc_a = _clean_curve_from_df(ga, cfg)
            lc_b = _clean_curve_from_df(gb, cfg)

            row["n_A"] = len(lc_a["t"])
            row["n_B"] = len(lc_b["t"])

            if row["n_A"] < min_points or row["n_B"] < min_points:
                raise ValueError(
                    f"not enough points: n_A={row['n_A']}, n_B={row['n_B']}"
                )

            res = TD.estimate_delay_pair(lc_a, lc_b, cfg)

            td = res.get("delay", res.get("TimeDelay", np.nan))
            inc = res.get(
                "uncertainty",
                res.get("Incertie", res.get("incertitude", np.nan)),
            )

            row.update(
                {
                    "TD": float(td) if np.isfinite(td) else np.nan,
                    "incertitude": float(inc) if np.isfinite(inc) else np.nan,
                    "q16": res.get("q16", np.nan),
                    "q50": res.get("q50", np.nan),
                    "q84": res.get("q84", np.nan),
                    "best_delay": res.get(
                        "delay_best",
                        res.get("best_delay", np.nan),
                    ),
                    "best_cost": res.get("best_cost", np.nan),
                }
            )

            if add_quality_flags:
                if not np.isfinite(row["TD"]):
                    qflag = "failed"
                elif not np.isfinite(row["incertitude"]):
                    qflag = "no_uncertainty"
                elif abs(row["TD"]) > 0 and row["incertitude"] / max(abs(row["TD"]), 1e-9) < 0.25:
                    qflag = "good"
                elif abs(row["TD"]) > 0 and row["incertitude"] / max(abs(row["TD"]), 1e-9) < 0.50:
                    qflag = "medium"
                else:
                    qflag = "weak"

                row["quality_flag"] = qflag

            if getattr(cfg, "make_plots", False) and hasattr(TD, "plot_pair_result"):
                if max_plots is None or n_plots < int(max_plots):
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    safe_src = str(src).replace("/", "_")
                    safe_a = str(comp_a).replace("/", "_")
                    safe_b = str(comp_b).replace("/", "_")

                    plot_path = plots_dir / f"source_{safe_src}__{safe_a}_vs_{safe_b}.png"

                    TD.plot_pair_result(
                        res,
                        source_id=src,
                        comp_a=comp_a,
                        comp_b=comp_b,
                        cfg=cfg,
                        outpath=plot_path if getattr(cfg, "save_plots", True) else None,
                    )

                    row["plot_path"] = str(plot_path)
                    n_plots += 1

        except Exception as e:
            row["status"] = f"failed: {type(e).__name__}: {e}"
            row["quality_flag"] = "failed"

        rows.append(row)
        n_done += 1

        if getattr(cfg, "verbose", True) and n_done % 25 == 0:
            print(f"Processed {n_done}/{len(pair_table)} candidate pairs...")

    out = pd.DataFrame(rows)

    first_cols = ["Source ID", "comp A", "comp B", "Proba", "TD", "incertitude"]
    other_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + other_cols]

    if output_csv is None:
        output_csv = cfg.output_csv

    output_csv = Path(output_csv)

    if not output_csv.is_absolute():
        output_csv = out_dir / output_csv

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    if getattr(cfg, "verbose", True):
        print(f"Saved candidate TD CSV: {output_csv}")
        print(f"Done. Number of tested pairs: {len(out)}")

    return out