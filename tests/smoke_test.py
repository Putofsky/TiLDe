"""Dependency-light functional checks for the supervisor handoff.

Run from the project root with:

    python tests/smoke_test.py

These tests use synthetic curves only; no confidential Gaia table is required.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "tilde-matplotlib"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from Utility.IdSep import build_component_dict
from Utility import time_delay_interpolation as interpolation
from Utility import time_delay_pspline as pspline


def synthetic_pair(delay: float = 20.0) -> pd.DataFrame:
    """Return two noiseless curves with a known positive B-minus-A delay."""

    time_a = np.linspace(0.0, 180.0, 46)
    time_b = time_a + float(delay)

    def latent(time: np.ndarray) -> np.ndarray:
        return np.sin(time / 16.0) + 0.25 * np.cos(time / 6.0)

    return pd.DataFrame(
        {
            "source_id": ["demo"] * (len(time_a) + len(time_b)),
            "lensComponentSourceId": (
                ["A"] * len(time_a) + ["B"] * len(time_b)
            ),
            "epoch_obs_jd": np.concatenate([time_a, time_b]),
            "flux_obs": np.concatenate([latent(time_a), latent(time_b - delay)]),
            "flux_obs_error": 0.03,
        }
    )


def test_component_grouping() -> None:
    """Verify that DataFrame input is accepted and sorted by time."""

    table = synthetic_pair()
    grouped = build_component_dict(
        table,
        field_col="source_id",
        component_col="lensComponentSourceId",
        sort_col="epoch_obs_jd",
    )
    assert set(grouped["demo"]) == {"A", "B"}
    assert grouped["demo"]["A"]["epoch_obs_jd"].is_monotonic_increasing


def test_interpolation_delay() -> None:
    """Recover the known delay with the classical interpolation baseline."""

    table = synthetic_pair(delay=20.0)
    system = interpolation.get_pair_from_df(table, "demo", "A", "B")
    result = interpolation.estimate_time_delay_linear_interpolation(
        system["curves"],
        dmin=-40,
        dmax=40,
        ngrid=81,
        common_grid_size=100,
        min_points=8,
        min_frac=0.5,
        min_span_frac=0.5,
        scalar_xatol=0.02,
        verbose=False,
    )
    assert abs(float(result["delay"]) - 20.0) < 1.0


def test_pspline_delay() -> None:
    """Exercise the complete P-spline estimator on a known synthetic delay."""

    table = synthetic_pair(delay=20.0)
    system = pspline.get_components_from_df(
        table,
        source_id="demo",
        comp_ids=["A", "B"],
        names=["A reference", "B"],
    )
    prepared = pspline.prepare_multicurves(system["curves"], mad_floor=0.05)
    all_time = np.concatenate([curve["t"] for curve in prepared])
    knots = pspline.quantile_knots(all_time, K=5)
    basis, knot_vector = pspline.spline_basis(all_time, knots, degree=3)

    assert len(prepared) == 2
    assert basis.shape[0] == len(all_time)
    assert basis.shape[1] >= 4
    assert np.all(np.isfinite(basis))
    assert len(knot_vector) > len(knots)

    settings = pspline.load_parameter_profile("quick")
    settings.update(
        dmin=-40,
        dmax=40,
        ngrid=41,
        K_loo_candidates=[15],
        K_loo_n_candidates=1,
        verbose=False,
    )
    result = pspline.estimate_time_delay_pspline(
        system["curves"],
        **settings,
    )
    assert abs(float(result["delay"]) - 20.0) < 1.0


def main() -> None:
    """Run all smoke tests without requiring pytest."""

    test_component_grouping()
    test_interpolation_delay()
    test_pspline_delay()
    print("TiLDe smoke tests: OK")


if __name__ == "__main__":
    main()
