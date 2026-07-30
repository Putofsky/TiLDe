"""Utilities for grouping measurements by Gaia field and component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_component_dict(
    df: Any,
    field_col: str,
    component_col: str,
    sort_col: str | None = None,
    reset_index: bool = True,
) -> dict:
    """Group a light-curve table into ``field -> component -> DataFrame``.

    Parameters
    ----------
    df:
        Input :class:`pandas.DataFrame` or path to a CSV file.
    field_col:
        Column identifying the lens system / Gaia field.
    component_col:
        Column identifying a component inside a field.
    sort_col:
        Optional measurement-order column, normally ``epoch_obs_jd``.
    reset_index:
        Reset the row index inside each returned component table.

    Returns
    -------
    dict
        Nested dictionary whose leaves are independent DataFrame copies.

    Raises
    ------
    TypeError
        If ``df`` is neither a DataFrame nor a CSV path.
    ValueError
        If a required column is absent.
    """

    if isinstance(df, (str, Path)):
        data = pd.read_csv(df, low_memory=False)
    elif isinstance(df, pd.DataFrame):
        data = df.copy()
    else:
        raise TypeError("df must be a pandas DataFrame or a CSV path.")

    required_cols = [field_col, component_col]
    if sort_col is not None:
        required_cols.append(sort_col)

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if sort_col is not None:
        data = data.sort_values([field_col, component_col, sort_col])
    else:
        data = data.sort_values([field_col, component_col])

    result: dict = {}

    for field_id, field_df in data.groupby(field_col, sort=False):
        result[field_id] = {}

        for component_id, component_df in field_df.groupby(
            component_col,
            sort=False,
        ):
            if reset_index:
                component_df = component_df.reset_index(drop=True)

            result[field_id][component_id] = component_df.copy()

    return result
