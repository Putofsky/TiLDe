from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


DEFAULT_TIMEZONE = "Europe/Paris"
DEFAULT_FIELD_COL = "source_id"
DEFAULT_COMPONENT_COL = "lensComponentSourceId"


def _read_any(
    data: Any,
    field_col: str = DEFAULT_FIELD_COL,
    component_col: str = DEFAULT_COMPONENT_COL,
) -> pd.DataFrame:
    """Accept CSV path, pandas DataFrame, or nested dict {source: {component: df}}."""

    if isinstance(data, (str, Path)):
        return pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, dict):
        parts: list[pd.DataFrame] = []

        for source_id, components in data.items():
            if not isinstance(components, dict):
                continue

            for component_id, component_df in components.items():
                tmp = pd.DataFrame(component_df).copy()

                if field_col not in tmp.columns:
                    tmp[field_col] = source_id
                if component_col not in tmp.columns:
                    tmp[component_col] = component_id

                parts.append(tmp)

        if parts:
            return pd.concat(parts, ignore_index=True)

    raise ValueError(
        "data must be a CSV path, a pandas DataFrame, or a nested dict "
        "{source_id: {component_id: DataFrame}}."
    )


def _guess_time_col(df: pd.DataFrame) -> str:
    candidates = [
        "jd_time",
        "jdTime",
        "JD_TIME",
        "jd",
        "JD",
        "julian_date",
        "JulianDate",
        "julianDate",
        "mjd",
        "MJD",
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
        key = str(col).lower()
        if "jd" in key or "julian" in key:
            return col

    for col in df.columns:
        key = str(col).lower()
        if "time" in key or "date" in key:
            return col

    raise ValueError(
        "Could not find a time column. Pass time_col='your_column'. "
        f"Available columns are: {list(df.columns)}"
    )


def _jd_or_datetime_to_local(series: pd.Series, timezone: str) -> pd.Series:
    """
    Convert JD/MJD time to local timezone.

    JD to Unix seconds:
        unix_seconds = (JD - 2440587.5) * 86400

    MJD is also handled:
        JD = MJD + 2400000.5
    """

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().any():
        jd = numeric.copy()

        # MJD values are usually around 50000-70000.
        # JD values are usually around 2400000+.
        mjd_mask = jd.between(30000, 100000)
        jd.loc[mjd_mask] = jd.loc[mjd_mask] + 2400000.5

        unix_seconds = (jd - 2440587.5) * 86400.0
        utc_time = pd.to_datetime(unix_seconds, unit="s", utc=True, errors="coerce")
    else:
        utc_time = pd.to_datetime(series, utc=True, errors="coerce")

    return utc_time.dt.tz_convert(timezone)


def _numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    numeric_cols: list[str] = []

    for col in df.columns:
        if col in exclude:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric_cols.append(col)

    return numeric_cols


def prepare_dataframe(
    data: Any,
    *,
    field_col: str = DEFAULT_FIELD_COL,
    component_col: str = DEFAULT_COMPONENT_COL,
    time_col: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> tuple[pd.DataFrame, str, list[str]]:
    df = _read_any(data, field_col=field_col, component_col=component_col)

    if field_col not in df.columns:
        raise ValueError(f"Missing field column {field_col!r}. Columns: {list(df.columns)}")

    if component_col not in df.columns:
        raise ValueError(f"Missing component column {component_col!r}. Columns: {list(df.columns)}")

    if time_col is None:
        time_col = _guess_time_col(df)

    if time_col not in df.columns:
        raise ValueError(f"Missing time column {time_col!r}. Columns: {list(df.columns)}")

    df = df.copy()

    # Convert JD/MJD/UTC datetime to our local time.
    df["_time_local"] = _jd_or_datetime_to_local(df[time_col], timezone)

    # Dash dropdowns work best with strings.
    df["_source_key"] = df[field_col].astype(str)
    df["_component_key"] = df[component_col].astype(str)

    # Drop bad times, then sort by source/component/time.
    df = df.dropna(subset=["_time_local"])
    df = df.sort_values([field_col, component_col, "_time_local"]).reset_index(drop=True)

    exclude = {
        field_col,
        component_col,
        time_col,
        "_time_local",
        "_source_key",
        "_component_key",
    }

    value_cols = _numeric_columns(df, exclude)

    if not value_cols:
        raise ValueError(
            "No numeric value columns found to plot. "
            "Your CSV must contain at least one numeric measurement column."
        )

    return df, time_col, value_cols


def make_app(
    data: Any,
    *,
    field_col: str = DEFAULT_FIELD_COL,
    component_col: str = DEFAULT_COMPONENT_COL,
    time_col: str | None = None,
    value_col: str | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    title: str = "EOLENS Plot",
) -> Dash:
    """
    Notebook use:

        from Utility import Plot

        app = Plot.make_app(data_path)
        app.run(debug=True)

    Or inline in newer Dash notebooks:

        app.run(jupyter_mode="inline")
    """

    df, resolved_time_col, value_cols = prepare_dataframe(
        data,
        field_col=field_col,
        component_col=component_col,
        time_col=time_col,
        timezone=timezone,
    )

    if value_col is None:
        value_col = value_cols[0]
    elif value_col not in value_cols:
        raise ValueError(f"value_col={value_col!r} is not numeric or was not found.")

    sources = sorted(df["_source_key"].dropna().unique().tolist())

    if not sources:
        raise ValueError("No source values found after cleaning the data.")

    first_source = sources[0]

    def component_options_for_source(source_key: str) -> list[dict[str, str]]:
        comps = (
            df.loc[df["_source_key"] == source_key, "_component_key"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        return [{"label": comp, "value": comp} for comp in comps]

    first_component_options = component_options_for_source(first_source)
    first_components = [opt["value"] for opt in first_component_options[:2]]

    app = Dash(__name__)

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "margin": "20px"},
        children=[
            html.H2(title),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr 1fr",
                    "gap": "12px",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div(
                        [
                            html.Label("Source"),
                            dcc.Dropdown(
                                id="source-dropdown",
                                options=[{"label": s, "value": s} for s in sources],
                                value=first_source,
                                clearable=False,
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            html.Label("Component(s)"),
                            dcc.Dropdown(
                                id="component-dropdown",
                                options=first_component_options,
                                value=first_components,
                                multi=True,
                                clearable=False,
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            html.Label("Y value"),
                            dcc.Dropdown(
                                id="value-dropdown",
                                options=[{"label": c, "value": c} for c in value_cols],
                                value=value_col,
                                clearable=False,
                            ),
                        ]
                    ),
                ],
            ),

            html.Div(
                style={"marginBottom": "12px"},
                children=[
                    html.Label("Plot mode"),
                    dcc.RadioItems(
                        id="mode-radio",
                        options=[
                            {"label": " lines + markers", "value": "lines+markers"},
                            {"label": " markers", "value": "markers"},
                            {"label": " lines", "value": "lines"},
                        ],
                        value="lines+markers",
                        inline=True,
                    ),
                ],
            ),

            dcc.Graph(id="main-graph", style={"height": "700px"}),

            html.Div(
                id="info-text",
                style={"marginTop": "8px", "color": "#555"},
            ),
        ],
    )

    @app.callback(
        Output("component-dropdown", "options"),
        Output("component-dropdown", "value"),
        Input("source-dropdown", "value"),
    )
    def update_components(source_key: str):
        options = component_options_for_source(source_key)
        selected = [opt["value"] for opt in options[:2]]
        return options, selected

    @app.callback(
        Output("main-graph", "figure"),
        Output("info-text", "children"),
        Input("source-dropdown", "value"),
        Input("component-dropdown", "value"),
        Input("value-dropdown", "value"),
        Input("mode-radio", "value"),
    )
    def update_graph(
        source_key: str,
        component_keys: list[str] | None,
        y_col: str,
        mode: str,
    ):
        if not component_keys:
            component_keys = []

        plot_df = df[
            (df["_source_key"] == source_key)
            & (df["_component_key"].isin(component_keys))
        ].copy()

        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[y_col]).sort_values("_time_local")

        fig = go.Figure()

        for component_key, sub in plot_df.groupby("_component_key", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=sub["_time_local"],
                    y=sub[y_col],
                    mode=mode,
                    name=str(component_key),
                    customdata=np.stack(
                        [sub[resolved_time_col].astype(str)],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "Local time: %{x}<br>"
                        f"{resolved_time_col}: "
                        "%{customdata[0]}<br>"
                        f"{y_col}: "
                        "%{y}<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        fig.update_layout(
            title=f"{y_col} by component for source {source_key}",
            xaxis_title=f"Local time ({timezone})",
            yaxis_title=y_col,
            hovermode="closest",
            legend_title=component_col,
            margin={"l": 60, "r": 20, "t": 60, "b": 60},
        )

        fig.update_xaxes(type="date")

        info = (
            f"Using time column {resolved_time_col!r}. "
            f"Rows plotted: {len(plot_df)}. "
            f"Sorted by converted local time in {timezone}."
        )

        return fig, info

    return app