from __future__ import annotations

from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from micromobility_analytics.analytics.aggregations import CohortGroupColumn, DistributionMetric
from micromobility_analytics.charts.theme import (
    MEMBER_CASUAL_PALETTE,
    RIDEABLE_TYPE_PALETTE,
    apply_plotly_theme,
    build_empty_plotly_figure,
    hex_to_rgba,
)
from micromobility_analytics.data.features import (
    COMMUTE_WINDOWS,
    EQUITY_TIERS,
    MEMBER_TYPES,
    RIDEABLE_TYPES,
    WEATHER_REGIMES,
)

_METRIC_LABELS: dict[str, str] = {
    "trip_duration_min": "Trip Duration (minutes)",
    "mean_speed_kmh": "Mean Cycling Speed (km/h)",
    "haversine_distance_km": "Straight-Line Distance (km)",
}

_CATEGORY_ORDERS: dict[str, list[str]] = {
    "weather_regime": list(WEATHER_REGIMES),
    "commute_window": list(COMMUTE_WINDOWS),
    "start_equity_tier": list(EQUITY_TIERS),
    "member_casual": list(MEMBER_TYPES),
    "rideable_type": list(RIDEABLE_TYPES),
}

_COHORT_AXIS_LABELS: dict[str, str] = {
    "weather_regime": "Weather Conditions",
}


def build_split_violin_figure(
    sampled_trips_df: pd.DataFrame,
    cohort_stats_df: pd.DataFrame | None = None,
    metric: DistributionMetric = "trip_duration_min",
    x_cohort: CohortGroupColumn = "weather_regime",
    split_by: Literal["member_casual", "rideable_type"] = "member_casual",
    log_scale: bool = False,
    height: int = 460,
) -> go.Figure:
    metric_label = _METRIC_LABELS.get(metric, metric)
    title = f"{metric_label} by Weather Conditions"

    if sampled_trips_df.empty:
        return build_empty_plotly_figure(title=title, height=height)

    df = sampled_trips_df.copy()
    if split_by == "member_casual":
        split_pair = ("member", "casual")
        palette = MEMBER_CASUAL_PALETTE
        legend_labels = {"member": "Member", "casual": "Casual"}
    elif split_by == "rideable_type":
        split_pair = ("electric_bike", "classic_bike")
        palette = RIDEABLE_TYPE_PALETTE
        legend_labels = {"electric_bike": "Electric Bike", "classic_bike": "Classic Bike"}
    else:
        msg = f"Unsupported split_by dimension: {split_by!r}"
        raise ValueError(msg)

    fig = go.Figure()
    sides: tuple[Literal["negative", "positive"], Literal["negative", "positive"]] = (
        "negative",
        "positive",
    )

    for split_val, side in zip(split_pair, sides, strict=True):
        sub = df.loc[df[split_by] == split_val]
        if sub.empty:
            continue
        color = palette[split_val]
        fig.add_trace(
            go.Violin(
                x=sub[x_cohort].astype("string"),
                y=sub[metric],
                legendgroup=split_val,
                scalegroup=split_val,
                name=legend_labels[split_val],
                side=side,
                line_color=color,
                fillcolor=hex_to_rgba(color, alpha=0.45),
                box_visible=True,
                meanline_visible=True,
                points=False,
                spanmode="hard",
            )
        )

    apply_plotly_theme(fig, title=title, height=height)
    if cohort_stats_df is not None and not cohort_stats_df.empty and x_cohort in cohort_stats_df:
        fig.update_layout(title_subtitle_text=(f"Sample of {len(df):,} trips"))
    fig.update_layout(
        margin={"l": 60, "r": 32, "t": 82, "b": 52},
        violinmode="overlay",
        violingap=0.15,
        violingroupgap=0.05,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
    )
    if x_cohort in _CATEGORY_ORDERS:
        fig.update_xaxes(
            title_text=_COHORT_AXIS_LABELS.get(x_cohort, x_cohort.replace("_", " ").title()),
            categoryorder="array",
            categoryarray=_CATEGORY_ORDERS[x_cohort],
        )
    fig.update_yaxes(
        title_text=metric_label,
        type="log" if log_scale else "linear",
    )
    return fig
