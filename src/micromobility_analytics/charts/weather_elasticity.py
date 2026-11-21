from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from micromobility_analytics.charts.theme import (
    MEMBER_CASUAL_PALETTE,
    SEQUENTIAL_CMAP,
    apply_plotly_theme,
    build_empty_plotly_figure,
)
from micromobility_analytics.data.features import WEATHER_REGIMES


def build_weather_time_heatmap(
    weather_density_df: pd.DataFrame,
    z_metric: str = "trips_per_observed_hour",
    height: int = 440,
) -> go.Figure:
    title = "Trips per Hour by Temperature and Time of day"
    if weather_density_df.empty:
        return build_empty_plotly_figure(title=title, height=height)

    df = weather_density_df.copy().reset_index(drop=True)
    df["ebike_share_pct"] = (df["ebike_share"] * 100.0).astype("float64")
    df["member_share_pct"] = (df["member_share"] * 100.0).astype("float64")

    customdata = df[
        [
            "trip_count",
            "observed_hours",
            "trips_per_observed_hour",
            "ebike_share_pct",
            "member_share_pct",
            "mean_precipitation_mm",
        ]
    ].to_numpy()

    colorbar_title = "Trips / Hour" if z_metric == "trips_per_observed_hour" else "Total Trips"
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=df["hour_of_day"],
                y=df["temp_bin_mid_c"],
                z=df[z_metric],
                colorscale=SEQUENTIAL_CMAP,
                colorbar={"title": colorbar_title},
                customdata=customdata,
                xgap=1,
                ygap=1,
                hovertemplate=(
                    "<b>Hour: %{x}:00 | Temp Bin: %{y:.1f}°C</b><br>"
                    "Normalized Rate: %{customdata[2]:.1f} trips / hour<br>"
                    "Total Trips: %{customdata[0]:,.0f} across %{customdata[1]:.0f} hours<br>"
                    "E-Bike Share: %{customdata[3]:.1f}% | Member Share: %{customdata[4]:.1f}%<br>"
                    "Mean Precip: %{customdata[5]:.2f} mm/h<extra></extra>"
                ),
            )
        ]
    )
    apply_plotly_theme(fig, title=title, height=height)
    fig.update_xaxes(
        title_text="Hour of Day",
        tickmode="linear",
        tick0=0,
        dtick=2,
    )
    fig.update_yaxes(title_text="Air Temperature Midpoint (°C)")
    return fig


def build_weather_elasticity_chart(
    weather_elasticity_df: pd.DataFrame,
    height: int = 440,
) -> go.Figure:
    title = "Ridership Change by Weather Conditions"
    if weather_elasticity_df.empty:
        return build_empty_plotly_figure(title=title, height=height)

    df = weather_elasticity_df.copy()
    fig = go.Figure()

    cohort_specs = [
        ("member", "electric_bike", "Member · E-Bike", "solid", "circle"),
        ("member", "classic_bike", "Member · Classic", "dash", "square"),
        ("casual", "electric_bike", "Casual · E-Bike", "solid", "diamond"),
        ("casual", "classic_bike", "Casual · Classic", "dash", "triangle-up"),
    ]

    for member_type, bike_type, label, line_dash, marker_symbol in cohort_specs:
        sub = df.loc[
            (df["member_casual"] == member_type) & (df["rideable_type"] == bike_type)
        ].copy()
        if sub.empty:
            continue

        color = MEMBER_CASUAL_PALETTE[member_type]
        customdata = sub[
            [
                "trip_count",
                "observed_hours",
                "trips_per_observed_hour",
                "pct_change_vs_baseline",
                "mean_duration_min",
            ]
        ].to_numpy()

        fig.add_trace(
            go.Scatter(
                x=sub["weather_regime"].astype("string"),
                y=sub["elasticity_index_pct"],
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 2.6, "dash": line_dash},
                marker={"size": 9, "symbol": marker_symbol, "color": color},
                customdata=customdata,
                hovertemplate=(
                    f"<b>{label} — %{{x}}</b><br>"
                    "Elasticity Index: %{y:.1f}% (%{customdata[3]:+.1f}% vs Clear/Dry)<br>"
                    "Rate: %{customdata[2]:.2f} trips / hour<br>"
                    "Volume: %{customdata[0]:,.0f} trips (%{customdata[1]:.0f} obs. hours)<br>"
                    "Mean Duration: %{customdata[4]:.1f} min<extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=100.0,
        line_dash="dot",
        line_color="#475569",
        annotation_text="Clear/Dry Baseline (100%)",
        annotation_position="top right",
    )

    apply_plotly_theme(fig, title=title, height=height)
    fig.update_xaxes(
        title_text="Weather Conditions",
        categoryorder="array",
        categoryarray=list(WEATHER_REGIMES),
    )
    fig.update_yaxes(title_text="Ridership Elasticity Index (% of Clear/Dry Rate)")
    fig.update_layout(
        margin={"l": 58, "r": 28, "t": 78, "b": 52},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11},
        },
    )
    return fig
