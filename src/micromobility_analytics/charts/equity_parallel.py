from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from micromobility_analytics.charts.theme import (
    EQUITY_TIER_PALETTE,
    apply_plotly_theme,
    build_empty_plotly_figure,
)
from micromobility_analytics.data.features import EQUITY_TIERS


def build_equity_parallel_coordinates(
    parcoords_df: pd.DataFrame,
    height: int = 460,
) -> go.Figure:
    title = "Station Metrics Overview"
    if parcoords_df.empty:
        return build_empty_plotly_figure(title=title, height=height)

    df = parcoords_df.copy().reset_index(drop=True)
    tier_code_map = {tier: idx for idx, tier in enumerate(EQUITY_TIERS)}
    if "equity_tier_code" not in df.columns:
        df["equity_tier_code"] = (
            df["equity_tier"].astype("string").map(tier_code_map).fillna(0).astype("int64")
        )

    df["ebike_share_pct"] = (df["ebike_share"] * 100.0).astype("float64")
    df["casual_share_pct"] = (df["casual_share"] * 100.0).astype("float64")

    # 0 = High Need, 1 = Moderate, 2 = Affluent
    discrete_colorscale = [
        [0.0, EQUITY_TIER_PALETTE["High Need"]],
        [0.33, EQUITY_TIER_PALETTE["High Need"]],
        [0.33, EQUITY_TIER_PALETTE["Moderate"]],
        [0.66, EQUITY_TIER_PALETTE["Moderate"]],
        [0.66, EQUITY_TIER_PALETTE["Affluent"]],
        [1.0, EQUITY_TIER_PALETTE["Affluent"]],
    ]

    dimensions = [
        {
            "label": "Median Income ($)",
            "values": df["median_household_income"].round(0),
            "tickformat": "$,.0f",
        },
        {
            "label": "Car-Free HH (%)",
            "values": df["pct_car_free_households"].round(1),
            "range": [35.0, 65.0],
            "tickformat": ".1f",
        },
        {
            "label": "Transit Dist (m)",
            "values": df["dist_to_transit_hub_m"].round(0),
            "tickformat": ",.0f",
        },
        {
            "label": "Trips / Day",
            "values": df["trips_per_day"].round(1),
            "tickformat": ".1f",
        },
        {
            "label": "E-Bike Share (%)",
            "values": df["ebike_share_pct"].round(1),
            "range": [0.0, 100.0],
            "tickformat": ".1f",
        },
        {
            "label": "Casual Share (%)",
            "values": df["casual_share_pct"].round(1),
            "range": [0.0, 60.0],
            "tickformat": ".1f",
        },
        {
            "label": "Flow Asymmetry",
            "values": df["flow_asymmetry_index"].round(3),
            "range": [-0.4, 0.4],
            "tickvals": [-0.4, -0.2, 0.0, 0.2, 0.4],
            "ticktext": ["-0.40", "-0.20", "0.00", "+0.20", "+0.40"],
            "tickformat": "+.2f",
        },
    ]

    fig = go.Figure(
        data=[
            go.Parcoords(
                line={
                    "color": df["equity_tier_code"],
                    "colorscale": discrete_colorscale,
                    "cmin": 0,
                    "cmax": 2,
                    "showscale": True,
                    "colorbar": {
                        "title": "Equity Tier",
                        "tickvals": [0, 1, 2],
                        "ticktext": list(EQUITY_TIERS),
                        "len": 0.75,
                    },
                },
                dimensions=dimensions,
                labelangle=0,
                labelfont={"size": 11.5, "color": "#0F172A"},
                tickfont={"size": 10, "color": "#334155"},
            )
        ]
    )
    apply_plotly_theme(fig, title=title, height=height)
    fig.update_layout(
        title={"y": 0.96, "yanchor": "top"},
        margin={"l": 72, "r": 108, "t": 98, "b": 44},
    )
    return fig


def build_equity_tier_comparison_chart(
    equity_summary_df: pd.DataFrame,
    height: int = 400,
) -> go.Figure:
    title = "Station Averages by Neighborhood Tier"
    if equity_summary_df.empty:
        return build_empty_plotly_figure(title=title, height=height)

    df = equity_summary_df.copy().reset_index(drop=True)
    tiers = df["equity_tier"].astype("string").tolist()
    colors = [EQUITY_TIER_PALETTE.get(t, "#0072B2") for t in tiers]
    ebike_pct = (df["mean_ebike_share"] * 100.0).astype("float64")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Mean Station Trips / Day",
            "Mean E-Bike Share (%)",
            "Mean Distance to Transit (m)",
        ),
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Bar(
            x=tiers,
            y=df["mean_trips_per_day"],
            marker_color=colors,
            text=df["mean_trips_per_day"].round(1),
            textposition="outside",
            name="Trips / Day",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=tiers,
            y=ebike_pct,
            marker_color=colors,
            text=ebike_pct.round(1),
            textposition="outside",
            name="E-Bike Share (%)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=tiers,
            y=df["mean_dist_to_transit_m"],
            marker_color=colors,
            text=df["mean_dist_to_transit_m"].round(0),
            textposition="outside",
            name="Transit Dist (m)",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    apply_plotly_theme(fig, title=title, height=height)
    fig.update_layout(margin={"l": 56, "r": 28, "t": 78, "b": 48})
    max_trips = float(df["mean_trips_per_day"].max()) if len(df) else 10.0
    max_ebike = float(ebike_pct.max()) if len(ebike_pct) else 100.0
    max_dist = float(df["mean_dist_to_transit_m"].max()) if len(df) else 1000.0
    fig.update_yaxes(range=[0.0, max(1.0, max_trips * 1.22)], row=1, col=1)
    fig.update_yaxes(range=[0.0, max(10.0, max_ebike * 1.22)], row=1, col=2)
    fig.update_yaxes(range=[0.0, max(100.0, max_dist * 1.22)], row=1, col=3)
    return fig
