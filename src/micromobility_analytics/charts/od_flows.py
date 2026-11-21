from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from micromobility_analytics.charts.theme import (
    CORRIDOR_ROLE_PALETTE,
    FONT_FAMILY,
    apply_plotly_theme,
    build_empty_plotly_figure,
    hex_to_rgba,
)


def build_od_sankey_figure(
    od_flows_df: pd.DataFrame,
    title: str = "Top Origin-Destination Bike Corridors",
    height: int = 480,
) -> go.Figure:
    if od_flows_df.empty:
        return build_empty_plotly_figure(
            title=title,
            message="No Origin–Destination corridors match the active filter criteria.",
            height=height,
        )

    df = od_flows_df.copy().reset_index(drop=True)
    sources = list(df["source_label"].astype("string"))
    targets = list(df["target_label"].astype("string"))

    unique_nodes: list[str] = []
    for label in [*sources, *targets]:
        if label not in unique_nodes:
            unique_nodes.append(label)

    node_index = {label: idx for idx, label in enumerate(unique_nodes)}
    source_indices = [node_index[src] for src in sources]
    target_indices = [node_index[tgt] for tgt in targets]

    node_colors = [
        "#0072B2" if label.startswith("Origin:") else "#009E73" for label in unique_nodes
    ]
    link_colors = [
        hex_to_rgba(CORRIDOR_ROLE_PALETTE.get(str(role), "#0072B2"), alpha=0.45)
        for role in df["corridor_role"]
    ]

    customdata = df[
        [
            "origin",
            "destination",
            "corridor_role",
            "flow_share_pct",
            "mean_duration_min",
            "mean_speed_kmh",
        ]
    ].copy()
    customdata["ebike_share_pct"] = (df["ebike_share"] * 100.0).astype("float64")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                textfont={"family": FONT_FAMILY, "size": 11, "color": "#0F172A"},
                node={
                    "pad": 18,
                    "thickness": 18,
                    "line": {"color": "#0F172A", "width": 0.8},
                    "label": unique_nodes,
                    "color": node_colors,
                    "hovertemplate": "%{label}<br>Total Flow: %{value:,} trips<extra></extra>",
                },
                link={
                    "source": source_indices,
                    "target": target_indices,
                    "value": df["trip_count"].tolist(),
                    "color": link_colors,
                    "customdata": customdata.to_numpy(),
                    "hovertemplate": (
                        "<b>%{customdata[0]} → %{customdata[1]}</b><br>"
                        "Corridor Role: %{customdata[2]}<br>"
                        "Trips: %{value:,} (%{customdata[3]:.1f}% of shown flow)<br>"
                        "Mean Duration: %{customdata[4]:.1f} min | "
                        "Speed: %{customdata[5]:.1f} km/h<br>"
                        "E-Bike Share: %{customdata[6]:.1f}%<extra></extra>"
                    ),
                },
            )
        ]
    )
    apply_plotly_theme(fig, title=title, height=height)
    fig.update_layout(
        font={"family": FONT_FAMILY, "size": 11, "color": "#0F172A"},
        margin={"l": 20, "r": 20, "t": 72, "b": 28},
    )
    return fig
