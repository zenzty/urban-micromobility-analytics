from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import plotly.graph_objects as go

FONT_FAMILY: str = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
EARTH_ORIGIN_SHIFT_M: float = 20037508.342789244

# Colorblind-safe categorical palettes
MEMBER_CASUAL_PALETTE: dict[str, str] = {
    "member": "#0072B2",  # Blue
    "casual": "#E69F00",  # Orange
}

RIDEABLE_TYPE_PALETTE: dict[str, str] = {
    "electric_bike": "#009E73",  # Bluish Green
    "classic_bike": "#56B4E9",  # Sky Blue
}

EQUITY_TIER_PALETTE: dict[str, str] = {
    "High Need": "#D55E00",  # Vermilion
    "Moderate": "#E69F00",  # Amber
    "Affluent": "#0072B2",  # Blue
}

CORRIDOR_ROLE_PALETTE: dict[str, str] = {
    "First-Mile to Transit": "#0072B2",
    "Last-Mile from Transit": "#009E73",
    "Transit Hub Connector": "#CC79A7",
    "Intra-Neighborhood": "#56B4E9",
    "Inter-Neighborhood": "#E69F00",
}

SEQUENTIAL_CMAP: str = "Viridis"
DIVERGING_CMAP: str = "RdBu"


def ensure_bokeh_extension() -> None:
    if "bokeh" not in hv.Store.renderers:
        hv.extension("bokeh", logo=False)


def project_lng_lat_to_web_mercator(
    lng: pd.Series,
    lat: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    lng_arr = lng.to_numpy(dtype="float64")
    lat_arr = np.clip(lat.to_numpy(dtype="float64"), -85.05112878, 85.05112878)

    easting_m = lng_arr * (EARTH_ORIGIN_SHIFT_M / 180.0)
    lat_rad = np.radians(lat_arr)
    northing_m = np.log(np.tan((np.pi / 4.0) + (lat_rad / 2.0))) * (EARTH_ORIGIN_SHIFT_M / np.pi)
    return (
        pd.Series(easting_m, index=lng.index, dtype="float64"),
        pd.Series(northing_m, index=lat.index, dtype="float64"),
    )


def hex_to_rgba(hex_color: str, alpha: float = 0.45) -> str:
    cleaned = hex_color.lstrip("#")
    if len(cleaned) != 6:
        return f"rgba(0, 114, 178, {alpha:.2f})"
    r = int(cleaned[0:2], 16)
    g = int(cleaned[2:4], 16)
    b = int(cleaned[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def apply_plotly_theme(
    fig: go.Figure,
    title: str,
    height: int = 440,
) -> go.Figure:
    fig.update_layout(
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 16, "family": FONT_FAMILY, "color": "#0F172A"},
        },
        font={"family": FONT_FAMILY, "size": 12, "color": "#1E293B"},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        height=height,
        margin={"l": 56, "r": 32, "t": 56, "b": 48},
        hoverlabel={
            "bgcolor": "#0F172A",
            "font_size": 12,
            "font_family": FONT_FAMILY,
            "font_color": "#F8FAFC",
        },
    )
    return fig


def build_empty_plotly_figure(
    title: str,
    message: str = "No trips match the active filter criteria.",
    height: int = 400,
) -> go.Figure:
    fig = go.Figure()
    apply_plotly_theme(fig, title=title, height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#64748B", "family": FONT_FAMILY},
    )
    return fig


def build_empty_holoviews_overlay(
    title: str,
    message: str = "No stations or trips match the active filter criteria.",
    height: int = 480,
) -> hv.Element:
    ensure_bokeh_extension()
    return hv.Text(0.0, 0.0, message).opts(
        hv.opts.Text(
            title=title,
            height=height,
            responsive=True,
            xaxis=None,
            yaxis=None,
            text_font_size="12pt",
            text_color="#64748B",
            bgcolor="#F8FAFC",
            backend="bokeh",
        )
    )
