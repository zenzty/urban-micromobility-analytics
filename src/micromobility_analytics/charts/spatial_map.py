from __future__ import annotations

from typing import Any, Literal

import holoviews as hv
import numpy as np
import pandas as pd
from bokeh.models import HoverTool

from micromobility_analytics.charts.theme import (
    DIVERGING_CMAP,
    EQUITY_TIER_PALETTE,
    SEQUENTIAL_CMAP,
    build_empty_holoviews_overlay,
    ensure_bokeh_extension,
    project_lng_lat_to_web_mercator,
)

ESRI_LIGHT_GRAY_TILE_URL: str = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Base/MapServer/tile/{Z}/{Y}/{X}"
)

StationMapColorMetric = Literal[
    "flow_asymmetry_index",
    "equity_tier",
    "ebike_share_pct",
    "dist_to_transit_hub_m",
]


def _build_station_hover_tool() -> HoverTool:
    return HoverTool(
        tooltips=[
            ("Station", "@station_name (@station_id)"),
            ("Neighborhood", "@neighborhood_name"),
            ("Equity Tier", "@equity_tier"),
            ("Total Trips", "@total_trips{0,0} (@trips_per_day{0.0}/day)"),
            ("Departures / Arrivals", "@total_departures{0,0} / @total_arrivals{0,0}"),
            ("Net Flow Balance", "@net_flow_balance{+0,0}"),
            ("Flow Asymmetry", "@flow_asymmetry_index{+0.00}"),
            ("E-Bike Share", "@ebike_share_pct{0.0}%"),
            ("Casual Share", "@casual_share_pct{0.0}%"),
            ("Nearest Transit Hub", "@nearest_transit_hub (@dist_to_transit_hub_m{0} m)"),
        ]
    )


def build_station_spatial_map(
    dynamic_stations_df: pd.DataFrame,
    color_by: StationMapColorMetric = "flow_asymmetry_index",
    selected_station_ids: frozenset[str] | None = None,
    selection_stream: hv.streams.Selection1D | None = None,
    height: int = 520,
) -> hv.Element:
    ensure_bokeh_extension()
    title = "Jersey City and Hoboken Citi Bike Stations"

    if dynamic_stations_df.empty:
        return build_empty_holoviews_overlay(title=title, height=height)

    df = dynamic_stations_df.copy().reset_index(drop=True)
    easting_m, northing_m = project_lng_lat_to_web_mercator(df["lng"], df["lat"])
    df["easting_m"] = easting_m
    df["northing_m"] = northing_m
    df["ebike_share_pct"] = (df["ebike_share"] * 100.0).astype("float64")
    df["casual_share_pct"] = (df["casual_share"] * 100.0).astype("float64")
    df["equity_tier"] = df["equity_tier"].astype("string")

    rates = df["trips_per_day"].to_numpy(dtype="float64")
    max_rate = float(np.max(rates)) if len(rates) > 0 else 0.0
    if max_rate > 0.0:
        norm_sqrt = np.sqrt(np.clip(rates, 0.0, None) / max_rate)
        marker_sizes = 9.0 + (17.0 * norm_sqrt)
    else:
        marker_sizes = np.full(len(df), 10.0, dtype="float64")
    df["marker_size"] = pd.Series(marker_sizes, index=df.index, dtype="float64")

    vdims = [
        "station_id",
        "station_name",
        "neighborhood_name",
        "equity_tier",
        "total_trips",
        "total_departures",
        "total_arrivals",
        "trips_per_day",
        "net_flow_balance",
        "flow_asymmetry_index",
        "ebike_share_pct",
        "casual_share_pct",
        "nearest_transit_hub",
        "dist_to_transit_hub_m",
        "marker_size",
    ]

    points = hv.Points(
        df,
        kdims=["easting_m", "northing_m"],
        vdims=vdims,
    )

    if selection_stream is not None:
        selection_stream.source = points

    selected_indices: list[int] = []
    if selected_station_ids:
        selected_indices = [
            int(idx)
            for idx, sid in enumerate(df["station_id"].astype("string"))
            if sid in selected_station_ids
        ]

    color_opts: dict[str, Any]
    if color_by == "flow_asymmetry_index":
        color_opts = {
            "color": "flow_asymmetry_index",
            "cmap": DIVERGING_CMAP,
            "clim": (-0.35, 0.35),
            "colorbar": True,
            "clabel": "Flow Asymmetry ((Arr - Dep) / Total)",
            "show_legend": False,
        }
    elif color_by == "equity_tier":
        color_opts = {
            "color": "equity_tier",
            "cmap": EQUITY_TIER_PALETTE,
            "colorbar": False,
            "show_legend": True,
            "legend_position": "bottom_left",
        }
    elif color_by == "ebike_share_pct":
        color_opts = {
            "color": "ebike_share_pct",
            "cmap": SEQUENTIAL_CMAP,
            "clim": (20.0, 90.0),
            "colorbar": True,
            "clabel": "E-Bike Share (%)",
            "show_legend": False,
        }
    elif color_by == "dist_to_transit_hub_m":
        color_opts = {
            "color": "dist_to_transit_hub_m",
            "cmap": "Cividis",
            "colorbar": True,
            "clabel": "Distance to Transit Hub (m)",
            "show_legend": False,
        }
    else:
        msg = f"Unsupported station map color metric: {color_by!r}"
        raise ValueError(msg)

    point_opts: dict[str, Any] = {
        "size": "marker_size",
        "alpha": 0.90,
        "nonselection_alpha": 0.25,
        "line_color": "#0F172A",
        "line_width": 1.2,
        "padding": 0.08,
        "tools": [_build_station_hover_tool(), "tap", "box_select", "lasso_select"],
        "active_tools": ["wheel_zoom"],
        "height": height,
        "responsive": True,
        "xaxis": None,
        "yaxis": None,
        "title": title,
        "backend": "bokeh",
        **color_opts,
    }
    if selected_indices:
        point_opts["selected"] = selected_indices

    styled_points = points.opts(hv.opts.Points(**point_opts))
    tiles = hv.Tiles(ESRI_LIGHT_GRAY_TILE_URL, name="EsriLightGray")
    return tiles * styled_points
