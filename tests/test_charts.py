from __future__ import annotations

import ast
from pathlib import Path

import holoviews as hv
import pandas as pd
import plotly.graph_objects as go

from micromobility_analytics.analytics import (
    compute_cohort_distributions,
    compute_dynamic_station_metrics,
    compute_equity_tier_summary,
    compute_od_flows,
    compute_weather_regime_elasticity,
    compute_weather_time_density,
    prepare_parallel_coords_table,
    sample_trips_for_distribution,
)
from micromobility_analytics.charts import (
    build_equity_parallel_coordinates,
    build_equity_tier_comparison_chart,
    build_od_sankey_figure,
    build_split_violin_figure,
    build_station_spatial_map,
    build_weather_elasticity_chart,
    build_weather_time_heatmap,
    hex_to_rgba,
    project_lng_lat_to_web_mercator,
)
from micromobility_analytics.config import PROJECT_ROOT
from tests.test_filters import _synthetic_enriched_dataset


def test_web_mercator_projection_and_rgba_helpers() -> None:
    lng = pd.Series([-74.0431])
    lat = pd.Series([40.7197])
    easting_m, northing_m = project_lng_lat_to_web_mercator(lng, lat)

    assert -8_250_000.0 < easting_m.iloc[0] < -8_235_000.0
    assert 4_965_000.0 < northing_m.iloc[0] < 4_978_000.0
    assert hex_to_rgba("#0072B2", alpha=0.5) == "rgba(0, 114, 178, 0.50)"


def test_v1_station_spatial_map_color_modes_and_selection_stream() -> None:
    trips_df, stations_df = _synthetic_enriched_dataset()
    dyn_stations = compute_dynamic_station_metrics(trips_df, stations_df)

    stream = hv.streams.Selection1D()
    overlay = build_station_spatial_map(
        dyn_stations,
        color_by="flow_asymmetry_index",
        selected_station_ids=frozenset({"JC005"}),
        selection_stream=stream,
    )
    assert isinstance(overlay, hv.Overlay)
    tile_layer = overlay.get(0)
    assert isinstance(tile_layer, hv.Tiles)
    assert "World_Light_Gray_Base" in str(tile_layer.data)
    assert isinstance(stream.source, hv.Points)
    assert stream.source.vdims[0].name == "station_id"

    for mode in ("equity_tier", "ebike_share_pct", "dist_to_transit_hub_m"):
        mode_map = build_station_spatial_map(dyn_stations, color_by=mode)  # type: ignore[arg-type]
        assert isinstance(mode_map, hv.Overlay)

    empty_map = build_station_spatial_map(dyn_stations.iloc[0:0])
    assert isinstance(empty_map, hv.Text)


def test_v2_od_sankey_figure_populated_and_empty() -> None:
    trips_df, _ = _synthetic_enriched_dataset()
    od_flows = compute_od_flows(trips_df, granularity="neighborhood")

    fig = build_od_sankey_figure(od_flows)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Sankey)
    assert len(fig.data[0].link.value) == len(od_flows)

    empty_fig = build_od_sankey_figure(od_flows.iloc[0:0])
    assert len(empty_fig.data) == 0
    assert len(empty_fig.layout.annotations) == 1


def test_v3_weather_heatmap_and_elasticity_charts() -> None:
    trips_df, _ = _synthetic_enriched_dataset()
    density_df = compute_weather_time_density(trips_df)
    elasticity_df = compute_weather_regime_elasticity(trips_df)

    heatmap_fig = build_weather_time_heatmap(density_df)
    assert isinstance(heatmap_fig, go.Figure)
    assert len(heatmap_fig.data) == 1
    assert isinstance(heatmap_fig.data[0], go.Heatmap)

    elast_fig = build_weather_elasticity_chart(elasticity_df)
    assert isinstance(elast_fig, go.Figure)
    assert len(elast_fig.data) >= 2

    assert len(build_weather_time_heatmap(density_df.iloc[0:0]).data) == 0
    assert len(build_weather_elasticity_chart(elasticity_df.iloc[0:0]).data) == 0


def test_v4_split_violin_figure_metrics_and_splits() -> None:
    trips_df, _ = _synthetic_enriched_dataset()
    sampled = sample_trips_for_distribution(trips_df, max_samples=100)
    stats = compute_cohort_distributions(trips_df, group_cols=("weather_regime", "member_casual"))

    fig_member = build_split_violin_figure(
        sampled,
        cohort_stats_df=stats,
        metric="trip_duration_min",
        x_cohort="weather_regime",
        split_by="member_casual",
    )
    assert isinstance(fig_member, go.Figure)
    assert len(fig_member.data) == 2
    assert all(isinstance(trace, go.Violin) for trace in fig_member.data)

    fig_bike = build_split_violin_figure(
        sampled,
        metric="mean_speed_kmh",
        x_cohort="commute_window",
        split_by="rideable_type",
        log_scale=True,
    )
    assert len(fig_bike.data) == 2
    assert fig_bike.layout.yaxis.type == "log"

    assert len(build_split_violin_figure(sampled.iloc[0:0]).data) == 0


def test_v5_equity_parallel_coordinates_and_tier_comparison() -> None:
    trips_df, stations_df = _synthetic_enriched_dataset()
    dyn = compute_dynamic_station_metrics(trips_df, stations_df)
    parcoords_df = prepare_parallel_coords_table(dyn)
    tier_summary = compute_equity_tier_summary(dyn)

    pc_fig = build_equity_parallel_coordinates(parcoords_df)
    assert isinstance(pc_fig, go.Figure)
    assert len(pc_fig.data) == 1
    assert isinstance(pc_fig.data[0], go.Parcoords)
    assert len(pc_fig.data[0].dimensions) == 7

    comp_fig = build_equity_tier_comparison_chart(tier_summary)
    assert isinstance(comp_fig, go.Figure)
    assert len(comp_fig.data) == 3

    assert len(build_equity_parallel_coordinates(parcoords_df.iloc[0:0]).data) == 0
    assert len(build_equity_tier_comparison_chart(tier_summary.iloc[0:0]).data) == 0


def test_charts_layer_has_zero_forbidden_ui_or_panel_imports() -> None:
    charts_dir: Path = PROJECT_ROOT / "src" / "micromobility_analytics" / "charts"
    forbidden = {"panel", "param", "micromobility_analytics.ui"}

    for py_file in charts_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    assert root_pkg not in forbidden, f"Forbidden import {alias.name} in {py_file}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_pkg = node.module.split(".")[0]
                assert root_pkg not in forbidden
                assert not node.module.startswith("micromobility_analytics.ui")
