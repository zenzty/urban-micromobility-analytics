from __future__ import annotations

import pandas as pd
import panel as pn

from micromobility_analytics.analytics import (
    CohortGroupColumn,
    DistributionMetric,
    OdGranularity,
    compute_cohort_distributions,
    compute_equity_tier_summary,
    compute_od_flows,
    compute_weather_regime_elasticity,
    compute_weather_time_density,
    prepare_parallel_coords_table,
    sample_trips_for_distribution,
)
from micromobility_analytics.charts import (
    CORRIDOR_ROLE_PALETTE,
    StationMapColorMetric,
    build_equity_parallel_coordinates,
    build_equity_tier_comparison_chart,
    build_od_sankey_figure,
    build_split_violin_figure,
    build_station_spatial_map,
    build_weather_elasticity_chart,
    build_weather_time_heatmap,
)
from micromobility_analytics.ui.state import DashboardState


def _build_corridor_legend_html() -> pn.pane.HTML:
    item_style = (
        "display:inline-flex;align-items:center;margin-right:10px;"
        "font-size:11px;color:#334155;white-space:nowrap;"
    )
    items = "".join(
        f'<span style="{item_style}">'
        f'<span style="width:9px;height:9px;border-radius:2px;background:{color};'
        f'display:inline-block;margin-right:4px;"></span>{role}</span>'
        for role, color in CORRIDOR_ROLE_PALETTE.items()
    )
    hdr_style = "font-size:10.5px;color:#64748b;margin-right:8px;text-transform:uppercase;"
    div_style = (
        "padding:6px 2px 0 2px;border-top:1px solid #f1f5f9;white-space:nowrap;overflow:hidden;"
    )
    return pn.pane.HTML(
        f'<div style="{div_style}"><strong style="{hdr_style}">Role:</strong>{items}</div>',
        sizing_mode="stretch_width",
        margin=(0, 0, 0, 0),
    )


def _caption_html(badge: str, question_tag: str) -> pn.pane.HTML:
    row_style = (
        "display:flex;justify-content:space-between;align-items:center;"
        "font-size:12px;color:#475569;margin-bottom:4px;"
    )
    q_style = "margin-right:0;color:#047857;background:#ecfdf5;border-color:#a7f3d0;"
    return pn.pane.HTML(
        f'<div style="{row_style}">'
        f'<span class="rubric-badge">{badge}</span>'
        f'<span class="rubric-badge" style="{q_style}">{question_tag}</span>'
        f"</div>",
        sizing_mode="stretch_width",
    )


def build_spatial_and_od_tab(state: DashboardState) -> pn.Column:
    # Geospatial Station Map with Selection1D brushing + OD Sankey network
    map_color_select = pn.widgets.Select.from_param(
        state.param.map_color_by,
        name="Color Stations By",
        width=210,
    )
    od_granularity_select = pn.widgets.Select.from_param(
        state.param.od_granularity,
        name="Group Corridors By",
        width=190,
    )
    od_top_k_slider = pn.widgets.IntSlider.from_param(
        state.param.od_top_k,
        name="Top Corridors Shown",
        width=180,
    )
    od_self_loop_toggle = pn.widgets.Checkbox.from_param(
        state.param.od_exclude_self_loops,
        name="Hide Same-Zone Trips",
        margin=(22, 0, 0, 8),
    )

    @pn.depends(
        state.param.months,
        state.param.member_casual,
        state.param.rideable_type,
        state.param.commute_window,
        state.param.weather_regime,
        state.param.equity_tiers,
        state.param.station_endpoint_mode,
        state.param.include_round_trips,
        state.param.transit_feeder_only,
        state.param.map_color_by,
    )
    def _v1_map_pane(*_: object) -> pn.pane.HoloViews:
        dyn_stations = state.dynamic_stations()
        color_metric: StationMapColorMetric = str(state.map_color_by)  # type: ignore[assignment]
        selected_ids = frozenset(state.selected_station_ids) if state.selected_station_ids else None
        overlay = build_station_spatial_map(
            dyn_stations,
            color_by=color_metric,
            selected_station_ids=selected_ids,
            selection_stream=state.selection_stream,
            height=500,
        )
        return pn.pane.HoloViews(overlay, sizing_mode="stretch_width")

    @pn.depends(
        state.param.months,
        state.param.member_casual,
        state.param.rideable_type,
        state.param.commute_window,
        state.param.weather_regime,
        state.param.equity_tiers,
        state.param.station_endpoint_mode,
        state.param.include_round_trips,
        state.param.transit_feeder_only,
        state.param.selected_station_ids,
        state.param.od_granularity,
        state.param.od_top_k,
        state.param.od_exclude_self_loops,
    )
    def _v2_sankey_pane(*_: object) -> pn.pane.Plotly:
        trips = state.filtered_trips()
        granularity: OdGranularity = str(state.od_granularity)  # type: ignore[assignment]
        od_df = compute_od_flows(
            trips,
            granularity=granularity,
            top_k=int(state.od_top_k),
            exclude_self_loops=bool(state.od_exclude_self_loops),
        )
        fig = build_od_sankey_figure(od_df, height=474)
        return pn.pane.Plotly(fig, sizing_mode="stretch_width", config={"displayModeBar": False})

    controls_bar = pn.Row(
        map_color_select,
        pn.Spacer(width=16),
        od_granularity_select,
        od_top_k_slider,
        od_self_loop_toggle,
        css_classes=["editorial-surface"],
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )

    left_col = pn.Column(
        _caption_html("STATION MAP", "Q1 / Q3"),
        _v1_map_pane,
        css_classes=["editorial-surface"],
        styles={"height": "100%"},
        sizing_mode="stretch_width",
    )
    right_col = pn.Column(
        _caption_html("CORRIDOR FLOWS", "Q1 / Q3"),
        _v2_sankey_pane,
        _build_corridor_legend_html(),
        css_classes=["editorial-surface"],
        styles={"height": "100%"},
        sizing_mode="stretch_width",
    )

    return pn.Column(
        controls_bar,
        pn.Row(left_col, right_col, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )


def build_weather_and_distributions_tab(state: DashboardState) -> pn.Column:
    # 2D Weather-Time Density Heatmap, Cohort Elasticity & Split Violin
    metric_select = pn.widgets.Select.from_param(
        state.param.violin_metric,
        name="Metric",
        width=210,
    )
    cohort_select = pn.widgets.Select.from_param(
        state.param.violin_x_cohort,
        name="Group By",
        width=200,
    )
    split_select = pn.widgets.Select.from_param(
        state.param.violin_split_by,
        name="Compare",
        width=190,
    )
    log_toggle = pn.widgets.Checkbox.from_param(
        state.param.violin_log_scale,
        name="Log Scale",
        margin=(22, 0, 0, 8),
    )

    @pn.depends(
        state.param.months,
        state.param.member_casual,
        state.param.rideable_type,
        state.param.commute_window,
        state.param.weather_regime,
        state.param.equity_tiers,
        state.param.station_endpoint_mode,
        state.param.include_round_trips,
        state.param.transit_feeder_only,
        state.param.selected_station_ids,
    )
    def _v3_row(*_: object) -> pn.Row:
        trips = state.filtered_trips()
        ref_trips = state.temporal_reference_trips()
        density_df = compute_weather_time_density(trips, reference_trips_df=ref_trips)
        elast_df = compute_weather_regime_elasticity(trips, reference_trips_df=ref_trips)

        heat_fig = build_weather_time_heatmap(density_df, height=420)
        elast_fig = build_weather_elasticity_chart(elast_df, height=420)

        left_card = pn.Column(
            _caption_html("HOURLY HEATMAP", "Q2"),
            pn.pane.Plotly(heat_fig, sizing_mode="stretch_width", config={"displayModeBar": False}),
            css_classes=["editorial-surface"],
            sizing_mode="stretch_width",
        )
        right_card = pn.Column(
            _caption_html("WEATHER IMPACT", "Q2"),
            pn.pane.Plotly(
                elast_fig, sizing_mode="stretch_width", config={"displayModeBar": False}
            ),
            css_classes=["editorial-surface"],
            sizing_mode="stretch_width",
        )
        return pn.Row(left_card, right_card, sizing_mode="stretch_width")

    @pn.depends(
        state.param.months,
        state.param.member_casual,
        state.param.rideable_type,
        state.param.commute_window,
        state.param.weather_regime,
        state.param.equity_tiers,
        state.param.station_endpoint_mode,
        state.param.include_round_trips,
        state.param.transit_feeder_only,
        state.param.selected_station_ids,
        state.param.violin_metric,
        state.param.violin_x_cohort,
        state.param.violin_split_by,
        state.param.violin_log_scale,
    )
    def _v4_violin_pane(*_: object) -> pn.pane.Plotly:
        trips = state.filtered_trips()
        metric: DistributionMetric = str(state.violin_metric)  # type: ignore[assignment]
        x_cohort: CohortGroupColumn = str(state.violin_x_cohort)  # type: ignore[assignment]
        split_by = str(state.violin_split_by)
        if split_by not in ("member_casual", "rideable_type"):
            split_by = "member_casual"

        strat_cols: tuple[CohortGroupColumn, ...] = (
            (x_cohort, split_by)  # type: ignore[assignment]
            if x_cohort != split_by
            else (x_cohort,)
        )
        sampled = sample_trips_for_distribution(
            trips,
            max_samples=12_000,
            stratify_cols=strat_cols,
            clip_upper_quantile=0.98,
        )
        stats = compute_cohort_distributions(trips, metric=metric, group_cols=strat_cols)
        fig = build_split_violin_figure(
            sampled,
            cohort_stats_df=stats,
            metric=metric,
            x_cohort=x_cohort,
            split_by=split_by,  # type: ignore[arg-type]
            log_scale=bool(state.violin_log_scale),
            height=450,
        )
        return pn.pane.Plotly(fig, sizing_mode="stretch_width", config={"displayModeBar": False})

    v4_controls = pn.Row(
        metric_select,
        cohort_select,
        split_select,
        log_toggle,
        sizing_mode="stretch_width",
        margin=(4, 0, 6, 0),
    )

    v4_section = pn.Column(
        _caption_html("TRIP DISTRIBUTIONS", "Q2 / Q4"),
        v4_controls,
        _v4_violin_pane,
        css_classes=["editorial-surface"],
        sizing_mode="stretch_width",
        margin=(10, 0, 0, 0),
    )

    return pn.Column(_v3_row, v4_section, sizing_mode="stretch_width")


def build_equity_and_hotspots_tab(state: DashboardState) -> pn.Column:
    # 7-Axis Equity Parallel Coordinates, Tier Comparison & Rebalancing Table

    @pn.depends(
        state.param.months,
        state.param.member_casual,
        state.param.rideable_type,
        state.param.commute_window,
        state.param.weather_regime,
        state.param.equity_tiers,
        state.param.station_endpoint_mode,
        state.param.include_round_trips,
        state.param.transit_feeder_only,
        state.param.selected_station_ids,
    )
    def _v5_content(*_: object) -> pn.Column:
        dyn_stations = compute_dynamic_station_metrics_for_equity(state)
        parcoords_df = prepare_parallel_coords_table(dyn_stations)
        eq_summary_df = compute_equity_tier_summary(dyn_stations)

        pc_fig = build_equity_parallel_coordinates(parcoords_df, height=450)
        comp_fig = build_equity_tier_comparison_chart(eq_summary_df, height=380)

        top_table_df = (
            dyn_stations[
                [
                    "station_id",
                    "station_name",
                    "neighborhood_name",
                    "equity_tier",
                    "median_household_income",
                    "dist_to_transit_hub_m",
                    "trips_per_day",
                    "ebike_share",
                    "net_flow_balance",
                    "flow_asymmetry_index",
                ]
            ]
            .copy()
            .sort_values("trips_per_day", ascending=False)
            .reset_index(drop=True)
        )
        top_table_df["median_household_income"] = top_table_df["median_household_income"].round(0)
        top_table_df["dist_to_transit_hub_m"] = top_table_df["dist_to_transit_hub_m"].round(0)
        top_table_df["trips_per_day"] = top_table_df["trips_per_day"].round(1)
        top_table_df["ebike_share"] = (top_table_df["ebike_share"] * 100.0).round(1)
        top_table_df["flow_asymmetry_index"] = top_table_df["flow_asymmetry_index"].round(3)
        top_table_df = top_table_df.rename(
            columns={
                "station_id": "ID",
                "station_name": "Station",
                "neighborhood_name": "Neighborhood",
                "equity_tier": "Equity Tier",
                "median_household_income": "Median Income ($)",
                "dist_to_transit_hub_m": "Transit Dist (m)",
                "trips_per_day": "Trips / Day",
                "ebike_share": "E-Bike (%)",
                "net_flow_balance": "Net Flow",
                "flow_asymmetry_index": "Asymmetry Index",
            }
        )

        pc_card = pn.Column(
            _caption_html("STATION PROFILES", "Q3"),
            pn.pane.Plotly(pc_fig, sizing_mode="stretch_width", config={"displayModeBar": False}),
            css_classes=["editorial-surface"],
            sizing_mode="stretch_width",
        )
        bottom_row = pn.Row(
            pn.Column(
                _caption_html("TIER SUMMARY", "Q3"),
                pn.pane.Plotly(
                    comp_fig, sizing_mode="stretch_width", config={"displayModeBar": False}
                ),
                css_classes=["editorial-surface"],
                styles={"height": "100%"},
                sizing_mode="stretch_width",
            ),
            pn.Column(
                _caption_html("STATION TABLE", "Q3"),
                pn.widgets.Tabulator(
                    top_table_df,
                    pagination="local",
                    page_size=10,
                    sizing_mode="stretch_width",
                    height=380,
                    show_index=False,
                ),
                css_classes=["editorial-surface"],
                styles={"height": "100%"},
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            margin=(10, 0, 0, 0),
        )
        return pn.Column(pc_card, bottom_row, sizing_mode="stretch_width")

    return pn.Column(_v5_content, sizing_mode="stretch_width")


def compute_dynamic_station_metrics_for_equity(state: DashboardState) -> pd.DataFrame:
    dyn = state.dynamic_stations()
    if state.selected_station_ids:
        sub = dyn.loc[dyn["station_id"].isin(list(state.selected_station_ids))].copy()
        if not sub.empty:
            return sub.reset_index(drop=True)
    return dyn


def build_methodology_and_audit_tab() -> pn.Column:
    lines = [
        "### 1. Data Sources & Spatial-Temporal Joins",
        "",
        "| Dataset | Provider | Coverage | Join Key |",
        "| :--- | :--- | :--- | :--- |",
        (
            "| **Citi Bike Trip Logs (`JC-` Slices)** | Lyft / Citi Bike System Data | "
            "`202401`, `202404`, `202407`, `202410` (`123` active days; `83` stations) | "
            "`started_at` floored to local hour (`America/New_York`); station IDs |"
        ),
        (
            "| **Historical Hourly Weather** | Open-Meteo Archive API | "
            "Hourly observations at Jersey City centroid (`40.7282, -74.0776`) | "
            "Exact `weather_hour` left-join with `merge_asof` DST fallback |"
        ),
        (
            "| **Census & Transit Proximity** | US Census ACS 5-Year & NJGIN/PATH | "
            "`12` Hudson County (`FIPS 34017`) tract zones & `12` transit hubs | "
            "Haversine distance matrix to nearest tract & transit hub (`m`) |"
        ),
        "",
        "---",
        "",
        "### 2. Data Cleaning & Validation Audit (`DQ-01` - `DQ-04`)",
        "",
        (
            "Across the 4 seasonal months (`202401`, `202404`, `202407`, `202410`): "
            "**360,527** raw records -> **355,259** clean records "
            "(**5,268** dropped / **1.46%**)."
        ),
        "",
        "| Rule | Filter Condition | Reason | Records Dropped (%) |",
        "| :--- | :--- | :--- | :--- |",
        (
            "| `DQ-01` | `ended_at <= started_at`, duration outside `[1, 240]` min, "
            "or prior-month spillover | Removes false starts (< 60s), unreturned bikes (> 4h), "
            "and 17 prior-month spillover rows | **3,454** (`0.96%`) |"
        ),
        (
            "| `DQ-02` | Missing `start_station_id`, `end_station_id`, or station names | "
            "Needed for corridor flows and neighborhood joins | "
            "**815** (`0.23%`) |"
        ),
        (
            "| `DQ-03` | Coordinates outside Hudson County (`[40.68, 40.77, -74.11, -74.015]`) "
            "or non-`JC`/`HB` station IDs | Removes out-of-bounds coordinates and "
            "970 cross-river NYC station trips | **970** (`0.27%`) |"
        ),
        (
            "| `DQ-04` | Straight-line speed `mean_speed_kmh > 45.0` | "
            "Filters impossible cycling speeds caused by timestamp/GPS errors | "
            "**29** (`0.01%`) |"
        ),
        "",
        "---",
        "",
        "### 3. Key Findings (`Q1` - `Q4`)",
        "",
        (
            "- **`Q1` - Transit Integration**: **79.2%** of trips start or end within "
            "`<= 350m` of a PATH, Light Rail, or Ferry stop (`27.5%` First-Mile "
            "to Transit, `27.0%` Last-Mile from Transit, `24.7%` Hub-to-Hub Connectors)."
        ),
        (
            "- **`Q2` - Weather Impact**: Accounting for how many hours each weather type occurs, "
            "member classic-bike rides stay at **66.6%** of their `Clear/Dry` "
            "hourly rate during `Light Rain` and **34.8%** during `Heavy Rain/Snow`, while "
            "`casual` classic-bike demand drops to **22.5%** in heavy rain/snow."
        ),
        (
            "- **`Q3` - Neighborhood Transit Equity**: Stations in `High Need` neighborhoods "
            "(`14` stations; median income `$54.3k - $69.4k`) average "
            "**1,023m** from rapid transit hubs (compared to **449m** in `Affluent` waterfront "
            "areas) and have the highest **e-bike share (73.4%** vs. **52.4%** in `Affluent` "
            "areas) to cover longer distances to transit."
        ),
        (
            "- **`Q4` - E-Bike vs. Classic Bike Usage**: E-bikes account for **58.1%** of "
            "fleet rides and maintain higher median speeds (`12.9` vs. `8.6 km/h`) and "
            "longer median distances (`1.24` vs. `1.00 km`) on uphill routes toward "
            "Journal Square, The Heights, and McGinley Square."
        ),
    ]
    return pn.Column(
        pn.pane.Markdown(
            "\n".join(lines),
            disable_math=True,
            css_classes=["editorial-surface"],
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
