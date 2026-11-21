from __future__ import annotations

import html

import panel as pn

from micromobility_analytics.analytics import KpiSummary
from micromobility_analytics.ui.state import DashboardState

WIDGET_PILL_CSS: str = """
.bk-btn-group {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
}
.bk-btn-group .bk-btn {
    background-color: #ffffff !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 5px !important;
    font-weight: 500 !important;
    font-size: 11.5px !important;
    padding: 5px 8px !important;
    box-shadow: none !important;
    transition: all 0.12s ease-in-out !important;
}
.bk-btn-group .bk-btn:hover {
    background-color: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}
.bk-btn-group .bk-btn.bk-active {
    background-color: #0072B2 !important;
    color: #ffffff !important;
    border-color: #005a8d !important;
    font-weight: 600 !important;
    box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.18) !important;
}
"""

RESET_BTN_CSS: str = """
.bk-btn {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #0f172a !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 7px 12px !important;
}
.bk-btn:hover {
    background-color: #0072B2 !important;
    border-color: #0072B2 !important;
}
"""

EDITORIAL_RAW_CSS: str = (
    """
:root {
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Cascadia Code', Consolas, monospace;
}
body {
    background-color: #f8fafc !important;
    font-family: var(--font-sans) !important;
    color: #0f172a;
}
#sidebar, .sidenav, .pn-sidebar {
    padding: 8px 14px 18px 16px !important;
    box-sizing: border-box !important;
}
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(6, minmax(120px, 1fr));
    gap: 10px;
    width: 100%;
    margin-bottom: 8px;
}
@media (max-width: 1180px) {
    .kpi-strip {
        grid-template-columns: repeat(3, minmax(140px, 1fr));
    }
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 12px 14px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.kpi-overline {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
}
.kpi-subline {
    font-size: 12px;
    color: #475569;
    font-variant-numeric: tabular-nums;
    margin-top: 4px;
}
.editorial-surface {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.section-overline {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748b;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 2px 4px 2px;
    margin: 10px 0 6px 0;
}
.rubric-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    color: #0072B2;
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 4px;
    padding: 1px 6px;
    margin-right: 6px;
    font-variant-numeric: tabular-nums;
}
.status-banner-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    width: 100%;
    align-items: stretch;
}
.status-banner {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12.5px;
    color: #1e293b;
    font-variant-numeric: tabular-nums;
    display: flex;
    align-items: center;
    box-sizing: border-box;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
"""
    + WIDGET_PILL_CSS
)


def build_kpi_cards_row(kpi: KpiSummary) -> pn.pane.HTML:
    # 6-card editorial KPI summary strip with tabular numerals
    cards = [
        (
            "VALIDATED TRIPS",
            f"{kpi.total_trips:,}",
            f"{kpi.trips_per_active_day:,.1f} trips / active day ({kpi.active_days} d)",
        ),
        (
            "ACTIVE STATIONS",
            f"{kpi.active_stations}",
            "Hudson County JC / HB core docks",
        ),
        (
            "E-BIKE FLEET SHARE",
            f"{kpi.ebike_share_pct:.1f}%",
            f"Member share: {kpi.member_share_pct:.1f}%",
        ),
        (
            "MEDIAN DURATION",
            f"{kpi.median_duration_min:.1f} min",
            f"Median speed: {kpi.median_speed_kmh:.1f} km/h",
        ),
        (
            "TRANSIT FEEDER SHARE",
            f"{kpi.transit_feeder_share_pct:.1f}%",
            "<= 350m to transit hub",
        ),
        (
            "HIGH-NEED TIER SHARE",
            f"{kpi.high_need_tier_share_pct:.1f}%",
            "Origin or destination in High-Need area",
        ),
    ]
    cards_html = "".join(
        f'<div class="kpi-card">'
        f'<div class="kpi-overline">{label}</div>'
        f'<div class="kpi-value">{val}</div>'
        f'<div class="kpi-subline">{sub}</div>'
        f"</div>"
        for label, val, sub in cards
    )
    return pn.pane.HTML(
        f'<div class="kpi-strip">{cards_html}</div>',
        sizing_mode="stretch_width",
        margin=(0, 0, 6, 0),
    )


def build_selection_status_banner(state: DashboardState) -> pn.Row:
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
    def _status_view(*_: object) -> pn.pane.HTML:
        kpi = state.kpi_summary()
        total_baseline = len(state.trips_df)
        pct_of_total = (kpi.total_trips / total_baseline * 100.0) if total_baseline > 0 else 0.0
        selected_names = state.selected_station_names()

        if selected_names:
            preview = ", ".join(html.escape(n) for n in selected_names[:4])
            if len(selected_names) > 4:
                preview += f" (+{len(selected_names) - 4} more)"
            brush_text = (
                f"<strong>Selected on Map ({len(selected_names)} stations, "
                f"mode={html.escape(str(state.station_endpoint_mode))}):</strong>&nbsp;{preview}"
            )
        else:
            brush_text = "Map Selection: All 83 stations included"

        left_text = (
            f"Showing <strong>{kpi.total_trips:,}</strong> of {total_baseline:,} trips "
            f"({pct_of_total:.1f}%) across <strong>{kpi.active_stations}</strong> active stations"
        )
        return pn.pane.HTML(
            f'<div class="status-banner-grid">'
            f'<div class="status-banner"><span>{left_text}</span></div>'
            f'<div class="status-banner"><span>{brush_text}</span></div>'
            f"</div>",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        )

    return pn.Row(_status_view, sizing_mode="stretch_width")


def _section_header(title: str) -> pn.pane.HTML:
    #  uppercase editorial sidebar section divider
    return pn.pane.HTML(
        f'<div class="section-overline">{html.escape(title)}</div>',
        sizing_mode="stretch_width",
        margin=(6, 0, 2, 0),
    )


def _sub_label(text: str) -> pn.pane.HTML:
    # Render a compact field label inside the sidebar
    return pn.pane.HTML(f"<small style='color:#475569;font-weight:600'>{html.escape(text)}</small>")


def build_sidebar_controls(state: DashboardState) -> pn.Column:
    # Render compact sidebar controls without nested multi-line scrollboxes
    preset_select = pn.widgets.Select.from_param(
        state.param.preset_scenario,
        name="Preset Filter",
        sizing_mode="stretch_width",
    )

    months_group = pn.widgets.CheckButtonGroup.from_param(
        state.param.months,
        name="Seasonal Slices (YYYYMM)",
        stylesheets=[WIDGET_PILL_CSS],
        sizing_mode="stretch_width",
    )
    member_group = pn.widgets.CheckButtonGroup.from_param(
        state.param.member_casual,
        name="Rider Subscription",
        stylesheets=[WIDGET_PILL_CSS],
        sizing_mode="stretch_width",
    )
    bike_group = pn.widgets.CheckButtonGroup.from_param(
        state.param.rideable_type,
        name="Equipment Type",
        stylesheets=[WIDGET_PILL_CSS],
        sizing_mode="stretch_width",
    )
    window_group = pn.widgets.CheckButtonGroup.from_param(
        state.param.commute_window,
        name="Commute Window",
        stylesheets=[WIDGET_PILL_CSS],
        sizing_mode="stretch_width",
    )

    weather_checks = pn.widgets.CheckBoxGroup.from_param(
        state.param.weather_regime,
        name="Weather Conditions",
        inline=False,
        sizing_mode="stretch_width",
    )
    equity_group = pn.widgets.CheckButtonGroup.from_param(
        state.param.equity_tiers,
        name="Neighborhood Equity Tiers",
        stylesheets=[WIDGET_PILL_CSS],
        sizing_mode="stretch_width",
    )
    endpoint_select = pn.widgets.RadioButtonGroup.from_param(
        state.param.station_endpoint_mode,
        name="Station Match Rule",
        stylesheets=[WIDGET_PILL_CSS],
        sizing_mode="stretch_width",
    )
    transit_toggle = pn.widgets.Checkbox.from_param(
        state.param.transit_feeder_only,
        name="Transit Hubs Only (<= 350m)",
    )
    round_trip_toggle = pn.widgets.Checkbox.from_param(
        state.param.include_round_trips,
        name="Include Station Round Trips",
    )

    reset_btn = pn.widgets.Button(
        label="Reset All Filters & Selection",
        stylesheets=[RESET_BTN_CSS],
        sizing_mode="stretch_width",
        margin=(10, 0, 6, 0),
    )
    reset_btn.on_click(state.reset_all_filters)

    return pn.Column(
        _section_header("PRESET"),
        preset_select,
        _section_header("TIME & RIDER FILTERS"),
        _sub_label("Months (2024)"),
        months_group,
        _sub_label("Rider Type"),
        member_group,
        _sub_label("Bike Type"),
        bike_group,
        _sub_label("Time of Day"),
        window_group,
        _section_header("WEATHER & NEIGHBORHOOD"),
        _sub_label("Weather Conditions"),
        weather_checks,
        _sub_label("Neighborhood Equity Tier"),
        equity_group,
        _sub_label("Station Match Rule"),
        endpoint_select,
        transit_toggle,
        round_trip_toggle,
        _section_header("RESET"),
        reset_btn,
        sizing_mode="stretch_width",
        margin=(8, 12, 12, 16),
    )
