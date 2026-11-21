from __future__ import annotations

import re
from pathlib import Path

import panel as pn

from micromobility_analytics.config import PROJECT_ROOT
from micromobility_analytics.ui import (
    PRESET_SCENARIOS,
    DashboardState,
    build_equity_and_hotspots_tab,
    build_kpi_cards_row,
    build_methodology_and_audit_tab,
    build_selection_status_banner,
    build_sidebar_controls,
    build_spatial_and_od_tab,
    build_weather_and_distributions_tab,
    create_dashboard_app,
)
from tests.test_filters import _synthetic_enriched_dataset


def test_dashboard_state_presets_two_stage_filtering_and_reset() -> None:
    trips_df, stations_df = _synthetic_enriched_dataset()
    state = DashboardState(trips_df=trips_df, stations_df=stations_df)

    assert len(state.filtered_trips()) == 5
    assert len(state.dynamic_stations()) == 4

    # Test Q1 preset: Peak Commute First/Last-Mile Transit Feeders (<=350m)
    state.preset_scenario = PRESET_SCENARIOS[1]
    assert list(state.commute_window) == ["AM Peak", "PM Peak"]
    assert state.transit_feeder_only is True
    assert state.include_round_trips is False
    assert state.od_granularity == "transit_hub_role"

    # Test Q2 preset: Adverse Weather Resilience
    state.preset_scenario = PRESET_SCENARIOS[2]
    assert list(state.weather_regime) == ["Light Rain", "Heavy Rain/Snow", "Extreme Temp"]

    # Test Q3 preset: High-Need Neighborhood Equity Focus
    state.preset_scenario = PRESET_SCENARIOS[3]
    assert list(state.equity_tiers) == ["High Need"]
    assert len(state.filtered_trips()) == 2  # R3 and R4 touch JC099 (High Need)

    # Test Q4 preset: E-Bike vs. Classic Weekend Leisure Rides
    state.preset_scenario = PRESET_SCENARIOS[4]
    assert list(state.commute_window) == ["Midday", "Night"]

    # Reset back to baseline
    state.reset_all_filters()
    assert state.preset_scenario == PRESET_SCENARIOS[0]
    assert len(state.filtered_trips()) == 5
    assert state.selected_station_ids == frozenset()


def test_dashboard_state_selection1d_linking_and_two_stage_isolation() -> None:
    trips_df, stations_df = _synthetic_enriched_dataset()
    state = DashboardState(trips_df=trips_df, stations_df=stations_df)

    # Sorted station_ids in synthetic dataset: ['HB101', 'JC001', 'JC005', 'JC099']
    # Index 0 -> 'HB101' (only R5 starts at HB101)
    state.selection_stream.event(index=[0])
    assert state.selected_station_ids == frozenset({"HB101"})
    assert state.selected_station_names() == ["Hoboken Terminal"]

    # Stage 1 (sidebar_filtered_trips) still has all 5 trips so V1 map shows all 4 stations!
    assert len(state.sidebar_filtered_trips()) == 5
    assert len(state.dynamic_stations()) == 4

    # Stage 2 (filtered_trips) is cross-filtered to the 1 trip touching HB101!
    assert len(state.filtered_trips()) == 1
    assert state.kpi_summary().total_trips == 1

    # Clear map selection restores full 5 trips
    state.clear_station_selection()
    assert state.selected_station_ids == frozenset()
    assert len(state.filtered_trips()) == 5


def test_ui_components_tabs_and_fast_list_template_construction() -> None:
    trips_df, stations_df = _synthetic_enriched_dataset()
    state = DashboardState(trips_df=trips_df, stations_df=stations_df)

    kpi_strip = build_kpi_cards_row(state.kpi_summary())
    banner = build_selection_status_banner(state)
    sidebar = build_sidebar_controls(state)

    assert isinstance(kpi_strip, pn.pane.HTML)
    assert "VALIDATED TRIPS" in kpi_strip.object
    assert isinstance(banner, pn.Row)
    assert isinstance(sidebar, pn.Column)

    tab1 = build_spatial_and_od_tab(state)
    tab2 = build_weather_and_distributions_tab(state)
    tab3 = build_equity_and_hotspots_tab(state)
    tab4 = build_methodology_and_audit_tab()
    assert isinstance(tab1, pn.Column)
    assert isinstance(tab2, pn.Column)
    assert isinstance(tab3, pn.Column)
    assert isinstance(tab4, pn.Column)

    app = create_dashboard_app(trips_df=trips_df, stations_df=stations_df)
    assert isinstance(app, pn.template.FastListTemplate)
    assert app.header_background == "#0f172a"
    assert app.accent_base_color == "#0072B2"
    assert app.theme_toggle is False

    # Verify human-readable option labels are bound instead of raw snake_case identifiers
    assert "Jan '24" in state.param.months.names
    assert "Member" in state.param.member_casual.names
    assert "E-Bike" in state.param.rideable_type.names
    assert "Flow Asymmetry" in state.param.map_color_by.names
    assert "Transit Feeder Role" in state.param.od_granularity.names
    assert "Trip Duration (min)" in state.param.violin_metric.names


def test_ui_layer_contains_zero_emojis() -> None:
    ui_dir: Path = PROJECT_ROOT / "src" / "micromobility_analytics" / "ui"
    emoji_pattern = re.compile(
        "["
        "\U0001f300-\U0001f6ff"
        "\U0001f700-\U0001f77f"
        "\U0001f780-\U0001f7ff"
        "\U0001f800-\U0001f8ff"
        "\U0001f900-\U0001f9ff"
        "\U0001fa00-\U0001faff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "]+",
        flags=re.UNICODE,
    )
    for py_file in ui_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        assert not emoji_pattern.search(text), f"Emoji found in {py_file}"
