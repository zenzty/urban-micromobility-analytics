from __future__ import annotations

import logging

import pandas as pd
import panel as pn

from micromobility_analytics.analytics import load_processed_datasets
from micromobility_analytics.charts import ensure_bokeh_extension
from micromobility_analytics.config import STATIONS_EQUITY_PATH, TRIPS_ENRICHED_PATH
from micromobility_analytics.ui.components import (
    EDITORIAL_RAW_CSS,
    build_kpi_cards_row,
    build_selection_status_banner,
    build_sidebar_controls,
)
from micromobility_analytics.ui.state import DashboardState
from micromobility_analytics.ui.views import (
    build_equity_and_hotspots_tab,
    build_methodology_and_audit_tab,
    build_spatial_and_od_tab,
    build_weather_and_distributions_tab,
)

logger = logging.getLogger(__name__)

pn.extension("plotly", "tabulator", raw_css=[EDITORIAL_RAW_CSS])
ensure_bokeh_extension()

TABS_FULL_WIDTH_CSS: str = """
.bk-header,
.bk-headers-wrapper,
.bk-headers {
    width: 100% !important;
    display: flex !important;
    flex: 1 1 100% !important;
}
.bk-tab {
    flex: 1 1 0px !important;
    text-align: center !important;
    justify-content: center !important;
}
:host > *[style*="visibility: hidden"] {
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
"""


@pn.cache
def get_cached_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load and cache the validated Parquet tables in memory across user sessions
    return load_processed_datasets(
        trips_path=TRIPS_ENRICHED_PATH,
        stations_path=STATIONS_EQUITY_PATH,
    )


def _build_missing_dataset_template(error_msg: str) -> pn.template.FastListTemplate:
    # diagnostic template when processed Parquet files have not been built yet
    msg_lines = [
        "### Processed Analytical Parquet Tables Not Found",
        "",
        f"`{error_msg}`",
        "",
        "Run the Layer 1 ETL & Pandera validation pipeline to build the processed Parquet files:",
        "",
        "```powershell",
        "uv run python scripts/build_dataset.py",
        "```",
    ]
    diagnostic = pn.pane.Markdown(
        "\n".join(msg_lines),
        css_classes=["editorial-surface"],
        sizing_mode="stretch_width",
    )
    return pn.template.FastListTemplate(
        title="Urban Micromobility, Weather & Transit Equity (2024)",
        header_background="#0f172a",
        accent_base_color="#0072B2",
        theme_toggle=False,
        main_max_width="1600px",
        main=[diagnostic],
    )


def create_dashboard_app(
    trips_df: pd.DataFrame | None = None,
    stations_df: pd.DataFrame | None = None,
) -> pn.template.FastListTemplate:
    # Assemble the complete Panel FastListTemplate application
    if trips_df is None or stations_df is None:
        try:
            trips_df, stations_df = get_cached_datasets()
        except FileNotFoundError as exc:
            logger.warning("Processed dataset files missing: %s", exc)
            return _build_missing_dataset_template(str(exc))

    state = DashboardState(trips_df=trips_df, stations_df=stations_df)

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
    def _reactive_kpi_strip(*_: object) -> pn.pane.HTML:
        return build_kpi_cards_row(state.kpi_summary())

    tabs = pn.Tabs(
        ("1. Station Map & Corridors", build_spatial_and_od_tab(state)),
        (
            "2. Weather & Trip Dynamics",
            build_weather_and_distributions_tab(state),
        ),
        ("3. Transit Equity & Stations", build_equity_and_hotspots_tab(state)),
        ("4. Data & Methodology", build_methodology_and_audit_tab()),
        dynamic=False,
        stylesheets=[TABS_FULL_WIDTH_CSS],
        sizing_mode="stretch_width",
    )

    main_layout = pn.Column(
        _reactive_kpi_strip,
        build_selection_status_banner(state),
        tabs,
        sizing_mode="stretch_width",
    )

    return pn.template.FastListTemplate(
        title="Urban Micromobility, Weather & Transit Equity (2024)",
        header_background="#0f172a",
        accent_base_color="#0072B2",
        theme_toggle=False,
        sidebar_width=330,
        main_max_width="1600px",
        sidebar=[build_sidebar_controls(state)],
        main=[main_layout],
    )


if __name__.startswith("bokeh") or __name__ == "__main__":
    create_dashboard_app().servable()
