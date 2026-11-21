from __future__ import annotations

import logging

import holoviews as hv
import pandas as pd
import param

from micromobility_analytics.analytics import (
    TRANSIT_FEEDER_THRESHOLD_M,
    KpiSummary,
    TripFilterCriteria,
    compute_dynamic_station_metrics,
    compute_kpi_summary,
    filter_trips,
)
from micromobility_analytics.config import settings
from micromobility_analytics.data.features import (
    COMMUTE_WINDOWS,
    EQUITY_TIERS,
    MEMBER_TYPES,
    RIDEABLE_TYPES,
    WEATHER_REGIMES,
)

logger = logging.getLogger(__name__)

PRESET_SCENARIOS: tuple[str, ...] = (
    "All Trips (2024)",
    "Peak Commute Transit Feeders (<=350m)",
    "Rain & Extreme Weather",
    "High-Need Neighborhoods",
    "Weekend & Midday Rides",
)


MONTH_OPTIONS: dict[str, str] = {
    "Jan '24": "202401",
    "Apr '24": "202404",
    "Jul '24": "202407",
    "Oct '24": "202410",
}
MEMBER_CASUAL_OPTIONS: dict[str, str] = {
    "Member": "member",
    "Casual": "casual",
}
RIDEABLE_TYPE_OPTIONS: dict[str, str] = {
    "Classic Bike": "classic_bike",
    "E-Bike": "electric_bike",
}
ENDPOINT_MODE_OPTIONS: dict[str, str] = {
    "Either": "either",
    "Origin": "origin",
    "Destination": "destination",
}
MAP_COLOR_OPTIONS: dict[str, str] = {
    "Flow Asymmetry": "flow_asymmetry_index",
    "Neighborhood Equity Tier": "equity_tier",
    "E-Bike Share (%)": "ebike_share_pct",
    "Transit Hub Distance (m)": "dist_to_transit_hub_m",
}
OD_GRANULARITY_OPTIONS: dict[str, str] = {
    "Neighborhood": "neighborhood",
    "Equity Tier": "equity_tier",
    "Transit Feeder Role": "transit_hub_role",
    "Station-to-Station": "station",
}
VIOLIN_METRIC_OPTIONS: dict[str, str] = {
    "Trip Duration (min)": "trip_duration_min",
    "Mean Speed (km/h)": "mean_speed_kmh",
    "Straight-Line Distance (km)": "haversine_distance_km",
}
VIOLIN_COHORT_OPTIONS: dict[str, str] = {
    "Weather Regime": "weather_regime",
    "Commute Window": "commute_window",
    "Origin Equity Tier": "start_equity_tier",
}
VIOLIN_SPLIT_OPTIONS: dict[str, str] = {
    "Member vs. Casual": "member_casual",
    "E-Bike vs. Classic": "rideable_type",
}


class DashboardState(param.Parameterized):
    preset_scenario: param.Selector = param.Selector(
        default=PRESET_SCENARIOS[0],
        objects=list(PRESET_SCENARIOS),
    )

    months = param.ListSelector(
        default=list(settings.default_jc_months),
        objects=MONTH_OPTIONS,
    )
    member_casual = param.ListSelector(
        default=list(MEMBER_TYPES),
        objects=MEMBER_CASUAL_OPTIONS,
    )
    rideable_type = param.ListSelector(
        default=list(RIDEABLE_TYPES),
        objects=RIDEABLE_TYPE_OPTIONS,
    )
    commute_window = param.ListSelector(
        default=list(COMMUTE_WINDOWS),
        objects=list(COMMUTE_WINDOWS),
    )
    weather_regime = param.ListSelector(
        default=list(WEATHER_REGIMES),
        objects=list(WEATHER_REGIMES),
    )
    equity_tiers = param.ListSelector(
        default=list(EQUITY_TIERS),
        objects=list(EQUITY_TIERS),
    )
    station_endpoint_mode: param.Selector = param.Selector(
        default="either",
        objects=ENDPOINT_MODE_OPTIONS,
    )
    include_round_trips = param.Boolean(default=True)
    transit_feeder_only = param.Boolean(default=False)

    selected_station_ids = param.Parameter(default=frozenset())

    map_color_by: param.Selector = param.Selector(
        default="flow_asymmetry_index",
        objects=MAP_COLOR_OPTIONS,
    )
    od_granularity: param.Selector = param.Selector(
        default="neighborhood",
        objects=OD_GRANULARITY_OPTIONS,
    )
    od_top_k = param.Integer(
        default=15,
        bounds=(5, 40),
    )
    od_exclude_self_loops = param.Boolean(default=False)
    violin_metric: param.Selector = param.Selector(
        default="trip_duration_min",
        objects=VIOLIN_METRIC_OPTIONS,
    )
    violin_x_cohort: param.Selector = param.Selector(
        default="weather_regime",
        objects=VIOLIN_COHORT_OPTIONS,
    )
    violin_split_by: param.Selector = param.Selector(
        default="member_casual",
        objects=VIOLIN_SPLIT_OPTIONS,
    )
    violin_log_scale = param.Boolean(default=False)

    def __init__(
        self,
        trips_df: pd.DataFrame,
        stations_df: pd.DataFrame,
        **params: object,
    ) -> None:
        self.trips_df: pd.DataFrame = trips_df
        self.stations_df: pd.DataFrame = (
            stations_df.sort_values("station_id").reset_index(drop=True)
            if not stations_df.empty
            else stations_df.copy()
        )
        self._station_id_Order: list[str] = (
            self.stations_df["station_id"].astype("string").tolist()
            if not self.stations_df.empty
            else []
        )
        self._updating_preset: bool = False
        super().__init__(**params)

        self.selection_stream = hv.streams.Selection1D()
        self.selection_stream.param.watch(self._on_map_selection_index_change, "index")

    def _on_map_selection_index_change(self, event: param.parameterized.Event) -> None:
        indices: list[int] = list(event.new) if event.new else []
        self.apply_selection_indices(indices)

    def apply_selection_indices(self, indices: list[int]) -> None:
        if not indices:
            self.selected_station_ids = frozenset()
            return
        matched = frozenset(
            self._station_id_Order[idx] for idx in indices if 0 <= idx < len(self._station_id_Order)
        )
        self.selected_station_ids = matched

    @param.depends("preset_scenario", watch=True)
    def _apply_preset_scenario(self) -> None:
        # Update filter parameters atomically when the user selects a guided scenario
        if self._updating_preset:
            return
        scenario = str(self.preset_scenario)
        with param.parameterized.batch_call_watchers(self):
            self.months = list(settings.default_jc_months)
            self.member_casual = list(MEMBER_TYPES)
            self.rideable_type = list(RIDEABLE_TYPES)
            self.commute_window = list(COMMUTE_WINDOWS)
            self.weather_regime = list(WEATHER_REGIMES)
            self.equity_tiers = list(EQUITY_TIERS)
            self.station_endpoint_mode = "either"
            self.include_round_trips = True
            self.transit_feeder_only = False
            self.selected_station_ids = frozenset()

            if scenario == PRESET_SCENARIOS[1]:
                # Q1: Peak Commute First/Last-Mile Transit Feeders (<=350m)
                self.commute_window = ["AM Peak", "PM Peak"]
                self.transit_feeder_only = True
                self.include_round_trips = False
                self.od_granularity = "transit_hub_role"
                self.map_color_by = "flow_asymmetry_index"
            elif scenario == PRESET_SCENARIOS[2]:
                # Q2: Adverse Weather Resilience (Rain & Extreme Cold/Heat)
                self.weather_regime = ["Light Rain", "Heavy Rain/Snow", "Extreme Temp"]
                self.violin_x_cohort = "weather_regime"
                self.violin_split_by = "member_casual"
            elif scenario == PRESET_SCENARIOS[3]:
                # Q3: High-Need Neighborhood Equity Focus
                self.equity_tiers = ["High Need"]
                self.station_endpoint_mode = "either"
                self.map_color_by = "equity_tier"
                self.od_granularity = "neighborhood"
            elif scenario == PRESET_SCENARIOS[4]:
                # Q4: E-Bike vs. Classic Weekend Leisure Rides
                self.commute_window = ["Midday", "Night"]
                self.violin_x_cohort = "commute_window"
                self.violin_split_by = "rideable_type"
                self.map_color_by = "ebike_share_pct"

    def to_filter_criteria(self, include_station_selection: bool = True) -> TripFilterCriteria:
        selected: frozenset[str] | None = None
        if include_station_selection and self.selected_station_ids:
            selected = frozenset(self.selected_station_ids)

        max_transit_m: float | None = (
            TRANSIT_FEEDER_THRESHOLD_M if bool(self.transit_feeder_only) else None
        )
        endpoint_mode = str(self.station_endpoint_mode)
        if endpoint_mode not in ("either", "origin", "destination"):
            endpoint_mode = "either"

        return TripFilterCriteria(
            months=tuple(str(m) for m in self.months) if self.months is not None else (),
            member_casual=(
                tuple(str(c) for c in self.member_casual) if self.member_casual is not None else ()
            ),
            rideable_type=(
                tuple(str(r) for r in self.rideable_type) if self.rideable_type is not None else ()
            ),
            commute_window=(
                tuple(str(w) for w in self.commute_window)
                if self.commute_window is not None
                else ()
            ),
            weather_regime=(
                tuple(str(w) for w in self.weather_regime)
                if self.weather_regime is not None
                else ()
            ),
            equity_tiers=(
                tuple(str(t) for t in self.equity_tiers) if self.equity_tiers is not None else ()
            ),
            selected_station_ids=selected,
            station_endpoint_mode=endpoint_mode,  # type: ignore[arg-type]
            include_round_trips=bool(self.include_round_trips),
            max_dist_to_transit_m=max_transit_m,
        )

    def sidebar_filtered_trips(self) -> pd.DataFrame:
        return filter_trips(
            self.trips_df,
            self.to_filter_criteria(include_station_selection=False),
        )

    def filtered_trips(self) -> pd.DataFrame:
        return filter_trips(
            self.trips_df,
            self.to_filter_criteria(include_station_selection=True),
        )

    def temporal_reference_trips(self) -> pd.DataFrame:
        month_criteria = TripFilterCriteria(
            months=tuple(str(m) for m in self.months) if self.months is not None else (),
        )
        return filter_trips(self.trips_df, month_criteria)

    def dynamic_stations(self) -> pd.DataFrame:
        return compute_dynamic_station_metrics(self.sidebar_filtered_trips(), self.stations_df)

    def kpi_summary(self) -> KpiSummary:
        return compute_kpi_summary(self.filtered_trips())

    def selected_station_names(self) -> list[str]:
        if not self.selected_station_ids or self.stations_df.empty:
            return []
        sub = self.stations_df.loc[
            self.stations_df["station_id"].isin(list(self.selected_station_ids)),
            "station_name",
        ]
        return sorted(sub.astype("string").tolist())

    def clear_station_selection(self, *_: object) -> None:
        # Clear any active  map brushing selection
        self.selection_stream.event(index=[])
        self.selected_station_ids = frozenset()
        self.param.trigger("map_color_by")

    def reset_all_filters(self, *_: object) -> None:
        # Restore all sidebar filters and view parameters to baseline defaults
        self._updating_preset = True
        try:
            with param.parameterized.batch_call_watchers(self):
                self.preset_scenario = PRESET_SCENARIOS[0]
                self.months = list(settings.default_jc_months)
                self.member_casual = list(MEMBER_TYPES)
                self.rideable_type = list(RIDEABLE_TYPES)
                self.commute_window = list(COMMUTE_WINDOWS)
                self.weather_regime = list(WEATHER_REGIMES)
                self.equity_tiers = list(EQUITY_TIERS)
                self.station_endpoint_mode = "either"
                self.include_round_trips = True
                self.transit_feeder_only = False
                self.selected_station_ids = frozenset()
                self.map_color_by = "flow_asymmetry_index"
                self.od_granularity = "neighborhood"
                self.od_top_k = 15
                self.od_exclude_self_loops = False
                self.violin_metric = "trip_duration_min"
                self.violin_x_cohort = "weather_regime"
                self.violin_split_by = "member_casual"
                self.violin_log_scale = False
            self.selection_stream.event(index=[])
            self.param.trigger("map_color_by")
        finally:
            self._updating_preset = False
