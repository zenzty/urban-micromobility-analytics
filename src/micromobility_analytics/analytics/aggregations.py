from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from micromobility_analytics.data.features import (
    EQUITY_TIERS,
    MEMBER_TYPES,
    RIDEABLE_TYPES,
    WEATHER_REGIMES,
)

logger = logging.getLogger(__name__)

TRANSIT_FEEDER_THRESHOLD_M: float = 350.0

OdGranularity = Literal["neighborhood", "equity_tier", "station", "transit_hub_role"]
DistributionMetric = Literal["trip_duration_min", "mean_speed_kmh", "haversine_distance_km"]
CohortGroupColumn = Literal[
    "member_casual",
    "rideable_type",
    "weather_regime",
    "commute_window",
    "start_equity_tier",
]

_PARCOORDS_AXES: tuple[str, ...] = (
    "median_household_income",
    "pct_car_free_households",
    "dist_to_transit_hub_m",
    "trips_per_day",
    "ebike_share",
    "casual_share",
    "net_flow_balance",
)


@dataclass(frozen=True)
class KpiSummary:
    total_trips: int
    active_stations: int
    active_days: int
    trips_per_active_day: float
    ebike_share_pct: float
    member_share_pct: float
    median_duration_min: float
    median_speed_kmh: float
    transit_feeder_share_pct: float
    high_need_tier_share_pct: float


def compute_kpi_summary(
    trips_df: pd.DataFrame,
    transit_threshold_m: float = TRANSIT_FEEDER_THRESHOLD_M,
) -> KpiSummary:
    if trips_df.empty:
        return KpiSummary(
            total_trips=0,
            active_stations=0,
            active_days=0,
            trips_per_active_day=0.0,
            ebike_share_pct=0.0,
            member_share_pct=0.0,
            median_duration_min=0.0,
            median_speed_kmh=0.0,
            transit_feeder_share_pct=0.0,
            high_need_tier_share_pct=0.0,
        )

    total = len(trips_df)
    active_stations = len(
        set(trips_df["start_station_id"].unique()).union(set(trips_df["end_station_id"].unique()))
    )
    active_days = max(1, int(trips_df["started_at"].dt.normalize().nunique()))
    trips_per_day = round(total / float(active_days), 2)

    ebike_pct = round(
        float((trips_df["rideable_type"] == "electric_bike").sum()) / total * 100.0, 2
    )
    member_pct = round(float((trips_df["member_casual"] == "member").sum()) / total * 100.0, 2)
    median_dur = round(float(trips_df["trip_duration_min"].median()), 2)
    median_spd = round(float(trips_df["mean_speed_kmh"].median()), 2)

    is_feeder = (trips_df["start_dist_to_transit_m"] <= transit_threshold_m) | (
        trips_df["end_dist_to_transit_m"] <= transit_threshold_m
    )
    feeder_pct = round(float(is_feeder.sum()) / total * 100.0, 2)

    is_high_need = (trips_df["start_equity_tier"] == "High Need") | (
        trips_df["end_equity_tier"] == "High Need"
    )
    high_need_pct = round(float(is_high_need.sum()) / total * 100.0, 2)

    return KpiSummary(
        total_trips=total,
        active_stations=active_stations,
        active_days=active_days,
        trips_per_active_day=trips_per_day,
        ebike_share_pct=ebike_pct,
        member_share_pct=member_pct,
        median_duration_min=median_dur,
        median_speed_kmh=median_spd,
        transit_feeder_share_pct=feeder_pct,
        high_need_tier_share_pct=high_need_pct,
    )


def compute_dynamic_station_metrics(
    trips_df: pd.DataFrame,
    stations_equity_df: pd.DataFrame,
) -> pd.DataFrame:
    static_cols = [
        "station_id",
        "station_name",
        "lat",
        "lng",
        "neighborhood_name",
        "census_tract_geoid",
        "equity_tier",
        "median_household_income",
        "pct_car_free_households",
        "population_density",
        "nearest_transit_hub",
        "dist_to_transit_hub_m",
    ]
    base = stations_equity_df[static_cols].copy()

    if trips_df.empty:
        base["total_departures"] = pd.Series(0, index=base.index, dtype="int64")
        base["total_arrivals"] = pd.Series(0, index=base.index, dtype="int64")
        base["total_trips"] = pd.Series(0, index=base.index, dtype="int64")
        base["net_flow_balance"] = pd.Series(0, index=base.index, dtype="int64")
        base["flow_asymmetry_index"] = pd.Series(0.0, index=base.index, dtype="float64")
        base["trips_per_day"] = pd.Series(0.0, index=base.index, dtype="float64")
        base["ebike_share"] = pd.Series(0.0, index=base.index, dtype="float64")
        base["casual_share"] = pd.Series(0.0, index=base.index, dtype="float64")
        base["mean_duration_min"] = pd.Series(0.0, index=base.index, dtype="float64")
        return base.sort_values("station_id").reset_index(drop=True)

    active_days = max(1, int(trips_df["started_at"].dt.normalize().nunique()))
    trips_work = trips_df.assign(
        _is_ebike=(trips_df["rideable_type"] == "electric_bike").astype("int64"),
        _is_casual=(trips_df["member_casual"] == "casual").astype("int64"),
    )

    departures = (
        trips_work.groupby("start_station_id", as_index=False, observed=True)
        .agg(
            total_departures=("ride_id", "count"),
            ebike_departures=("_is_ebike", "sum"),
            casual_departures=("_is_casual", "sum"),
            dep_mean_duration=("trip_duration_min", "mean"),
        )
        .rename(columns={"start_station_id": "station_id"})
    )

    arrivals = (
        trips_work.groupby("end_station_id", as_index=False, observed=True)
        .agg(
            total_arrivals=("ride_id", "count"),
            arr_mean_duration=("trip_duration_min", "mean"),
        )
        .rename(columns={"end_station_id": "station_id"})
    )

    merged = base.merge(departures, on="station_id", how="left").merge(
        arrivals, on="station_id", how="left"
    )

    for col in ["total_departures", "total_arrivals", "ebike_departures", "casual_departures"]:
        merged[col] = merged[col].fillna(0).astype("int64")

    merged["total_trips"] = (merged["total_departures"] + merged["total_arrivals"]).astype("int64")
    merged["net_flow_balance"] = (merged["total_arrivals"] - merged["total_departures"]).astype(
        "int64"
    )
    merged["flow_asymmetry_index"] = pd.Series(
        np.where(
            merged["total_trips"] > 0,
            merged["net_flow_balance"] / merged["total_trips"],
            0.0,
        ),
        index=merged.index,
        dtype="float64",
    )
    merged["trips_per_day"] = (merged["total_departures"] / float(active_days)).astype("float64")
    merged["ebike_share"] = pd.Series(
        np.where(
            merged["total_departures"] > 0,
            merged["ebike_departures"] / merged["total_departures"],
            0.0,
        ),
        index=merged.index,
        dtype="float64",
    )
    merged["casual_share"] = pd.Series(
        np.where(
            merged["total_departures"] > 0,
            merged["casual_departures"] / merged["total_departures"],
            0.0,
        ),
        index=merged.index,
        dtype="float64",
    )
    merged["mean_duration_min"] = (
        merged["dep_mean_duration"]
        .fillna(merged["arr_mean_duration"])
        .fillna(0.0)
        .astype("float64")
    )

    out_cols = [
        *static_cols,
        "total_departures",
        "total_arrivals",
        "total_trips",
        "net_flow_balance",
        "flow_asymmetry_index",
        "trips_per_day",
        "ebike_share",
        "casual_share",
        "mean_duration_min",
    ]
    return merged[out_cols].sort_values("station_id").reset_index(drop=True)


def _classify_corridor_role(
    start_near_hub: pd.Series,
    end_near_hub: pd.Series,
    same_neighborhood: pd.Series,
) -> pd.Series:
    labels = np.select(
        [
            (~start_near_hub) & end_near_hub,
            start_near_hub & (~end_near_hub),
            start_near_hub & end_near_hub,
            same_neighborhood,
        ],
        [
            "First-Mile to Transit",
            "Last-Mile from Transit",
            "Transit Hub Connector",
            "Intra-Neighborhood",
        ],
        default="Inter-Neighborhood",
    )
    return pd.Series(labels, index=start_near_hub.index, dtype="string")


def compute_od_flows(
    trips_df: pd.DataFrame,
    granularity: OdGranularity = "neighborhood",
    top_k: int = 25,
    exclude_self_loops: bool = False,
    transit_threshold_m: float = TRANSIT_FEEDER_THRESHOLD_M,
) -> pd.DataFrame:
    out_cols = [
        "origin",
        "destination",
        "source_label",
        "target_label",
        "corridor_role",
        "trip_count",
        "flow_share_pct",
        "mean_duration_min",
        "mean_speed_kmh",
        "ebike_share",
    ]
    if trips_df.empty or top_k <= 0:
        return pd.DataFrame(columns=out_cols)

    work = trips_df.copy()
    if granularity == "neighborhood":
        work["origin"] = work["start_neighborhood"].astype("string")
        work["destination"] = work["end_neighborhood"].astype("string")
    elif granularity == "equity_tier":
        work["origin"] = work["start_equity_tier"].astype("string")
        work["destination"] = work["end_equity_tier"].astype("string")
    elif granularity == "station":
        work["origin"] = work["start_station_name"].astype("string")
        work["destination"] = work["end_station_name"].astype("string")
    elif granularity == "transit_hub_role":
        start_near = work["start_dist_to_transit_m"] <= transit_threshold_m
        end_near = work["end_dist_to_transit_m"] <= transit_threshold_m
        work["origin"] = pd.Series(
            np.where(start_near, "Transit Hub (<=350m)", "Neighborhood Feeder (>350m)"),
            index=work.index,
            dtype="string",
        )
        work["destination"] = pd.Series(
            np.where(end_near, "Transit Hub (<=350m)", "Neighborhood Feeder (>350m)"),
            index=work.index,
            dtype="string",
        )
    else:
        msg = f"Unsupported OD granularity: {granularity!r}"
        raise ValueError(msg)

    if exclude_self_loops:
        work = work.loc[work["origin"] != work["destination"]].copy()
        if work.empty:
            return pd.DataFrame(columns=out_cols)

    total_scoped_trips = len(work)
    work["_is_ebike"] = (work["rideable_type"] == "electric_bike").astype("float64")
    work["_corridor_role"] = _classify_corridor_role(
        work["start_dist_to_transit_m"] <= transit_threshold_m,
        work["end_dist_to_transit_m"] <= transit_threshold_m,
        work["start_neighborhood"] == work["end_neighborhood"],
    )

    grouped = (
        work.groupby(["origin", "destination"], as_index=False, observed=True)
        .agg(
            trip_count=("ride_id", "count"),
            mean_duration_min=("trip_duration_min", "mean"),
            mean_speed_kmh=("mean_speed_kmh", "mean"),
            ebike_share=("_is_ebike", "mean"),
            corridor_role=(
                "_corridor_role",
                lambda s: str(s.mode().iloc[0]) if not s.empty else "Inter-Neighborhood",
            ),
        )
        .sort_values(["trip_count", "origin", "destination"], ascending=[False, True, True])
        .head(top_k)
        .reset_index(drop=True)
    )

    grouped["origin"] = grouped["origin"].astype("string")
    grouped["destination"] = grouped["destination"].astype("string")
    grouped["source_label"] = ("Origin: " + grouped["origin"]).astype("string")
    grouped["target_label"] = ("Dest: " + grouped["destination"]).astype("string")
    grouped["corridor_role"] = grouped["corridor_role"].astype("string")
    grouped["trip_count"] = grouped["trip_count"].astype("int64")
    grouped["flow_share_pct"] = (grouped["trip_count"] / float(total_scoped_trips) * 100.0).astype(
        "float64"
    )
    grouped["mean_duration_min"] = grouped["mean_duration_min"].astype("float64")
    grouped["mean_speed_kmh"] = grouped["mean_speed_kmh"].astype("float64")
    grouped["ebike_share"] = grouped["ebike_share"].astype("float64")

    return grouped[out_cols]


def _align_reference_timeframe(
    trips_df: pd.DataFrame,
    reference_trips_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if reference_trips_df is None or reference_trips_df.empty or trips_df.empty:
        return trips_df

    active_dates = trips_df["started_at"].dt.normalize().unique()
    mask = reference_trips_df["started_at"].dt.normalize().isin(active_dates)
    aligned = reference_trips_df.loc[mask]
    return aligned if not aligned.empty else trips_df


def compute_weather_regime_elasticity(
    trips_df: pd.DataFrame,
    reference_trips_df: pd.DataFrame | None = None,
    baseline_regime: str = "Clear/Dry",
) -> pd.DataFrame:
    out_cols = [
        "weather_regime",
        "member_casual",
        "rideable_type",
        "trip_count",
        "observed_hours",
        "trips_per_observed_hour",
        "baseline_trips_per_hour",
        "elasticity_index_pct",
        "pct_change_vs_baseline",
        "mean_duration_min",
        "mean_speed_kmh",
    ]
    if trips_df.empty:
        return pd.DataFrame(columns=out_cols)

    ref_df = _align_reference_timeframe(trips_df, reference_trips_df)

    regime_exposure = (
        ref_df[["weather_hour", "weather_regime"]]
        .drop_duplicates()
        .groupby("weather_regime", as_index=False, observed=False)
        .agg(observed_hours=("weather_hour", "count"))
    )

    cohort_counts = trips_df.groupby(
        ["weather_regime", "member_casual", "rideable_type"],
        as_index=False,
        observed=False,
    ).agg(
        trip_count=("ride_id", "count"),
        mean_duration_min=("trip_duration_min", "mean"),
        mean_speed_kmh=("mean_speed_kmh", "mean"),
    )

    merged = cohort_counts.merge(regime_exposure, on="weather_regime", how="left")
    merged["trip_count"] = merged["trip_count"].fillna(0).astype("int64")
    merged["observed_hours"] = merged["observed_hours"].fillna(0).astype("int64")
    merged["mean_duration_min"] = merged["mean_duration_min"].fillna(0.0).astype("float64")
    merged["mean_speed_kmh"] = merged["mean_speed_kmh"].fillna(0.0).astype("float64")

    merged["trips_per_observed_hour"] = pd.Series(
        np.where(
            merged["observed_hours"] > 0,
            merged["trip_count"] / merged["observed_hours"],
            0.0,
        ),
        index=merged.index,
        dtype="float64",
    )

    baseline_rows = merged.loc[
        merged["weather_regime"] == baseline_regime,
        ["member_casual", "rideable_type", "trips_per_observed_hour"],
    ].rename(columns={"trips_per_observed_hour": "baseline_trips_per_hour"})

    merged = merged.merge(baseline_rows, on=["member_casual", "rideable_type"], how="left")
    merged["baseline_trips_per_hour"] = merged["baseline_trips_per_hour"].fillna(0.0)

    merged["elasticity_index_pct"] = pd.Series(
        np.where(
            merged["baseline_trips_per_hour"] > 0.0,
            (merged["trips_per_observed_hour"] / merged["baseline_trips_per_hour"]) * 100.0,
            0.0,
        ),
        index=merged.index,
        dtype="float64",
    )
    merged["pct_change_vs_baseline"] = pd.Series(
        np.where(
            merged["baseline_trips_per_hour"] > 0.0,
            merged["elasticity_index_pct"] - 100.0,
            0.0,
        ),
        index=merged.index,
        dtype="float64",
    )

    merged["weather_regime"] = pd.Categorical(
        merged["weather_regime"],
        categories=list(WEATHER_REGIMES),
        ordered=True,
    )
    merged["member_casual"] = pd.Categorical(
        merged["member_casual"],
        categories=list(MEMBER_TYPES),
        ordered=False,
    )
    merged["rideable_type"] = pd.Categorical(
        merged["rideable_type"],
        categories=list(RIDEABLE_TYPES),
        ordered=False,
    )

    return (
        merged[out_cols]
        .sort_values(["weather_regime", "member_casual", "rideable_type"])
        .reset_index(drop=True)
    )


def compute_weather_time_density(
    trips_df: pd.DataFrame,
    temp_bin_width_c: float = 2.0,
    reference_trips_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out_cols = [
        "hour_of_day",
        "temp_bin_mid_c",
        "trip_count",
        "observed_hours",
        "trips_per_observed_hour",
        "mean_precipitation_mm",
        "mean_wind_speed_kmh",
        "ebike_share",
        "member_share",
    ]
    if trips_df.empty:
        return pd.DataFrame(columns=out_cols)
    if temp_bin_width_c <= 0.0:
        msg = "temp_bin_width_c must be strictly positive."
        raise ValueError(msg)

    ref_df = _align_reference_timeframe(trips_df, reference_trips_df)

    def _bin_mid(series: pd.Series) -> pd.Series:
        floored = np.floor(series.to_numpy(dtype="float64") / temp_bin_width_c) * temp_bin_width_c
        return pd.Series(floored + (temp_bin_width_c / 2.0), index=series.index, dtype="float64")

    ref_hours = (
        ref_df[["weather_hour", "hour_of_day", "temperature_c"]]
        .drop_duplicates(subset=["weather_hour"])
        .copy()
    )
    ref_hours["temp_bin_mid_c"] = _bin_mid(ref_hours["temperature_c"])
    exposure = ref_hours.groupby(
        ["hour_of_day", "temp_bin_mid_c"], as_index=False, observed=True
    ).agg(observed_hours=("weather_hour", "count"))

    work = trips_df.copy()
    work["temp_bin_mid_c"] = _bin_mid(work["temperature_c"])
    work["_is_ebike"] = (work["rideable_type"] == "electric_bike").astype("float64")
    work["_is_member"] = (work["member_casual"] == "member").astype("float64")

    trip_grid = work.groupby(["hour_of_day", "temp_bin_mid_c"], as_index=False, observed=True).agg(
        trip_count=("ride_id", "count"),
        mean_precipitation_mm=("precipitation_mm", "mean"),
        mean_wind_speed_kmh=("wind_speed_kmh", "mean"),
        ebike_share=("_is_ebike", "mean"),
        member_share=("_is_member", "mean"),
    )

    merged = trip_grid.merge(exposure, on=["hour_of_day", "temp_bin_mid_c"], how="left")
    merged["observed_hours"] = merged["observed_hours"].fillna(1).clip(lower=1).astype("int64")
    merged["trip_count"] = merged["trip_count"].astype("int64")
    merged["trips_per_observed_hour"] = (merged["trip_count"] / merged["observed_hours"]).astype(
        "float64"
    )

    return merged[out_cols].sort_values(["hour_of_day", "temp_bin_mid_c"]).reset_index(drop=True)


def compute_cohort_distributions(
    trips_df: pd.DataFrame,
    metric: DistributionMetric = "trip_duration_min",
    group_cols: Sequence[CohortGroupColumn] = ("member_casual", "rideable_type"),
) -> pd.DataFrame:
    group_list = list(group_cols)
    stat_cols = ["count", "median", "iqr", "p10", "p25", "p75", "p90", "mean", "std"]
    out_cols = [*group_list, *stat_cols]

    if trips_df.empty:
        return pd.DataFrame(columns=out_cols)
    if metric not in ("trip_duration_min", "mean_speed_kmh", "haversine_distance_km"):
        msg = f"Unsupported distribution metric: {metric!r}"
        raise ValueError(msg)

    grouped = trips_df.groupby(group_list, observed=True)[metric]
    summary = grouped.agg(
        count="count",
        median="median",
        p10=lambda s: float(np.percentile(s.to_numpy(dtype="float64"), 10)) if len(s) else 0.0,
        p25=lambda s: float(np.percentile(s.to_numpy(dtype="float64"), 25)) if len(s) else 0.0,
        p75=lambda s: float(np.percentile(s.to_numpy(dtype="float64"), 75)) if len(s) else 0.0,
        p90=lambda s: float(np.percentile(s.to_numpy(dtype="float64"), 90)) if len(s) else 0.0,
        mean="mean",
        std=lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
    ).reset_index()

    summary["count"] = summary["count"].astype("int64")
    summary["iqr"] = (summary["p75"] - summary["p25"]).astype("float64")
    for col in ["median", "iqr", "p10", "p25", "p75", "p90", "mean", "std"]:
        summary[col] = summary[col].fillna(0.0).astype("float64")

    return summary[out_cols].sort_values(group_list).reset_index(drop=True)


def sample_trips_for_distribution(
    trips_df: pd.DataFrame,
    max_samples: int = 15_000,
    stratify_cols: Sequence[CohortGroupColumn] = ("member_casual", "rideable_type"),
    clip_upper_quantile: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if trips_df.empty or max_samples <= 0:
        return trips_df.iloc[0:0].copy().reset_index(drop=True)

    work = trips_df.copy()
    if clip_upper_quantile is not None:
        if not (0.5 <= clip_upper_quantile < 1.0):
            msg = "clip_upper_quantile must be in [0.5, 1.0)."
            raise ValueError(msg)
        dur_cap = float(work["trip_duration_min"].quantile(clip_upper_quantile))
        spd_cap = float(work["mean_speed_kmh"].quantile(clip_upper_quantile))
        work = work.loc[
            (work["trip_duration_min"] <= dur_cap) & (work["mean_speed_kmh"] <= spd_cap)
        ].copy()

    if len(work) <= max_samples:
        return work.reset_index(drop=True)

    fraction = max_samples / float(len(work))
    strat_list = [col for col in stratify_cols if col in work.columns]
    if not strat_list:
        return work.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    sampled_parts: list[pd.DataFrame] = []
    for _, group_df in work.groupby(strat_list, observed=True):
        if group_df.empty:
            continue
        target_n = max(1, int(round(len(group_df) * fraction)))
        target_n = min(len(group_df), target_n)
        sampled_parts.append(group_df.sample(n=target_n, random_state=random_state))

    combined = pd.concat(sampled_parts, ignore_index=True)
    if len(combined) > max_samples:
        combined = combined.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    return combined.reset_index(drop=True)


def compute_equity_tier_summary(
    dynamic_stations_df: pd.DataFrame,
) -> pd.DataFrame:
    out_cols = [
        "equity_tier",
        "station_count",
        "total_trips",
        "mean_trips_per_day",
        "median_household_income",
        "mean_car_free_pct",
        "mean_dist_to_transit_m",
        "mean_ebike_share",
        "mean_casual_share",
        "net_flow_balance",
    ]
    if dynamic_stations_df.empty:
        return pd.DataFrame(columns=out_cols)

    summary = dynamic_stations_df.groupby("equity_tier", as_index=False, observed=False).agg(
        station_count=("station_id", "count"),
        total_trips=("total_trips", "sum"),
        mean_trips_per_day=("trips_per_day", "mean"),
        median_household_income=("median_household_income", "median"),
        mean_car_free_pct=("pct_car_free_households", "mean"),
        mean_dist_to_transit_m=("dist_to_transit_hub_m", "mean"),
        mean_ebike_share=("ebike_share", "mean"),
        mean_casual_share=("casual_share", "mean"),
        net_flow_balance=("net_flow_balance", "sum"),
    )

    summary["equity_tier"] = pd.Categorical(
        summary["equity_tier"],
        categories=list(EQUITY_TIERS),
        ordered=True,
    )
    summary["station_count"] = summary["station_count"].fillna(0).astype("int64")
    summary["total_trips"] = summary["total_trips"].fillna(0).astype("int64")
    summary["net_flow_balance"] = summary["net_flow_balance"].fillna(0).astype("int64")

    for float_col in [
        "mean_trips_per_day",
        "median_household_income",
        "mean_car_free_pct",
        "mean_dist_to_transit_m",
        "mean_ebike_share",
        "mean_casual_share",
    ]:
        summary[float_col] = summary[float_col].fillna(0.0).astype("float64")

    return summary[out_cols].sort_values("equity_tier").reset_index(drop=True)


def prepare_parallel_coords_table(
    dynamic_stations_df: pd.DataFrame,
    normalize_axes: bool = True,
) -> pd.DataFrame:
    if dynamic_stations_df.empty:
        return dynamic_stations_df.copy()

    result = dynamic_stations_df.copy()
    tier_code_map = {tier: idx for idx, tier in enumerate(EQUITY_TIERS)}
    result["equity_tier_code"] = (
        result["equity_tier"].astype("string").map(tier_code_map).fillna(0).astype("int64")
    )

    if normalize_axes:
        for col in _PARCOORDS_AXES:
            values = result[col].to_numpy(dtype="float64")
            col_min = float(np.min(values))
            col_max = float(np.max(values))
            span = col_max - col_min
            norm_vals = (values - col_min) / span if span > 0.0 else np.zeros_like(values)
            result[f"{col}_norm"] = pd.Series(norm_vals, index=result.index, dtype="float64")

    return result.sort_values(["equity_tier_code", "station_id"]).reset_index(drop=True)
