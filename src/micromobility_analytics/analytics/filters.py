from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from micromobility_analytics.config import STATIONS_EQUITY_PATH, TRIPS_ENRICHED_PATH

logger = logging.getLogger(__name__)

StationEndpointMode = Literal["either", "origin", "destination"]


@dataclass(frozen=True)
class TripFilterCriteria:
    months: tuple[str, ...] | None = None
    start_date: str | pd.Timestamp | None = None
    end_date: str | pd.Timestamp | None = None
    member_casual: tuple[str, ...] | None = None
    rideable_type: tuple[str, ...] | None = None
    commute_window: tuple[str, ...] | None = None
    weather_regime: tuple[str, ...] | None = None
    equity_tiers: tuple[str, ...] | None = None
    selected_station_ids: frozenset[str] | None = None
    station_endpoint_mode: StationEndpointMode = "either"
    include_round_trips: bool = True
    max_dist_to_transit_m: float | None = None


def load_processed_datasets(
    trips_path: Path = TRIPS_ENRICHED_PATH,
    stations_path: Path = STATIONS_EQUITY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not trips_path.exists():
        msg = f"Processed trips Parquet not found at {trips_path}. Run scripts/build_dataset.py."
        raise FileNotFoundError(msg)
    if not stations_path.exists():
        msg = (
            f"Processed stations Parquet not found at {stations_path}. "
            "Run scripts/build_dataset.py."
        )
        raise FileNotFoundError(msg)

    trips_df = pd.read_parquet(trips_path)
    stations_df = pd.read_parquet(stations_path)
    logger.info(
        "Loaded processed datasets: %d trips, %d stations.",
        len(trips_df),
        len(stations_df),
    )
    return trips_df, stations_df


def _endpoint_mask(
    origin_cond: pd.Series,
    dest_cond: pd.Series,
    mode: StationEndpointMode,
) -> pd.Series:
    if mode == "origin":
        return origin_cond
    if mode == "destination":
        return dest_cond
    if mode == "either":
        return origin_cond | dest_cond
    msg = f"Unsupported station_endpoint_mode: {mode!r}"
    raise ValueError(msg)


def filter_trips(
    trips_df: pd.DataFrame,
    criteria: TripFilterCriteria | None = None,
) -> pd.DataFrame:
    if criteria is None or trips_df.empty:
        return trips_df.copy().reset_index(drop=True)

    mask = pd.Series(True, index=trips_df.index, dtype=bool)

    if criteria.months is not None:
        if len(criteria.months) == 0:
            return trips_df.iloc[0:0].copy().reset_index(drop=True)
        mask &= trips_df["started_at"].dt.strftime("%Y%m").isin(list(criteria.months))

    if criteria.start_date is not None:
        start_ts = pd.Timestamp(criteria.start_date)
        mask &= trips_df["started_at"] >= start_ts

    if criteria.end_date is not None:
        end_ts = pd.Timestamp(criteria.end_date)
        if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= trips_df["started_at"] <= end_ts

    if criteria.member_casual is not None:
        if len(criteria.member_casual) == 0:
            return trips_df.iloc[0:0].copy().reset_index(drop=True)
        mask &= trips_df["member_casual"].isin(list(criteria.member_casual))

    if criteria.rideable_type is not None:
        if len(criteria.rideable_type) == 0:
            return trips_df.iloc[0:0].copy().reset_index(drop=True)
        mask &= trips_df["rideable_type"].isin(list(criteria.rideable_type))

    if criteria.commute_window is not None:
        if len(criteria.commute_window) == 0:
            return trips_df.iloc[0:0].copy().reset_index(drop=True)
        mask &= trips_df["commute_window"].isin(list(criteria.commute_window))

    if criteria.weather_regime is not None:
        if len(criteria.weather_regime) == 0:
            return trips_df.iloc[0:0].copy().reset_index(drop=True)
        mask &= trips_df["weather_regime"].isin(list(criteria.weather_regime))

    if criteria.equity_tiers is not None:
        if len(criteria.equity_tiers) == 0:
            return trips_df.iloc[0:0].copy().reset_index(drop=True)
        tier_list = list(criteria.equity_tiers)
        mask &= _endpoint_mask(
            trips_df["start_equity_tier"].isin(tier_list),
            trips_df["end_equity_tier"].isin(tier_list),
            criteria.station_endpoint_mode,
        )

    if criteria.selected_station_ids is not None:
        if len(criteria.selected_station_ids) == 0:
            return trips_df.iloc[0:0].copy().reset_index(drop=True)
        station_list = list(criteria.selected_station_ids)
        mask &= _endpoint_mask(
            trips_df["start_station_id"].isin(station_list),
            trips_df["end_station_id"].isin(station_list),
            criteria.station_endpoint_mode,
        )

    if not criteria.include_round_trips:
        mask &= ~trips_df["is_round_trip"]

    if criteria.max_dist_to_transit_m is not None:
        threshold = float(criteria.max_dist_to_transit_m)
        mask &= _endpoint_mask(
            trips_df["start_dist_to_transit_m"] <= threshold,
            trips_df["end_dist_to_transit_m"] <= threshold,
            criteria.station_endpoint_mode,
        )

    return trips_df.loc[mask].copy().reset_index(drop=True)
