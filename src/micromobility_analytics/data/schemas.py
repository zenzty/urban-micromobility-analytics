from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

from micromobility_analytics.config import settings
from micromobility_analytics.data.features import (
    COMMUTE_WINDOWS,
    DAYS_OF_WEEK,
    EQUITY_TIERS,
    MEMBER_TYPES,
    RIDEABLE_TYPES,
    WEATHER_REGIMES,
)

_MIN_LAT, _MAX_LAT, _MIN_LNG, _MAX_LNG = settings.metro_bbox

_FINITE_CHECK = pa.Check(
    lambda s: bool(np.isfinite(s.to_numpy(dtype="float64")).all()),
    error="Float column must contain only finite values (no NaN or +/-inf).",
)

_STATION_PREFIX_CHECK = pa.Check(
    lambda s: s.astype("string").str.startswith(settings.valid_station_prefixes, na=False),
    error=f"Station IDs must start with one of {settings.valid_station_prefixes}.",
)


def _ordered_categorical_check(expected: tuple[str, ...]) -> pa.Check:
    # Verify that a Series is an ordered Categorical with exact categories
    return pa.Check(
        lambda s: (
            isinstance(s.dtype, pd.CategoricalDtype)
            and bool(s.cat.ordered)
            and tuple(s.cat.categories) == expected
        ),
        error=f"Expected ordered categorical with categories {expected}.",
    )


TRIPS_ENRICHED_SCHEMA = pa.DataFrameSchema(
    columns={
        "ride_id": pa.Column("string", nullable=False, unique=True),
        "rideable_type": pa.Column(
            "category",
            checks=[pa.Check.isin(list(RIDEABLE_TYPES))],
            nullable=False,
        ),
        "member_casual": pa.Column(
            "category",
            checks=[pa.Check.isin(list(MEMBER_TYPES))],
            nullable=False,
        ),
        "started_at": pa.Column("datetime64[ns]", nullable=False),
        "ended_at": pa.Column("datetime64[ns]", nullable=False),
        "weather_hour": pa.Column("datetime64[ns]", nullable=False),
        "start_station_id": pa.Column(
            "string",
            checks=[_STATION_PREFIX_CHECK],
            nullable=False,
        ),
        "start_station_name": pa.Column("string", nullable=False),
        "end_station_id": pa.Column(
            "string",
            checks=[_STATION_PREFIX_CHECK],
            nullable=False,
        ),
        "end_station_name": pa.Column("string", nullable=False),
        "start_lat": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(_MIN_LAT, _MAX_LAT)],
            nullable=False,
        ),
        "start_lng": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(_MIN_LNG, _MAX_LNG)],
            nullable=False,
        ),
        "end_lat": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(_MIN_LAT, _MAX_LAT)],
            nullable=False,
        ),
        "end_lng": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(_MIN_LNG, _MAX_LNG)],
            nullable=False,
        ),
        "trip_duration_min": pa.Column(
            "float64",
            checks=[
                _FINITE_CHECK,
                pa.Check.in_range(
                    settings.min_trip_duration_min,
                    settings.max_trip_duration_min,
                ),
            ],
            nullable=False,
        ),
        "haversine_distance_km": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.ge(0.0)],
            nullable=False,
        ),
        "mean_speed_kmh": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(0.0, settings.max_mean_speed_kmh)],
            nullable=False,
        ),
        "is_round_trip": pa.Column("bool", nullable=False),
        "hour_of_day": pa.Column(
            "int8",
            checks=[pa.Check.in_range(0, 23)],
            nullable=False,
        ),
        "day_of_week": pa.Column(
            "category",
            checks=[pa.Check.isin(list(DAYS_OF_WEEK)), _ordered_categorical_check(DAYS_OF_WEEK)],
            nullable=False,
        ),
        "is_weekend": pa.Column("bool", nullable=False),
        "commute_window": pa.Column(
            "category",
            checks=[
                pa.Check.isin(list(COMMUTE_WINDOWS)),
                _ordered_categorical_check(COMMUTE_WINDOWS),
            ],
            nullable=False,
        ),
        "temperature_c": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(-30.0, 50.0)],
            nullable=False,
        ),
        "apparent_temperature_c": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(-40.0, 55.0)],
            nullable=False,
        ),
        "precipitation_mm": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.ge(0.0)],
            nullable=False,
        ),
        "wind_speed_kmh": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.ge(0.0)],
            nullable=False,
        ),
        "weather_regime": pa.Column(
            "category",
            checks=[
                pa.Check.isin(list(WEATHER_REGIMES)),
                _ordered_categorical_check(WEATHER_REGIMES),
            ],
            nullable=False,
        ),
        "start_neighborhood": pa.Column("string", nullable=False),
        "start_equity_tier": pa.Column(
            "category",
            checks=[pa.Check.isin(list(EQUITY_TIERS)), _ordered_categorical_check(EQUITY_TIERS)],
            nullable=False,
        ),
        "start_dist_to_transit_m": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.ge(0.0)],
            nullable=False,
        ),
        "end_neighborhood": pa.Column("string", nullable=False),
        "end_equity_tier": pa.Column(
            "category",
            checks=[pa.Check.isin(list(EQUITY_TIERS)), _ordered_categorical_check(EQUITY_TIERS)],
            nullable=False,
        ),
        "end_dist_to_transit_m": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.ge(0.0)],
            nullable=False,
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["ended_at"] > df["started_at"],
            error="ended_at must be strictly greater than started_at.",
        ),
        pa.Check(
            lambda df: df["weather_hour"] == df["started_at"].dt.floor("h"),
            error="weather_hour must equal started_at floored to the hour.",
        ),
        pa.Check(
            lambda df: df["is_round_trip"] == (df["start_station_id"] == df["end_station_id"]),
            error="is_round_trip must match start_station_id == end_station_id.",
        ),
    ],
    strict=True,
    coerce=False,
)

STATIONS_EQUITY_SCHEMA = pa.DataFrameSchema(
    columns={
        "station_id": pa.Column(
            "string",
            checks=[_STATION_PREFIX_CHECK],
            nullable=False,
            unique=True,
        ),
        "station_name": pa.Column("string", nullable=False),
        "lat": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(_MIN_LAT, _MAX_LAT)],
            nullable=False,
        ),
        "lng": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(_MIN_LNG, _MAX_LNG)],
            nullable=False,
        ),
        "neighborhood_name": pa.Column("string", nullable=False),
        "census_tract_geoid": pa.Column(
            "string",
            checks=[pa.Check.str_matches(r"^34017\d{6}$")],
            nullable=False,
        ),
        "equity_tier": pa.Column(
            "category",
            checks=[pa.Check.isin(list(EQUITY_TIERS)), _ordered_categorical_check(EQUITY_TIERS)],
            nullable=False,
        ),
        "median_household_income": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.gt(0.0)],
            nullable=False,
        ),
        "pct_car_free_households": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(0.0, 100.0)],
            nullable=False,
        ),
        "population_density": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.gt(0.0)],
            nullable=False,
        ),
        "nearest_transit_hub": pa.Column("string", nullable=False),
        "dist_to_transit_hub_m": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(0.0, 5000.0)],
            nullable=False,
        ),
        "total_departures": pa.Column(
            "int64",
            checks=[pa.Check.ge(0)],
            nullable=False,
        ),
        "total_arrivals": pa.Column(
            "int64",
            checks=[pa.Check.ge(0)],
            nullable=False,
        ),
        "total_trips": pa.Column(
            "int64",
            checks=[pa.Check.gt(0)],
            nullable=False,
        ),
        "net_flow_balance": pa.Column("int64", nullable=False),
        "trips_per_day": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.ge(0.0)],
            nullable=False,
        ),
        "ebike_share": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(0.0, 1.0)],
            nullable=False,
        ),
        "casual_share": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.in_range(0.0, 1.0)],
            nullable=False,
        ),
        "mean_duration_min": pa.Column(
            "float64",
            checks=[_FINITE_CHECK, pa.Check.gt(0.0)],
            nullable=False,
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["total_trips"] == (df["total_departures"] + df["total_arrivals"]),
            error="total_trips must equal total_departures + total_arrivals.",
        ),
        pa.Check(
            lambda df: df["net_flow_balance"] == (df["total_arrivals"] - df["total_departures"]),
            error="net_flow_balance must equal total_arrivals - total_departures.",
        ),
    ],
    strict=True,
    coerce=False,
)


def validate_trips_enriched(df: pd.DataFrame) -> pd.DataFrame:
    # Validate the enriched trip fact table against TRIPS_ENRICHED_SCHEMA
    return TRIPS_ENRICHED_SCHEMA.validate(df)


def validate_stations_equity(df: pd.DataFrame) -> pd.DataFrame:
    # Validate the station equity summary table against STATIONS_EQUITY_SCHEMA
    return STATIONS_EQUITY_SCHEMA.validate(df)
