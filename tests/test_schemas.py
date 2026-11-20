from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from micromobility_analytics.data.features import DAYS_OF_WEEK, build_enriched_tables
from micromobility_analytics.data.loader import load_census_and_transit_reference
from micromobility_analytics.data.schemas import (
    validate_stations_equity,
    validate_trips_enriched,
)


def _sample_enriched_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_trips = pd.DataFrame(
        {
            "ride_id": ["S1", "S2"],
            "rideable_type": ["classic_bike", "electric_bike"],
            "member_casual": ["member", "casual"],
            "started_at": pd.to_datetime(["2024-07-10 08:15:00", "2024-07-10 17:30:00"]),
            "ended_at": pd.to_datetime(["2024-07-10 08:25:00", "2024-07-10 17:45:00"]),
            "start_station_id": ["JC005", "JC001"],
            "start_station_name": ["Grove St PATH", "Exchange Place"],
            "end_station_id": ["JC001", "JC005"],
            "end_station_name": ["Exchange Place", "Grove St PATH"],
            "start_lat": [40.7197, 40.7168],
            "start_lng": [-74.0431, -74.0323],
            "end_lat": [40.7168, 40.7197],
            "end_lng": [-74.0323, -74.0431],
        }
    )
    weather = pd.DataFrame(
        {
            "weather_hour": pd.to_datetime(["2024-07-10 08:00:00", "2024-07-10 17:00:00"]),
            "temperature_c": [25.0, 29.0],
            "apparent_temperature_c": [26.0, 31.0],
            "precipitation_mm": [0.0, 0.0],
            "wind_speed_kmh": [9.0, 14.0],
            "relative_humidity_pct": [60.0, 58.0],
            "weather_code": [1, 2],
        }
    )
    zones_df, hubs_df = load_census_and_transit_reference()
    return build_enriched_tables(clean_trips, weather, zones_df, hubs_df)


def test_valid_enriched_tables_pass_pandera_schemas() -> None:
    trips_df, stations_df = _sample_enriched_pair()

    validated_trips = validate_trips_enriched(trips_df)
    validated_stations = validate_stations_equity(stations_df)

    assert len(validated_trips) == 2
    assert len(validated_stations) == 2


def test_invalid_trips_fail_pandera_validation() -> None:
    trips_df, _ = _sample_enriched_pair()

    # 1. Speed exceeding max_mean_speed_kmh
    bad_speed_df = trips_df.copy()
    bad_speed_df.loc[0, "mean_speed_kmh"] = 60.0
    with pytest.raises(pandera.errors.SchemaError):
        validate_trips_enriched(bad_speed_df)

    # 2. Duration exceeding max_trip_duration_min
    bad_duration_df = trips_df.copy()
    bad_duration_df.loc[0, "trip_duration_min"] = 300.0
    with pytest.raises(pandera.errors.SchemaError):
        validate_trips_enriched(bad_duration_df)

    # 3. Relational invariant: ended_at <= started_at
    bad_timestamps_df = trips_df.copy()
    bad_timestamps_df["ended_at"] = bad_timestamps_df["started_at"]
    with pytest.raises(pandera.errors.SchemaError):
        validate_trips_enriched(bad_timestamps_df)

    # 4. Non-finite float (+inf on ge(0.0) column)
    bad_inf_df = trips_df.copy()
    bad_inf_df.loc[0, "precipitation_mm"] = np.inf
    with pytest.raises(pandera.errors.SchemaError):
        validate_trips_enriched(bad_inf_df)

    # 5. Out-of-network station prefix (Manhattan code)
    bad_prefix_df = trips_df.copy()
    bad_prefix_df.loc[0, "end_station_id"] = "5470.10"
    with pytest.raises(pandera.errors.SchemaError):
        validate_trips_enriched(bad_prefix_df)

    # 6. Unordered categorical on day_of_week
    bad_cat_df = trips_df.copy()
    bad_cat_df["day_of_week"] = pd.Categorical(
        bad_cat_df["day_of_week"],
        categories=list(DAYS_OF_WEEK),
        ordered=False,
    )
    with pytest.raises(pandera.errors.SchemaError):
        validate_trips_enriched(bad_cat_df)


def test_invalid_stations_fail_pandera_validation() -> None:
    _, stations_df = _sample_enriched_pair()

    # 1. Invalid mode share > 1.0
    bad_share_df = stations_df.copy()
    bad_share_df.loc[0, "ebike_share"] = 1.25
    with pytest.raises(pandera.errors.SchemaError):
        validate_stations_equity(bad_share_df)

    # 2. Out-of-network Census tract GEOID (New York County 36061 instead of Hudson 34017)
    bad_geoid_df = stations_df.copy()
    bad_geoid_df.loc[0, "census_tract_geoid"] = "36061000100"
    with pytest.raises(pandera.errors.SchemaError):
        validate_stations_equity(bad_geoid_df)

    # 3. Broken relational invariant: net_flow_balance != total_arrivals - total_departures
    bad_flow_df = stations_df.copy()
    bad_flow_df.loc[0, "net_flow_balance"] = 999
    with pytest.raises(pandera.errors.SchemaError):
        validate_stations_equity(bad_flow_df)
