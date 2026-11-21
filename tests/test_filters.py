from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from micromobility_analytics.analytics.filters import (
    TripFilterCriteria,
    filter_trips,
    load_processed_datasets,
)
from micromobility_analytics.data.features import build_enriched_tables
from micromobility_analytics.data.loader import load_census_and_transit_reference


def _synthetic_enriched_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_trips = pd.DataFrame(
        {
            "ride_id": ["R1", "R2", "R3", "R4", "R5"],
            "rideable_type": [
                "classic_bike",
                "electric_bike",
                "electric_bike",
                "classic_bike",
                "electric_bike",
            ],
            "member_casual": ["member", "casual", "member", "casual", "member"],
            "started_at": pd.to_datetime(
                [
                    "2024-01-15 08:15:00",  # Mon AM Peak, Clear/Dry, JC005 -> JC001
                    "2024-01-15 08:30:00",  # Mon AM Peak, Clear/Dry, JC005 -> JC005 (round trip)
                    "2024-01-15 17:20:00",  # Mon PM Peak, Light Rain, JC001 -> JC099 (High Need)
                    "2024-07-20 14:10:00",  # Sat Midday, Clear/Dry, JC099 -> JC005
                    "2024-07-20 22:05:00",  # Sat Night, Heavy Rain/Snow, HB101 -> JC001
                ]
            ),
            "ended_at": pd.to_datetime(
                [
                    "2024-01-15 08:25:00",
                    "2024-01-15 08:45:00",
                    "2024-01-15 17:40:00",
                    "2024-07-20 14:30:00",
                    "2024-07-20 22:25:00",
                ]
            ),
            "start_station_id": ["JC005", "JC005", "JC001", "JC099", "HB101"],
            "start_station_name": [
                "Grove St PATH",
                "Grove St PATH",
                "Exchange Place",
                "Bergen Ave",
                "Hoboken Terminal",
            ],
            "end_station_id": ["JC001", "JC005", "JC099", "JC005", "JC001"],
            "end_station_name": [
                "Exchange Place",
                "Grove St PATH",
                "Bergen Ave",
                "Grove St PATH",
                "Exchange Place",
            ],
            "start_lat": [40.7197, 40.7197, 40.7168, 40.7130, 40.7349],
            "start_lng": [-74.0431, -74.0431, -74.0323, -74.0620, -74.0291],
            "end_lat": [40.7168, 40.7197, 40.7130, 40.7197, 40.7168],
            "end_lng": [-74.0323, -74.0431, -74.0620, -74.0431, -74.0323],
        }
    )
    weather = pd.DataFrame(
        {
            "weather_hour": pd.to_datetime(
                [
                    "2024-01-15 08:00:00",
                    "2024-01-15 17:00:00",
                    "2024-07-20 14:00:00",
                    "2024-07-20 22:00:00",
                ]
            ),
            "temperature_c": [4.0, 6.0, 27.0, 22.0],
            "apparent_temperature_c": [2.0, 4.0, 28.0, 22.0],
            "precipitation_mm": [0.0, 1.2, 0.0, 5.0],
            "wind_speed_kmh": [10.0, 18.0, 9.0, 28.0],
            "relative_humidity_pct": [55.0, 82.0, 60.0, 92.0],
            "weather_code": [0, 61, 1, 65],
        }
    )
    zones_df, hubs_df = load_census_and_transit_reference()
    return build_enriched_tables(clean_trips, weather, zones_df, hubs_df)


def test_filter_trips_default_or_none_returns_all_rows() -> None:
    trips_df, _ = _synthetic_enriched_dataset()

    assert len(filter_trips(trips_df, None)) == 5
    assert len(filter_trips(trips_df, TripFilterCriteria())) == 5


def test_filter_trips_temporal_cohort_and_weather_slicing() -> None:
    trips_df, _ = _synthetic_enriched_dataset()

    # Filter by month and rider type
    jan_members = filter_trips(
        trips_df,
        TripFilterCriteria(months=("202401",), member_casual=("member",)),
    )
    assert list(jan_members["ride_id"]) == ["R1", "R3"]

    # Filter by date range, bike type, and weather regime
    rain_ebikes = filter_trips(
        trips_df,
        TripFilterCriteria(
            start_date="2024-01-15",
            end_date="2024-07-20",
            rideable_type=("electric_bike",),
            weather_regime=("Light Rain", "Heavy Rain/Snow"),
        ),
    )
    assert list(rain_ebikes["ride_id"]) == ["R3", "R5"]

    # Filter excluding round trips
    no_round = filter_trips(trips_df, TripFilterCriteria(include_round_trips=False))
    assert list(no_round["ride_id"]) == ["R1", "R3", "R4", "R5"]


def test_filter_trips_spatial_endpoint_modes_and_transit_proximity() -> None:
    trips_df, _ = _synthetic_enriched_dataset()

    origin_jc099 = filter_trips(
        trips_df,
        TripFilterCriteria(
            selected_station_ids=frozenset({"JC099"}),
            station_endpoint_mode="origin",
        ),
    )
    assert list(origin_jc099["ride_id"]) == ["R4"]

    dest_jc099 = filter_trips(
        trips_df,
        TripFilterCriteria(
            selected_station_ids=frozenset({"JC099"}),
            station_endpoint_mode="destination",
        ),
    )
    assert list(dest_jc099["ride_id"]) == ["R3"]

    either_jc099 = filter_trips(
        trips_df,
        TripFilterCriteria(
            selected_station_ids=frozenset({"JC099"}),
            station_endpoint_mode="either",
        ),
    )
    assert list(either_jc099["ride_id"]) == ["R3", "R4"]

    # Origin strictly within 100m of a transit hub
    near_transit_origin = filter_trips(
        trips_df,
        TripFilterCriteria(max_dist_to_transit_m=100.0, station_endpoint_mode="origin"),
    )
    assert list(near_transit_origin["ride_id"]) == ["R1", "R2", "R3", "R5"]


def test_filter_trips_empty_selection_or_invalid_mode() -> None:
    trips_df, _ = _synthetic_enriched_dataset()

    empty_selection = filter_trips(
        trips_df,
        TripFilterCriteria(selected_station_ids=frozenset()),
    )
    assert empty_selection.empty

    empty_months = filter_trips(trips_df, TripFilterCriteria(months=()))
    assert empty_months.empty


def test_load_processed_datasets_roundtrip_and_missing_error(tmp_path: Path) -> None:
    trips_df, stations_df = _synthetic_enriched_dataset()
    trips_path = tmp_path / "trips.parquet"
    stations_path = tmp_path / "stations.parquet"

    with pytest.raises(FileNotFoundError, match="Processed trips Parquet not found"):
        load_processed_datasets(trips_path=trips_path, stations_path=stations_path)

    trips_df.to_parquet(trips_path, index=False)
    with pytest.raises(FileNotFoundError, match="Processed stations Parquet not found"):
        load_processed_datasets(trips_path=trips_path, stations_path=stations_path)

    stations_df.to_parquet(stations_path, index=False)
    loaded_trips, loaded_stations = load_processed_datasets(
        trips_path=trips_path,
        stations_path=stations_path,
    )
    assert len(loaded_trips) == 5
    assert len(loaded_stations) == 4
