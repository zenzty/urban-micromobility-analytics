from __future__ import annotations

import pandas as pd
import pytest

from micromobility_analytics.data.features import (
    assign_station_spatial_equity,
    build_enriched_tables,
    classify_weather_regime,
    compute_haversine_km,
    engineer_trip_features,
)
from micromobility_analytics.data.loader import load_census_and_transit_reference


def test_compute_haversine_km_known_points() -> None:
    lat1 = pd.Series([40.7197, 40.7197])
    lon1 = pd.Series([-74.0431, -74.0431])
    lat2 = pd.Series([40.7197, 40.7168])
    lon2 = pd.Series([-74.0431, -74.0323])

    distances = compute_haversine_km(lat1, lon1, lat2, lon2)

    assert distances.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert 0.90 < distances.iloc[1] < 1.05


def test_engineer_trip_features_commute_windows_and_round_trip() -> None:
    trips = pd.DataFrame(
        {
            "ride_id": ["AM", "MID", "PM", "NIGHT", "WKND_AM", "WKND_NIGHT"],
            "rideable_type": ["classic_bike"] * 6,
            "member_casual": ["member"] * 6,
            "started_at": pd.to_datetime(
                [
                    "2024-04-15 08:15:00",  # Monday AM Peak
                    "2024-04-15 12:30:00",  # Monday Midday
                    "2024-04-15 17:45:00",  # Monday PM Peak
                    "2024-04-15 22:10:00",  # Monday Night
                    "2024-04-20 08:30:00",  # Saturday -> Midday
                    "2024-04-20 23:05:00",  # Saturday Night
                ]
            ),
            "ended_at": pd.to_datetime(
                [
                    "2024-04-15 08:25:00",
                    "2024-04-15 12:40:00",
                    "2024-04-15 17:55:00",
                    "2024-04-15 22:20:00",
                    "2024-04-20 08:45:00",
                    "2024-04-20 23:20:00",
                ]
            ),
            "start_station_id": ["JC005", "JC005", "JC001", "JC001", "JC005", "JC001"],
            "start_station_name": ["Grove"] * 6,
            "end_station_id": ["JC005", "JC001", "JC005", "JC001", "JC001", "JC005"],
            "end_station_name": ["Dest"] * 6,
            "start_lat": [40.7197] * 6,
            "start_lng": [-74.0431] * 6,
            "end_lat": [40.7197, 40.7168, 40.7197, 40.7168, 40.7168, 40.7197],
            "end_lng": [-74.0431, -74.0323, -74.0431, -74.0323, -74.0323, -74.0431],
        }
    )

    enriched = engineer_trip_features(trips)

    assert list(enriched["commute_window"]) == [
        "AM Peak",
        "Midday",
        "PM Peak",
        "Night",
        "Midday",
        "Night",
    ]
    assert list(enriched["is_round_trip"]) == [True, False, False, True, False, False]
    assert list(enriched["is_weekend"]) == [False, False, False, False, True, True]
    assert enriched["hour_of_day"].dtype.name == "int8"
    assert enriched["day_of_week"].cat.ordered is True
    assert enriched["commute_window"].cat.ordered is True


def test_classify_weather_regime_all_categories() -> None:
    weather = pd.DataFrame(
        {
            "weather_hour": pd.date_range("2024-04-15 08:00:00", periods=5, freq="h"),
            "temperature_c": [18.0, 16.0, 14.0, 12.0, -2.5],
            "apparent_temperature_c": [18.0, 15.0, 13.0, 10.0, -7.0],
            "precipitation_mm": [0.0, 0.0, 1.2, 4.0, 0.0],
            "wind_speed_kmh": [10.0, 12.0, 15.0, 25.0, 18.0],
            "weather_code": [0, 3, 61, 65, 0],
        }
    )

    classified = classify_weather_regime(weather)

    assert list(classified["weather_regime"]) == [
        "Clear/Dry",
        "Cloudy",
        "Light Rain",
        "Heavy Rain/Snow",
        "Extreme Temp",
    ]
    assert classified["weather_regime"].cat.ordered is True


def test_assign_station_spatial_equity_rejects_cross_river_nyc_stations() -> None:
    zones_df, hubs_df = load_census_and_transit_reference()
    nyc_station_df = pd.DataFrame(
        {
            "station_id": ["5470.10"],
            "station_name": ["W 15 St & 7 Ave (Manhattan)"],
            "lat": [40.7393],
            "lng": [-73.9993],
        }
    )
    with pytest.raises(ValueError, match="out-of-network stations"):
        assign_station_spatial_equity(nyc_station_df, zones_df, hubs_df)


def test_assign_station_spatial_equity_empty_input() -> None:
    zones_df, hubs_df = load_census_and_transit_reference()
    empty_df = pd.DataFrame(columns=["station_id", "station_name", "lat", "lng"])
    result = assign_station_spatial_equity(empty_df, zones_df, hubs_df)
    assert result.empty
    assert "equity_tier" in result.columns


def test_trips_per_day_uses_non_contiguous_active_dates_denominator() -> None:
    clean_trips = pd.DataFrame(
        {
            "ride_id": ["T1", "T2", "T3", "T4"],
            "rideable_type": ["classic_bike"] * 4,
            "member_casual": ["member"] * 4,
            "started_at": pd.to_datetime(
                [
                    "2024-01-10 08:00:00",
                    "2024-01-10 09:00:00",
                    "2024-07-10 08:00:00",
                    "2024-07-10 18:00:00",
                ]
            ),
            "ended_at": pd.to_datetime(
                [
                    "2024-01-10 08:15:00",
                    "2024-01-10 09:15:00",
                    "2024-07-10 08:15:00",
                    "2024-07-10 18:15:00",
                ]
            ),
            "start_station_id": ["JC005", "JC005", "JC005", "JC001"],
            "start_station_name": [
                "Grove St PATH",
                "Grove St PATH",
                "Grove St PATH",
                "Exchange Place",
            ],
            "end_station_id": ["JC001", "JC001", "JC001", "JC005"],
            "end_station_name": [
                "Exchange Place",
                "Exchange Place",
                "Exchange Place",
                "Grove St PATH",
            ],
            "start_lat": [40.7197, 40.7197, 40.7197, 40.7168],
            "start_lng": [-74.0431, -74.0431, -74.0431, -74.0323],
            "end_lat": [40.7168, 40.7168, 40.7168, 40.7197],
            "end_lng": [-74.0323, -74.0323, -74.0323, -74.0431],
        }
    )
    weather = pd.DataFrame(
        {
            "weather_hour": pd.to_datetime(
                [
                    "2024-01-10 08:00:00",
                    "2024-01-10 09:00:00",
                    "2024-07-10 08:00:00",
                    "2024-07-10 18:00:00",
                ]
            ),
            "temperature_c": [4.0, 5.0, 26.0, 28.0],
            "apparent_temperature_c": [2.0, 3.0, 27.0, 29.0],
            "precipitation_mm": [0.0, 0.0, 0.0, 0.0],
            "wind_speed_kmh": [12.0, 10.0, 8.0, 9.0],
            "relative_humidity_pct": [60.0, 58.0, 65.0, 62.0],
            "weather_code": [0, 0, 1, 1],
        }
    )
    zones_df, hubs_df = load_census_and_transit_reference()
    _, stations_equity = build_enriched_tables(clean_trips, weather, zones_df, hubs_df)

    grove = stations_equity.loc[stations_equity["station_id"] == "JC005"].iloc[0]
    exchange = stations_equity.loc[stations_equity["station_id"] == "JC001"].iloc[0]

    assert grove["total_departures"] == 3
    assert grove["trips_per_day"] == pytest.approx(3.0 / 2.0)
    assert exchange["total_departures"] == 1
    assert exchange["trips_per_day"] == pytest.approx(1.0 / 2.0)


def test_build_enriched_tables_joins_weather_and_od_equity_with_nearest_hour_fallback() -> None:
    clean_trips = pd.DataFrame(
        {
            "ride_id": ["T1", "T2", "T3"],
            "rideable_type": ["electric_bike", "classic_bike", "electric_bike"],
            "member_casual": ["casual", "member", "member"],
            "started_at": pd.to_datetime(
                [
                    "2024-04-15 08:10:00",
                    "2024-04-15 08:40:00",
                    "2024-04-15 10:15:00",  # 10:00 missing in weather -> aligns to 09:00
                ]
            ),
            "ended_at": pd.to_datetime(
                [
                    "2024-04-15 08:25:00",
                    "2024-04-15 08:50:00",
                    "2024-04-15 10:30:00",
                ]
            ),
            "start_station_id": ["JC005", "JC005", "JC099"],
            "start_station_name": ["Grove St PATH", "Grove St PATH", "Bergen Ave"],
            "end_station_id": ["JC099", "JC099", "JC005"],
            "end_station_name": ["Bergen Ave", "Bergen Ave", "Grove St PATH"],
            "start_lat": [40.7197, 40.7197, 40.7130],
            "start_lng": [-74.0431, -74.0431, -74.0620],
            "end_lat": [40.7130, 40.7130, 40.7197],
            "end_lng": [-74.0620, -74.0620, -74.0431],
        }
    )
    weather = pd.DataFrame(
        {
            "weather_hour": pd.to_datetime(["2024-04-15 08:00:00", "2024-04-15 09:00:00"]),
            "temperature_c": [15.0, 17.0],
            "apparent_temperature_c": [14.5, 17.0],
            "precipitation_mm": [0.0, 0.5],
            "wind_speed_kmh": [10.0, 12.0],
            "relative_humidity_pct": [55.0, 65.0],
            "weather_code": [1, 51],
        }
    )
    zones_df, hubs_df = load_census_and_transit_reference()

    trips_enriched, stations_equity = build_enriched_tables(
        clean_trips_df=clean_trips,
        weather_df=weather,
        zones_df=zones_df,
        hubs_df=hubs_df,
    )

    assert len(trips_enriched) == 3
    assert set(
        [
            "start_neighborhood",
            "start_equity_tier",
            "start_dist_to_transit_m",
            "end_neighborhood",
            "end_equity_tier",
            "end_dist_to_transit_m",
        ]
    ).issubset(trips_enriched.columns)
    assert trips_enriched.loc[0, "start_equity_tier"] == "Affluent"
    assert trips_enriched.loc[0, "end_equity_tier"] == "High Need"
    assert trips_enriched.loc[0, "weather_regime"] == "Clear/Dry"
    assert trips_enriched.loc[2, "weather_regime"] == "Light Rain"

    grove = stations_equity.loc[stations_equity["station_id"] == "JC005"].iloc[0]
    bergen = stations_equity.loc[stations_equity["station_id"] == "JC099"].iloc[0]

    assert grove["total_departures"] == 2
    assert grove["total_arrivals"] == 1
    assert grove["net_flow_balance"] == -1
    assert grove["ebike_share"] == pytest.approx(0.5)
    assert grove["casual_share"] == pytest.approx(0.5)

    assert bergen["total_departures"] == 1
    assert bergen["total_arrivals"] == 2
    assert bergen["net_flow_balance"] == 1
    assert bergen["equity_tier"] == "High Need"
