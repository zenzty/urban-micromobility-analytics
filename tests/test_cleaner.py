from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from micromobility_analytics.data.cleaner import clean_trips
from micromobility_analytics.data.loader import fetch_hourly_weather, read_citibike_zip


def _base_trip_row(**overrides: object) -> dict[str, object]:
    # valid raw Citi Bike trip record
    row: dict[str, object] = {
        "ride_id": "R0001",
        "rideable_type": "classic_bike",
        "started_at": "2024-04-15 08:10:00",
        "ended_at": "2024-04-15 08:20:00",
        "start_station_name": "Grove St PATH",
        "start_station_id": "JC005",
        "end_station_name": "Exchange Place",
        "end_station_id": "JC001",
        "start_lat": 40.7197,
        "start_lng": -74.0431,
        "end_lat": 40.7168,
        "end_lng": -74.0323,
        "member_casual": "member",
    }
    row.update(overrides)
    return row


def test_clean_trips_applies_dq_rules_and_reports_audit() -> None:
    rows = [
        # Valid row
        _base_trip_row(ride_id="VALID_1"),
        # Duplicate ride_id (dropped in deduplication)
        _base_trip_row(ride_id="VALID_1"),
        # DQ-01: negative duration (ended_at <= started_at)
        _base_trip_row(
            ride_id="BAD_DUR_NEG",
            started_at="2024-04-15 08:20:00",
            ended_at="2024-04-15 08:10:00",
        ),
        # DQ-01: too short (< 1.0 min)
        _base_trip_row(
            ride_id="BAD_DUR_SHORT",
            started_at="2024-04-15 08:10:00",
            ended_at="2024-04-15 08:10:30",
        ),
        # DQ-01: too long (> 240.0 min)
        _base_trip_row(
            ride_id="BAD_DUR_LONG",
            started_at="2024-04-15 08:10:00",
            ended_at="2024-04-15 13:00:00",
        ),
        # DQ-01: prior-month midnight spillover (2024-06-30 ending 2024-07-01)
        _base_trip_row(
            ride_id="BAD_SPILLOVER",
            started_at="2024-06-30 23:55:00",
            ended_at="2024-07-01 00:05:00",
        ),
        # DQ-02: missing end_station_id
        _base_trip_row(ride_id="BAD_STATION", end_station_id=""),
        # DQ-03: out-of-bounds GPS (0.0, 0.0)
        _base_trip_row(ride_id="BAD_GEO_ZERO", end_lat=0.0, end_lng=0.0),
        # DQ-03: cross-river Manhattan drop-off station (non-JC/HB station ID and NYC lng)
        _base_trip_row(
            ride_id="BAD_CROSS_RIVER_NYC",
            end_station_id="5470.10",
            end_station_name="W 15 St & 7 Ave",
            end_lat=40.7393,
            end_lng=-73.9993,
        ),
        # DQ-04: impossible speed (> 45 km/h inside Hudson County: ~6 km in 2 min = ~180 km/h)
        _base_trip_row(
            ride_id="BAD_SPEED",
            started_at="2024-04-15 08:10:00",
            ended_at="2024-04-15 08:12:00",
            start_lat=40.7000,
            start_lng=-74.0800,
            end_lat=40.7500,
            end_lng=-74.0300,
        ),
        # Valid docked_bike normalized to classic_bike
        _base_trip_row(ride_id="VALID_DOCKED", rideable_type="docked_bike"),
        # Valid fractional-millisecond ISO timestamp (post-mid-2024 Citi Bike format)
        _base_trip_row(
            ride_id="VALID_MILLIS",
            started_at="2024-07-31 12:06:49.628",
            ended_at="2024-07-31 12:23:59.876",
        ),
        # Unsupported category outlier dropped after DQ rules
        _base_trip_row(ride_id="BAD_CATEGORY", member_casual="unknown_tier"),
    ]
    raw_df = pd.DataFrame(rows)

    cleaned_df, report = clean_trips(raw_df)

    assert len(cleaned_df) == 3
    assert list(cleaned_df["ride_id"]) == ["VALID_1", "VALID_DOCKED", "VALID_MILLIS"]
    assert list(cleaned_df["rideable_type"]) == ["classic_bike", "classic_bike", "classic_bike"]

    assert report.initial_rows == 13
    assert report.duplicates_dropped == 1
    assert report.rule_audits["DQ-01"].rows_dropped == 4
    assert report.rule_audits["DQ-02"].rows_dropped == 1
    assert report.rule_audits["DQ-03"].rows_dropped == 2
    assert report.rule_audits["DQ-04"].rows_dropped == 1
    assert report.other_invalid_dropped == 1
    assert report.final_rows == 3
    assert report.total_dropped == 10
    assert report.total_dropped_pct == pytest.approx((10 / 13) * 100.0, rel=1e-4)


def test_clean_trips_empty_dataframe_returns_zero_audit() -> None:
    empty_df = pd.DataFrame(columns=list(_base_trip_row().keys()))
    cleaned_df, report = clean_trips(empty_df)

    assert cleaned_df.empty
    assert report.initial_rows == 0
    assert report.final_rows == 0
    assert report.total_dropped == 0
    assert report.total_dropped_pct == 0.0


def test_read_citibike_zip_ignores_macosx_metadata_and_concats_parts(tmp_path: Path) -> None:
    zip_path = tmp_path / "JC-202401-citibike-tripdata.csv.zip"
    header = (
        "ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,"
        "end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual\n"
    )
    row1 = (
        "R1,classic_bike,2024-01-01 08:00:00,2024-01-01 08:10:00,"
        "Grove St PATH,JC005,Exchange Place,JC001,40.7197,-74.0431,40.7168,-74.0323,member\n"
    )
    row2 = (
        "R2,electric_bike,2024-01-02 09:00:00,2024-01-02 09:12:00,"
        "Hoboken Terminal,HB101,Grove St PATH,JC005,40.7349,-74.0291,40.7197,-74.0431,casual\n"
    )
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("__MACOSX/", b"")
        zf.writestr("__MACOSX/._JC-202401-citibike-tripdata.csv", b"\x00\x05\x16\x07corrupt")
        zf.writestr("nested/._hidden.csv", b"not-a-csv")
        zf.writestr("README.txt", b"ignore non-csv")
        zf.writestr("part1.csv", header + row1)
        zf.writestr("part2.csv", header + row2)

    loaded_df = read_citibike_zip(zip_path)

    assert len(loaded_df) == 2
    assert list(loaded_df["ride_id"]) == ["R1", "R2"]
    assert list(loaded_df["start_station_id"]) == ["JC005", "HB101"]


def test_read_citibike_zip_raises_when_no_valid_csv_members(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("__MACOSX/._file.csv", b"corrupt")
        zf.writestr("._appledouble.csv", b"corrupt")

    with pytest.raises(ValueError, match="No valid CSV members"):
        read_citibike_zip(zip_path)


def test_fetch_hourly_weather_reads_cached_json_and_deduplicates(tmp_path: Path) -> None:
    cache_file = tmp_path / "open_meteo_jc_2024-01-01_2024-01-01.json"
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [2.5, 2.5, 1.8],
            "apparent_temperature": [-1.0, -1.0, -2.0],
            "precipitation": [None, None, 0.4],
            "wind_speed_10m": [11.0, 11.0, 13.5],
            "relative_humidity_2m": [70.0, 70.0, 75.0],
            "weather_code": [None, None, 51],
        }
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    weather_df = fetch_hourly_weather(
        start_date="2024-01-01",
        end_date="2024-01-01",
        raw_dir=tmp_path,
        force_download=False,
    )

    assert len(weather_df) == 2
    assert weather_df.loc[0, "precipitation_mm"] == 0.0
    assert weather_df.loc[0, "weather_code"] == 0
    assert weather_df.loc[1, "weather_code"] == 51
