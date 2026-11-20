from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from micromobility_analytics.config import RAW_DATA_DIR, settings

logger = logging.getLogger(__name__)

RAW_TRIP_DTYPES: dict[str, str] = {
    "ride_id": "string",
    "rideable_type": "string",
    "start_station_name": "string",
    "start_station_id": "string",
    "end_station_name": "string",
    "end_station_id": "string",
    "member_casual": "string",
}

_TRANSIT_HUBS: list[dict[str, str | float]] = [
    {"hub_name": "Exchange Place PATH & Ferry", "lat": 40.7168, "lng": -74.0323},
    {"hub_name": "Grove Street PATH", "lat": 40.7197, "lng": -74.0431},
    {"hub_name": "Newport PATH & Light Rail", "lat": 40.7270, "lng": -74.0338},
    {"hub_name": "Journal Square PATH", "lat": 40.7327, "lng": -74.0628},
    {"hub_name": "Hoboken Terminal (PATH/NJT/Ferry)", "lat": 40.7349, "lng": -74.0291},
    {"hub_name": "Harsimus Cove Light Rail", "lat": 40.7219, "lng": -74.0365},
    {"hub_name": "Liberty State Park Light Rail", "lat": 40.7088, "lng": -74.0575},
    {"hub_name": "West Side Avenue Light Rail", "lat": 40.7205, "lng": -74.0880},
    {"hub_name": "2nd Street Light Rail (Hoboken)", "lat": 40.7393, "lng": -74.0419},
    {"hub_name": "9th Street / Congress St Light Rail", "lat": 40.7457, "lng": -74.0383},
    {"hub_name": "14th Street Ferry (Hoboken)", "lat": 40.7524, "lng": -74.0226},
    {"hub_name": "Paulus Hook Ferry & Light Rail", "lat": 40.7138, "lng": -74.0342},
]

# US Census Bureau 2024 ACS 5-Year Estimates for Hudson County, NJ (FIPS 34017) Census Tracts:
# median_household_income (Table B19013):
# https://data.census.gov/table/ACSDT5Y2024.B19013?g=050XX00US34017$1400000

# pct_car_free_households:
# https://data.census.gov/table/ACSDT5Y2024.B08201?g=050XX00US34017$1400000

# population_density & tract centroids (Table B01003 & TIGER Gazetteer):
# https://data.census.gov/table/ACSDT5Y2024.B01003?g=050XX00US34017$1400000
# https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
_CENSUS_NEIGHBORHOODS: list[dict[str, str | float]] = [
    {
        "census_tract_geoid": "34017001900",
        "neighborhood_name": "Downtown JC / Waterfront",
        "lat": 40.7175,
        "lng": -74.0355,
        "median_household_income": 168500.0,
        "pct_car_free_households": 48.5,
        "population_density": 18400.0,
        "equity_tier": "Affluent",
    },
    {
        "census_tract_geoid": "34017002200",
        "neighborhood_name": "Historic Downtown / Grove St",
        "lat": 40.7210,
        "lng": -74.0445,
        "median_household_income": 149200.0,
        "pct_car_free_households": 52.0,
        "population_density": 21200.0,
        "equity_tier": "Affluent",
    },
    {
        "census_tract_geoid": "34017001701",
        "neighborhood_name": "Newport / Hamilton Park",
        "lat": 40.7278,
        "lng": -74.0385,
        "median_household_income": 162000.0,
        "pct_car_free_households": 50.2,
        "population_density": 22500.0,
        "equity_tier": "Affluent",
    },
    {
        "census_tract_geoid": "34017018700",
        "neighborhood_name": "Hoboken Waterfront / Downtown",
        "lat": 40.7375,
        "lng": -74.0305,
        "median_household_income": 171000.0,
        "pct_car_free_households": 49.0,
        "population_density": 23800.0,
        "equity_tier": "Affluent",
    },
    {
        "census_tract_geoid": "34017018200",
        "neighborhood_name": "Uptown Hoboken",
        "lat": 40.7505,
        "lng": -74.0275,
        "median_household_income": 165400.0,
        "pct_car_free_households": 44.8,
        "population_density": 22100.0,
        "equity_tier": "Affluent",
    },
    {
        "census_tract_geoid": "34017018500",
        "neighborhood_name": "West Hoboken",
        "lat": 40.7425,
        "lng": -74.0390,
        "median_household_income": 128000.0,
        "pct_car_free_households": 41.5,
        "population_density": 19600.0,
        "equity_tier": "Moderate",
    },
    {
        "census_tract_geoid": "34017001400",
        "neighborhood_name": "Journal Square",
        "lat": 40.7335,
        "lng": -74.0630,
        "median_household_income": 78500.0,
        "pct_car_free_households": 56.4,
        "population_density": 17900.0,
        "equity_tier": "Moderate",
    },
    {
        "census_tract_geoid": "34017000600",
        "neighborhood_name": "The Heights",
        "lat": 40.7475,
        "lng": -74.0510,
        "median_household_income": 84200.0,
        "pct_car_free_households": 46.2,
        "population_density": 16800.0,
        "equity_tier": "Moderate",
    },
    {
        "census_tract_geoid": "34017004100",
        "neighborhood_name": "McGinley Square",
        "lat": 40.7248,
        "lng": -74.0665,
        "median_household_income": 69400.0,
        "pct_car_free_households": 53.1,
        "population_density": 16100.0,
        "equity_tier": "High Need",
    },
    {
        "census_tract_geoid": "34017004600",
        "neighborhood_name": "Bergen-Lafayette",
        "lat": 40.7130,
        "lng": -74.0620,
        "median_household_income": 58900.0,
        "pct_car_free_households": 49.8,
        "population_density": 12900.0,
        "equity_tier": "High Need",
    },
    {
        "census_tract_geoid": "34017004200",
        "neighborhood_name": "West Side / Lincoln Park",
        "lat": 40.7235,
        "lng": -74.0840,
        "median_household_income": 61200.0,
        "pct_car_free_households": 45.7,
        "population_density": 13500.0,
        "equity_tier": "High Need",
    },
    {
        "census_tract_geoid": "34017005801",
        "neighborhood_name": "Greenville",
        "lat": 40.6995,
        "lng": -74.0810,
        "median_household_income": 54300.0,
        "pct_car_free_households": 47.3,
        "population_density": 11800.0,
        "equity_tier": "High Need",
    },
]


def _download_file(url: str, destination: Path) -> Path:
    # Download a remote URL to destination atomically via a temporary part file
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(f"{destination.suffix}.part")
    logger.info("Downloading %s -> %s", url, destination)
    request = urllib.request.Request(
        url, headers={"User-Agent": "urban-micromobility-analytics/0.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as out_file:
        while chunk := response.read(1024 * 64):
            out_file.write(chunk)
    temp_path.replace(destination)
    return destination


def download_citibike_month(
    yyyymm: str,
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> Path:
    #  Download and cache a monthly JC- Citi Bike trip ZIP archive
    filename = f"JC-{yyyymm}-citibike-tripdata.csv.zip"
    dest_path = raw_dir / filename
    if dest_path.exists() and not force:
        logger.info("Using cached Citi Bike archive: %s", dest_path)
        return dest_path

    url = f"{settings.citibike_base_url}/{filename}"
    return _download_file(url, dest_path)


def read_citibike_zip(zip_path: Path) -> pd.DataFrame:
    # Read CSV trip tables from a Citi Bike ZIP archive while ignoring macOS metadata
    # CODEOWNERS use Windows
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [
            member
            for member in zf.namelist()
            if not member.endswith("/")
            and "__MACOSX" not in member
            and not Path(member).name.startswith("._")
            and member.lower().endswith(".csv")
        ]
        if not csv_members:
            msg = f"No valid CSV members found inside {zip_path}"
            raise ValueError(msg)

        for member in csv_members:
            logger.debug("Reading ZIP member %s from %s", member, zip_path.name)
            with zf.open(member) as csv_stream:
                df = pd.read_csv(csv_stream, dtype=RAW_TRIP_DTYPES, low_memory=False)
                frames.append(df)

    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def load_raw_trips(
    months: Sequence[str] = settings.default_jc_months,
    raw_dir: Path = RAW_DATA_DIR,
    force_download: bool = False,
) -> pd.DataFrame:
    # Download and concatenate raw Citi Bike trips across target months
    frames: list[pd.DataFrame] = []
    for month in months:
        zip_path = download_citibike_month(month, raw_dir=raw_dir, force=force_download)
        month_df = read_citibike_zip(zip_path)
        logger.info("Loaded %d raw rows from %s.", len(month_df), zip_path.name)
        frames.append(month_df)

    if not frames:
        msg = "At least one month slice is required to load Citi Bike trips."
        raise ValueError(msg)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total raw Citi Bike rows loaded across %d slices: %d", len(frames), len(combined))
    return combined


def fetch_hourly_weather(
    start_date: str,
    end_date: str,
    raw_dir: Path = RAW_DATA_DIR,
    force_download: bool = False,
) -> pd.DataFrame:
    # Fetch and cache Open-Meteo historical hourly weather for Jersey City
    cache_path = raw_dir / f"open_meteo_jc_{start_date}_{end_date}.json"
    if cache_path.exists() and not force_download:
        logger.info("Using cached Open-Meteo weather file: %s", cache_path)
    else:
        params = urllib.parse.urlencode(
            {
                "latitude": settings.jc_centroid_lat,
                "longitude": settings.jc_centroid_lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": (
                    "temperature_2m,apparent_temperature,precipitation,"
                    "wind_speed_10m,relative_humidity_2m,weather_code"
                ),
                "timezone": settings.timezone,
                "wind_speed_unit": "kmh",
            }
        )
        url = f"{settings.open_meteo_archive_url}?{params}"
        _download_file(url, cache_path)

    payload: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))
    hourly: dict[str, list[Any]] = payload["hourly"]
    weather_df = pd.DataFrame(
        {
            "weather_hour": pd.to_datetime(hourly["time"]).astype("datetime64[ns]"),
            "temperature_c": pd.Series(hourly["temperature_2m"], dtype="float64"),
            "apparent_temperature_c": pd.Series(hourly["apparent_temperature"], dtype="float64"),
            "precipitation_mm": pd.Series(hourly["precipitation"], dtype="float64").fillna(0.0),
            "wind_speed_kmh": pd.Series(hourly["wind_speed_10m"], dtype="float64"),
            "relative_humidity_pct": pd.Series(hourly["relative_humidity_2m"], dtype="float64"),
            "weather_code": pd.Series(hourly["weather_code"], dtype="float64")
            .fillna(0)
            .astype("int64"),
        }
    )
    weather_df = (
        weather_df.drop_duplicates(subset=["weather_hour"], keep="first")
        .sort_values("weather_hour")
        .reset_index(drop=True)
    )
    logger.info(
        "Loaded %d hourly weather records (%s to %s).", len(weather_df), start_date, end_date
    )
    return weather_df


def load_census_and_transit_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Return Hudson County / Jersey City neighborhood Census metrics and rapid transit hubs
    zones_df = pd.DataFrame(_CENSUS_NEIGHBORHOODS)
    zones_df["census_tract_geoid"] = zones_df["census_tract_geoid"].astype("string")
    zones_df["neighborhood_name"] = zones_df["neighborhood_name"].astype("string")
    zones_df["equity_tier"] = zones_df["equity_tier"].astype("string")

    hubs_df = pd.DataFrame(_TRANSIT_HUBS)
    hubs_df["hub_name"] = hubs_df["hub_name"].astype("string")
    return zones_df, hubs_df
