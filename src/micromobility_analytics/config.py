from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
TRIPS_ENRICHED_PATH: Path = PROCESSED_DATA_DIR / "trips_enriched.parquet"
STATIONS_EQUITY_PATH: Path = PROCESSED_DATA_DIR / "stations_equity.parquet"


class AppSettings(BaseSettings):
    app_title: str = "Urban Micromobility Analytics"
    default_jc_months: tuple[str, ...] = ("202401", "202404", "202407", "202410")
    # Citi Bike: https://citibikenyc.com/system-data
    citibike_base_url: str = "https://s3.amazonaws.com/tripdata"
    # Open-Meteo: https://archive-api.open-meteo.com/v1/archive
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    # Jersey City centroid
    jc_centroid_lat: float = 40.7282
    jc_centroid_lon: float = -74.0776
    timezone: str = "America/New_York"
    # Eliminates out-of-bounds GPS errors and 970 cross-river NYC station trips
    # that would otherwise be falsely snapped to Hudson County Census tracts.
    metro_bbox: tuple[float, float, float, float] = (40.68, 40.77, -74.11, -74.015)
    valid_station_prefixes: tuple[str, ...] = ("JC", "HB")
    # Filters false starts/dock-bounces (< 60s), timestamp glitches, lost-bike outliers (> 4h),
    # and 17 prior-month midnight-boundary spillover rows that otherwise distort monthly cohorts
    # and the 123-day trips_per_day denominator.
    min_trip_duration_min: float = 1.0
    max_trip_duration_min: float = 240.0
    # Filters timestamp/GPS glitches where straight-line distance vs. duration implies physically
    # impossible cycling speeds.
    max_mean_speed_kmh: float = 45.0
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = AppSettings()
