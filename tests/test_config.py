from micromobility_analytics.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    STATIONS_EQUITY_PATH,
    TRIPS_ENRICHED_PATH,
    settings,
)


def test_project_paths_exist() -> None:
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.parent == DATA_DIR
    assert PROCESSED_DATA_DIR.parent == DATA_DIR
    assert TRIPS_ENRICHED_PATH.parent == PROCESSED_DATA_DIR
    assert STATIONS_EQUITY_PATH.parent == PROCESSED_DATA_DIR


def test_default_pipeline_settings() -> None:
    assert len(settings.default_jc_months) == 4
    assert settings.min_trip_duration_min == 1.0
    assert settings.max_trip_duration_min == 240.0
    assert settings.max_mean_speed_kmh == 45.0
