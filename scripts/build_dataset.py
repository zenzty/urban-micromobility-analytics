from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from micromobility_analytics.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    STATIONS_EQUITY_PATH,
    TRIPS_ENRICHED_PATH,
    settings,
)
from micromobility_analytics.data import (
    CleaningAuditReport,
    build_enriched_tables,
    clean_trips,
    fetch_hourly_weather,
    load_census_and_transit_reference,
    load_raw_trips,
    validate_stations_equity,
    validate_trips_enriched,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Parse CLI arguments for the dataset build pipeline
    parser = argparse.ArgumentParser(
        description="Build validated Citi Bike, weather, and equity Parquet datasets.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=list(settings.default_jc_months),
        help="Target YYYYMM Citi Bike JC slices (default: %(default)s).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw Citi Bike and Open-Meteo files even if cached.",
    )
    return parser.parse_args(argv)


def _log_audit_report(report: CleaningAuditReport) -> None:
    # Emit a structured summary of data cleaning counts and drop percentages
    logger.info("=== Data Cleaning Audit Summary ===")
    logger.info("Initial raw rows      : %d", report.initial_rows)
    logger.info("Duplicate/null ride_id: %d", report.duplicates_dropped)
    for rule_id, audit in report.rule_audits.items():
        logger.info(
            "%-6s %-55s dropped=%7d (%.4f%%)",
            rule_id,
            audit.description,
            audit.rows_dropped,
            audit.pct_of_initial,
        )
    if report.other_invalid_dropped > 0:
        logger.info("Other category drops  : %d", report.other_invalid_dropped)
    logger.info(
        "Final clean rows      : %d (total dropped: %d / %.4f%%)",
        report.final_rows,
        report.total_dropped,
        report.total_dropped_pct,
    )


def run_pipeline(months: Sequence[str], force_download: bool = False) -> None:
    # Execute the full ETL pipeline and persist validated Parquet outputs
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Layer 1 ETL build for months: %s", ", ".join(months))
    raw_trips_df = load_raw_trips(
        months=months,
        raw_dir=RAW_DATA_DIR,
        force_download=force_download,
    )

    clean_trips_df, audit_report = clean_trips(raw_trips_df, valid_months=months)
    _log_audit_report(audit_report)

    if clean_trips_df.empty:
        msg = "No valid trips remained after data cleaning; aborting dataset build."
        raise RuntimeError(msg)

    start_date = clean_trips_df["started_at"].min().strftime("%Y-%m-%d")
    end_date = clean_trips_df["started_at"].max().strftime("%Y-%m-%d")

    weather_df = fetch_hourly_weather(
        start_date=start_date,
        end_date=end_date,
        raw_dir=RAW_DATA_DIR,
        force_download=force_download,
    )
    zones_df, hubs_df = load_census_and_transit_reference()

    trips_enriched_df, stations_equity_df = build_enriched_tables(
        clean_trips_df=clean_trips_df,
        weather_df=weather_df,
        zones_df=zones_df,
        hubs_df=hubs_df,
    )

    logger.info("Validating processed tables against Pandera schemas...")
    validated_trips = validate_trips_enriched(trips_enriched_df)
    validated_stations = validate_stations_equity(stations_equity_df)

    validated_trips.to_parquet(TRIPS_ENRICHED_PATH, index=False)
    validated_stations.to_parquet(STATIONS_EQUITY_PATH, index=False)

    logger.info(
        "Saved %d rows to %s",
        len(validated_trips),
        TRIPS_ENRICHED_PATH,
    )
    logger.info(
        "Saved %d rows to %s",
        len(validated_stations),
        STATIONS_EQUITY_PATH,
    )


def main(argv: Sequence[str] | None = None) -> int:
    # Configure logging and execute the dataset build CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = _parse_args(argv)
    run_pipeline(months=args.months, force_download=args.force_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
