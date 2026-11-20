from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from micromobility_analytics.config import settings
from micromobility_analytics.data.features import (
    MEMBER_TYPES,
    RIDEABLE_TYPES,
    add_kinematic_features,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleAudit:
    # Audit metrics for a single data quality rule

    rule_id: str
    description: str
    rows_dropped: int
    pct_of_initial: float


@dataclass(frozen=True)
class CleaningAuditReport:
    # Structured summary of all cleaning steps applied to raw Citi Bike data

    initial_rows: int
    duplicates_dropped: int
    other_invalid_dropped: int
    final_rows: int
    total_dropped: int
    total_dropped_pct: float
    rule_audits: dict[str, RuleAudit]


def _pct(count: int, total: int) -> float:
    # Return percentage rounded to four decimal places, safe against zero denominator
    if total <= 0:
        return 0.0
    return round((count / total) * 100.0, 4)


def _is_missing_text(series: pd.Series) -> pd.Series:
    # Detect null, empty, whitespace-only, or sentinel text values
    text = series.astype("string").str.strip()
    return text.isna() | (text == "") | text.str.lower().isin(["nan", "none", "null"])


def _apply_rule(
    df: pd.DataFrame,
    invalid_mask: pd.Series,
    rule_id: str,
    description: str,
    initial_rows: int,
) -> tuple[pd.DataFrame, RuleAudit]:
    # Filter out rows matching invalid_mask and construct a RuleAudit record
    dropped = int(invalid_mask.sum())
    filtered = df.loc[~invalid_mask].copy()
    audit = RuleAudit(
        rule_id=rule_id,
        description=description,
        rows_dropped=dropped,
        pct_of_initial=_pct(dropped, initial_rows),
    )
    logger.info(
        "%s (%s): dropped %d rows (%.4f%% of initial).",
        rule_id,
        description,
        dropped,
        audit.pct_of_initial,
    )
    return filtered, audit


def clean_trips(
    raw_df: pd.DataFrame,
    valid_months: Sequence[str] | None = settings.default_jc_months,
) -> tuple[pd.DataFrame, CleaningAuditReport]:
    # Clean raw Citi Bike trip records using DQ-01...DQ-04 and return an audit report
    initial_rows = len(raw_df)
    df = raw_df.copy()

    string_cols = [
        "ride_id",
        "rideable_type",
        "member_casual",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
    ]
    for col in string_cols:
        df[col] = df[col].astype("string").str.strip()

    df["started_at"] = pd.to_datetime(df["started_at"], format="ISO8601", errors="coerce").astype(
        "datetime64[ns]"
    )
    df["ended_at"] = pd.to_datetime(df["ended_at"], format="ISO8601", errors="coerce").astype(
        "datetime64[ns]"
    )

    for coord_col in ["start_lat", "start_lng", "end_lat", "end_lng"]:
        df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce").astype("float64")

    # Deduplicate and remove rows with missing ride_id before rule evaluation.
    missing_ride_id = _is_missing_text(df["ride_id"])
    df = df.loc[~missing_ride_id].drop_duplicates(subset=["ride_id"], keep="first").copy()
    duplicates_dropped = initial_rows - len(df)
    if duplicates_dropped > 0:
        logger.info("Removed %d missing or duplicate ride_id rows.", duplicates_dropped)

    # DQ-01: Invalid timestamps, prior-month boundary spillover, or duration outside bounds.
    duration_min = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
    df["trip_duration_min"] = duration_min.astype("float64")
    out_of_month = (
        ~df["started_at"].dt.strftime("%Y%m").isin(list(valid_months))
        if valid_months is not None
        else pd.Series(False, index=df.index)
    )
    dq01_invalid = (
        df["started_at"].isna()
        | df["ended_at"].isna()
        | (df["ended_at"] <= df["started_at"])
        | (df["trip_duration_min"] < settings.min_trip_duration_min)
        | (df["trip_duration_min"] > settings.max_trip_duration_min)
        | out_of_month
    )
    df, audit_dq01 = _apply_rule(
        df,
        dq01_invalid,
        "DQ-01",
        (
            f"Duration outside [{settings.min_trip_duration_min}, "
            f"{settings.max_trip_duration_min}] min or prior-month spillover"
        ),
        initial_rows,
    )

    # DQ-02: Missing station IDs or names.
    dq02_invalid = (
        _is_missing_text(df["start_station_id"])
        | _is_missing_text(df["end_station_id"])
        | _is_missing_text(df["start_station_name"])
        | _is_missing_text(df["end_station_name"])
    )
    df, audit_dq02 = _apply_rule(
        df,
        dq02_invalid,
        "DQ-02",
        "Missing start/end station ID or name",
        initial_rows,
    )

    # DQ-03: Coordinates or station IDs outside Hudson County (JC/HB) service area.
    min_lat, max_lat, min_lng, max_lng = settings.metro_bbox
    prefixes = settings.valid_station_prefixes
    dq03_invalid = (
        df["start_lat"].isna()
        | df["start_lng"].isna()
        | df["end_lat"].isna()
        | df["end_lng"].isna()
        | ~df["start_lat"].between(min_lat, max_lat)
        | ~df["end_lat"].between(min_lat, max_lat)
        | ~df["start_lng"].between(min_lng, max_lng)
        | ~df["end_lng"].between(min_lng, max_lng)
        | ~df["start_station_id"].str.startswith(prefixes, na=False)
        | ~df["end_station_id"].str.startswith(prefixes, na=False)
    )
    df, audit_dq03 = _apply_rule(
        df,
        dq03_invalid,
        "DQ-03",
        "Coordinates or station IDs outside Hudson County (JC/HB) service area",
        initial_rows,
    )

    # DQ-04: Unrealistic cycling speed (> max_mean_speed_kmh).
    df = add_kinematic_features(df)
    dq04_invalid = (
        df["mean_speed_kmh"].isna()
        | (df["mean_speed_kmh"] < 0.0)
        | (df["mean_speed_kmh"] > settings.max_mean_speed_kmh)
    )
    df, audit_dq04 = _apply_rule(
        df,
        dq04_invalid,
        "DQ-04",
        f"Unrealistic mean_speed_kmh > {settings.max_mean_speed_kmh}",
        initial_rows,
    )

    # Normalize docked_bike -> classic_bike and drop unexpected category outliers if any.
    df["rideable_type"] = df["rideable_type"].replace({"docked_bike": "classic_bike"})
    invalid_categories = ~df["rideable_type"].isin(RIDEABLE_TYPES) | ~df["member_casual"].isin(
        MEMBER_TYPES
    )
    other_invalid_dropped = int(invalid_categories.sum())
    if other_invalid_dropped > 0:
        logger.warning(
            "Dropped %d rows with unsupported rideable_type or member_casual values.",
            other_invalid_dropped,
        )
        df = df.loc[~invalid_categories].copy()

    df = df.reset_index(drop=True)
    final_rows = len(df)
    total_dropped = initial_rows - final_rows

    report = CleaningAuditReport(
        initial_rows=initial_rows,
        duplicates_dropped=duplicates_dropped,
        other_invalid_dropped=other_invalid_dropped,
        final_rows=final_rows,
        total_dropped=total_dropped,
        total_dropped_pct=_pct(total_dropped, initial_rows),
        rule_audits={
            "DQ-01": audit_dq01,
            "DQ-02": audit_dq02,
            "DQ-03": audit_dq03,
            "DQ-04": audit_dq04,
        },
    )
    return df, report
