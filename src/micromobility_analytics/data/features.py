from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from micromobility_analytics.config import settings

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM: float = 6371.0088

RIDEABLE_TYPES: tuple[str, ...] = ("classic_bike", "electric_bike")
MEMBER_TYPES: tuple[str, ...] = ("member", "casual")
COMMUTE_WINDOWS: tuple[str, ...] = ("AM Peak", "Midday", "PM Peak", "Night")
WEATHER_REGIMES: tuple[str, ...] = (
    "Clear/Dry",
    "Cloudy",
    "Light Rain",
    "Heavy Rain/Snow",
    "Extreme Temp",
)
EQUITY_TIERS: tuple[str, ...] = ("High Need", "Moderate", "Affluent")
DAYS_OF_WEEK: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def compute_haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:

    lat1_rad = np.radians(lat1.to_numpy(dtype="float64"))
    lon1_rad = np.radians(lon1.to_numpy(dtype="float64"))
    lat2_rad = np.radians(lat2.to_numpy(dtype="float64"))
    lon2_rad = np.radians(lon2.to_numpy(dtype="float64"))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))  # prevent float overshoot > 1.0
    return pd.Series(EARTH_RADIUS_KM * c, index=lat1.index, dtype="float64")


def add_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "trip_duration_min" not in result.columns:
        duration_sec = (result["ended_at"] - result["started_at"]).dt.total_seconds()
        result["trip_duration_min"] = (duration_sec / 60.0).astype("float64")
    else:
        result["trip_duration_min"] = result["trip_duration_min"].astype("float64")

    if "haversine_distance_km" not in result.columns:
        result["haversine_distance_km"] = compute_haversine_km(
            result["start_lat"],
            result["start_lng"],
            result["end_lat"],
            result["end_lng"],
        )
    else:
        result["haversine_distance_km"] = result["haversine_distance_km"].astype("float64")

    if "mean_speed_kmh" not in result.columns:
        duration_hours = result["trip_duration_min"] / 60.0
        speed_values = np.where(
            duration_hours > 0.0,
            result["haversine_distance_km"] / duration_hours,
            0.0,
        )
        result["mean_speed_kmh"] = pd.Series(speed_values, index=result.index, dtype="float64")
    else:
        result["mean_speed_kmh"] = result["mean_speed_kmh"].astype("float64")

    return result


def engineer_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    result = add_kinematic_features(df)

    result["started_at"] = pd.to_datetime(result["started_at"]).astype("datetime64[ns]")
    result["ended_at"] = pd.to_datetime(result["ended_at"]).astype("datetime64[ns]")
    result["is_round_trip"] = (result["start_station_id"] == result["end_station_id"]).astype(bool)

    # Round down to the nearest hour for Open-Meteo API
    result["weather_hour"] = result["started_at"].dt.floor("h").astype("datetime64[ns]")
    result["hour_of_day"] = result["started_at"].dt.hour.astype("int8")

    day_names = result["started_at"].dt.day_name()
    result["day_of_week"] = pd.Categorical(
        day_names,
        categories=list(DAYS_OF_WEEK),
        ordered=True,
    )
    # pandas day_of_week: Monday=0 ... Saturday=5, Sunday=6
    is_weekend = result["started_at"].dt.day_of_week.isin([5, 6]).astype(bool)
    result["is_weekend"] = is_weekend

    hours = result["hour_of_day"]
    is_am_peak = (~is_weekend) & hours.isin([6, 7, 8, 9])
    is_pm_peak = (~is_weekend) & hours.isin([16, 17, 18, 19])
    is_midday = ((~is_weekend) & hours.isin([10, 11, 12, 13, 14, 15])) | (
        is_weekend & hours.between(6, 19)
    )
    window_labels = np.select(
        [is_am_peak, is_pm_peak, is_midday],
        ["AM Peak", "PM Peak", "Midday"],
        default="Night",
    )
    result["commute_window"] = pd.Categorical(
        window_labels,
        categories=list(COMMUTE_WINDOWS),
        ordered=True,
    )
    return result


def classify_weather_regime(weather_df: pd.DataFrame) -> pd.DataFrame:
    result = weather_df.copy()
    result["weather_hour"] = pd.to_datetime(result["weather_hour"]).astype("datetime64[ns]")

    precip = result["precipitation_mm"].astype("float64")
    temp = result["temperature_c"].astype("float64")
    apparent_temp = result["apparent_temperature_c"].astype("float64")
    wmo = result["weather_code"].astype("int64")

    # WMO 4677 codes from Open-Meteo (https://open-meteo.com/en/docs):
    # 2, 3 = partly cloudy / overcast
    # 45, 48 = fog
    # 51, 53, 55 = drizzle
    # 61 = slight rain
    # >= 63 = moderate/heavy rain, freezing rain, snow, or thunderstorms
    is_heavy = (precip >= 2.5) | (wmo >= 63)
    is_light_rain = ((precip > 0.0) & (precip < 2.5)) | wmo.isin([51, 53, 55, 61])
    is_extreme_temp = (temp < 0.0) | (temp > 32.0) | (apparent_temp < -5.0) | (apparent_temp > 35.0)
    is_cloudy = wmo.isin([2, 3, 45, 48])

    regimes = np.select(
        [is_heavy, is_light_rain, is_extreme_temp, is_cloudy],
        ["Heavy Rain/Snow", "Light Rain", "Extreme Temp", "Cloudy"],
        default="Clear/Dry",
    )
    result["weather_regime"] = pd.Categorical(
        regimes,
        categories=list(WEATHER_REGIMES),
        ordered=True,
    )
    return result


def _distance_matrix_km(
    src_lats: pd.Series,
    src_lngs: pd.Series,
    dst_lats: pd.Series,
    dst_lngs: pd.Series,
) -> np.ndarray:

    n_src = len(src_lats)
    columns: list[np.ndarray] = []
    for target_lat, target_lng in zip(dst_lats, dst_lngs, strict=True):
        target_lat_s = pd.Series(np.full(n_src, target_lat, dtype="float64"), index=src_lats.index)
        target_lng_s = pd.Series(np.full(n_src, target_lng, dtype="float64"), index=src_lngs.index)
        dist_km = compute_haversine_km(src_lats, src_lngs, target_lat_s, target_lng_s)
        columns.append(dist_km.to_numpy(dtype="float64"))
    return np.column_stack(columns)


def assign_station_spatial_equity(
    station_coords_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    hubs_df: pd.DataFrame,
) -> pd.DataFrame:
    result = station_coords_df.copy()
    result["station_id"] = result["station_id"].astype("string")
    result["station_name"] = result["station_name"].astype("string")
    result["lat"] = result["lat"].astype("float64")
    result["lng"] = result["lng"].astype("float64")

    if result.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "lat",
                "lng",
                "neighborhood_name",
                "census_tract_geoid",
                "equity_tier",
                "median_household_income",
                "pct_car_free_households",
                "population_density",
                "nearest_transit_hub",
                "dist_to_transit_hub_m",
            ]
        )

    min_lat, max_lat, min_lng, max_lng = settings.metro_bbox
    prefixes = settings.valid_station_prefixes
    out_of_service_mask = (
        ~result["station_id"].str.startswith(prefixes, na=False)
        | ~result["lat"].between(min_lat, max_lat)
        | ~result["lng"].between(min_lng, max_lng)
    )
    if out_of_service_mask.any():
        bad_ids = list(result.loc[out_of_service_mask, "station_id"].head(5))
        msg = (
            f"Cannot assign Hudson County Census equity to {int(out_of_service_mask.sum())} "
            f"out-of-network stations (sample IDs: {bad_ids}). Filter to JC/HB stations first."
        )
        raise ValueError(msg)

    hub_dists_m = (
        _distance_matrix_km(
            result["lat"],
            result["lng"],
            hubs_df["lat"],
            hubs_df["lng"],
        )
        * 1000.0
    )
    nearest_hub_idx = np.argmin(hub_dists_m, axis=1)
    result["nearest_transit_hub"] = pd.Series(
        hubs_df["hub_name"].to_numpy()[nearest_hub_idx],
        index=result.index,
        dtype="string",
    )
    result["dist_to_transit_hub_m"] = pd.Series(
        hub_dists_m[np.arange(len(result)), nearest_hub_idx],
        index=result.index,
        dtype="float64",
    )

    zone_dists_km = _distance_matrix_km(
        result["lat"],
        result["lng"],
        zones_df["lat"],
        zones_df["lng"],
    )
    nearest_zone_idx = np.argmin(zone_dists_km, axis=1)

    result["neighborhood_name"] = pd.Series(
        zones_df["neighborhood_name"].to_numpy()[nearest_zone_idx],
        index=result.index,
        dtype="string",
    )
    result["census_tract_geoid"] = pd.Series(
        zones_df["census_tract_geoid"].to_numpy()[nearest_zone_idx],
        index=result.index,
        dtype="string",
    )
    result["equity_tier"] = pd.Categorical(
        zones_df["equity_tier"].to_numpy()[nearest_zone_idx],
        categories=list(EQUITY_TIERS),
        ordered=True,
    )
    result["median_household_income"] = pd.Series(
        zones_df["median_household_income"].to_numpy(dtype="float64")[nearest_zone_idx],
        index=result.index,
        dtype="float64",
    )
    result["pct_car_free_households"] = pd.Series(
        zones_df["pct_car_free_households"].to_numpy(dtype="float64")[nearest_zone_idx],
        index=result.index,
        dtype="float64",
    )
    result["population_density"] = pd.Series(
        zones_df["population_density"].to_numpy(dtype="float64")[nearest_zone_idx],
        index=result.index,
        dtype="float64",
    )
    return result


def _extract_unique_stations(trips_df: pd.DataFrame) -> pd.DataFrame:
    starts = trips_df[["start_station_id", "start_station_name", "start_lat", "start_lng"]].rename(
        columns={
            "start_station_id": "station_id",
            "start_station_name": "station_name",
            "start_lat": "lat",
            "start_lng": "lng",
        }
    )
    ends = trips_df[["end_station_id", "end_station_name", "end_lat", "end_lng"]].rename(
        columns={
            "end_station_id": "station_id",
            "end_station_name": "station_name",
            "end_lat": "lat",
            "end_lng": "lng",
        }
    )
    combined = pd.concat([starts, ends], ignore_index=True)
    return combined.groupby("station_id", as_index=False).agg(
        station_name=("station_name", "first"),
        lat=("lat", "median"),
        lng=("lng", "median"),
    )


def _join_hourly_weather(trips_df: pd.DataFrame, weather_classified: pd.DataFrame) -> pd.DataFrame:
    weather_cols = [
        "weather_hour",
        "temperature_c",
        "apparent_temperature_c",
        "precipitation_mm",
        "wind_speed_kmh",
        "weather_regime",
    ]
    weather_dedup = (
        weather_classified[weather_cols]
        .drop_duplicates(subset=["weather_hour"], keep="first")
        .sort_values("weather_hour")
        .reset_index(drop=True)
    )

    merged = trips_df.merge(weather_dedup, on="weather_hour", how="left")
    missing_mask = merged["temperature_c"].isna()
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        logger.warning(
            "Found %d trips without exact weather_hour match; applying nearest-hour alignment.",
            missing_count,
        )
        missing_subset = merged.loc[missing_mask, ["weather_hour"]].copy()
        missing_subset["_orig_idx"] = missing_subset.index
        missing_sorted = missing_subset.sort_values("weather_hour")
        aligned = pd.merge_asof(
            missing_sorted,
            weather_dedup,
            on="weather_hour",
            direction="nearest",
        ).set_index("_orig_idx")
        for col in [
            "temperature_c",
            "apparent_temperature_c",
            "precipitation_mm",
            "wind_speed_kmh",
            "weather_regime",
        ]:
            merged.loc[aligned.index, col] = aligned[col]

    return merged


def _build_stations_summary(
    trips_df: pd.DataFrame,
    station_equity: pd.DataFrame,
) -> pd.DataFrame:
    # Divide by observed active days instead of 365
    n_active_days = max(1, int(trips_df["started_at"].dt.normalize().nunique()))

    trips_for_agg = trips_df.assign(
        _is_ebike=(trips_df["rideable_type"] == "electric_bike").astype("int64"),
        _is_casual=(trips_df["member_casual"] == "casual").astype("int64"),
    )

    departures = (
        trips_for_agg.groupby("start_station_id", as_index=False)
        .agg(
            total_departures=("ride_id", "count"),
            ebike_departures=("_is_ebike", "sum"),
            casual_departures=("_is_casual", "sum"),
            dep_mean_duration=("trip_duration_min", "mean"),
        )
        .rename(columns={"start_station_id": "station_id"})
    )

    arrivals = (
        trips_for_agg.groupby("end_station_id", as_index=False)
        .agg(
            total_arrivals=("ride_id", "count"),
            arr_mean_duration=("trip_duration_min", "mean"),
        )
        .rename(columns={"end_station_id": "station_id"})
    )

    summary = station_equity.merge(departures, on="station_id", how="left").merge(
        arrivals, on="station_id", how="left"
    )

    for count_col in [
        "total_departures",
        "total_arrivals",
        "ebike_departures",
        "casual_departures",
    ]:
        summary[count_col] = summary[count_col].fillna(0).astype("int64")

    summary["total_trips"] = (summary["total_departures"] + summary["total_arrivals"]).astype(
        "int64"
    )
    summary["net_flow_balance"] = (summary["total_arrivals"] - summary["total_departures"]).astype(
        "int64"
    )
    summary["trips_per_day"] = (summary["total_departures"] / float(n_active_days)).astype(
        "float64"
    )
    summary["ebike_share"] = pd.Series(
        np.where(
            summary["total_departures"] > 0,
            summary["ebike_departures"] / summary["total_departures"],
            0.0,
        ),
        index=summary.index,
        dtype="float64",
    )
    summary["casual_share"] = pd.Series(
        np.where(
            summary["total_departures"] > 0,
            summary["casual_departures"] / summary["total_departures"],
            0.0,
        ),
        index=summary.index,
        dtype="float64",
    )
    # Fallback to arrival duration if a station had arrivals only and 0 departures
    summary["mean_duration_min"] = (
        summary["dep_mean_duration"].fillna(summary["arr_mean_duration"]).astype("float64")
    )

    columns = [
        "station_id",
        "station_name",
        "lat",
        "lng",
        "neighborhood_name",
        "census_tract_geoid",
        "equity_tier",
        "median_household_income",
        "pct_car_free_households",
        "population_density",
        "nearest_transit_hub",
        "dist_to_transit_hub_m",
        "total_departures",
        "total_arrivals",
        "total_trips",
        "net_flow_balance",
        "trips_per_day",
        "ebike_share",
        "casual_share",
        "mean_duration_min",
    ]
    return summary[columns].sort_values("station_id").reset_index(drop=True)


def build_enriched_tables(
    clean_trips_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    hubs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trips = engineer_trip_features(clean_trips_df)
    weather_classified = classify_weather_regime(weather_df)

    unique_stations = _extract_unique_stations(trips)
    station_equity = assign_station_spatial_equity(unique_stations, zones_df, hubs_df)

    trips_with_weather = _join_hourly_weather(trips, weather_classified)

    origin_lookup = station_equity[
        ["station_id", "neighborhood_name", "equity_tier", "dist_to_transit_hub_m"]
    ].rename(
        columns={
            "station_id": "start_station_id",
            "neighborhood_name": "start_neighborhood",
            "equity_tier": "start_equity_tier",
            "dist_to_transit_hub_m": "start_dist_to_transit_m",
        }
    )
    dest_lookup = station_equity[
        ["station_id", "neighborhood_name", "equity_tier", "dist_to_transit_hub_m"]
    ].rename(
        columns={
            "station_id": "end_station_id",
            "neighborhood_name": "end_neighborhood",
            "equity_tier": "end_equity_tier",
            "dist_to_transit_hub_m": "end_dist_to_transit_m",
        }
    )

    trips_enriched = trips_with_weather.merge(
        origin_lookup, on="start_station_id", how="left"
    ).merge(dest_lookup, on="end_station_id", how="left")

    string_cols = [
        "ride_id",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
        "start_neighborhood",
        "end_neighborhood",
    ]
    for col in string_cols:
        trips_enriched[col] = trips_enriched[col].astype("string")

    trips_enriched["rideable_type"] = pd.Categorical(
        trips_enriched["rideable_type"],
        categories=list(RIDEABLE_TYPES),
        ordered=False,
    )
    trips_enriched["member_casual"] = pd.Categorical(
        trips_enriched["member_casual"],
        categories=list(MEMBER_TYPES),
        ordered=False,
    )
    trips_enriched["weather_regime"] = pd.Categorical(
        trips_enriched["weather_regime"],
        categories=list(WEATHER_REGIMES),
        ordered=True,
    )
    trips_enriched["start_equity_tier"] = pd.Categorical(
        trips_enriched["start_equity_tier"],
        categories=list(EQUITY_TIERS),
        ordered=True,
    )
    trips_enriched["end_equity_tier"] = pd.Categorical(
        trips_enriched["end_equity_tier"],
        categories=list(EQUITY_TIERS),
        ordered=True,
    )

    trip_columns = [
        "ride_id",
        "rideable_type",
        "member_casual",
        "started_at",
        "ended_at",
        "weather_hour",
        "start_station_id",
        "start_station_name",
        "end_station_id",
        "end_station_name",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
        "trip_duration_min",
        "haversine_distance_km",
        "mean_speed_kmh",
        "is_round_trip",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "commute_window",
        "temperature_c",
        "apparent_temperature_c",
        "precipitation_mm",
        "wind_speed_kmh",
        "weather_regime",
        "start_neighborhood",
        "start_equity_tier",
        "start_dist_to_transit_m",
        "end_neighborhood",
        "end_equity_tier",
        "end_dist_to_transit_m",
    ]
    trips_enriched = trips_enriched[trip_columns].reset_index(drop=True)
    stations_summary = _build_stations_summary(trips_enriched, station_equity)

    logger.info(
        "Built enriched tables: %d trips and %d unique stations.",
        len(trips_enriched),
        len(stations_summary),
    )
    return trips_enriched, stations_summary
