from __future__ import annotations

import pytest

from micromobility_analytics.analytics.aggregations import (
    compute_cohort_distributions,
    compute_dynamic_station_metrics,
    compute_equity_tier_summary,
    compute_kpi_summary,
    compute_od_flows,
    compute_weather_regime_elasticity,
    compute_weather_time_density,
    prepare_parallel_coords_table,
    sample_trips_for_distribution,
)
from micromobility_analytics.analytics.filters import TripFilterCriteria, filter_trips
from tests.test_filters import _synthetic_enriched_dataset


def test_compute_kpi_summary_populated_and_empty() -> None:
    trips_df, _ = _synthetic_enriched_dataset()

    kpi = compute_kpi_summary(trips_df)
    assert kpi.total_trips == 5
    assert kpi.active_stations == 4
    assert kpi.active_days == 2
    assert kpi.trips_per_active_day == pytest.approx(2.5)
    assert kpi.ebike_share_pct == pytest.approx(60.0)
    assert kpi.member_share_pct == pytest.approx(60.0)
    assert kpi.transit_feeder_share_pct == pytest.approx(100.0)
    assert kpi.high_need_tier_share_pct == pytest.approx(40.0)

    empty_kpi = compute_kpi_summary(trips_df.iloc[0:0])
    assert empty_kpi.total_trips == 0
    assert empty_kpi.active_stations == 0
    assert empty_kpi.trips_per_active_day == 0.0


def test_compute_dynamic_station_metrics_flow_conservation_and_zero_trip_stations() -> None:
    trips_df, stations_df = _synthetic_enriched_dataset()

    dyn = compute_dynamic_station_metrics(trips_df, stations_df)
    assert len(dyn) == 4
    # Flow conservation across the full closed network: sum(departures) == sum(arrivals)
    assert int(dyn["total_departures"].sum()) == 5
    assert int(dyn["total_arrivals"].sum()) == 5
    assert int(dyn["net_flow_balance"].sum()) == 0
    assert dyn["flow_asymmetry_index"].between(-1.0, 1.0).all()

    hb101 = dyn.loc[dyn["station_id"] == "HB101"].iloc[0]
    assert hb101["total_departures"] == 1
    assert hb101["total_arrivals"] == 0
    assert hb101["flow_asymmetry_index"] == pytest.approx(-1.0)

    sub_trips = filter_trips(trips_df, TripFilterCriteria(months=("202401",)))
    dyn_jan = compute_dynamic_station_metrics(sub_trips, stations_df)
    hb101_jan = dyn_jan.loc[dyn_jan["station_id"] == "HB101"].iloc[0]
    assert hb101_jan["total_trips"] == 0
    assert hb101_jan["flow_asymmetry_index"] == 0.0
    assert hb101_jan["mean_duration_min"] == 0.0


def test_compute_od_flows_sankey_labels_and_self_loop_exclusion() -> None:
    trips_df, _ = _synthetic_enriched_dataset()

    # 1. Station granularity with self-loops included vs excluded
    flows_all = compute_od_flows(trips_df, granularity="station", exclude_self_loops=False)
    assert "source_label" in flows_all.columns
    assert "target_label" in flows_all.columns
    assert (flows_all["source_label"] != flows_all["target_label"]).all()
    assert flows_all["flow_share_pct"].sum() == pytest.approx(100.0)

    flows_no_self = compute_od_flows(trips_df, granularity="station", exclude_self_loops=True)
    assert not (flows_no_self["origin"] == flows_no_self["destination"]).any()
    assert int(flows_no_self["trip_count"].sum()) == 4

    # 2. Equity tier and transit_hub_role granularities
    tier_flows = compute_od_flows(trips_df, granularity="equity_tier", top_k=2)
    assert len(tier_flows) <= 2

    role_flows = compute_od_flows(trips_df, granularity="transit_hub_role")
    assert set(role_flows["origin"]).issubset(
        {"Transit Hub (<=350m)", "Neighborhood Feeder (>350m)"}
    )


def test_weather_elasticity_exposure_normalization_and_temporal_alignment() -> None:
    trips_df, _ = _synthetic_enriched_dataset()

    # Clear/Dry has 2 distinct weather_hours (2024-01-15 08:00 and 2024-07-20 14:00)
    elasticity = compute_weather_regime_elasticity(trips_df)
    clear_rows = elasticity.loc[elasticity["weather_regime"] == "Clear/Dry"]
    assert (clear_rows["observed_hours"] == 2).all()

    # Member + classic_bike in Clear/Dry (1 trip over 2 hours = 0.5 trips/hr)
    mc_clear = clear_rows.loc[
        (clear_rows["member_casual"] == "member") & (clear_rows["rideable_type"] == "classic_bike")
    ].iloc[0]
    assert mc_clear["trips_per_observed_hour"] == pytest.approx(0.5)
    assert mc_clear["elasticity_index_pct"] == pytest.approx(100.0)
    assert mc_clear["pct_change_vs_baseline"] == pytest.approx(0.0)

    jan_casual = filter_trips(
        trips_df,
        TripFilterCriteria(months=("202401",), member_casual=("casual",)),
    )
    jan_elast = compute_weather_regime_elasticity(jan_casual, reference_trips_df=trips_df)
    obs_by_regime = (
        jan_elast.groupby("weather_regime", observed=False)["observed_hours"].first().to_dict()
    )
    assert obs_by_regime["Clear/Dry"] == 1
    assert obs_by_regime["Light Rain"] == 1
    assert obs_by_regime["Heavy Rain/Snow"] == 0

    # 2D weather-time density grid
    density = compute_weather_time_density(trips_df, temp_bin_width_c=2.0)
    assert len(density) == 4
    assert (density["trips_per_observed_hour"] >= 1.0).all()


def test_cohort_distributions_and_stratified_sampling() -> None:
    trips_df, _ = _synthetic_enriched_dataset()

    dist = compute_cohort_distributions(
        trips_df,
        metric="trip_duration_min",
        group_cols=("member_casual", "rideable_type"),
    )
    assert int(dist["count"].sum()) == 5
    assert (dist["p75"] >= dist["p25"]).all()
    assert (dist["iqr"] == (dist["p75"] - dist["p25"])).all()

    sampled = sample_trips_for_distribution(
        trips_df,
        max_samples=3,
        stratify_cols=("member_casual",),
        clip_upper_quantile=0.95,
        random_state=42,
    )
    assert 1 <= len(sampled) <= 3


def test_equity_tier_summary_and_parallel_coords_normalization() -> None:
    trips_df, stations_df = _synthetic_enriched_dataset()
    dyn = compute_dynamic_station_metrics(trips_df, stations_df)

    tier_summary = compute_equity_tier_summary(dyn)
    assert list(tier_summary["equity_tier"]) == ["High Need", "Moderate", "Affluent"]
    assert int(tier_summary["station_count"].sum()) == 4

    parcoords = prepare_parallel_coords_table(dyn, normalize_axes=True)
    for col in [
        "median_household_income_norm",
        "pct_car_free_households_norm",
        "dist_to_transit_hub_m_norm",
        "trips_per_day_norm",
        "ebike_share_norm",
        "casual_share_norm",
        "net_flow_balance_norm",
    ]:
        assert col in parcoords.columns
        assert parcoords[col].between(0.0, 1.0).all()
