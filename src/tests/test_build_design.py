"""``features.build_design`` — the canonical design-matrix builder.

These tests pin ``build_design`` to the exact hand-rolled chain the notebooks
used to duplicate, so the rewrite to a single core builder is provably
behaviour-preserving, and lock the one reconciled divergence (engineered drops
raw ``*_deg`` wind bearings; raw keeps every wind column verbatim).
"""
import numpy as np
import pandas as pd
import pytest

from forecast import (
    add_neighbour_features,
    assemble_inputs,
    build_buoy_features,
    build_design,
    build_seq_features,
)


@pytest.fixture
def sources(synthetic_df):
    """A primary wave frame + one neighbour series + a wind frame, shared index."""
    wave = synthetic_df(120, seed=0)
    neighbour = synthetic_df(120, seed=1)["hsig_m"]
    neighbours = {"tweed-heads": neighbour.reindex(wave.index)}
    # Mirror load_wind's namespaced output, and include a raw ``*_deg`` bearing
    # so the engineered/raw divergence is exercised.
    rng = np.random.default_rng(2)
    n = len(wave)
    wind = pd.DataFrame(
        {
            "mountain-creek_wind_speed_ms": 4.0 + rng.standard_normal(n),
            "mountain-creek_wind_dir_deg_sin": np.sin(np.linspace(0, 6, n)),
            "mountain-creek_wind_dir_deg_cos": np.cos(np.linspace(0, 6, n)),
            "mountain-creek_wind_dir_deg": (180 + 10 * rng.standard_normal(n)) % 360,
        },
        index=wave.index,
    )
    return wave, neighbours, wind


def test_build_design_engineered_matches_manual_chain(sources):
    wave, neighbours, wind = sources

    merged, neighbour_cols, _ = assemble_inputs(wave, neighbours, wind)
    primary_only = merged[[c for c in merged.columns if c not in neighbour_cols]]
    expected = build_buoy_features(primary_only)
    expected = add_neighbour_features(expected, merged, neighbour_cols)
    wind_cols = [c for c in wind.columns if not c.endswith("_deg")]
    expected = add_neighbour_features(expected, wind, wind_cols)

    got = build_design(wave, neighbours, wind, kind="engineered")
    pd.testing.assert_frame_equal(got, expected)


def test_build_design_raw_matches_manual_chain(sources):
    wave, neighbours, wind = sources

    merged, _, _ = assemble_inputs(wave, neighbours, wind)
    expected = build_seq_features(merged)
    for col in wind.columns:
        expected[col] = wind[col]

    got = build_design(wave, neighbours, wind, kind="raw")
    pd.testing.assert_frame_equal(got, expected)


def test_engineered_excludes_raw_wind_bearing_but_keeps_lags(sources):
    wave, neighbours, wind = sources
    X = build_design(wave, neighbours, wind, kind="engineered")
    # Raw bearing dropped; its sin/cos (and their lags) carry direction.
    assert not any(c == "mountain-creek_wind_dir_deg" for c in X.columns)
    assert "mountain-creek_wind_dir_deg_sin_lag_1" in X.columns
    # Engineered hallmarks: primary + neighbour lag columns.
    assert "hsig_m_lag_1" in X.columns
    assert "tweed-heads_hsig_m_lag_1" in X.columns


def test_raw_keeps_wind_verbatim_and_has_no_lags(sources):
    wave, neighbours, wind = sources
    X = build_design(wave, neighbours, wind, kind="raw")
    # Every wind column rides through unchanged, including the raw bearing.
    for col in wind.columns:
        assert col in X.columns
    # Circular encoding applied, but no engineered lag/rolling columns.
    assert "peak_dir_deg_sin" in X.columns
    assert not any("_lag_" in c or "_roll" in c for c in X.columns)


def test_build_design_no_wind(sources):
    wave, neighbours, _ = sources
    X = build_design(wave, neighbours, None, kind="engineered")
    assert "tweed-heads_hsig_m_lag_1" in X.columns
    assert not any(c.startswith("mountain-creek_") for c in X.columns)


def test_build_design_invalid_kind_raises(sources):
    wave, neighbours, wind = sources
    with pytest.raises(ValueError, match="kind must be"):
        build_design(wave, neighbours, wind, kind="bogus")
