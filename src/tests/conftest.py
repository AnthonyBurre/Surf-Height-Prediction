"""Shared pytest fixtures for the test suite.

Pytest auto-discovers this file, so any test in ``src/tests/`` can name
these fixtures as parameters without an explicit import.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from forecast import chronological_split, make_target


# ---------------------------------------------------------------------------
# Wind raw record fixture
# ---------------------------------------------------------------------------

_WIND_RAW_RECORDS = [
    {
        "_id": 1,
        "Date": "2024-01-01T00:00:00",
        "Time": "00:00",
        "Wind Direction (degTN)": 180,
        "Wind Speed (m/s)": 2.5,
        "Wind Sigma Theta (deg)": 18.0,
        "Wind Speed Std Dev (m/s)": 0.7,
        "Ozone (ppm)": 0.02,
        "PM10 (ug/m^3)": 12.0,
    },
    {
        "_id": 2,
        "Date": "2024-01-01T00:00:00",
        "Time": "01:00",
        "Wind Direction (degTN)": 185,
        "Wind Speed (m/s)": 2.7,
        "Wind Sigma Theta (deg)": 17.0,
        "Wind Speed Std Dev (m/s)": 0.8,
        "Ozone (ppm)": 0.02,
        "PM10 (ug/m^3)": 11.5,
    },
]


@pytest.fixture
def wind_raw_records() -> list[dict]:
    return deepcopy(_WIND_RAW_RECORDS)


# ---------------------------------------------------------------------------
# Wave pipeline helpers
# ---------------------------------------------------------------------------


_WAVE_STANDARD_ROWS = [
    {
        "Date/Time": pd.Timestamp("2017-01-01 00:00:00"),
        "Hs": 1.10, "Hmax": 1.90, "Tz": 5.50,
        "Tp": 9.00, "Peak Direction": 95.0, "SST": 25.0,
    },
    {
        "Date/Time": pd.Timestamp("2017-01-01 00:30:00"),
        "Hs": 1.15, "Hmax": 1.95, "Tz": 5.55,
        "Tp": 9.10, "Peak Direction": 98.0, "SST": 25.1,
    },
]


@pytest.fixture
def standard_rows() -> list[dict]:
    return deepcopy(_WAVE_STANDARD_ROWS)


@pytest.fixture
def raw_frame():
    def _make(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)
    return _make


@pytest.fixture
def ts_aest():
    """AEST tz-aware timestamp matching cleaned-frame index values."""
    def _make(s: str) -> pd.Timestamp:
        return pd.Timestamp(s, tz="Australia/Brisbane")
    return _make


# ---------------------------------------------------------------------------
# Forecast scaffolding
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df():
    """Factory for deterministic 30-min wave-buoy-like frames.

    Default ``n=200`` matches the original duplicated helpers; callers
    can pass a smaller ``n`` for cheap tests (the underlying RNG is seeded).
    """
    def _make(n: int = 200, freq: str = "30min", seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n, freq=freq, tz="Australia/Brisbane")
        t = np.arange(n)
        diurnal = 0.5 * np.sin(2 * np.pi * t / 48)  # 24h cycle
        df = pd.DataFrame(
            {
                "hsig_m": 1.2 + diurnal + 0.05 * rng.standard_normal(n),
                "hmax_m": 2.0 + diurnal + 0.1 * rng.standard_normal(n),
                "tz_s": 5.5 + 0.2 * rng.standard_normal(n),
                "tp_s": 9.0 + 0.3 * rng.standard_normal(n),
                "peak_dir_deg": (90 + 5 * rng.standard_normal(n)) % 360,
                "sst_c": 25.0 + 0.1 * rng.standard_normal(n),
            },
            index=idx,
        )
        df.index.name = "datetime"
        return df
    return _make


@pytest.fixture
def split(synthetic_df):
    """Default chronological split used by experiment-logging tests."""
    df = synthetic_df(120)
    y = make_target(df, horizon_steps=4)
    Xtr, Xte, ytr, yte = chronological_split(df, y, test_frac=0.25)
    return df, Xtr, Xte, ytr, yte
