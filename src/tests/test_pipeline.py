from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from qld_ckan import unify_frames
from qld_ckan.wave.pipeline import clean, run, unify


# ---------------------------------------------------------------------------
# qld_ckan.unify_frames — shared helper exercised through the wave wrapper
# elsewhere; covered directly here so future sources don't have to.
# ---------------------------------------------------------------------------


def test_unify_frames_concatenates_with_reset_index():
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"x": [3]})
    out = unify_frames([a, b])
    assert list(out["x"]) == [1, 2, 3]
    assert list(out.index) == [0, 1, 2]


def test_unify_frames_raises_on_empty_list():
    with pytest.raises(ValueError, match="No data was downloaded"):
        unify_frames([])


# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------


def test_clean_renames_all_columns(raw_frame, standard_rows):
    result = clean(raw_frame(standard_rows))
    assert set(result.columns) == {"hsig_m", "hmax_m", "tz_s", "tp_s", "peak_dir_deg", "sst_c"}


def test_clean_sets_datetime_index(raw_frame, standard_rows):
    result = clean(raw_frame(standard_rows))
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "datetime"


def test_clean_sorts_index(raw_frame, standard_rows):
    result = clean(raw_frame(list(reversed(standard_rows))))
    assert result.index.is_monotonic_increasing


def test_clean_drops_rows_with_nat_datetime(raw_frame, standard_rows):
    standard_rows.append({
        "Date/Time": pd.NaT,
        "Hs": 1.0, "Hmax": 1.5, "Tz": 5.0,
        "Tp": 8.0, "Peak Direction": 90.0, "SST": 24.0,
    })
    result = clean(raw_frame(standard_rows))
    assert len(result) == 2
    assert result.index.isna().sum() == 0


def test_clean_preserves_row_count_when_no_bad_rows(raw_frame, standard_rows):
    assert len(clean(raw_frame(standard_rows))) == 2


def test_clean_preserves_values(raw_frame, standard_rows, ts_aest):
    result = clean(raw_frame(standard_rows))
    assert result.loc[ts_aest("2017-01-01 00:00:00"), "hsig_m"] == pytest.approx(1.10)


def test_clean_index_is_aest_tz_aware(raw_frame, standard_rows):
    """Source timestamps are naive AEST (UTC+10); after clean the index must
    be tz-aware Australia/Brisbane with the original wall-clock times
    intact. Asserting the tz directly catches a regression that drops the
    tz_localize step (the index name 'datetime' is just a string and would
    still match)."""
    result = clean(raw_frame(standard_rows))
    assert str(result.index.tz) == "Australia/Brisbane"
    assert result.index[0] == pd.Timestamp("2017-01-01 00:00", tz="Australia/Brisbane")


@pytest.mark.parametrize(
    "raw_col, clean_col",
    [
        ("Hs", "hsig_m"),
        ("Hmax", "hmax_m"),
        ("Tz", "tz_s"),
        ("Tp", "tp_s"),
        ("Peak Direction", "peak_dir_deg"),
        ("SST", "sst_c"),
    ],
)
def test_clean_replaces_sentinel_with_nan(raw_col, clean_col, raw_frame, standard_rows, ts_aest):
    """Every measurement column must honour the -99.9 sentinel — a missed
    column lets garbage through as a real reading (e.g. peak_dir_deg=-99.9
    survives encode_circular as a normal sin/cos pair)."""
    standard_rows[0][raw_col] = -99.9
    result = clean(raw_frame(standard_rows))
    assert np.isnan(result.loc[ts_aest("2017-01-01 00:00:00"), clean_col])
    # other measurement columns at the same row unchanged
    assert not np.isnan(result.loc[ts_aest("2017-01-01 00:30:00"), clean_col])


def test_clean_coerces_string_numerics(raw_frame, standard_rows, ts_aest):
    # Simulate CKAN returning numerics as strings, including the sentinel
    standard_rows[0]["Hs"] = "1.10"
    standard_rows[1]["Hs"] = "-99.9"
    result = clean(raw_frame(standard_rows))
    assert result["hsig_m"].dtype.kind == "f"
    assert result.loc[ts_aest("2017-01-01 00:00:00"), "hsig_m"] == pytest.approx(1.10)
    assert np.isnan(result.loc[ts_aest("2017-01-01 00:30:00"), "hsig_m"])


def test_clean_reindexes_gaps_as_nan_rows(raw_frame, ts_aest):
    # Two timestamps 1.5 hours apart leave two missing 30-min slots between.
    rows = [
        {"Date/Time": pd.Timestamp("2017-01-01 00:00:00"), "Hs": 1.10, "Hmax": 1.9,
         "Tz": 5.5, "Tp": 9.0, "Peak Direction": 95.0, "SST": 25.0},
        {"Date/Time": pd.Timestamp("2017-01-01 01:30:00"), "Hs": 1.20, "Hmax": 2.0,
         "Tz": 5.6, "Tp": 9.1, "Peak Direction": 96.0, "SST": 25.1},
    ]
    result = clean(raw_frame(rows))
    assert len(result) == 4
    assert np.isnan(result.loc[ts_aest("2017-01-01 00:30:00"), "hsig_m"])
    assert np.isnan(result.loc[ts_aest("2017-01-01 01:00:00"), "hsig_m"])


def test_clean_drops_duplicate_timestamps(raw_frame, standard_rows, ts_aest):
    # Duplicate the first row's timestamp with different values
    standard_rows.append({
        "Date/Time": pd.Timestamp("2017-01-01 00:00:00"),
        "Hs": 9.99, "Hmax": 9.99, "Tz": 9.99,
        "Tp": 9.99, "Peak Direction": 9.99, "SST": 9.99,
    })
    result = clean(raw_frame(standard_rows))
    # Only 2 unique timestamps, and the first-seen value wins.
    assert len(result) == 2
    assert result.loc[ts_aest("2017-01-01 00:00:00"), "hsig_m"] == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# unify()
# ---------------------------------------------------------------------------


def test_unify_concatenates_frames_from_fetch_all(raw_frame, standard_rows):
    frame_a = raw_frame(standard_rows)
    frame_b = raw_frame([{
        "Date/Time": pd.Timestamp("2018-01-01 00:00:00"),
        "Hs": 1.20, "Hmax": 2.00, "Tz": 5.60,
        "Tp": 9.50, "Peak Direction": 100.0, "SST": 26.0,
    }])

    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame_a, frame_b]):
        result = unify()

    assert len(result) == 3
    assert list(result.columns) == list(frame_a.columns)


def test_unify_resets_index(raw_frame, standard_rows):
    frame = raw_frame(standard_rows)
    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame, frame]):
        result = unify()
    assert list(result.index) == list(range(len(result)))


def test_unify_raises_when_no_data_downloaded():
    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[]):
        with pytest.raises(ValueError, match="No data was downloaded"):
            unify()


def test_unify_passes_custom_resource_ids_to_fetch_all(raw_frame, standard_rows):
    custom_ids = {2020: "fake-resource-id"}
    frame = raw_frame(standard_rows)

    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame]) as mock_fetch:
        unify(resource_ids=custom_ids)

    mock_fetch.assert_called_once_with(custom_ids)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_writes_csv_and_creates_parent_dir(tmp_path, raw_frame, standard_rows):
    output = tmp_path / "nested" / "out.csv"
    frame = raw_frame(standard_rows)

    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame]):
        result = run(output_path=output)

    assert output.exists()
    assert len(result) == 2

    reloaded = pd.read_csv(output, parse_dates=["datetime"], index_col="datetime")
    assert set(reloaded.columns) == set(result.columns)
    assert len(reloaded) == len(result)


def test_run_accepts_string_output_path(tmp_path, raw_frame, standard_rows):
    output = str(tmp_path / "out.csv")
    frame = raw_frame(standard_rows)

    with patch("qld_ckan.wave.pipeline.fetch_all", return_value=[frame]):
        run(output_path=output)

    assert (tmp_path / "out.csv").exists()
